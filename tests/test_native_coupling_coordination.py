"""Public source and complete-event integration, no optimizer updates."""
from copy import deepcopy
import numpy as np
import pytest
from tests.test_native_coupling import native_fixture


def test_source_explicit_execution_budget_is_consumed_and_fingerprinted():
    from src.models.native_coupling import NativeMarkerSource
    from src.models.marker_state import MarkerExecutionBudget
    mesh,source,state,u=native_fixture()
    budget=MarkerExecutionBudget(2,2,1,1,10.,2*1024**3)
    bounded=NativeMarkerSource(mesh,atomic=source.atomic,atomic_version=source.configuration['atomic'],
        potentials_ev=source.potentials_ev,F0=source.F0,diffusion_ju=0.,execution_budget=budget)
    expected,_=source.advance(state,1e-9,u,u)
    actual,diagnostic=bounded.advance(state,1e-9,u,u)
    np.testing.assert_array_equal(actual.cloud.P_imp,expected.cloud.P_imp)
    np.testing.assert_allclose(actual.feedback.values,expected.feedback.values,rtol=1e-12,atol=1e-20)
    assert bounded.configuration['execution_budget']==vars(budget)
    with pytest.raises(ValueError,match='substep'):
        bounded.advance(state,2e-9,u,u)
    assert state.clock.sim_time_si==0


def test_source_locator_skips_remote_curved_candidates_and_rejects_outside():
    from src.models.native_coupling import NativeSourceMesh
    from tests.test_native_gauss import native_coefficients,rectangle_geometry
    original,_,_,_=native_fixture()
    curved=rectangle_geometry().coordinates.copy()
    curved[0,:,0]=native_coefficients(lambda s,t:2+s-.4*s*s,
        lambda s,t:1-.8*s,lambda s,t:0.,lambda s,t:0.)
    remote=rectangle_geometry().coordinates.copy();remote[0,0]+=8
    mesh=NativeSourceMesh(np.concatenate([curved,remote],axis=-1),np.ones((4,4,2)),
        np.arange(8).reshape(2,4).T,np.arange(1,33).reshape(4,8),units=original.units,projection='none')
    positions=np.array([[10.3,.4,0.],[2.264,.4,0.],[30.,.4,0.]])
    elements,st=mesh.locate(positions)
    np.testing.assert_array_equal(elements,[2,1,0])
    np.testing.assert_allclose(st[:2],[[.3,.4],[.3,.4]],atol=1e-10)


def test_production_complete_event_requires_and_atomically_refreshes_diagnostics():
    from tests.test_native_production_events import production_fixture,analytic_equilibrium
    from src.models.native_coupling import advance_coupled_reference
    class Equilibrium:
        configuration={'version':'analytic-equilibrium-fixture'}
        def __call__(self,fields,time_si):return analytic_equilibrium(fields,time_si)
    model,source,shard,state=production_fixture()
    with pytest.raises(ValueError,match='equilibrium provider'):
        advance_coupled_reference(state,model=model,source=source,shard_model=shard)
    first,_=advance_coupled_reference(state,model=model,source=source,shard_model=shard,
        equilibrium_provider=Equilibrium())
    assert first.shards['nonlocal_diagnostic'].time_si==first.fluid_time_si
    assert state.shards['nonlocal_diagnostic'].time_si==state.fluid_time_si
    class Failing(Equilibrium):
        def __call__(self,fields,time_si):raise RuntimeError('new-field ES failed')
    before=deepcopy(np.random.get_state())
    with pytest.raises(RuntimeError,match='new-field ES failed'):
        advance_coupled_reference(first,model=model,source=source,shard_model=shard,
            equilibrium_provider=Failing())
    np.testing.assert_array_equal(before[1],np.random.get_state()[1])
    assert before[2:]==np.random.get_state()[2:]
    assert first.plan_cursor==1
    class Different(Equilibrium):
        configuration={'version':'different-equilibrium'}
    with pytest.raises(ValueError,match='configuration changed'):
        advance_coupled_reference(first,model=model,source=source,shard_model=shard,
            equilibrium_provider=Different())


def test_source_locator_hints_do_not_hide_a_crossed_element():
    mesh,source,state,u=native_fixture()
    elements,st=mesh.locate(np.array([[2.3,.4,0.],[4.,.4,0.]]),
        element_hints=np.array([1,1]),local_hints=np.array([[.1,.1],[.9,.4]]))
    np.testing.assert_array_equal(elements,[1,0])
    np.testing.assert_allclose(st[0],[.3,.4],atol=1e-12)


def test_production_raw_history_births_use_the_same_event_sampler():
    from tests.test_native_production_events import production_fixture,event
    from src.models.native_coupling import NativeMarkerSource
    from src.models.native_state import NativeStateContext
    model,old_source,shard,state=production_fixture()
    source=NativeMarkerSource(old_source.mesh,atomic=old_source.atomic,
        atomic_version=old_source.configuration['atomic'],potentials_ev=old_source.potentials_ev,
        F0=old_source.F0,diffusion_ju=0.,time_sampling='jorek_raw_deltas_v1',field_static=True)
    model.source_configuration=deepcopy(source.configuration)
    config=deepcopy(state.native_context.configuration);config['model']=model.checkpoint_configuration
    state.native_context=NativeStateContext(state.native_context.units,state.native_context.modes,config)
    np.random.seed(1931)
    out,_=event(state,model,source,shard)
    assert out.marker.cloud.alive.sum()==1
    assert out.shards['nonlocal_diagnostic'].time_si==out.fluid_time_si
