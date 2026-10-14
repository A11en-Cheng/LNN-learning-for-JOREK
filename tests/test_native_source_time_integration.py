"""Source and complete-event seams consume saved rawD, not rounded endpoints."""
from copy import deepcopy
import numpy as np
import pytest
from tests.test_native_coupling import native_fixture
from src.models.native_source_time import NativeSourceTimeSampler
from src.models.native_coupling import NativeMarkerSource


def configured(source):
    return NativeMarkerSource(source.mesh,atomic=source.atomic,atomic_version=source.configuration['atomic'],
        potentials_ev=source.potentials_ev,F0=source.F0,diffusion_ju=source.diffusion_ju,
        time_sampling='jorek_raw_deltas_v1',field_static=False)


def test_saved_rawD_changes_actual_source_even_when_field_endpoints_equal():
    _,legacy,state,u=native_fixture();source=configured(legacy)
    raw=np.zeros_like(u);raw[4,0]=.25
    sampler=NativeSourceTimeSampler(u,raw,time_prev_si=0.,time_now_si=1e-9,tstep_si=1e-9,static=False)
    result,diag=source.advance(state,1e-9,u,u,time_sampler=sampler)
    # sample at t=0 uses rho=.75, rho_imp=.5 => ne=2e19, SCD=1e-12, dt=1e-9.
    assert result.cloud.P_imp[0,1]==pytest.approx(.02,abs=1e-14)
    assert source.configuration!=legacy.configuration
    with pytest.raises(ValueError,match='sampler'):source.advance(state,1e-9,u,u)
    with pytest.raises(ValueError,match='mode|sampler'):legacy.advance(state,1e-9,u,u,time_sampler=sampler)
    stale=NativeSourceTimeSampler(u,raw,time_prev_si=0.,time_now_si=2e-9,tstep_si=1e-9,static=False)
    with pytest.raises(ValueError,match='bound|clock'):source.advance(state,1e-9,u,u,time_sampler=stale)
    np.testing.assert_array_equal(state.cloud.P_imp[0,:2],[1,0])


def test_complete_reference_and_torch_adapters_bind_raw_history_and_completed_step():
    from tests.test_base_coupled_training import event_fixture,sequence
    from src.training.base_coupled import ReferenceEvents,TorchRecomputedSourceEvents
    for torch_path in (False,True):
        model,source,shard,state=event_fixture(n=2)
        source=configured(source);model.source_configuration=deepcopy(source.configuration)
        state.native_context.configuration['model']=model.checkpoint_configuration
        seen=[];advance=source.advance
        def record(m,target,previous,current,**kwargs):
            s=kwargs['time_sampler'];seen.append(s)
            np.testing.assert_array_equal(s.raw_deltas,state.fluid_deltas)
            assert s.tstep_si==state.fluid_dt_previous_si
            return advance(m,target,previous,current,**kwargs)
        source.advance=record
        adapter=(TorchRecomputedSourceEvents if torch_path else ReferenceEvents)(model,source,shard)
        initial=adapter.initialize(state) if torch_path else state
        result,_=adapter.advance(initial,sequence(n=2,burnin=0).controls[0])
        assert len(seen)==1


def test_inspected_static_live_branch_uses_current_field_with_nonzero_saved_history():
    _,legacy,state,u=native_fixture()
    source=NativeMarkerSource(legacy.mesh,atomic=legacy.atomic,atomic_version=legacy.configuration['atomic'],
        potentials_ev=legacy.potentials_ev,F0=legacy.F0,diffusion_ju=legacy.diffusion_ju,
        time_sampling='jorek_raw_deltas_v1',field_static=True)
    raw=np.zeros_like(u);raw[4,0]=.25
    sampler=NativeSourceTimeSampler(u,raw,time_prev_si=0.,time_now_si=1e-9,tstep_si=1e-9,static=True)
    result,_=source.advance(state,1e-9,u,u,time_sampler=sampler)
    assert result.cloud.P_imp[0,1]==pytest.approx(.04,abs=1e-14)
    np.testing.assert_array_equal(sampler.fields_at(0.),u)


def test_saved_history_mode_requires_explicit_static_branch_selection():
    _,legacy,_,_=native_fixture()
    with pytest.raises(ValueError,match='static|sampling'):
        NativeMarkerSource(legacy.mesh,atomic=legacy.atomic,atomic_version=legacy.configuration['atomic'],
            potentials_ev=legacy.potentials_ev,F0=legacy.F0,diffusion_ju=legacy.diffusion_ju,
            time_sampling='jorek_raw_deltas_v1')
