"""Production mechanism event adapters: explicit ES, rawD, graph and rollback.

No training or accuracy gate: the analytic equilibrium is an explicit fixture
dependency, not an inferred or certified production equilibrium solver.
"""
from copy import deepcopy
import random
import numpy as np
import pytest
import torch

from tests.test_native_production_events import production_fixture,analytic_equilibrium
from src.training.base_coupled import ReferenceEvents,TorchRecomputedSourceEvents


class AnalyticEquilibrium:
    def __init__(self,version='analytic-fixture-v1'):
        self.configuration=dict(version=version,geometry='rectangle psi=R-2',production_certified=False)

    def __call__(self,fields,time_si):
        return analytic_equilibrium(fields,time_si)


def fixture():
    from src.models.native_coupling import NativeMarkerSource
    model,legacy,shard,state=production_fixture()
    source=NativeMarkerSource(legacy.mesh,atomic=legacy.atomic,atomic_version=legacy.configuration['atomic'],
        potentials_ev=legacy.potentials_ev,F0=legacy.F0,diffusion_ju=0.,
        time_sampling='jorek_raw_deltas_v1',field_static=False)
    model.source_configuration=deepcopy(source.configuration)
    state.native_context.configuration['model']=model.checkpoint_configuration
    state.marker.provenance['fluid_control_origin']='external_control'
    state.fluid_deltas[6,0]=.0002
    return model,source,shard,state


def control(index=0,phase='ordinary'):
    from src.data.base_coupled import StepControl
    return StepControl((index+1)*1e-9,1e-9,(index+2)*1e-9,'external_control','analytic-fixture',phase=phase)


@pytest.mark.parametrize('adapter_type',[ReferenceEvents,TorchRecomputedSourceEvents])
def test_production_missing_equilibrium_fails_before_event(adapter_type):
    model,source,shard,_=production_fixture()
    with pytest.raises(ValueError,match='equilibrium|provider|ES'):
        adapter_type(model,source,shard)


@pytest.mark.parametrize('adapter_type',[ReferenceEvents,TorchRecomputedSourceEvents])
def test_complete_production_event_uses_saved_history_births_and_refreshes_diagnostics(adapter_type):
    from src.models.native_state import configuration_digest
    model,source,shard,runtime=fixture();provider=AnalyticEquilibrium()
    adapter=adapter_type(model,source,shard,equilibrium_provider=provider)
    initial=adapter.initialize(runtime) if isinstance(adapter,TorchRecomputedSourceEvents) else runtime
    np.random.seed(1931)
    result,_=adapter.advance(initial,control())
    out=getattr(result,'runtime',result)
    cloud=out.marker.cloud
    # Independent cylindrical proposal and psi=R-2 velocity at saved sim_time=0.
    proposal=np.random.RandomState(1931).random_sample(7)
    radius=np.sqrt(4+5*proposal[0])
    velocity=np.array([0.,-.0008,.0008*source.F0])/radius/source.mesh.units.t_norm_si
    assert cloud.alive.sum()==1
    np.testing.assert_allclose(cloud.V_prev[0],velocity,rtol=2e-15,atol=1e-15)
    np.testing.assert_allclose(cloud.x[0],[radius,proposal[1]+velocity[1]*1e-9,
        2*np.pi*proposal[2]+velocity[2]*1e-9/radius],rtol=2e-15,atol=1e-15)
    assert cloud.weight[0]==float(np.float32(1e15))
    diagnostic=out.shards['nonlocal_diagnostic']
    assert diagnostic.time_si==out.fluid_time_si==2e-9
    assert diagnostic.field_digest==configuration_digest(out.fields_current)
    assert diagnostic.feedback_digest==configuration_digest(out.marker.feedback.values)
    assert diagnostic.feedback_time_si==out.marker.feedback.time_si==1e-9
    assert out.shards['source_volume_time_si']==out.fluid_time_si
    assert out.shards['equilibrium_provider_digest']==configuration_digest(provider.configuration)
    np.testing.assert_allclose(diagnostic.temperature_ev,2000.,rtol=2e-15)
    np.testing.assert_allclose(diagnostic.electron_density_m3,1.25*source.mesh.units.n_norm_m3,rtol=2e-15)
    assert not runtime.marker.cloud.alive.any() and runtime.shards['nonlocal_diagnostic'].time_si==1e-9
    assert 'equilibrium_provider_digest' not in runtime.shards


def assert_rng_equal(expected):
    assert random.getstate()==expected[0]
    actual=np.random.get_state()
    assert actual[0]==expected[1][0] and actual[2:]==expected[1][2:]
    np.testing.assert_array_equal(actual[1],expected[1][1])
    torch.testing.assert_close(torch.get_rng_state(),expected[2],rtol=0,atol=0)


def rng_state():
    return deepcopy((random.getstate(),np.random.get_state(),torch.get_rng_state()))


def assert_runtime_equal(left,right):
    from src.models.native_state import configuration_digest
    for name in ('fields_current','fields_previous','fluid_deltas','field_memory'):
        np.testing.assert_array_equal(getattr(left,name),getattr(right,name))
    for name in left.marker.cloud.array_names():
        np.testing.assert_array_equal(getattr(left.marker.cloud,name),getattr(right.marker.cloud,name))
    for name in ('values','deltas'):
        np.testing.assert_array_equal(getattr(left.marker.feedback,name),getattr(right.marker.feedback,name))
    assert left.marker.feedback.time_si==right.marker.feedback.time_si
    assert left.marker.feedback.gather_time_si==right.marker.feedback.gather_time_si
    assert left.marker.clock==right.marker.clock
    for name in ('fluid_time_si','fluid_index','fluid_dt_previous_si','plan_cursor','step_plan_si'):
        assert getattr(left,name)==getattr(right,name)
    for name in ('positions','velocities','radius','ablation_atoms_s','source_volume_m3'):
        np.testing.assert_array_equal(left.shards[name],right.shards[name])
    for name in ('equilibrium_provider_digest','source_volume_time_si','source_volume_provenance'):
        assert left.shards.get(name)==right.shards.get(name)
    assert configuration_digest(vars(left.shards['nonlocal_diagnostic'].equilibrium))==configuration_digest(
        vars(right.shards['nonlocal_diagnostic'].equilibrium))
    for name in ('psi','temperature_ev','electron_density_m3','shell_volume_m3'):
        np.testing.assert_array_equal(getattr(left.shards['nonlocal_diagnostic'],name),
                                      getattr(right.shards['nonlocal_diagnostic'],name))
    for name in ('time_si','field_digest','feedback_digest','feedback_time_si','provenance','mesh_digest'):
        assert getattr(left.shards['nonlocal_diagnostic'],name)==getattr(right.shards['nonlocal_diagnostic'],name)


class FailingEquilibrium(AnalyticEquilibrium):
    def __call__(self,fields,time_si):
        random.random();random.gauss(0.,1.)
        np.random.random(5);np.random.normal(size=3)
        torch.rand(3)
        raise RuntimeError('external ES failed after field and shard update')


@pytest.mark.parametrize('adapter_type',[ReferenceEvents,TorchRecomputedSourceEvents])
def test_diagnostic_failure_rolls_back_complete_event_and_all_cpu_rng(adapter_type):
    model,source,shard,runtime=fixture()
    bad=adapter_type(model,source,shard,equilibrium_provider=FailingEquilibrium())
    initial=bad.initialize(runtime) if isinstance(bad,TorchRecomputedSourceEvents) else runtime
    before_runtime=deepcopy(runtime)
    weights={name:value.detach().clone() for name,value in model.state_dict().items()}
    random.seed(71);random.gauss(0.,1.)  # populated Python Gaussian cache
    np.random.seed(1931);np.random.normal()  # populated NumPy Gaussian cache
    before=rng_state()
    with pytest.raises(RuntimeError,match='external ES failed'):
        bad.advance(initial,control())
    assert_rng_equal(before)
    assert_runtime_equal(runtime,before_runtime)
    for name,value in weights.items():torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
    good=adapter_type(model,source,shard,equilibrium_provider=AnalyticEquilibrium())
    recovered,_=good.advance(initial,control())
    after=rng_state()
    random.setstate(before[0]);np.random.set_state(before[1]);torch.set_rng_state(before[2])
    uninterrupted,_=good.advance(initial,control())
    assert_runtime_equal(getattr(recovered,'runtime',recovered),getattr(uninterrupted,'runtime',uninterrupted))
    assert_rng_equal(after)


def test_torch_two_production_events_preserve_field_raw_history_and_memory_graph():
    model,source,shard,runtime=fixture()
    # Deterministic small nonzero decoder tests graph transport, never accuracy.
    with torch.no_grad():model.field.decoder.weight.fill_(1e-4)
    adapter=TorchRecomputedSourceEvents(model,source,shard,equilibrium_provider=AnalyticEquilibrium())
    initial=adapter.initialize(runtime)
    for value in (initial.values,initial.raw_deltas,initial.memory):value.requires_grad_()
    np.random.seed(1931)
    first,_=adapter.advance(initial,control())
    for value in (first.values,first.raw_deltas,first.memory):value.retain_grad()
    second,_=adapter.advance(first,control(1))
    (second.values.square().sum()+second.raw_deltas.square().sum()+second.memory.square().sum()).backward()
    for value in (initial.values,initial.raw_deltas,initial.memory,first.values,first.raw_deltas,first.memory):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert second.runtime.marker.cloud.alive.sum()==2
    assert second.runtime.shards['nonlocal_diagnostic'].time_si==second.runtime.fluid_time_si
    assert second.runtime.fluid_time_si==pytest.approx(3e-9,rel=0,abs=1e-23)


def sequence():
    from src.data.base_coupled import Support,make_sequence
    return make_sequence(run='analytic-fixture',task='self_generated',controls=tuple(control(i) for i in range(3)),
        nodes=(),burnin_steps=0,support=(Support('analytic-fixture',0.,4e-9,'all_history','analytic-fixture'),),
        blocks=(Support('analytic-fixture',0.,4e-9,'train','analytic-fixture'),),
        state_evidence={'complete_state':'saved'},required_state=('complete_state',),
        model_reasons=(),parameters={'ns_radius_min':1.})


@pytest.mark.parametrize('adapter_type',[ReferenceEvents,TorchRecomputedSourceEvents])
def test_production_session_restores_rng_diagnostics_and_rejects_other_es_configuration(tmp_path,adapter_type):
    from src.training.base_coupled import run_sequence,save_session,load_session
    model,source,shard,runtime=fixture();seq=sequence()
    adapter=adapter_type(model,source,shard,equilibrium_provider=AnalyticEquilibrium())
    initial=adapter.initialize(runtime) if isinstance(adapter,TorchRecomputedSourceEvents) else runtime
    np.random.seed(1931)
    first=run_sequence(seq,adapter,initial,stop_cursor=1)
    path=tmp_path/'production-session.pt';save_session(path,seq,adapter,first)
    expected=run_sequence(seq,adapter,session=first);expected_rng=rng_state()
    fresh,other_source,other_shard,_=fixture()
    restored_adapter=adapter_type(fresh,other_source,other_shard,equilibrium_provider=AnalyticEquilibrium())
    loaded=load_session(path,seq,restored_adapter)
    actual=run_sequence(seq,restored_adapter,session=loaded)
    assert_runtime_equal(getattr(actual.state,'runtime',actual.state),getattr(expected.state,'runtime',expected.state))
    assert_rng_equal(expected_rng)
    wrong=adapter_type(fresh,other_source,other_shard,equilibrium_provider=AnalyticEquilibrium('other-ES'))
    assert wrong.configuration_fingerprint!=adapter.configuration_fingerprint
    assert wrong.fingerprint!=adapter.fingerprint
    before=rng_state()
    with pytest.raises(ValueError,match='fingerprint'):load_session(path,seq,wrong)
    assert_rng_equal(before)
    # The shard-level binding also rejects swapping providers outside a session.
    with pytest.raises(ValueError,match='equilibrium provider configuration'):
        wrong.advance(first.state,control(1))
    assert_rng_equal(before)


@pytest.mark.parametrize('adapter_type',[ReferenceEvents,TorchRecomputedSourceEvents])
def test_registered_production_startup_refreshes_without_births(adapter_type):
    model,source,shard,runtime=fixture()
    runtime.marker.clock.sim_time_si=runtime.fluid_time_si
    runtime.marker=runtime.marker.jorek_restart()
    adapter=adapter_type(model,source,shard,equilibrium_provider=AnalyticEquilibrium())
    initial=adapter.initialize(runtime) if isinstance(adapter,TorchRecomputedSourceEvents) else runtime
    before=rng_state()
    result,diagnostic=adapter.advance(initial,control(phase='registered_restart'))
    out=getattr(result,'runtime',result)
    assert diagnostic['event']=='registered_restart_zero_projection_startup'
    assert not out.marker.cloud.alive.any() and not out.marker.feedback.values.any()
    assert out.shards['nonlocal_diagnostic'].time_si==2e-9
    assert out.shards['source_volume_time_si']==2e-9
    assert not out.marker.clock.startup_pending
    assert_rng_equal(before)
