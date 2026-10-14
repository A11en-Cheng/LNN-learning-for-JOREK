"""Complete source advancement seam: bounded all-slot execution, no training."""
from copy import deepcopy
import numpy as np
import pytest

from src.models.marker_state import MarkerCloud, MarkerRuntime, MarkerClock, ProjectionFeedback, advance_window


def fixture(slots=7, live=5, z=1):
    cloud = MarkerCloud.empty(z, slots)
    cloud.i_elm[:live] = 1
    cloud.x[:] = [2., 0., 0.]
    cloud.weight[:live] = 2.
    cloud.V_prev[:] = [1., 0., 0.]
    cloud.i_life[:] = np.arange(slots) + 11
    return MarkerRuntime(cloud, ProjectionFeedback(np.zeros((1,5)), np.zeros((1,5)), 0., 'fixture'),
                         MarkerClock(0., 0., .1, 0., 1.))


def advance(state, target=.1, **kwargs):
    def sample(t, cloud, ids, feedback):
        n=len(ids)
        z=cloud.P_imp.shape[1]-1
        scd=np.zeros((n,z));scd[:,0]=2.
        acd=np.zeros((n,z));acd[:,0]=.5
        radiation=np.zeros((n,z+1));radiation[:,:2]=[3.,7.]
        return dict(velocity=np.tile([1.,0.,0.],(n,1)), ne_si=np.ones(n),
                    scd=scd, acd=acd, radiation_w_per_atom=radiation)
    return advance_window(state, target, sample=sample,
        locate=lambda x:(np.ones(len(x),dtype=int), np.zeros((len(x),2))),
        deposit=lambda c,i,load:load.sum(0,keepdims=True), project=lambda rhs:rhs,
        potentials_ev=np.full(state.cloud.P_imp.shape[1]-1,10.), t_norm=1., central_density=1., **kwargs)


def budget(**overrides):
    from src.models.marker_state import MarkerExecutionBudget
    return MarkerExecutionBudget(**{**dict(max_slots=1000000, max_live=60000, chunk_size=2,
        max_particle_steps=10, max_seconds=10., max_rss_bytes=2*1024**3), **overrides})


def test_chunked_complete_cloud_preserves_slots_and_hand_computed_charge_energy_moments():
    state=fixture()
    out, diag=advance(state, execution_budget=budget())
    # Independent one-step master equation and old-neutral radiation oracle.
    np.testing.assert_allclose(out.cloud.P_imp[:5], np.tile([.8,.2],(5,1)))
    np.testing.assert_allclose(out.cloud.x[:5,0], 2.1)
    np.testing.assert_allclose(diag['energy'][:5], .4*10*1.602176634e-19)
    np.testing.assert_allclose(diag['rad'][:5], .6)
    np.testing.assert_allclose(diag['moments'][:5], np.tile([0.,0.,.2],(5,1)))
    assert len(out.cloud.weight)==7 and out.cloud.i_life.tolist()==list(range(11,18))
    np.testing.assert_array_equal(state.cloud.x[:,0], 2.)
    direct, reference=advance(state)
    for name in MarkerCloud.array_names():
        np.testing.assert_array_equal(getattr(out.cloud,name),getattr(direct.cloud,name))
    np.testing.assert_allclose(out.feedback.values,direct.feedback.values,rtol=2e-15,atol=0.)
    assert diag['execution']['chunks']==3 and diag['execution']['processed_particle_steps']==5


@pytest.mark.parametrize('live',[11168,52604])
def test_default_guard_and_explicit_complete_cloud_capacity_have_distinct_semantics(live):
    state=fixture(slots=1000000,live=live,z=10)
    with pytest.raises(ValueError,match='live-marker limit'):
        advance(state)
    out,diag=advance(state,execution_budget=budget(chunk_size=1024))
    assert len(out.cloud.weight)==1000000 and out.cloud.alive.sum()==live
    assert diag['execution']['processed_particle_steps']==live
    assert out.cloud.P_imp.shape==(1000000,11)
    np.testing.assert_allclose(out.cloud.P_imp[:live,1], .2)
    np.testing.assert_array_equal(out.cloud.i_life,state.cloud.i_life)
    np.testing.assert_array_equal(out.cloud.x[live:],state.cloud.x[live:])


def test_bounded_failure_preserves_input_and_nonintegral_window_residual():
    state=fixture();original=deepcopy(state)
    with pytest.raises(ValueError,match='slot/live'):
        advance(state,execution_budget=budget(max_live=4,chunk_size=2))
    with pytest.raises(RuntimeError,match='time budget'):
        advance(state,execution_budget=budget(max_seconds=1e-12))
    with pytest.raises((RuntimeError,ValueError),match='RSS|memory'):
        advance(state,execution_budget=budget(max_rss_bytes=1))
    for name in MarkerCloud.array_names():
        np.testing.assert_array_equal(getattr(state.cloud,name),getattr(original.cloud,name))
    first,_=advance(state,target=.25,execution_budget=budget())
    second,_=advance(first,target=.5,execution_budget=budget(chunk_size=3))
    direct,_=advance(state,target=.5)
    assert first.clock.step_rest_time==pytest.approx(.05)
    assert second.clock.sim_time_si==direct.clock.sim_time_si
    assert second.clock.particle_time_si==direct.clock.particle_time_si
    assert second.clock.window_index==2 and direct.clock.window_index==1
    for name in MarkerCloud.array_names():
        np.testing.assert_array_equal(getattr(second.cloud,name),getattr(direct.cloud,name))
    np.testing.assert_allclose(second.feedback.deltas,second.feedback.values-first.feedback.values)
