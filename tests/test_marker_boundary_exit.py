"""Verified exterior geometry, never a failed locator interpreted as death."""
from copy import deepcopy
import numpy as np
import pytest

from src.models.marker_transport import LocalSubcycling, initialize_transport_history
from tests.test_native_coupling import native_fixture
from tests.test_native_gauss import native_coefficients


def boundary_source(neighbours=None):
    from src.models.native_coupling import NativeMarkerSource
    mesh, old, s, u = native_fixture()
    u[7,:,0,:] = native_coefficients(lambda a,b:a,
        lambda a,b:1., lambda a,b:0., lambda a,b:0.)
    u[6] = 0
    s.cloud.x[0] = [2.0001,.4,0.]
    s.cloud.st[0] = [.0001,.4]
    s.cloud.V_prev[0] = [-1.,0.,0.]
    policy = LocalSubcycling(boundary_exit=True)
    source = NativeMarkerSource(mesh, atomic=old.atomic, atomic_version=old.configuration['atomic'],
        potentials_ev=old.potentials_ev, F0=old.F0, diffusion_ju=5e-6,
        spatial_backend='batched_v1', transport_policy=policy,
        saved_neighbours=np.zeros((4,1),np.int32) if neighbours is None else neighbours)
    return source, initialize_transport_history(s,policy,previous_dt_si=1e-9), u


def test_public_source_boundary_singularity_exits_with_fractional_last_load():
    source, state, u = boundary_source()
    before = deepcopy(state)
    result, d = source.advance(state,1e-9,u,u)
    assert not result.cloud.alive[0]
    assert result.cloud.i_elm[0] == -1
    assert result.cloud.st[0,0] == 0.
    assert result.cloud.x[0,0] == pytest.approx(2.,abs=1e-12)
    exits = d['boundary_exits']
    assert len(exits) == 1 and exits[0]['slot'] == 0 and exits[0]['edge'] == 3
    assert 0 < exits[0]['accepted_dt_si'] < exits[0]['proposed_dt_si']
    alive_dt = exits[0]['time_si']
    assert d['moments'][0,2] == pytest.approx(state.cloud.weight[0]*alive_dt,rel=1e-12)
    assert source.mesh.integrate(result.feedback.values)[4] == pytest.approx(
        state.cloud.weight[0]*alive_dt/(8e19*1e-9),rel=2e-12)
    assert result.transport_history.previous_dt_si[0] == exits[0]['accepted_dt_si']
    assert result.clock.sim_time_si == 1e-9 and d['gather_time_seconds'] == 1e-9
    assert d['execution']['boundary_exits'] == 1
    np.testing.assert_array_equal(state.cloud.x,before.cloud.x)
    np.testing.assert_array_equal(state.cloud.P_imp,before.cloud.P_imp)


def test_unknown_neighbour_is_not_a_verified_exterior_or_a_limit_waiver():
    source, state, u = boundary_source(np.full((4,1),-1,np.int32))
    with pytest.raises(ValueError,match='refinement limit'):
        source.advance(state,1e-9,u,u)
    assert state.cloud.alive[0] and state.transport_history.level[0] == 0


def test_explicit_topology_required_and_old_history_is_not_silently_upgraded():
    from src.models.native_coupling import NativeMarkerSource
    mesh, old, state, u = native_fixture()
    policy = LocalSubcycling(boundary_exit=True)
    with pytest.raises(ValueError,match='neighbours'):
        NativeMarkerSource(mesh,atomic=old.atomic,atomic_version=old.configuration['atomic'],
            potentials_ev=old.potentials_ev,F0=old.F0,diffusion_ju=0.,
            spatial_backend='batched_v1',transport_policy=policy)
    old_state = initialize_transport_history(state,LocalSubcycling(),previous_dt_si=1e-9)
    source, _, u = boundary_source()
    with pytest.raises(ValueError,match='configuration'):
        source.advance(old_state,1e-9,u,u)


def test_independent_fractional_charge_radiation_old_moments_and_resume():
    import pickle
    from tests.test_marker_state import runtime
    from src.models.marker_state import advance_window
    from src.models.marker_boundary_exit import NativeMarkerBoundaryExit
    mesh, _, _, _ = native_fixture()
    provider = NativeMarkerBoundaryExit(mesh,np.zeros((4,1),np.int32))
    policy = LocalSubcycling(boundary_exit=True,min_level=2,max_level=4)
    s=runtime();s.cloud.x[0]=[2.97,.4,0.];s.cloud.st[0]=[.97,.4]
    s.cloud.V_prev[0]=[1.,0.,0.]
    s=initialize_transport_history(s,policy,previous_dt_si=.1)
    positions=[]
    def sample(t,c,ids,feedback):
        return dict(velocity=np.tile([1.,0.,0.],(len(ids),1)),ne_si=np.ones(len(ids)),
            scd=np.full((len(ids),1),2.),acd=np.zeros((len(ids),1)),
            radiation_w_per_atom=np.full((len(ids),2),3.),transport_rate_si=np.full(len(ids),2.))
    def deposit(c,ids,load):
        positions.extend(c.x[ids].copy());return load.sum(0,keepdims=True)
    kwargs=dict(sample=sample,locate=mesh.locate_batch,deposit=deposit,project=lambda x:x,
        potentials_ev=[10.],t_norm=1.,central_density=1.,transport_policy=policy,
        boundary_exit=lambda c,ids,v,h,hp:provider.exits(mesh,c,ids,v,h,hp))
    out,d=advance_window(s,.1,**kwargs)
    np.testing.assert_allclose(positions,[[2.97,.4,0.],[2.995,.4,0.]],atol=1e-14)
    np.testing.assert_allclose(out.cloud.P_imp[0],[.95*.99,1-.95*.99],atol=2e-15)
    assert d['rad'][0]==pytest.approx(3*2*.03,rel=1e-13)
    np.testing.assert_allclose(d['moments'][0],[.05*2*.005,.05*2*.005,2*.03],atol=2e-15)
    assert d['energy'][0]==pytest.approx((1-.95*.99)*2*10*1.602176634e-19,rel=1e-13)
    assert d['gather_time_seconds']==.1
    assert d['boundary_exits'][0]['accepted_dt_si']==pytest.approx(.005,abs=1e-15)
    assert out.cloud.i_elm[0]==-1 and out.cloud.st[0,0]==1.
    out.clock.field_start_si=.1;out.clock.field_end_si=.2
    restored=pickle.loads(pickle.dumps(out))
    continuous,dc=advance_window(out,.2,**kwargs)
    resumed,dr=advance_window(restored,.2,**kwargs)
    from tools.diagnose_native_real_chain import _array_digest
    from tools.benchmark_native_cpu_threads import stable_diagnostics
    assert _array_digest(continuous)==_array_digest(resumed)
    assert _array_digest(stable_diagnostics(dc))==_array_digest(stable_diagnostics(dr))
    assert not dc['moments'].any() and not dc['energy'].any() and dc['boundary_exits']==[]


def test_locator_failure_without_geometry_exit_does_not_kill_or_commit():
    from tests.test_marker_state import runtime
    from src.models.marker_state import advance_window
    from src.models.marker_boundary_exit import NativeMarkerBoundaryExit
    from tools.diagnose_native_real_chain import _array_digest
    mesh, _, _, _=native_fixture()
    provider=NativeMarkerBoundaryExit(mesh,np.zeros((4,1),np.int32))
    policy=LocalSubcycling(boundary_exit=True,min_level=2,max_level=4)
    s=runtime();s.cloud.x[0]=[2.3,.4,0.];s.cloud.st[0]=[.3,.4];s.cloud.V_prev[0]=[1.,0.,0.]
    s=initialize_transport_history(s,policy,previous_dt_si=.1);before=_array_digest(s)
    def sample(t,c,ids,f):
        return dict(velocity=np.tile([1.,0.,0.],(len(ids),1)),ne_si=np.ones(len(ids)),
            scd=np.zeros((len(ids),1)),acd=np.zeros((len(ids),1)),
            radiation_w_per_atom=np.zeros((len(ids),2)),transport_rate_si=np.zeros(len(ids)))
    with pytest.raises(ValueError,match='unverified marker loss'):
        advance_window(s,.1,sample=sample,locate=lambda x:(np.zeros(len(x),int),np.zeros((len(x),2))),
            deposit=lambda c,ids,load:load.sum(0,keepdims=True),project=lambda x:x,potentials_ev=[10.],
            t_norm=1.,central_density=1.,transport_policy=policy,
            boundary_exit=lambda c,ids,v,h,hp:provider.exits(mesh,c,ids,v,h,hp))
    assert _array_digest(s)==before
