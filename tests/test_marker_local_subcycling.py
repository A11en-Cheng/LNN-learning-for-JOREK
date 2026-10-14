"""Opt-in numerical transport: charge, old-position loads and history together."""
from copy import deepcopy
import pickle

import numpy as np
import pytest

from src.models.marker_state import advance_window, push_velocity
from src.models.marker_transport import LocalSubcycling, initialize_transport_history, push_variable_direction
from test_marker_state import runtime, cloud


def run(state, policy, *, rate=0., target=.1, births=None, lose=False):
    deposited=[]
    def sample(t,c,ids,feedback):
        n=len(ids)
        return dict(velocity=np.tile([1.,0.,0.],(n,1)),ne_si=np.ones(n),
            scd=np.full((n,1),2.),acd=np.zeros((n,1)),
            radiation_w_per_atom=np.ones((n,2))*3,
            transport_rate_si=np.full(n,rate))
    def deposit(c,ids,load):
        deposited.append(c.x[ids].copy())
        return load.sum(0,keepdims=True)
    result=advance_window(state,target,sample=sample,
        locate=lambda x:(np.full(len(x),-1 if lose else 1),np.zeros((len(x),2))),
        deposit=deposit,project=lambda x:x,potentials_ev=[10.],t_norm=1.,
        central_density=1.,births=births,transport_policy=policy)
    return *result,deposited


def test_subcycling_consumes_each_accepted_step_charge_load_and_old_position():
    policy=LocalSubcycling(min_level=2,max_level=4)
    s=initialize_transport_history(runtime(),policy,previous_dt_si=.1)
    out,d,points=run(s,policy,rate=2.)
    assert len(points)==4
    assert out.cloud.P_imp[0,0]==pytest.approx((1-.025*2)**4)
    assert d['rad'][0]==pytest.approx(3*2*.1)
    assert d['moments'][0,2]==pytest.approx(2*.1)
    assert d['energy'][0]==pytest.approx(out.cloud.P_imp[0,1]*2*10*1.602176634e-19)
    assert points[0][0,0]==2.
    assert out.cloud.x[0,0]==pytest.approx(2.1)
    assert out.transport_history.previous_dt_si[0]==.025
    assert out.transport_history.level[0]==2
    assert s.transport_history.level[0]==0


def test_unflagged_original_path_and_equal_interval_pusher_are_bitwise():
    policy=LocalSubcycling()
    original=runtime()
    a,da,_=run(original,None)
    b,db,_=run(initialize_transport_history(original,policy,previous_dt_si=.1),policy)
    for key in original.cloud.array_names():
        np.testing.assert_array_equal(getattr(a.cloud,key),getattr(b.cloud,key))
    np.testing.assert_array_equal(a.feedback.values,b.feedback.values)
    x=np.array([[2.,0.,0.]])
    old=np.array([[0.,3.,0.]])
    velocity=np.array([[2.,0.,0.]])
    np.testing.assert_array_equal(push_variable_direction(x,old,velocity,.1,np.array([.1])),
        push_velocity(x,old,velocity,.1))
    np.testing.assert_allclose(push_variable_direction(x,old,velocity,.05,np.array([.1])),
        [[2.125,-.025,0.]])


def test_resume_birth_identity_and_missing_or_mismatched_history():
    policy=LocalSubcycling(min_level=2,max_level=4)
    s=runtime();s.cloud=cloud(2);s.cloud.i_elm[1]=-1
    s=initialize_transport_history(s,policy,previous_dt_si=.1)
    first,_,_=run(s,policy,rate=2.)
    restored=pickle.loads(pickle.dumps(first,protocol=5))
    direct,_,_=run(first,policy,target=.2,births=cloud())
    resumed,_,_=run(restored,policy,target=.2,births=cloud())
    for key in first.cloud.array_names():
        np.testing.assert_array_equal(getattr(direct.cloud,key),getattr(resumed.cloud,key))
    assert direct.transport_history.level.tolist()==[2,0]
    assert direct.transport_history.previous_dt_si.tolist()==[.025,.1]
    np.testing.assert_array_equal(direct.transport_history.lifetime,direct.cloud.i_life)
    with pytest.raises(ValueError,match='history'):
        run(runtime(),policy)
    with pytest.raises(ValueError,match='history'):
        run(first,None)
    with pytest.raises(ValueError,match='configuration'):
        run(first,LocalSubcycling(min_level=3,max_level=4))
    with pytest.raises(ValueError,match='already'):
        initialize_transport_history(first,policy,previous_dt_si=.1)


def test_lost_particle_deposits_only_accepted_fine_interval_and_limits_rollback():
    policy=LocalSubcycling(min_level=2,max_level=4)
    s=initialize_transport_history(runtime(),policy,previous_dt_si=.1)
    lost,d,points=run(s,policy,rate=2.,lose=True)
    assert len(points)==1
    assert not lost.cloud.alive.any()
    assert d['moments'][0,2]==pytest.approx(2*.025)
    before=deepcopy(s)
    with pytest.raises(ValueError,match='refinement limit'):
        run(s,policy,rate=1e9)
    np.testing.assert_array_equal(s.cloud.x,before.cloud.x)
    np.testing.assert_array_equal(s.transport_history.level,before.transport_history.level)


@pytest.mark.parametrize('corrupt',['integer_dt','intermediate_level','missing_import'])
def test_corrupt_history_fails_before_advancing(corrupt):
    policy=LocalSubcycling()
    s=initialize_transport_history(runtime(),policy,previous_dt_si=.1)
    if corrupt=='integer_dt':s.transport_history.previous_dt_si=np.ones(1,dtype=int)
    if corrupt=='intermediate_level':s.transport_history.level[:]=1
    if corrupt=='missing_import':s.transport_history.initialization={}
    with pytest.raises(ValueError,match='history'):
        run(s,policy)


def test_refined_work_budget_and_sampling_past_available_history_fail_atomically():
    policy=LocalSubcycling(min_level=2,max_level=4,max_refined_particle_steps=1)
    s=initialize_transport_history(runtime(),policy,previous_dt_si=.1)
    with pytest.raises(ValueError,match='work budget'):
        run(s,policy,rate=2.)
    assert s.cloud.x[0,0]==2. and s.transport_history.level[0]==0
    policy=LocalSubcycling(min_level=2,max_level=4)
    s=initialize_transport_history(runtime(),policy,previous_dt_si=.1)
    s.clock.field_end_si=.06
    with pytest.raises(ValueError,match='available field history'):
        run(s,policy,rate=2.)
    assert s.cloud.x[0,0]==2. and s.cloud.P_imp[0,0]==1.


def test_model_checkpoint_retains_transport_history_and_rejects_native_restart(tmp_path):
    import torch
    from src.models.source_module import AugmentedState
    from src.models.marker_state import save_model_rollout,load_model_rollout
    policy=LocalSubcycling(min_level=2,max_level=4)
    state=initialize_transport_history(runtime(),policy,previous_dt_si=.1)
    first,_,_=run(state,policy,rate=2.)
    model=torch.nn.Linear(1,1)
    augmented=AugmentedState(torch.ones(1,1,9),torch.ones(1,1,16),torch.ones(1,1,dtype=torch.bool),
        torch.tensor([[.7]]),1,.1,first,torch.zeros(1,1,9),0.)
    path=tmp_path/'transport.pt';save_model_rollout(path,augmented,model)
    restored=load_model_rollout(path,model)
    direct,_,_=run(first,policy,target=.2)
    resumed,_,_=run(restored.marker_state,policy,target=.2)
    from tools.diagnose_native_real_chain import _array_digest
    assert _array_digest(direct)==_array_digest(resumed)
    with pytest.raises(ValueError,match='restart'):
        first.jorek_restart()
    corrupted=deepcopy(augmented);corrupted.marker_state.transport_history.previous_dt_si[0]=0
    with pytest.raises(ValueError,match='history'):
        save_model_rollout(tmp_path/'corrupt.pt',corrupted,model)


def test_native_source_public_seam_triggers_from_raw_gradient_and_projects_fine_loads():
    from tests.test_native_coupling import native_fixture
    from tests.test_native_gauss import native_coefficients
    from src.models.native_coupling import NativeMarkerSource
    mesh,native,s,u=native_fixture()
    # Steep linear density at the floor; raw gradients remain unclipped.
    slope=.001
    u[7,:,0,:]=native_coefficients(lambda r,z:2.5e-7+slope*(r-.3),
        lambda r,z:slope,lambda r,z:0.,lambda r,z:0.)
    policy=LocalSubcycling(min_level=2,max_level=8)
    source=NativeMarkerSource(mesh,atomic=native.atomic,atomic_version='fixture_prescribed_SCD_m3_s',
        potentials_ev=native.potentials_ev,F0=native.F0,diffusion_ju=5e-6,
        spatial_backend='batched_v1',transport_policy=policy)
    s=initialize_transport_history(s,policy,previous_dt_si=s.clock.particle_dt_si)
    result,d=source.advance(s,1e-9,u,u)
    assert result.transport_history.level[0]>=2
    assert d['execution']['refined_particle_steps']>=4
    assert mesh.integrate(result.feedback.values)[4]==pytest.approx(1e12/8e19,rel=3e-12)
    assert d['gather_time_seconds']==1e-9
    assert (result.cloud.P_imp[0]>=0).all()
    assert result.cloud.P_imp[0].sum()==pytest.approx(1.)
    with pytest.raises(ValueError,match='history'):
        source.advance(native_fixture()[2],1e-9,u,u)
    source.transport_policy=LocalSubcycling(min_level=3,max_level=8)
    with pytest.raises(ValueError,match='configuration'):
        source.advance(s,1e-9,u,u)


@pytest.mark.parametrize('boundary_exit',[False,True])
def test_full_coupled_torch_events_keep_refined_history_bptt_and_resume(tmp_path,boundary_exit):
    import torch
    from tests.test_base_coupled_training import event_fixture
    from tests.test_native_gauss import native_coefficients
    from src.data.base_coupled import StepControl
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from src.models.native_coupling import NativeMarkerSource
    from src.models.marker_state import save_model_rollout,load_model_rollout
    from tools.diagnose_native_real_chain import _array_digest
    model,old,shard,state=event_fixture(2)
    policy=LocalSubcycling(min_level=2,max_level=8,boundary_exit=boundary_exit)
    source=NativeMarkerSource(old.mesh,atomic=old.atomic,atomic_version=old.configuration['atomic'],
        potentials_ev=old.potentials_ev,F0=old.F0,diffusion_ju=5e-6,
        spatial_backend='batched_v1',transport_policy=policy,
        saved_neighbours=(np.zeros_like(old.mesh.vertex) if boundary_exit else None))
    model.source_configuration=deepcopy(source.configuration)
    state.native_context.configuration['model']=model.checkpoint_configuration
    u=state.fields_current
    u[7,:,0,:]=native_coefficients(lambda r,z:2.5e-7+.001*(r-.3),
        lambda r,z:.001,lambda r,z:0.,lambda r,z:0.)
    state.fields_previous=u.copy()
    c=state.marker.cloud;c.i_elm[0]=1;c.st[0]=[.3,.4];c.x[0]=[2.3,.4,0.]
    c.weight[0]=1e12;c.i_life[0]=1;c.V_prev[0]=[0.,0.,-1.]
    state.marker=initialize_transport_history(state.marker,policy,previous_dt_si=1e-9)
    adapter=TorchRecomputedSourceEvents(model,source,shard)
    first,_=adapter.advance(adapter.initialize(state),StepControl(1e-9,1e-9,2e-9,'external_control','fixture'))
    assert first.runtime.marker.transport_history.level[0]>=2
    first.values.retain_grad()
    path=tmp_path/'coupled.pt';save_model_rollout(path,first.runtime,model)
    control=StepControl(2e-9,1e-9,3e-9,'external_control','fixture')
    direct,_=adapter.advance(first,control)
    direct.values.square().sum().backward()
    assert first.values.grad is not None and first.values.grad.abs().sum()>0
    # Restoring weights while a graph is live invalidates Torch buffer versions.
    restored=load_model_rollout(path,model,native_context=first.runtime.native_context)
    resumed,_=adapter.advance(adapter.initialize(restored),control)
    assert _array_digest(direct.runtime)==_array_digest(resumed.runtime)
    corrupted=deepcopy(first.runtime);corrupted.marker.transport_history=None
    with pytest.raises(ValueError,match='history'):
        adapter.initialize(corrupted)
