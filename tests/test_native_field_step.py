"""Untrained structure / prescribed-weight tests, not closed-loop validation."""
from copy import deepcopy
from pathlib import Path
import math

import h5py
import numpy as np
import pytest
import torch

from src.models.native_state import NativeUnits
from src.models.stencil_conv import build_fem_stencil_geometry
from stencil_fixtures import structured_mesh


def make_step(*, n=3, **kwargs):
    from src.models.native_field_step import NativeFieldStep, NativeUpdateMapping
    vertex,pos=structured_mesh(n,n)
    geometry=build_fem_stencil_geometry(vertex,pos)
    indices=np.arange(1,4*len(pos)+1).reshape(4,len(pos))
    mapping=NativeUpdateMapping(indices, constrained=np.zeros(len(pos),bool),
        treat_axis=False,keep_n0_const=False)
    return NativeFieldStep(geometry,mapping,units=NativeUnits(1.,8e19,1.,5/3),
        state_scale=np.ones((9,4)),increment_scale=np.ones((9,4)),
        hidden=8,n_layers=1,n_iters=2,**kwargs)


def test_all_degrees_are_jointly_updated_and_differentiable():
    torch.manual_seed(42)
    model=make_step()
    u=torch.linspace(-.2,.3,9*4*9,dtype=torch.float64).reshape(9,4,1,9).requires_grad_()
    history=torch.full_like(u,.125,requires_grad=True)
    out=model(u,history,dt_si=.5,previous_dt_si=1.,scheme='Gears')
    assert out.values.shape == out.fluid_deltas.shape == out.increment.shape == u.shape
    assert out.increment.requires_grad and out.fluid_deltas.requires_grad
    cross=torch.autograd.grad(out.increment[8,3].sum(),u,retain_graph=True)[0]
    assert cross[0,1].abs().sum()>0  # psi derivative coefficient affects Te derivative update
    for d in range(4): assert out.increment[:,d].abs().max()>0
    out.increment.square().sum().backward()
    for d in range(4):
        assert u.grad[:,d].abs().sum()>0
        assert history.grad[:,d].abs().sum()>0
    assert model.encoder.weight.grad.abs().sum()>0
    assert model.decoder.weight.grad.abs().sum()>0
    for name,parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum()>0, name
    assert (u.grad[:,:,0].abs().sum(-1)>0).all()
    assert (history.grad[:,:,0].abs().sum(-1)>0).all()
    assert (out.increment[:,:,0].abs().sum(-1)>0).all()
    for d in range(4):
        perturbed=u.detach().clone(); perturbed[:,d]+=.1
        changed=model(perturbed,history.detach(),dt_si=.5,previous_dt_si=1.,scheme='Gears')
        assert not torch.equal(changed.increment,out.increment)


def test_shared_indices_and_frozen_history_follow_distinct_update_rules():
    from src.models.native_field_step import NativeUpdateMapping
    indices=np.arange(1,9).reshape(4,2)
    indices[0,1]=indices[0,0]  # one native degree shared by two node records
    mapping=NativeUpdateMapping(indices,constrained=np.zeros(2,bool),treat_axis=False,keep_n0_const=False)
    u=torch.zeros(9,4,1,2,dtype=torch.float64)
    d=torch.full_like(u,.25)
    proposed=torch.ones_like(u,requires_grad=True)
    proposed=proposed*torch.tensor([1.,3.],dtype=u.dtype)
    out=mapping(u,d,proposed)
    torch.testing.assert_close(out.increment[:,0],torch.full_like(out.increment[:,0],2.))
    torch.testing.assert_close(out.increment[:,1:,0,0],torch.ones_like(out.increment[:,1:,0,0]))
    torch.testing.assert_close(out.increment[:,1:,0,1],torch.full_like(out.increment[:,1:,0,1],3.))
    frozen=NativeUpdateMapping(indices,constrained=np.zeros(2,bool),treat_axis=False,keep_n0_const=True)
    held=frozen(u,d,proposed)
    assert torch.equal(held.values,u) and torch.equal(held.fluid_deltas,d)
    assert not held.increment.any()  # history preserved, NOT zeroed
    # Floating addition cannot be used to recover the true new history.
    indices=np.arange(1,9).reshape(4,2)
    free=NativeUpdateMapping(indices,constrained=np.zeros(2,bool),treat_axis=False,keep_n0_const=False)
    large=torch.full_like(u,1e16)
    rounded=free(large,d,torch.ones_like(u))
    assert torch.equal(rounded.values,large)
    assert torch.equal(rounded.fluid_deltas,torch.ones_like(u))
    for flags in ({'constrained':np.array([True,False]),'treat_axis':False},
                  {'constrained':np.zeros(2,bool),'treat_axis':True}):
        with pytest.raises(ValueError,match='unsupported'):
            NativeUpdateMapping(indices,keep_n0_const=False,**flags)


def test_prescribed_weights_scaling_and_variable_time_have_independent_oracle():
    model=make_step(stencil_strength=0.)
    with torch.no_grad():
        model.state_scale.fill_(2.); model.center.fill_(10.)
        model.increment_scale.copy_(torch.tensor([2.,3.,4.,5.]).reshape(1,4,1,1).expand(9,4,1,1))
        model.encoder.weight.zero_(); model.encoder.bias.zero_()
        model.encoder.weight[0,[0,36,72,75,76]]=1.
        model.decoder.weight.zero_(); model.decoder.bias.zero_(); model.decoder.weight[:,0]=1.
    u=torch.full((9,4,1,9),11.,dtype=torch.float64)
    history=torch.ones_like(u)
    out=model(u,history,dt_si=.5,previous_dt_si=1.,scheme='Gears')
    # U=.5,D=.5,h=.5,theta=1,zeta=1/3 -> rate=tanh(17/6); integrate h=.5.
    for degree,scale in enumerate((2.,3.,4.,5.)):
        torch.testing.assert_close(out.increment[:,degree],torch.full_like(out.increment[:,degree],scale*.5*math.tanh(17/6)),rtol=2e-14,atol=1e-14)
    changed=model(u,history,dt_si=1.,previous_dt_si=.5,scheme='Gears')
    # .5+.5+1+1+2/3=11/3; reference interval=1, h=1.
    assert changed.increment[0,0,0,0].item() == pytest.approx(2*math.tanh(11/3),rel=2e-14)
    nohistory=model(u,torch.zeros_like(history),dt_si=.5,previous_dt_si=1.,scheme='Gears')
    assert nohistory.increment[0,0,0,0].item() == pytest.approx(math.tanh(7/3),rel=2e-14)
    assert not torch.equal(out.increment,nohistory.increment)
    with pytest.raises(ValueError,match='scheme'):
        model(u,history,dt_si=.5,previous_dt_si=1.,scheme='unknown')


def test_supplied_time_coefficients_must_match_explicit_steps():
    from src.models.marker_state import FluidTimeCoefficients
    model=make_step()
    u=torch.zeros(9,4,1,9,dtype=torch.float64)
    with pytest.raises(ValueError,match='time coefficients'):
        model(u,u,dt_si=.5,previous_dt_si=1.,scheme='Gears',
              time_coefficients=FluidTimeCoefficients(1.,.5,1.))


def reference_fixture(model):
    from src.models.native_state import NativeStateContext
    from src.models.marker_state import ReferenceRollout,MarkerRuntime,MarkerCloud,ProjectionFeedback,MarkerClock
    n=model.geometry.n_nodes
    context=NativeStateContext(model.units,(0,),{
        'mesh':{'vertex':model.geometry.vertex,'pos':model.geometry.pos},
        'boundary':{'policy':'learned_unverified'},'physics':{'feedback':'not_consumed'},
        'numerics':{'scheme':'Gears'},'model':model.checkpoint_configuration})
    u=np.linspace(-.2,.3,9*4*n).reshape(9,4,1,n)
    feedback=np.zeros((5,4,1,n))
    marker=MarkerRuntime(MarkerCloud.empty(),ProjectionFeedback(feedback,feedback.copy(),1.),
                         MarkerClock(1.,.05,.1,1.,2.))
    state=ReferenceRollout(marker,u,u-.1,{'radius':np.ones(1)},2.,7,(.5,1.),
        fluid_dt_previous_si=1.,fluid_deltas=np.full_like(u,.125),time_evol_scheme='Gears')
    return state.with_native_context(context),context


def zero_particle_event(marker,target,previous,current,shards):
    marker.feedback.time_si=marker.clock.sim_time_si=target
    return marker,{}


def test_reference_solver_resume_and_failed_event_leave_committed_state_unchanged(tmp_path):
    from src.models.marker_state import reference_particle_then_fluid,save_model_rollout,load_model_rollout,fluid_time_coefficients
    torch.manual_seed(7)
    model=make_step()
    state,context=reference_fixture(model)
    solver=model.reference_solver(context)
    kwargs=dict(particle_step=zero_particle_event,solve_fluid=solver,
                update_shards=lambda shards,u,t,dt: {'radius':shards['radius']*.5})
    first,_=reference_particle_then_fluid(state,**kwargs)
    file=tmp_path/'learned.pt'; save_model_rollout(file,first,model)
    torch.manual_seed(999)
    other=make_step()
    restored=load_model_rollout(file,other,native_context=context)
    direct,_=reference_particle_then_fluid(first,**kwargs)
    resumed,_=reference_particle_then_fluid(restored,**{**kwargs,'solve_fluid':other.reference_solver(context)})
    assert direct.fields_current.tobytes() == resumed.fields_current.tobytes()
    assert direct.fluid_deltas.tobytes() == resumed.fluid_deltas.tobytes()
    assert direct.plan_cursor == resumed.plan_cursor == 2
    assert not restored.marker.clock.startup_pending
    original=deepcopy(state)
    def broken(shards,u,t,dt):
        shards['radius'][:]=99
        raise RuntimeError('failed shard update')
    with pytest.raises(RuntimeError,match='shard'):
        reference_particle_then_fluid(state,**{**kwargs,'update_shards':broken})
    assert state.fields_current.tobytes() == original.fields_current.tobytes()
    assert state.fluid_deltas.tobytes() == original.fluid_deltas.tobytes()
    assert state.marker.clock == original.marker.clock
    np.testing.assert_array_equal(state.shards['radius'],[1.])
    feedback=deepcopy(state.marker.feedback); feedback.values.fill(1.)
    with pytest.raises(ValueError,match='feedback'):
        solver(state.fields_current,state.shards,feedback,.5,1.,
               fluid_deltas=state.fluid_deltas,time_coefficients=fluid_time_coefficients('Gears',.5,1.))


def test_real_phasea_single_current_frame_forward_backward_without_storage_adapter(monkeypatch):
    from src.data.native_state import read_native_field_state,read_native_update_mapping
    from src.models.native_field_step import NativeFieldStep
    from src.models.native_state import NativeFieldState
    path=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/jorek004112.h5')
    opened=[]; real_file=h5py.File
    def checked_file(p,*args,**kwargs):
        assert Path(p)==path  # no future/previous fields or feedback snapshots
        opened.append(Path(p)); return real_file(p,*args,**kwargs)
    monkeypatch.setattr(h5py,'File',checked_file)
    with h5py.File(path) as f:
        units=NativeUnits(float(f['t_norm'][0]),float(f['central_density'][0])*1e20,
                          float(f['central_mass'][0]),5/3)
        pos=f['x'][:,0,0,:].T; vertex=f['vertex'][()]-1
        previous_dt=float(f['tstep'][0])*units.t_norm_si
    state=read_native_field_state(path,units=units,modes=(0,))
    mapping=read_native_update_mapping(path,keep_n0_const=False,boundary_policy='learned_unverified')
    model=NativeFieldStep(build_fem_stencil_geometry(vertex,pos),mapping,units=units,
        state_scale=np.full((9,4),10.),increment_scale=np.full((9,4),.001),
        hidden=12,n_layers=1,n_iters=2)
    monkeypatch.setattr(NativeFieldState,'to_node_layout',lambda *args,**kw: pytest.fail('storage adapter in learning path'))
    monkeypatch.setattr(torch.Tensor,'numpy',lambda *args,**kw: pytest.fail('NumPy conversion in learning path'))
    u=torch.tensor(state.values,requires_grad=True); d=torch.tensor(state.raw_deltas,requires_grad=True)
    out=model(u,d,dt_si=.05*units.t_norm_si,previous_dt_si=previous_dt,scheme='Crank-Nicholson')
    out.increment.square().mean().backward()
    assert torch.isfinite(out.values).all() and torch.isfinite(u.grad).all() and torch.isfinite(d.grad).all()
    for degree in range(4):
        assert u.grad[:,degree].abs().sum()>0 and d.grad[:,degree].abs().sum()>0
        assert out.increment[:,degree].abs().max()>0
    assert all(p==path for p in opened)


def test_normalization_overflow_and_wrong_context_geometry_are_rejected():
    model=make_step()
    u=torch.full((9,4,1,9),1e300,dtype=torch.float64)
    with torch.no_grad():
        model.state_scale.fill_(1e-300)
        model.encoder.weight.fill_(1.)
    with pytest.raises(ValueError,match='normalization'):
        model(u,torch.ones_like(u),dt_si=.5,previous_dt_si=1.,scheme='Gears')
    model=make_step()
    _,context=reference_fixture(model)
    context.configuration['mesh']['pos'][0,0]+=1.
    with pytest.raises(ValueError,match='geometry'):
        model.reference_solver(context)


@pytest.mark.parametrize('scheme,theta',[('implicit Euler',1.),('Crank-Nicholson',.5),('Gears',1.)])
@pytest.mark.parametrize('h,k',[(.25,1.),(1.,.5),(2.,2.)])
def test_all_time_formats_and_step_ratios_have_known_weight_response(scheme,theta,h,k):
    model=make_step(stencil_strength=0.)
    model.units=NativeUnits(2.,8e19,1.,5/3)
    with torch.no_grad():
        model.encoder.weight.zero_(); model.encoder.bias.zero_()
        model.encoder.weight[0,72:]=torch.tensor([.1,.2,.3,.4,.5],dtype=torch.float64)
        model.decoder.weight.zero_(); model.decoder.bias.zero_(); model.decoder.weight[:,0]=1.
    u=torch.zeros(9,4,1,9,dtype=torch.float64)
    result=model(u,u,dt_si=2*h,previous_dt_si=2*k,scheme=scheme)
    zeta=h/(h+k) if scheme=='Gears' else 0.
    expected=h*math.tanh(.1*h+.2*k+.3*h/k+.4*theta+.5*zeta)
    torch.testing.assert_close(result.increment,torch.full_like(u,expected),rtol=2e-14,atol=1e-14)


@pytest.mark.parametrize('kind',['H2','degree','missing_history','nonfinite','negative_dt'])
def test_unsupported_or_missing_native_inputs_fail(kind):
    model=make_step()
    u=torch.zeros(9,4,1,9,dtype=torch.float64); d=u.clone(); dt=.5
    if kind=='H2': u=u.expand(9,4,2,9)
    elif kind=='degree': u=u[:,:3]
    elif kind=='missing_history': d=None
    elif kind=='nonfinite': d.fill_(float('nan'))
    elif kind=='negative_dt': dt=-.5
    with pytest.raises(ValueError): model(u,d,dt_si=dt,previous_dt_si=1.,scheme='Gears')


def test_unimplemented_boundary_and_jeccd_requests_are_rejected():
    from src.models.native_field_step import NativeUpdateMapping
    model=make_step()
    _,context=reference_fixture(model)
    context.configuration['boundary']['policy']='jorek_physical'
    with pytest.raises(ValueError,match='boundary'):
        model.reference_solver(context)
    with pytest.raises(ValueError,match='JECCD'):
        NativeUpdateMapping(np.arange(1,37).reshape(4,9),constrained=np.zeros(9,bool),
                            treat_axis=False,keep_n0_const=False,jeccd=True)


def test_non_torch_solver_and_mismatched_scaling_checkpoint_are_rejected(tmp_path):
    from src.models.marker_state import save_model_rollout
    model=make_step()
    u=torch.zeros(9,4,1,9,dtype=torch.float64)
    model.stencil_layers[0].solver='direct'
    with pytest.raises(ValueError,match='Richardson'):
        model(u,u,dt_si=.5,previous_dt_si=1.,scheme='Gears')
    model=make_step()
    state,_=reference_fixture(model)
    with torch.no_grad(): model.state_scale.fill_(2.)
    with pytest.raises(ValueError,match='configuration'):
        save_model_rollout(tmp_path/'wrong.pt',state,model)
    assert not (tmp_path/'wrong.pt').exists()
