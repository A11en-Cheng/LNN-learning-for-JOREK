"""Small native source/field integration; never full JOREK or production-cloud validation."""
from copy import deepcopy
import numpy as np
import pytest
from tests.test_native_gauss import rectangle_geometry
from src.models.native_state import NativeUnits
from src.models.marker_feedback import MU_ZERO,ATOMIC_MASS_UNIT
from src.models.marker_state import MarkerCloud,MarkerRuntime,ProjectionFeedback,MarkerClock


def native_fixture():
    from src.models.native_coupling import NativeSourceMesh,NativeMarkerSource
    geometry=rectangle_geometry()
    units=NativeUnits(np.sqrt(MU_ZERO*ATOMIC_MASS_UNIT*8e19),8e19,1.,5/3)
    mesh=NativeSourceMesh(geometry.coordinates,geometry.size[:,:,None],np.arange(4)[:,None],
        np.arange(1,17).reshape(4,4),units=units)
    # A deliberately prescribed atomic reference, not an ADAS production claim.
    def atomic(te_ev,ne):
        n=len(ne)
        scd=np.zeros((n,10));scd[:,0]=1e-12
        return scd,np.zeros_like(scd),np.zeros((n,11))
    source=NativeMarkerSource(mesh,atomic=atomic,atomic_version='fixture_prescribed_SCD_m3_s',
        potentials_ev=np.full(10,20.),F0=-1.,diffusion_ju=0.)
    cloud=MarkerCloud.empty(10,2);cloud.i_elm[0]=1;cloud.st[0]=[.3,.4]
    cloud.x[0]=[2.3,.4,0.];cloud.weight[0]=1e12;cloud.i_life[0]=4
    cloud.V_prev[0]=[0.,0.,-1e-3/2.3/units.t_norm_si]
    runtime=MarkerRuntime(cloud,ProjectionFeedback(np.zeros((5,4,1,4)),np.zeros((5,4,1,4)),0.),
        MarkerClock(0.,0.,1e-9,0.,1e-9))
    u=np.zeros((9,4,1,4));u[4,0]=1.;u[7,0]=.5;u[8,0]=.01;u[6,0]=1e-3
    return mesh,source,runtime,u


def test_native_source_evolves_persistent_cloud_and_projects_all_degrees():
    mesh,source,state,u=native_fixture();before=deepcopy(state)
    result,diagnostic=source.advance(state,1e-9,u,u)
    # Neutral -> ion fraction SCD * n_bg * dt = 1e-12 * 4e19 * 1e-9 = .04.
    np.testing.assert_allclose(result.cloud.P_imp[0,:2],[.96,.04],rtol=0,atol=1e-14)
    assert result.feedback.values.shape==(5,4,1,4)
    assert np.any(result.feedback.values[:,1:])
    np.testing.assert_array_equal(result.feedback.deltas,result.feedback.values-before.feedback.values)
    # Consistent projection preserves total normalized number: w/n_norm.
    integral=mesh.integrate(result.feedback.values)[4]
    assert integral==pytest.approx(1e12/8e19,rel=2e-12)
    assert diagnostic['energy'][0]==pytest.approx(.04*20*1e12*1.602176565e-19,rel=1e-13)
    assert result.cloud.i_life[0]==4 and result.cloud.weight[0]==1e12
    assert state.clock.sim_time_si==0 and state.cloud.P_imp[0,0]==1
    result.clock.field_start_si=1e-9;result.clock.field_end_si=2e-9
    second,_=source.advance(result,2e-9,u,u)  # no new births; cloud persists
    assert second.cloud.P_imp[0,1]>.04 and second.cloud.alive.sum()==1
    shifted=deepcopy(state);shifted.cloud.st[0]=[.7,.4];shifted.cloud.x[0,0]=2.7
    other,_=source.advance(shifted,1e-9,u,u)
    assert not np.array_equal(other.feedback.values,result.feedback.values)


def test_source_birth_slots_missing_history_and_source_failure_are_atomic():
    mesh,source,state,u=native_fixture()
    born=MarkerCloud.empty(10,1);born.i_elm[:]=1;born.st[:]=[.6,.6]
    born.x[:]=[2.6,.6,0.];born.weight[:]=2e12;born.V_prev[:]=[0,0,-1.]
    candidate,_=source.advance(state,1e-9,u,u,births=born)
    assert candidate.cloud.alive.sum()==2
    assert candidate.cloud.i_life[0]==4 and candidate.cloud.i_life[1]==1
    assert candidate.cloud.P_imp[0,1]>.0 and candidate.cloud.P_imp[1,1]>.0
    assert state.cloud.alive.sum()==1 and state.cloud.i_life[1]==0
    invalid=deepcopy(state);invalid.feedback.deltas=None
    with pytest.raises(ValueError,match='coefficients/history'):source.advance(invalid,1e-9,u,u)
    wrong_time=deepcopy(state);wrong_time.feedback.time_si=1.
    with pytest.raises(ValueError,match='event time'):
        source.advance(wrong_time,1e-9,u,u)
    failed=deepcopy(source)
    def bad_atomic(te,ne):raise RuntimeError('atomic failure')
    failed.atomic=bad_atomic
    with pytest.raises(RuntimeError,match='atomic failure'):failed.advance(state,1e-9,u,u,births=born)
    assert state.cloud.alive.sum()==1 and state.clock.sim_time_si==0
    assert not state.feedback.values.any()
    restarted=state.jorek_restart()
    with pytest.raises(ValueError,match='startup'):source.advance(restarted,1e-9,u,u)


def test_native_point_sampler_higher_degrees_and_shared_projection_have_analytic_oracle():
    from tests.test_native_gauss import native_coefficients
    mesh,source,state,u=native_fixture()
    # f=R+2Z+RZ; at (2.3,.4), f=4.02 and gradient=(1.4,4.3).
    a=native_coefficients(lambda s,t:2+s+2*t+(2+s)*t,
        lambda s,t:1+t,lambda s,t:4+s,lambda s,t:1.)[None,:,None,:]
    val,grad,*_=mesh.point(0,[.3,.4],a)
    np.testing.assert_allclose(val,[4.02],rtol=0,atol=2e-14)
    np.testing.assert_allclose(grad,[[1.4,4.3]],rtol=0,atol=2e-14)
    np.testing.assert_allclose(mesh.integrate(a),[12.],rtol=0,atol=3e-14)
    # Integral independently: int_2^3 int_0^1 R*(R+2Z+RZ) dZ dR
    # = 1.5*int R^2 + int R = 9.5+2.5=12.


def test_real_adas_provider_units_and_no_surrogate_fallback():
    from src.models.native_coupling import NativeNeonAtomicRates
    from src.models.source_module import load_neon_tables
    tables=load_neon_tables();provider=NativeNeonAtomicRates(tables)
    # Exact table knot; cm3/s -> m3/s factor is independently 1e-6.
    i,j=2,3
    ne=10.**tables.scd[0][i]*1e6;te=10.**tables.scd[1][j]
    ion,recomb,rad=provider(np.array([te]),np.array([ne]))
    np.testing.assert_allclose(ion[0],10.**tables.scd[2][:,j,i]*1e-6,rtol=2e-14)
    assert ion.shape==recomb.shape==(1,10) and rad.shape==(1,11)
    assert (rad>=0).all() and rad[0,10]==0
    bad=deepcopy(tables);object.__setattr__(bad,'source','surrogate')
    with pytest.raises(ValueError,match='no surrogate'):NativeNeonAtomicRates(bad)


def test_torch_feedback_loads_have_independent_energy_history_and_charge_oracle():
    import torch
    from src.models.native_coupling import TorchMarkerLoads
    from src.models.marker_feedback import MarkerFeedbackParameters
    from src.models.marker_state import fluid_time_coefficients
    mesh,source,state,u=native_fixture()
    consumer=TorchMarkerLoads(mesh,parameters=MarkerFeedbackParameters(.8,1.,5/3,'Ne',10,
        'element_matrix',True,True,True),temperature_floor=1e-8)
    u=torch.tensor(u,requires_grad=True)
    p=torch.zeros((5,4,1,4),dtype=u.dtype);p[:,0,0]=torch.tensor([2.,3.,4.,2.,1.])[:,None]
    p.requires_grad_();raw=torch.zeros_like(p);raw[3,0,0]=.1;raw.requires_grad_()
    dt=mesh.units.t_norm_si*.2;time=fluid_time_coefficients('Gears',dt,dt*2)
    out=consumer(u,p,raw,time=time,dt_si=dt)
    # Integral R on this rectangle is 2.5; direct=-5*.2 -> integrated=-2.5.
    const=out.direct.new_zeros(mesh.n_dofs);const[mesh.inverse[0]]=1.
    assert float((out.direct@const).detach())==pytest.approx(-2.5,rel=1e-13)
    ratio=1/20*(2/2.01455279245)
    assert float((out.history@const).detach())==pytest.approx(-ratio*.05*.01*.5*2.5,rel=1e-13)
    torch.testing.assert_close(out.alpha_e,torch.full_like(out.alpha_e,ratio*2-1))
    torch.testing.assert_close(out.ne,torch.full_like(out.ne,1+(ratio*2-1)*.5))
    # Charge-square channel only changes Zeff here, not the two direct loads.
    changed=p.detach().clone();changed[2,0]*=2
    other=consumer(u,changed,raw,time=time,dt_si=dt)
    assert torch.equal(other.direct,out.direct) and torch.equal(other.history,out.history)
    assert not torch.equal(other.zeff,out.zeff)
    (out.direct.square().sum()+out.history.square().sum()+out.ne.square().sum()+out.zeff.square().sum()).backward()
    assert p.grad.abs().sum()>0 and raw.grad.abs().sum()>0 and u.grad.abs().sum()>0


def test_coupled_field_consumes_energy_and_independent_density_sources_with_torch_gradients():
    import torch
    from src.models.native_coupling import SmallNativeResponse,NativeDensitySource,CoupledNativeField
    from src.models.marker_feedback import MarkerFeedbackParameters
    from tests.test_native_temporal import gp_temporal_fixture
    base,params,window=gp_temporal_fixture()
    mesh,source,state,u=native_fixture()
    base.units=mesh.units
    from src.models.native_features import TorchNativeGauss
    base.feature_path.evaluator=TorchNativeGauss(mesh.coordinates,mesh.size,mesh.vertex,units=mesh.units)
    response=SmallNativeResponse(mesh,parameters=MarkerFeedbackParameters(.8,1.,5/3,'Ne',10,
        'element_matrix',True,True,True),temperature_floor=1e-8)
    model=CoupledNativeField(base,response,mode='frozen99_reference',source_configuration=source.configuration,
        shard_configuration={'model':'test_explicit_native_density'})
    with torch.no_grad():base.decoder.weight.zero_();base.decoder.bias.zero_()
    u=torch.tensor(u,requires_grad=True);raw=torch.zeros_like(u)
    memory=torch.zeros(4,8,dtype=u.dtype,requires_grad=True)
    p=torch.zeros((5,4,1,4),dtype=u.dtype);p[0,0]=2;p.requires_grad_()
    pd=torch.zeros_like(p);pd.requires_grad_()
    density=torch.zeros((2,4,1,4),dtype=u.dtype);density[0,0]=.1;density[1,0]=.2;density.requires_grad_()
    dt=mesh.units.t_norm_si*.2
    out=model(u,raw,dt_si=dt,previous_dt_si=dt,scheme='implicit Euler',memory=memory,feature_parameters=params,
        marker_feedback=p,projection_deltas=pd,shard_source=NativeDensitySource(density))
    # Neutral closure: capacity=rho-rhoimp=.5. ΔTe=-2*.2/.5=-.8.
    torch.testing.assert_close(out.increment[8,0],torch.full_like(out.increment[8,0],-.8),atol=2e-12,rtol=0)
    torch.testing.assert_close(out.increment[4,0],torch.full_like(out.increment[4,0],.06),atol=2e-12,rtol=0)
    torch.testing.assert_close(out.increment[7,0],torch.full_like(out.increment[7,0],.04),atol=2e-12,rtol=0)
    assert torch.equal(out.fluid_deltas,out.increment)
    out.values.square().sum().backward()
    assert p.grad.abs().sum()>0 and pd.grad.abs().sum()>0 and density.grad.abs().sum()>0


def test_two_actual_source_field_events_checkpoint_and_failures_are_atomic(tmp_path):
    import torch
    from src.models.native_coupling import (SmallNativeResponse,CoupledNativeField,SmallShardModel,
        advance_coupled_reference)
    from src.models.marker_feedback import MarkerFeedbackParameters
    from src.models.marker_state import ReferenceRollout,save_model_rollout,load_model_rollout
    from src.models.native_state import NativeStateContext
    from tests.test_native_temporal import gp_temporal_fixture
    mesh,source,marker,u=native_fixture();base,params,_=gp_temporal_fixture()
    base.units=mesh.units
    from src.models.native_features import TorchNativeGauss
    base.feature_path.evaluator=TorchNativeGauss(mesh.coordinates,mesh.size,mesh.vertex,units=mesh.units)
    with torch.no_grad():base.decoder.weight.zero_();base.decoder.bias.zero_()
    shard_model=SmallShardModel(mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
    shards=dict(positions=np.array([[2.6,.6,0.]]),velocities=np.zeros((1,3)),
        radius=np.array([1e-5]),ablation_atoms_s=np.array([1e12]))
    response=SmallNativeResponse(mesh,parameters=MarkerFeedbackParameters(.8,1.,5/3,'Ne',10,
        'element_matrix',True,True,True),temperature_floor=1e-8)
    model=CoupledNativeField(base,response,mode='frozen99_reference',source_configuration=source.configuration,
        shard_configuration=shard_model.configuration)
    marker.cloud=MarkerCloud.empty(10,8)
    context=NativeStateContext(mesh.units,(0,),dict(mesh={'x':mesh.coordinates,'size':mesh.size,'vertex':mesh.vertex+1,'index':base.update_mapping.indices.cpu().numpy()},
        boundary={'policy':'learned_unverified'},physics={'feedback':'explicit_two_sources','feature_parameter_rows':list(params.rows)},
        numerics={'scheme':'implicit Euler'},model=model.checkpoint_configuration))
    state=ReferenceRollout(marker,u.copy(),u.copy(),shards,1e-9,1,(1e-9,1e-9,1e-9),
        fluid_dt_previous_si=1e-9,fluid_deltas=np.full_like(u,.002),time_evol_scheme='implicit Euler',
        native_context=context,field_memory=np.zeros((4,8)))
    first,diagnostic=advance_coupled_reference(state,model=model,source=source,shard_model=shard_model)
    assert first.marker.cloud.alive.sum()==1 and np.any(first.marker.feedback.values)
    assert not np.array_equal(first.fields_current,state.fields_current)
    assert first.shards['ablation_atoms_s'][0]!=shards['ablation_atoms_s'][0]
    assert first.plan_cursor==1 and state.plan_cursor==0 and state.marker.cloud.alive.sum()==0
    observed=[];second_source=deepcopy(source);actual_advance=second_source.advance
    def record(marker,target,previous,current,**kwargs):
        observed.append((previous.copy(),current.copy()))
        return actual_advance(marker,target,previous,current,**kwargs)
    second_source.advance=record
    expected,_=advance_coupled_reference(first,model=model,source=second_source,shard_model=shard_model)
    np.testing.assert_array_equal(observed[0][0],first.fields_previous)
    np.testing.assert_array_equal(observed[0][1],first.fields_current)

    path=tmp_path/'coupled.pt';save_model_rollout(path,first,model)
    field_only=deepcopy(base);untouched={k:v.clone() for k,v in field_only.state_dict().items()}
    with pytest.raises(ValueError,match='configuration'):
        load_model_rollout(path,field_only,native_context=context)
    assert all(torch.equal(v,field_only.state_dict()[k]) for k,v in untouched.items())
    fresh=deepcopy(model)
    restored=load_model_rollout(path,fresh,native_context=context)
    actual,_=advance_coupled_reference(restored,model=fresh,source=source,shard_model=shard_model)
    for name in ('fields_current','fluid_deltas','field_memory'):
        np.testing.assert_array_equal(getattr(actual,name),getattr(expected,name))
    np.testing.assert_array_equal(actual.marker.feedback.values,expected.marker.feedback.values)
    np.testing.assert_array_equal(actual.marker.cloud.P_imp,expected.marker.cloud.P_imp)
    assert actual.marker.clock==expected.marker.clock
    # Registered program restart zeros projection and performs startup fluid first;
    # model restoration above preserves projection and performs particle first.
    restarted=deepcopy(first);restarted.marker=first.marker.jorek_restart()
    restart_out,restart_diag=advance_coupled_reference(restarted,model=model,source=source,shard_model=shard_model)
    assert restart_diag['event']=='registered_restart_zero_projection_startup'
    assert not restart_out.marker.feedback.values.any()
    np.testing.assert_array_equal(restart_out.marker.cloud.P_imp,first.marker.cloud.P_imp)
    assert not restart_out.marker.clock.startup_pending
    assert restart_out.fluid_time_si==pytest.approx(first.marker.clock.sim_time_si+1e-9)
    assert not np.array_equal(actual.marker.cloud.P_imp,restart_out.marker.cloud.P_imp)
    for stage in ('source','field','shard'):
        bad_source=deepcopy(source);bad_shard=deepcopy(shard_model);bad=deepcopy(first)
        if stage=='source':
            def fail(*args):raise RuntimeError('source fail')
            bad_source.atomic=fail
        elif stage=='field':bad.fields_current[4,0]=-.5
        else:
            def fail(*args):raise RuntimeError('shard fail')
            bad_shard.update=fail
        before=deepcopy(bad)
        with pytest.raises((RuntimeError,ValueError)):
            advance_coupled_reference(bad,model=model,source=bad_source,shard_model=bad_shard)
        np.testing.assert_array_equal(bad.fields_current,before.fields_current)
        np.testing.assert_array_equal(bad.marker.cloud.P_imp,before.marker.cloud.P_imp)
        assert bad.plan_cursor==before.plan_cursor


def test_given_feedback_coupled_torch_two_step_bptt_and_explicit_detach():
    import torch
    from src.models.native_coupling import SmallNativeResponse,CoupledNativeField,NativeDensitySource
    from src.models.marker_feedback import MarkerFeedbackParameters
    from tests.test_native_temporal import gp_temporal_fixture
    mesh,source,_,u=native_fixture();base,params,_=gp_temporal_fixture()
    base.units=mesh.units
    from src.models.native_features import TorchNativeGauss
    base.feature_path.evaluator=TorchNativeGauss(mesh.coordinates,mesh.size,mesh.vertex,units=mesh.units)
    with torch.no_grad():base.decoder.weight.mul_(.01);base.decoder.bias.mul_(.01)
    response=SmallNativeResponse(mesh,parameters=MarkerFeedbackParameters(.8,1.,5/3,'Ne',10,
        'element_matrix',True,True,True),temperature_floor=1e-8)
    model=CoupledNativeField(base,response,mode='frozen99_reference',source_configuration=source.configuration,shard_configuration={'fixture':1})
    u=torch.tensor(u,requires_grad=True);d=torch.full_like(u,.001,requires_grad=True)
    h=torch.full((4,8),.1,dtype=u.dtype,requires_grad=True)
    p=torch.zeros((5,4,1,4),dtype=u.dtype);p[0,0]=1e-4;p.requires_grad_()
    pd=torch.zeros_like(p);s=NativeDensitySource(torch.zeros((2,4,1,4),dtype=u.dtype))
    dt=mesh.units.t_norm_si*.001
    kwargs=dict(dt_si=dt,previous_dt_si=dt,scheme='Gears',feature_parameters=params,
        marker_feedback=p,projection_deltas=pd,shard_source=s)
    a=model(u,d,memory=h,**kwargs);a.values.retain_grad();a.memory.retain_grad()
    b=model(a.values,a.fluid_deltas,memory=a.memory,**kwargs)
    b.values.square().sum().backward()
    for x in (u,d,h,p,a.values,a.memory):assert x.grad is not None and x.grad.abs().sum()>0
    assert all(param.grad is not None and param.grad.abs().sum()>0 for param in base.temporal_core.parameters())
    c=model(a.values.detach(),a.fluid_deltas.detach(),memory=a.memory.detach(),**kwargs)
    grads=torch.autograd.grad(c.values.square().sum(),(u,d,h),allow_unused=True)
    assert grads==(None,None,None)


def test_density_source_mass_conversion_erosion_and_stopped_births_have_independent_oracle():
    from src.models.native_coupling import SmallShardModel
    mesh,source,marker,u=native_fixture()
    model=SmallShardModel(mesh,width_m=.2,material_density_m3=1e30,ablation_model=3)
    s=dict(positions=np.array([[2.5,.5,0.]]),velocities=np.array([[1.,0.,0.]]),
        radius=np.array([.01]),ablation_atoms_s=np.array([1e15]))
    density=model.density_source(s)
    integral=mesh.integrate(density.values.numpy())[1]
    assert integral==pytest.approx(1e15*mesh.units.t_norm_si*20/8e19/(2*np.pi),rel=2e-12)
    updated=model.update(s,u,0.,1e-9)
    assert updated['positions'][0,0]==pytest.approx(2.5+1e-9)
    assert updated['radius'][0]==pytest.approx(.01-1e-9*1e15/(4*np.pi*.01**2*1e30),abs=1e-18)
    stopped=deepcopy(s);stopped['ablation_atoms_s'][:]=0
    born=model.births(stopped,marker,1e-9,u,u,source)
    assert not born.alive.any()
    # Existing marker still advances even though there are no new births.
    out,_=source.advance(marker,1e-9,u,u,births=born)
    assert out.cloud.alive.sum()==1 and out.cloud.P_imp[0,1]>0

@pytest.mark.parametrize('run_name,index,zero_index',[
    ('phaseA_dataset_1d22_ang15_massloss_neon_v400',4111,4110),
    ('phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner',2704,2703),
])
def test_real_native_torch_loads_and_initialization_gaps(run_name,index,zero_index):
    """Past saved ZERO allows this one local delta reconstruction, not arbitrary files.

    Real local load component only: no full real cloud rollout qualification.
    """
    from pathlib import Path
    import h5py
    import torch
    from src.data.native_element import read_native_element
    from src.data.native_state import audit_native_segment,read_native_feedback_state
    from src.models.native_coupling import NativeSourceMesh,TorchMarkerLoads
    from src.models.native_gauss import marker_electron_energy_local_loads,GaussRule,FluidCorrectionParameters
    from src.models.marker_feedback import MarkerFeedbackParameters,ProjectionUnits
    from src.models.marker_state import fluid_time_coefficients
    import json
    run=Path('data/EHL-2')/run_name
    events=json.loads(Path('data/restart_events.json').read_text())['runs'][run_name]['restart_events']
    rows=audit_native_segment(run,(zero_index,index,index+1),restart_events=events)
    assert rows[1]['restart'] and rows[1]['fluid_interval_jorek']==pytest.approx(0.,abs=1e-7)
    assert rows[2]['fluid_interval_jorek']==pytest.approx(.05 if index==4111 else 5.)
    with h5py.File(run/f'jorek{index:06d}.h5') as f:
        u=f['values'][()];vertices=f['vertex'][()].astype(int)-1
        units=NativeUnits(float(f['t_norm'][0]),float(f['central_density'][0])*1e20,float(f['central_mass'][0]),5/3)
    with h5py.File(run/f'projections{index:06d}.h5') as f:p=f['values'][()]
    with h5py.File(run/f'projections{zero_index:06d}.h5') as f:past=f['values'][()]
    assert not past.any()
    with pytest.raises(ValueError,match='history'):
        read_native_feedback_state(run/f'projections{index:06d}.h5',units=units,modes=(0,))
    e=int(np.argmax(np.abs(p).sum((0,1,2))[vertices].sum(0)))
    element=read_native_element(run/f'jorek{index:06d}.h5',element_index=e)
    ul,pl,dl=[element.gather(a) for a in (u,p,p-past)]
    mesh=NativeSourceMesh(element.geometry.coordinates,element.geometry.size[:,:,None],np.arange(4)[:,None],
        np.arange(1,17).reshape(4,4),units=units)
    params=MarkerFeedbackParameters(units.n_norm_m3/1e20,units.central_mass,5/3,'Ne',10,'element_matrix',True,True,True)
    dt=.5*units.t_norm_si;time=fluid_time_coefficients('Gears',dt,dt/2)
    consumer=TorchMarkerLoads(mesh,parameters=params,temperature_floor=3.2085e-4)
    inputs=[torch.tensor(a,requires_grad=True) for a in (ul,pl,dl)]
    actual=consumer(*inputs,time=time,dt_si=dt)
    expected=marker_electron_energy_local_loads(element.geometry,ul,pl,dl,rule=GaussRule(4),toroidal=element.toroidal,
        units=ProjectionUnits('jorek_marker_projection',units.n_norm_m3,units.t_norm_si,units.energy_norm_j_m3),
        parameters=params,correction=FluidCorrectionParameters(3.2085e-4,1e-5),time=time,dt_si=dt)
    # Local JOREK [vertex,degree] versus global unique-index [degree,node].
    for a,b in ((actual.direct,expected.direct),(actual.history,expected.projection_history)):
        np.testing.assert_allclose(a.detach().numpy().reshape(4,4).T[:,:,None],b,rtol=3e-11,atol=1e-18)
    (actual.direct.square().sum()+actual.history.square().sum()).backward()
    assert all(a.grad is not None and torch.isfinite(a.grad).all() for a in inputs)


def test_spatial_source_response_and_geometry_mismatch_rejection():
    import torch
    from src.models.native_coupling import SmallNativeResponse,CoupledNativeField,NativeDensitySource,NativeSourceMesh
    from src.models.marker_feedback import MarkerFeedbackParameters
    from src.models.marker_state import fluid_time_coefficients
    from tests.test_native_temporal import gp_temporal_fixture
    mesh,source,state,u=native_fixture();other=deepcopy(state)
    other.cloud.st[0]=[.7,.4];other.cloud.x[0,0]=2.7
    a,_=source.advance(state,1e-9,u,u);b,_=source.advance(other,1e-9,u,u)
    params=MarkerFeedbackParameters(.8,1.,5/3,'Ne',10,'element_matrix',True,True,True)
    response=SmallNativeResponse(mesh,parameters=params,temperature_floor=1e-8)
    dt=1e-9;time=fluid_time_coefficients('implicit Euler',dt,dt)
    s=NativeDensitySource(torch.zeros((2,4,1,4),dtype=torch.float64))
    deltas=[response.increment(torch.tensor(u),torch.tensor(x.feedback.values),torch.tensor(x.feedback.deltas),s,time=time,dt_si=dt)[0] for x in (a,b)]
    assert not torch.allclose(deltas[0][8],deltas[1][8],rtol=1e-6,atol=1e-16)
    assert torch.equal(deltas[0][:8],deltas[1][:8])
    base,_,_=gp_temporal_fixture();base.units=mesh.units
    from src.models.native_features import TorchNativeGauss
    base.feature_path.evaluator=TorchNativeGauss(mesh.coordinates,mesh.size,mesh.vertex,units=mesh.units)
    shifted=mesh.coordinates.copy();shifted[0,0]+=1.
    bad_mesh=NativeSourceMesh(shifted,mesh.size,mesh.vertex,mesh.indices,units=mesh.units)
    with pytest.raises(ValueError,match='geometry'):
        CoupledNativeField(base,SmallNativeResponse(bad_mesh,parameters=params,temperature_floor=1e-8),mode='frozen99_reference',
            source_configuration=source.configuration,shard_configuration={'explicit':1})
