"""Untrained joint-source mechanism tests at the coupled field/event seams."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from tests.test_native_coupling import native_fixture
from tests.test_native_temporal import gp_temporal_fixture


def joint_fixture(seed=23):
    from src.models.native_coupling import CoupledNativeField,TorchMarkerLoads,NativeDensitySource
    from src.models.native_source_features import SourceFeatureScales,SourceEventTime,FieldEventContext
    from src.models.native_features import TorchNativeGauss
    from src.models.marker_feedback import MarkerFeedbackParameters
    torch.manual_seed(seed)
    mesh,source,marker,u=native_fixture();base,params,_=gp_temporal_fixture()
    base.units=mesh.units
    base.feature_path.evaluator=TorchNativeGauss(mesh.coordinates,mesh.size,mesh.vertex,units=mesh.units)
    closure=TorchMarkerLoads(mesh,parameters=MarkerFeedbackParameters(.8,1.,5/3,'Ne',10,
        'element_matrix',True,True,True),temperature_floor=1e-8)
    scales=SourceFeatureScales.physical_reference()
    model=CoupledNativeField(base,closure,source_configuration=source.configuration,
        shard_configuration={'test':'explicit_density'},source_scales=scales)
    u=torch.tensor(u,requires_grad=True);raw=torch.full_like(u,.001,requires_grad=True)
    p=torch.zeros((5,4,1,4),dtype=u.dtype)
    p[:,0,0]=torch.tensor([.02,.03,8.,4.,2.],dtype=u.dtype)[:,None]  # Q=2, Q2=4, no floor/cap
    p.requires_grad_();pd=torch.full_like(p,.001,requires_grad=True)
    density=torch.zeros((2,4,1,4),dtype=u.dtype);density[:,0]=.01;density.requires_grad_()
    memory=torch.full((4,8),.1,dtype=u.dtype,requires_grad=True)
    dt=mesh.units.t_norm_si*.01
    kwargs=dict(dt_si=dt,previous_dt_si=dt*2,scheme='Gears',feature_parameters=params,
        marker_feedback=p,projection_deltas=pd,shard_source=NativeDensitySource(density),
        field_event=FieldEventContext(dt*5,dt*5,'ordinary','explicit_test_state',dt*.8),
        source_time=SourceEventTime(event_time_si=dt*5,gather_time_si=dt*.8,units=mesh.units),memory=memory)
    return model,mesh,source,marker,u,raw,kwargs


def test_joint_main_path_needs_no_capacity_and_zero_decoder_has_no_physical_skip(monkeypatch):
    model,mesh,source,marker,u,raw,kwargs=joint_fixture()
    def forbidden(*args,**kw):raise AssertionError('capacity solve in joint learning')
    monkeypatch.setattr(torch.linalg,'solve',forbidden)
    out=model(u,raw,**kwargs)
    assert out.values.shape==(9,4,1,4) and out.source_features.shape==(4,101)
    assert (out.increment.abs().sum(-1)>0).all()
    with torch.no_grad():model.field.decoder.weight.zero_();model.field.decoder.bias.zero_()
    zero=model(u,raw,**kwargs)
    torch.testing.assert_close(zero.values,u,rtol=0,atol=0)
    assert not zero.increment.any() and not zero.fluid_deltas.any()
    assert model.checkpoint_configuration['coupling_mode']=='joint_learning'

@pytest.mark.parametrize('seed',[23,41])
def test_all_sources_histories_and_high_degrees_reach_joint_prediction_and_gradients(seed):
    model,mesh,source,marker,u,raw,k=joint_fixture(seed)
    baseline=model(u,raw,**k)
    # Source semantics: literal Q=4/2, Q2=8/2; no density injection inferred from c5.
    torch.testing.assert_close(baseline.source_features[:,84],torch.full((4,),2.,dtype=u.dtype))
    torch.testing.assert_close(baseline.source_features[:,85],torch.full((4,),4.,dtype=u.dtype))
    for name,count in [('marker_feedback',5),('projection_deltas',5),('shard_source',2)]:
        for channel in range(count):
            changed=dict(k)
            tensor=(k[name].values if name=='shard_source' else k[name]).detach().clone()
            tensor[channel,1,0,0]+=.02  # identical dof0, nonzero higher degree
            if name=='shard_source':changed[name]=type(k[name])(tensor)
            else:changed[name]=tensor
            other=model(u,raw,**changed)
            assert not torch.equal(other.source_features,baseline.source_features),(name,channel)
            assert not torch.allclose(other.increment,baseline.increment,rtol=1e-10,atol=1e-14),(name,channel)
            if name=='marker_feedback' and channel in (2,3,4):
                # c3/4/5 do not silently become power or an independent shard rate.
                torch.testing.assert_close(other.source_features[:,[48,49,58,59]],baseline.source_features[:,[48,49,58,59]])
    (baseline.increment.square().sum()+baseline.memory.square().sum()).backward()
    for a in (u,raw,k['marker_feedback'],k['projection_deltas'],k['shard_source'].values,k['memory']):
        assert a.grad is not None and torch.isfinite(a.grad).all()
        assert (a.grad.abs().sum(-1)>0).all()
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.source_encoding.projection.parameters())
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.temporal_core.parameters())
    changed_raw=model(u,raw+.01,**k)
    assert not torch.equal(changed_raw.increment,baseline.increment)


def test_joint_positive_time_limit_complete_noop_and_two_step_bptt():
    model,mesh,source,marker,u,raw,k=joint_fixture()
    k['dt_si']=mesh.units.t_norm_si*.02
    a=model(u,raw,**k);a.values.retain_grad();a.memory.retain_grad()
    b=model(a.values,a.fluid_deltas,**{**k,'memory':a.memory,'dt_si':k['dt_si']*.7,'previous_dt_si':k['dt_si']})
    assert not torch.equal(b.source_features[:,89],a.source_features[:,89])  # Pe from actual predicted U
    b.increment.square().sum().backward()
    for x in (u,raw,k['memory'],k['marker_feedback'],a.values,a.memory):
        assert x.grad is not None and x.grad.abs().sum()>0 and torch.isfinite(x.grad).all()
    a=model(u,raw,**k)
    b=model(a.values.detach(),a.fluid_deltas.detach(),**{**k,'memory':a.memory.detach(),
        'marker_feedback':k['marker_feedback'].detach(),'projection_deltas':k['projection_deltas'].detach(),
        'shard_source':type(k['shard_source'])(k['shard_source'].values.detach())})
    assert all(x is None for x in torch.autograd.grad(b.values.square().sum(),(u,raw,k['memory']),allow_unused=True))
    from dataclasses import replace
    no=model(u,raw,**{**k,'dt_si':0.,'field_event':replace(k['field_event'],phase='zero_time_probe')})
    assert no.values is u and no.fluid_deltas is raw and no.memory is k['memory'] and not no.increment.any()
    # Fixed previous dt, raw histories, nonzero memory/decoder bias. Do not freeze memory.
    changes=[]
    for q in (1e-2,1e-3,1e-4,1e-5):
        o=model(u,raw,**{**k,'dt_si':mesh.units.t_norm_si*q})
        changes.append((float(o.increment.detach().norm()),float((o.memory-k['memory']).detach().norm())))
        assert torch.isfinite(o.increment/q).all() and torch.isfinite((o.memory-k['memory'])/q).all()
    assert all(y[0]<x[0] and y[1]<x[1] for x,y in zip(changes,changes[1:]))
    assert changes[-1][0]<changes[0][0]*.002 and changes[-1][1]<changes[0][1]*.002


def test_source_units_are_explicit_and_mismatch_rejected():
    from dataclasses import replace
    from src.models.native_source_features import SourceEventTime
    model,mesh,_,_,u,raw,k=joint_fixture()
    wrong=replace(mesh.units,n_norm_m3=mesh.units.n_norm_m3*2)
    with pytest.raises(ValueError,match='units'):
        model(u,raw,**{**k,'source_time':SourceEventTime(k['source_time'].event_time_si,k['source_time'].gather_time_si,units=wrong)})


def test_joint_self_generated_events_resume_restart_and_three_stage_rollback(tmp_path):
    from src.models.native_coupling import SmallShardModel,advance_coupled_reference,SmallNativeResponse,CoupledNativeField
    from src.models.marker_state import ReferenceRollout,MarkerCloud,save_model_rollout,load_model_rollout
    from src.models.native_state import NativeStateContext
    model,mesh,source,marker,u,d,k=joint_fixture()
    with torch.no_grad():model.field.decoder.weight.mul_(.001);model.field.decoder.bias.mul_(.001)
    shard=SmallShardModel(mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
    model.shard_configuration=deepcopy(shard.configuration)
    shards=dict(positions=np.array([[2.6,.6,0.]]),velocities=np.zeros((1,3)),radius=np.array([1e-5]),ablation_atoms_s=np.array([1e12]))
    marker.cloud=MarkerCloud.empty(10,8)
    context=NativeStateContext(mesh.units,(0,),dict(mesh={'x':mesh.coordinates,'size':mesh.size,'vertex':mesh.vertex+1,'index':model.field.update_mapping.indices.cpu().numpy()},
        boundary={'policy':'learned_unverified'},physics={'feature_parameter_rows':list(k['feature_parameters'].rows)},
        numerics={'scheme':'implicit Euler'},model=model.checkpoint_configuration))
    state=ReferenceRollout(marker,u.detach().numpy().copy(),u.detach().numpy().copy(),shards,1e-9,1,(1e-9,)*3,
        fluid_dt_previous_si=1e-9,fluid_deltas=d.detach().numpy().copy(),time_evol_scheme='implicit Euler',
        native_context=context,field_memory=k['memory'].detach().numpy().copy())
    chronology=[];initial_source=deepcopy(source);initial_shard=deepcopy(shard)
    source_advance=initial_source.advance;shard_update=initial_shard.update
    def source_event(*a,**kw):
        chronology.append('source');return source_advance(*a,**kw)
    def shard_event(s,old_fields,time,dt):
        chronology.append('shard');np.testing.assert_array_equal(old_fields,state.fields_current)
        return shard_update(s,old_fields,time,dt)
    initial_source.advance=source_event;initial_shard.update=shard_event
    handle=model.register_forward_pre_hook(lambda m,a:chronology.append('field'))
    first,_=advance_coupled_reference(state,model=model,source=initial_source,shard_model=initial_shard)
    handle.remove()
    assert chronology==['source','field','shard']
    assert first.marker.cloud.alive.sum()==1 and np.any(first.marker.feedback.values)
    assert not np.array_equal(first.fields_current,state.fields_current)
    assert state.plan_cursor==0 and state.marker.cloud.alive.sum()==0
    seen=[];record_source=deepcopy(source);real=record_source.advance
    def observe(m,target,previous,current,**kw):
        seen.append(current.copy());return real(m,target,previous,current,**kw)
    record_source.advance=observe
    second,_=advance_coupled_reference(first,model=model,source=record_source,shard_model=shard)
    np.testing.assert_array_equal(seen[0],first.fields_current)
    path=tmp_path/'joint.pt';save_model_rollout(path,first,model)
    fresh=deepcopy(model);restored=load_model_rollout(path,fresh,native_context=context)
    resumed,_=advance_coupled_reference(restored,model=fresh,source=source,shard_model=shard)
    for name in ('fields_current','fields_previous','fluid_deltas','field_memory'):
        np.testing.assert_array_equal(getattr(second,name),getattr(resumed,name))
    for name in resumed.marker.cloud.array_names():
        np.testing.assert_array_equal(getattr(resumed.marker.cloud,name),getattr(second.marker.cloud,name))
    np.testing.assert_array_equal(second.marker.feedback.values,resumed.marker.feedback.values)
    np.testing.assert_array_equal(second.marker.feedback.deltas,resumed.marker.feedback.deltas)
    assert second.marker.clock==resumed.marker.clock
    restart=deepcopy(first);restart.marker=first.marker.jorek_restart()
    startup,diag=advance_coupled_reference(restart,model=model,source=source,shard_model=shard)
    assert diag['event']=='registered_restart_zero_projection_startup' and not startup.marker.feedback.values.any()
    np.testing.assert_array_equal(startup.marker.cloud.P_imp,first.marker.cloud.P_imp)
    assert not np.array_equal(second.marker.cloud.P_imp,startup.marker.cloud.P_imp)
    # New-mode restore refuses both field-only and explicit hybrid reference.
    from src.models.marker_feedback import MarkerFeedbackParameters
    ref=CoupledNativeField(deepcopy(model.field),SmallNativeResponse(mesh,parameters=MarkerFeedbackParameters(.8,1.,5/3,'Ne',10,
        'element_matrix',True,True,True),temperature_floor=1e-8),mode='frozen99_reference',
        source_configuration=source.configuration,shard_configuration=shard.configuration)
    for wrong in (deepcopy(model.field),ref):
        weights={n:t.clone() for n,t in wrong.state_dict().items()}
        with pytest.raises(ValueError,match='configuration'):load_model_rollout(path,wrong,native_context=context)
        assert all(torch.equal(t,wrong.state_dict()[n]) for n,t in weights.items())
    old_state=deepcopy(first)
    old_state.native_context.configuration['model']=ref.checkpoint_configuration
    old_path=tmp_path/'reference-mode.pt';save_model_rollout(old_path,old_state,ref)
    before={n:t.clone() for n,t in model.state_dict().items()}
    with pytest.raises(ValueError,match='configuration'):load_model_rollout(old_path,model,native_context=context)
    assert all(torch.equal(t,model.state_dict()[n]) for n,t in before.items())
    for stage in ('source','field','shard'):
        bad_source=deepcopy(source);bad_shard=deepcopy(shard);bad=deepcopy(first)
        if stage=='source':
            def fail(*a):raise RuntimeError('source fail')
            bad_source.atomic=fail
        elif stage=='field':bad.fluid_deltas=None
        else:
            def fail(*a):raise RuntimeError('shard fail')
            bad_shard.update=fail
        before=deepcopy(bad);weights={n:t.clone() for n,t in model.state_dict().items()}
        with pytest.raises((ValueError,RuntimeError)):
            advance_coupled_reference(bad,model=model,source=bad_source,shard_model=bad_shard)
        np.testing.assert_array_equal(bad.fields_current,before.fields_current)
        np.testing.assert_array_equal(bad.marker.cloud.P_imp,before.marker.cloud.P_imp)
        np.testing.assert_array_equal(bad.marker.feedback.values,before.marker.feedback.values)
        assert bad.marker.clock==before.marker.clock and bad.plan_cursor==before.plan_cursor
        assert all(torch.equal(t,model.state_dict()[n]) for n,t in weights.items())


def test_joint_source_scaling_pooling_and_moving_cloud_have_analytic_spatial_oracle():
    from src.models.native_coupling import CoupledNativeField,NativeDensitySource
    from src.models.native_source_features import SourceFeatureScales,SourceEventTime,FieldEventContext
    from tests.test_native_gauss import native_coefficients
    model,mesh,source,marker,u,raw,k=joint_fixture()
    scales=SourceFeatureScales(np.full((12,4),2.),np.full(12,4.),np.full((12,2),3.),np.ones(14),np.array([2.,5.]),
        'explicit_physical_reference_separate_values_gradients_test_v1')
    scaled=CoupledNativeField(deepcopy(model.field),deepcopy(model.response),source_configuration=source.configuration,
        shard_configuration=model.shard_configuration,source_scales=scales)
    p=k['marker_feedback'].detach().clone()
    # c1=s=R-2: exact derivative (1,0), independent of projection coefficient scaling.
    p[0,:,0]=torch.tensor(native_coefficients(lambda s,t:s,lambda s,t:1.,lambda s,t:0.,lambda s,t:0.))
    out=scaled(u,raw,**{**k,'marker_feedback':p})
    torch.testing.assert_close(out.source_features[:,60],torch.full((4,),1/3,dtype=u.dtype),rtol=0,atol=1e-14)
    torch.testing.assert_close(out.source_features[:,61],torch.zeros(4,dtype=u.dtype),rtol=0,atol=1e-14)
    # ∫ R*s*hat / ∫ R*hat: left=5/14, right=11/16; then / value scale4.
    torch.testing.assert_close(out.source_features[:,48],torch.tensor([5/56,11/64,11/64,5/56],dtype=u.dtype),rtol=0,atol=1e-14)
    torch.testing.assert_close(out.source_features[:,1],torch.tensor([1/6,-1/6,-1/6,1/6],dtype=u.dtype),rtol=0,atol=0)
    torch.testing.assert_close(out.source_features[:,98],torch.full((4,),.004,dtype=u.dtype))
    torch.testing.assert_close(out.source_features[:,99],torch.full((4,),.01,dtype=u.dtype))
    changed=deepcopy(marker);changed.cloud.x[0,0]=2.7;changed.cloud.st[0,0]=.7
    clouds=[source.advance(m,1e-9,u.detach().numpy(),u.detach().numpy())[0] for m in (marker,changed)]
    outputs=[]
    for c in clouds:
        outputs.append(model(u,raw,**{**k,'marker_feedback':torch.tensor(c.feedback.values),
            'projection_deltas':torch.tensor(c.feedback.deltas),'field_event':FieldEventContext(c.feedback.time_si,c.feedback.time_si,'ordinary','explicit_test_state',c.feedback.gather_time_si),
            'source_time':SourceEventTime(c.feedback.time_si,c.feedback.gather_time_si,units=mesh.units)}))
    assert not torch.equal(outputs[0].source_features,outputs[1].source_features)
    assert not torch.allclose(outputs[0].increment,outputs[1].increment,rtol=1e-10,atol=1e-14)


@pytest.mark.parametrize('missing',['marker_feedback','projection_deltas','shard_source','source_time','feature_parameters'])
def test_joint_missing_inputs_fail_without_mutation(missing):
    model,mesh,_,_,u,raw,k=joint_fixture();before=u.detach().clone();weights=deepcopy(model.state_dict())
    k[missing]=None
    with pytest.raises(ValueError):model(u,raw,**k)
    assert torch.equal(u,before) and all(torch.equal(t,model.state_dict()[n]) for n,t in weights.items())


def test_joint_invalid_modes_clock_scaling_and_parameter_routes():
    from dataclasses import replace
    from src.models.native_coupling import CoupledNativeField,NativeDensitySource,SmallNativeResponse
    from src.models.native_source_features import SourceFeatureScales
    model,mesh,source,marker,u,raw,k=joint_fixture()
    for clock in (replace(k['source_time'],gather_time_si=None),replace(k['source_time'],phase='unknown'),
        replace(k['source_time'],phase='registered_restart_zero_projection',gather_time_si=None)):
        with pytest.raises(ValueError):model(u,raw,**{**k,'source_time':clock})
    with pytest.raises(ValueError,match='mass source'):
        model(u,raw,**{**k,'shard_source':NativeDensitySource(k['shard_source'].values,units='unknown')})
    for mode,scales in [('unknown',SourceFeatureScales.physical_reference()),('frozen99_reference',SourceFeatureScales.physical_reference()),
        ('joint_learning',None),('joint_learning',replace(SourceFeatureScales.physical_reference(),provenance='validation_statistics'))]:
        with pytest.raises(ValueError):
            CoupledNativeField(deepcopy(model.field),deepcopy(model.response),mode=mode,source_scales=scales,
                source_configuration=source.configuration,shard_configuration=model.shard_configuration)
    # Give eta a valid nonzero compatible psi gradient and check its actual feature route.
    with torch.no_grad():u[0,1]=.1
    rows=deepcopy(k['feature_parameters'].rows)
    tensor_parameters=[]
    for row in rows:
        row['value']=torch.tensor(row['value'],dtype=u.dtype,requires_grad=True)
        tensor_parameters.append(row['value'])
    params=replace(k['feature_parameters'],rows=rows)
    out=model(u,raw,**{**k,'feature_parameters':params})
    grads=torch.autograd.grad(out.increment.square().sum(),tensor_parameters)
    assert all(g is not None and torch.isfinite(g) and g.abs()>0 for g in grads)
    for index in range(len(rows)):
        changed=deepcopy(k['feature_parameters'].rows);changed[index]['value']*=2
        other=model(u,raw,**{**k,'feature_parameters':replace(k['feature_parameters'],rows=changed)})
        assert not torch.equal(out.increment,other.increment)


def test_interpreted_power_and_projection_history_remain_separate_features():
    model,mesh,_,_,u,raw,k=joint_fixture()
    p=k['marker_feedback'].detach().clone();p[1,0]=-.03
    # Uniform c2<0 is clamped only in interpreted power, not saved coefficients.
    pd=torch.zeros_like(p);pd[3,0]=.1;pd[4,0]=.02
    o=model(u,raw,**{**k,'marker_feedback':p,'projection_deltas':pd})
    assert o.source_features.shape==(4,101)
    torch.testing.assert_close(o.source_features[:,96],torch.full((4,),-.02,dtype=u.dtype))
    # ratio=.5, Q=2, N=2 => deltaQ=(.1-2*.02)*.5/2=.015
    mass_ratio=1/20*(2/2.01455279245)
    torch.testing.assert_close(o.source_features[:,97],torch.full((4,),-mass_ratio*.015*.01*.5,dtype=u.dtype))
    assert torch.equal(p[1,0],torch.full_like(p[1,0],-.03))


def test_corrupt_source_scales_do_not_partially_restore_model(tmp_path):
    from src.models.native_torch_rollout import TorchFieldState
    from src.models.native_state import configuration_digest
    from src.models.marker_state import save_model_rollout,load_model_rollout
    model,mesh,_,_,u,raw,k=joint_fixture()
    state=TorchFieldState(u,raw,k['memory'],mesh.units,0.,k['previous_dt_si'],0,'Gears',configuration_digest(model.checkpoint_configuration))
    path=tmp_path/'joint-corrupt.pt';save_model_rollout(path,state,model)
    record=torch.load(path,weights_only=False)
    record['module']['source_encoding.scale_gauss_values'][0]=2.  # metadata still declares 1
    torch.save(record,path)
    before={n:t.clone() for n,t in model.state_dict().items()};rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError,match='configuration'):load_model_rollout(path,model)
    assert all(torch.equal(t,model.state_dict()[n]) for n,t in before.items())
    assert torch.equal(rng,torch.get_rng_state())
