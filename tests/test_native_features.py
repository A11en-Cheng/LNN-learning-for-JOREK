"""Compatible differentiable observations; no prediction/closed-loop claims."""
import numpy as np
import pytest
import torch
from src.models.native_state import NativeUnits
from tests.test_native_gauss import rectangle_geometry,native_coefficients

UNITS=NativeUnits(1.,8e19,1.,5/3)


def evaluator(geometry=None):
    from src.models.native_features import TorchNativeGauss
    g=geometry or rectangle_geometry()
    return TorchNativeGauss(g.coordinates,g.size[:,:,None],np.arange(4)[:,None],units=UNITS)


def test_full_cubic_linear_values_and_physical_directions():
    ev=evaluator()
    a=native_coefficients(lambda s,t:4+2*s-3*t,lambda s,t:2.,lambda s,t:-3.,lambda s,t:0.)
    u=torch.tensor(np.repeat(a[None,:,None,:],9,axis=0),dtype=torch.float64,requires_grad=True)
    out=ev(u,units=UNITS)
    expected=4+2*ev.coordinates[0,:,0]-4-3*ev.coordinates[0,:,1]
    torch.testing.assert_close(out.values[0,:,0],expected)
    torch.testing.assert_close(out.gradients[0,:,0],torch.tensor([2.,-3.],dtype=u.dtype).expand(16,2))
    (out.values.square().sum()+out.gradients.square().sum()).backward()
    assert (u.grad.abs().sum(-1)>0).all()


def test_weighted_train_only_scaling_and_parameter_schema(tmp_path):
    from src.models.native_features import NativeFeaturePath,select_feature_parameters
    from tests.test_native_windows import sequence,make_windows
    from dataclasses import replace
    rs=sequence(5)
    for r in rs:
        r.state=replace(r.state,values=np.repeat(r.state.values,2,axis=-1),raw_deltas=np.repeat(r.state.raw_deltas,2,axis=-1),units=UNITS)
        r.parameters=parameters()
    windows=make_windows(rs).windows
    path=NativeFeaturePath.fit(evaluator(),windows,scenario_enabled=True,physics_enabled=True)
    u=torch.tensor(rs[1].state.values,requires_grad=True)
    d=torch.tensor(rs[1].state.raw_deltas,requires_grad=True)
    params=select_feature_parameters(rs[1].parameters,scenario_enabled=True,physics_enabled=True)
    f=path(u,d,units=UNITS,parameters=params)
    assert f.node_observations.shape==(4,27) and f.scenario.shape==(4,)
    assert f.magnetic.shape==(4,2) and f.raw_fluid_deltas is d
    torch.testing.assert_close(path.restore_values(f.scaled_values),f.gauss.values)
    torch.testing.assert_close(path.restore_gradients(f.scaled_gradients),f.gauss.gradients)
    test=__import__('copy').deepcopy(windows[0]); test.split='validation'
    test.support_records[0].state.values.fill(1e12)
    other=NativeFeaturePath.fit(evaluator(),windows+[test],scenario_enabled=True,physics_enabled=True)
    torch.testing.assert_close(path.value_scale,other.value_scale,rtol=0,atol=0)
    torch.testing.assert_close(path.gradient_scale,other.gradient_scale,rtol=0,atol=0)
    path.save_statistics(tmp_path/'statistics.json')
    restored=NativeFeaturePath.from_statistics(evaluator(),tmp_path/'statistics.json')
    torch.testing.assert_close(restored.gradient_scale,path.gradient_scale,rtol=0,atol=0)


def parameters():
    from src.data.native_parameters import route_parameters
    return tuple(route_parameters({'names':['n_spi','pellet_density','eta'],
        'vector':[846.,4.3e8,1.443e-10],
        'provenance':{'n_spi':'namelist','pellet_density':'namelist','eta':'h5'}},
        source='checkpoint',namelist_source='production'))


def test_curved_signed_geometry_matches_analytic_and_numpy_and_gradcheck():
    from src.models.native_gauss import NativeElementGeometry,evaluate_native_element,GaussRule,ToroidalSampling
    R=lambda s,t:2+s+.1*s*s*t
    Z=lambda s,t:t+.2*s*t*t
    Rs=lambda s,t:1+.2*s*t; Rt=lambda s,t:.1*s*s; Rst=lambda s,t:.2*s
    Zs=lambda s,t:.2*t*t; Zt=lambda s,t:1+.4*s*t; Zst=lambda s,t:.4*t
    f=lambda s,t:R(s,t)+2*Z(s,t)
    fs=lambda s,t:Rs(s,t)+2*Zs(s,t)
    ft=lambda s,t:Rt(s,t)+2*Zt(s,t)
    fst=lambda s,t:Rst(s,t)+2*Zst(s,t)
    size=np.arange(1,17).reshape(4,4)/8.; size[1::2,::2]*=-1
    x=np.stack([native_coefficients(R,Rs,Rt,Rst),native_coefficients(Z,Zs,Zt,Zst)])[:,:,None,:]/size[None,:,None,:]
    g=NativeElementGeometry(x,size,3); ev=evaluator(g)
    c=native_coefficients(f,fs,ft,fst)/size
    u=torch.tensor(np.repeat(c[None,:,None,:],9,0),requires_grad=True)
    out=ev(u,units=UNITS)
    torch.testing.assert_close(out.values[0,:,0],ev.coordinates[0,:,0]+2*ev.coordinates[0,:,1],rtol=2e-14,atol=2e-14)
    torch.testing.assert_close(out.gradients[0,:,0],u.new_tensor([1.,2.]).expand(16,2),rtol=2e-14,atol=2e-14)
    reference=evaluate_native_element(g,u.detach().numpy(),rule=GaussRule(4),toroidal=ToroidalSampling((0,),1,1))
    torch.testing.assert_close(out.values[0],torch.tensor(reference.values),rtol=2e-14,atol=2e-14)
    torch.testing.assert_close(out.gradients[0],torch.tensor(np.stack([reference.d_x,reference.d_y],-1)),rtol=2e-14,atol=2e-14)
    assert torch.autograd.gradcheck(lambda a:(ev(a,units=UNITS).values,ev(a,units=UNITS).gradients),(u,),eps=1e-6,atol=2e-8,rtol=2e-6)
    pert=u.detach().clone(); pert[:,3]+=.1
    changed=ev(pert,units=UNITS)
    assert not torch.equal(changed.values,out.values) and not torch.equal(changed.gradients,out.gradients)


def test_positive_weighted_pool_preserves_constants_and_rejects_bad_geometry():
    ev=evaluator()
    f=torch.ones(1,16,3,dtype=torch.float64)
    torch.testing.assert_close(ev.aggregate(f),torch.ones(4,3,dtype=torch.float64))
    # R=2+s, left/right partition integrals 7/12 and 2/3 (t integrates1/2).
    torch.testing.assert_close(ev.pool_norm,torch.tensor([7/12,2/3,2/3,7/12],dtype=torch.float64))
    with pytest.raises(ValueError,match='units'): ev(torch.ones(9,4,1,4,dtype=torch.float64),units=NativeUnits(2.,8e19,1.,5/3))
    with pytest.raises(ValueError,match='full Torch'): ev(torch.ones(9,1,1,4,dtype=torch.float64),units=UNITS)
    bad=rectangle_geometry(); bad.coordinates.fill(0.)
    with pytest.raises(ValueError,match='singular'): evaluator(bad)


def feature_fixture(*,scenario=True,physics=True):
    from src.models.native_features import NativeFeaturePath,select_feature_parameters
    from tests.test_native_windows import sequence,make_windows
    from dataclasses import replace
    rs=sequence(5)
    for r in rs:
        a=np.arange(144,dtype=float).reshape(9,4,1,4)/144.+r.state.step_index/20.
        r.state=replace(r.state,values=a,raw_deltas=np.ones_like(a)*.125,units=UNITS)
        r.parameters=parameters()
    windows=make_windows(rs).windows
    path=NativeFeaturePath.fit(evaluator(),windows,scenario_enabled=scenario,physics_enabled=physics)
    params=select_feature_parameters(parameters(),scenario_enabled=scenario,physics_enabled=physics)
    return path,params,windows


def test_feature_path_integrates_full_residual_step_and_gradients():
    from src.models.native_field_step import NativeFieldStep,NativeUpdateMapping
    from src.models.stencil_conv import build_fem_stencil_geometry
    path,params,windows=feature_fixture()
    g=rectangle_geometry()
    geometry=build_fem_stencil_geometry(np.arange(4)[:,None],g.coordinates[:,0,0,:].T)
    mapping=NativeUpdateMapping(np.arange(1,17).reshape(4,4),constrained=np.zeros(4,bool),treat_axis=False,keep_n0_const=False)
    model=NativeFieldStep(geometry,mapping,units=UNITS,state_scale=np.ones((9,4)),
        increment_scale=np.ones((9,4)),hidden=8,n_layers=1,n_iters=2,feature_path=path)
    state=windows[0].current_input().state
    u=torch.tensor(state.values,requires_grad=True); d=torch.tensor(state.raw_deltas,requires_grad=True)
    out=model(u,d,dt_si=.5,previous_dt_si=1.,scheme='Gears',feature_parameters=params)
    out.increment.square().sum().backward()
    assert (u.grad.abs().sum(-1)>0).all() and (d.grad.abs().sum(-1)>0).all()
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0,name
    torch.testing.assert_close(out.values,u+out.increment)
    assert torch.equal(out.fluid_deltas,out.increment)
    assert model.checkpoint_configuration['features']['parameter_version']=='native-feature-parameters-v1'
    with torch.no_grad(): model.decoder.weight.zero_(); model.decoder.bias.zero_()
    no_op=model(u,d,dt_si=.5,previous_dt_si=1.,scheme='Gears',feature_parameters=params)
    assert torch.equal(no_op.values,u)


def test_parameter_reordering_routes_and_disabled_groups():
    from copy import deepcopy
    from src.models.native_features import select_feature_parameters,FeatureParameters
    path,p,windows=feature_fixture()
    state=windows[0].current_input().state
    u=torch.tensor(state.values); d=torch.tensor(state.raw_deltas)
    base=path(u,d,units=UNITS,parameters=p)
    reversed_p=select_feature_parameters(tuple(reversed(parameters())),scenario_enabled=True,physics_enabled=True)
    same=path(u,d,units=UNITS,parameters=reversed_p)
    assert torch.equal(base.scenario,same.scenario) and torch.equal(base.magnetic,same.magnetic)
    rows=deepcopy(list(parameters())); rows[2]['value']*=2.
    changed=path(u,d,units=UNITS,parameters=select_feature_parameters(rows,scenario_enabled=True,physics_enabled=True))
    torch.testing.assert_close(changed.magnetic,2*base.magnetic)
    assert torch.equal(changed.scenario,base.scenario) and torch.equal(changed.node_observations,base.node_observations)
    rows=deepcopy(list(parameters())); rows[0]['value']+=1
    changed=path(u,d,units=UNITS,parameters=select_feature_parameters(rows,scenario_enabled=True,physics_enabled=True))
    assert not torch.equal(changed.scenario,base.scenario)
    assert torch.equal(changed.magnetic,base.magnetic)
    with pytest.raises(ValueError,match='unknown'): select_feature_parameters([{'name':'run_id'}],scenario_enabled=True,physics_enabled=True)
    with pytest.raises(ValueError,match='missing'): select_feature_parameters([],scenario_enabled=True,physics_enabled=True)
    rows=deepcopy(list(parameters())); rows[1]['status']='unconfirmed_default'
    with pytest.raises(ValueError,match='unconfirmed'):
        path(u,d,units=UNITS,parameters=select_feature_parameters(rows,scenario_enabled=True,physics_enabled=True))
    rows=deepcopy(list(parameters())); rows[1]['value']=torch.tensor(4.3e8,dtype=torch.float64,requires_grad=True)
    q=path(u,d,units=UNITS,parameters=select_feature_parameters(rows,scenario_enabled=True,physics_enabled=True))
    q.scenario.sum().backward()
    assert rows[1]['value'].grad.abs()>0
    disabled,dp,_=feature_fixture(scenario=False,physics=False)
    out=disabled(u,d,units=UNITS,parameters=dp)
    assert out.scenario is None and out.magnetic is None
    assert disabled.checkpoint_configuration['disabled_groups']==['scenario','physical_eta']


def test_two_real_phase_a_windows_features_and_backward(monkeypatch):
    import h5py
    from pathlib import Path
    from src.data.native_window_io import read_segment
    from src.data.native_windows import build_windows,SplitBlock,WindowConfig
    from src.data.native_features import read_native_feature_evaluator
    from src.models.native_features import NativeFeaturePath,select_feature_parameters
    for suffix,first,h in [('',4111,.05),('_injInner',2704,5.)]:
        run=Path('data/EHL-2')/('phaseA_dataset_1d22_ang15_massloss_neon_v400'+suffix)
        rs,_=read_segment(run,range(first,first+4),restart_events=(first,),gamma=5/3,gamma_provenance='explicit_test_config',modes=(0,))
        units=rs[0].state.units
        windows=build_windows(rs,[SplitBlock(run.name,'train',first,first+3)],WindowConfig(1,2,1,0),
            time_plan={(run.name,i):h*units.t_norm_si for i in range(first,first+4)},
            plan_provenance='declared_segment_schedule',scheme='Gears').windows
        ev=read_native_feature_evaluator(run/f'jorek{first:06d}.h5',units=units)
        path=NativeFeaturePath.fit(ev,windows,scenario_enabled=True,physics_enabled=True)
        view=windows[0].current_input()
        u=torch.tensor(view.state.values,requires_grad=True); d=torch.tensor(view.state.raw_deltas,requires_grad=True)
        p=select_feature_parameters(view.parameters,scenario_enabled=True,physics_enabled=True)
        with monkeypatch.context() as m:
            m.setattr(torch.Tensor,'numpy',lambda *a,**k:pytest.fail('dynamic numpy path'))
            m.setattr(torch.Tensor,'detach',lambda *a,**k:pytest.fail('dynamic detach path'))
            m.setattr(h5py,'File',lambda *a,**k:pytest.fail('feature forward file read'))
            features=path(u,d,units=units,parameters=p)
            loss=features.node_observations.square().mean()+features.scenario.square().mean()+features.magnetic.square().mean()
            loss.backward()
        assert torch.isfinite(u.grad).all() and (u.grad.abs().sum(-1)>0).all()
        assert features.raw_fluid_deltas is d
        # The joint field path independently verifies history gradients as well.
        from src.models.native_field_step import NativeFieldStep
        from src.models.stencil_conv import build_fem_stencil_geometry
        from src.data.native_state import read_native_update_mapping
        from src.data.native_windows import NativeScaler
        scales=NativeScaler.fit({'train':list(windows[0].support_records)})
        geometry=build_fem_stencil_geometry(ev.vertex.numpy().T,ev.node_positions.numpy())
        mapping=read_native_update_mapping(run/f'jorek{first:06d}.h5',keep_n0_const=False,boundary_policy='learned_unverified')
        model=NativeFieldStep(geometry,mapping,units=units,
            state_scale=scales.value_scale[:,:,0,0],increment_scale=scales.history_scale[:,:,0,0],
            center=scales.value_center[:,:,0,0],hidden=12,n_layers=1,n_iters=2,feature_path=path)
        u.grad=None; d.grad=None
        with monkeypatch.context() as m:
            m.setattr(torch.Tensor,'numpy',lambda *a,**k:pytest.fail('dynamic numpy path'))
            m.setattr(torch.Tensor,'detach',lambda *a,**k:pytest.fail('dynamic detach path'))
            m.setattr(h5py,'File',lambda *a,**k:pytest.fail('field step file read'))
            out=model(u,d,dt_si=view.dt_si,previous_dt_si=view.previous_dt_si,scheme=view.scheme,feature_parameters=p)
            out.increment.square().mean().backward()
        assert torch.isfinite(u.grad).all() and (u.grad.abs().sum(-1)>0).all()
        assert torch.isfinite(d.grad).all() and (d.grad.abs().sum(-1)>0).all()
        assert all(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters())
        before=features.node_observations.clone()
        windows[0].targets[0].values.fill(999.)
        after=path(u,d,units=units,parameters=p)
        assert torch.equal(before,after.node_observations)
        predicted=u+.01*torch.sin(u)
        fresh=path(predicted,d,units=units,parameters=p)
        assert not torch.equal(before,fresh.node_observations)


def test_reference_events_and_feature_checkpoint_resume(tmp_path):
    from src.models.native_field_step import NativeFieldStep,NativeUpdateMapping
    from src.models.stencil_conv import build_fem_stencil_geometry
    from tests.test_native_field_step import reference_fixture,zero_particle_event
    from src.models.marker_state import reference_particle_then_fluid,save_model_rollout,load_model_rollout,fluid_time_coefficients
    from src.models.native_state import NativeStateContext
    from copy import deepcopy
    path,p,_=feature_fixture()
    g=rectangle_geometry(); geometry=build_fem_stencil_geometry(np.arange(4)[:,None],g.coordinates[:,0,0,:].T)
    def make():
        return NativeFieldStep(geometry,NativeUpdateMapping(np.arange(1,17).reshape(4,4),
            constrained=np.zeros(4,bool),treat_axis=False,keep_n0_const=False),units=UNITS,
            state_scale=np.ones((9,4)),increment_scale=np.ones((9,4)),hidden=8,n_layers=1,n_iters=2,
            feature_path=deepcopy(path))
    model=make(); state,old_context=reference_fixture(model)
    cfg=deepcopy(old_context.configuration)
    cfg['mesh']={'x':g.coordinates,'size':g.size[:,:,None],'vertex':np.arange(4)[:,None]+1}
    cfg['physics']['feature_parameter_rows']=list(parameters())
    ctx=NativeStateContext(UNITS,(0,),cfg); state=state.with_native_context(ctx)
    solver=model.reference_solver(ctx)
    event=dict(particle_step=zero_particle_event,solve_fluid=solver,update_shards=lambda s,u,t,dt:s)
    first,_=reference_particle_then_fluid(state,**event)
    save_model_rollout(tmp_path/'features.pt',first,model)
    other=make(); restored=load_model_rollout(tmp_path/'features.pt',other,native_context=ctx)
    direct,_=reference_particle_then_fluid(first,**event)
    resumed,_=reference_particle_then_fluid(restored,**{**event,'solve_fluid':other.reference_solver(ctx)})
    assert direct.fields_current.tobytes()==resumed.fields_current.tobytes()
    assert direct.fluid_deltas.tobytes()==resumed.fluid_deltas.tobytes()
    with pytest.raises(ValueError,match='zero feedback'):
        solver(state.fields_current,state.shards,None,.5,1.,fluid_deltas=state.fluid_deltas,
               time_coefficients=fluid_time_coefficients('Gears',.5,1.))


# Uses the existing fixture compiling unmodified local JOREK basis/gauss.
from tests.test_native_gauss import fortran_reference


def test_torch_full_native_gauss_matches_independent_fortran(fortran_reference):
    from src.data.native_element import read_native_element
    from src.models.native_features import TorchNativeGauss
    import h5py
    file='data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/jorek004112.h5'
    element=read_native_element(file,element_index=100)
    with h5py.File(file) as f:
        units=NativeUnits(float(f['t_norm'][0]),float(f['central_density'][0])*1e20,float(f['central_mass'][0]),5/3)
        u=element.gather(f['values'][()])
    ev=TorchNativeGauss(element.geometry.coordinates,element.geometry.size[:,:,None],np.arange(4)[:,None],units=units)
    g=ev(torch.tensor(u),units=units)
    zero=np.zeros((5,4,1,4))
    reference,_=fortran_reference(element.geometry,u,zero,zero,.5,2.,.05)
    expected=np.concatenate([reference[:,6:15],reference[:,33:42],reference[:,42:51]],-1)
    actual=torch.cat([g.values[0],g.gradients[0,:,:,0],g.gradients[0,:,:,1]],-1).numpy()
    scale=np.maximum(np.max(abs(expected),axis=0),1e-13)
    assert np.max(abs(actual-expected)/scale)<2e-11


def test_gauss_statistics_have_independent_physical_units_oracle():
    from src.models.native_features import NativeFeaturePath
    from tests.test_native_windows import sequence,make_windows
    from dataclasses import replace
    a=native_coefficients(lambda s,t:4+2*s-3*t,lambda s,t:2.,lambda s,t:-3.,lambda s,t:0.)
    rs=sequence(5)
    for r in rs:
        u=np.repeat(a[None,:,None,:],9,0)
        r.state=replace(r.state,values=u,raw_deltas=np.zeros_like(u),units=UNITS)
    windows=make_windows(rs).windows
    path=NativeFeaturePath.fit(evaluator(),windows,scenario_enabled=False,physics_enabled=False)
    assert len(path.training_support)==5
    torch.testing.assert_close(path.value_center,torch.full((9,),107/30,dtype=torch.float64),rtol=2e-14,atol=2e-14)
    torch.testing.assert_close(path.value_scale,torch.full((9,),np.sqrt(971)/30,dtype=torch.float64),rtol=2e-14,atol=2e-14)
    torch.testing.assert_close(path.gradient_scale,torch.tensor([2.,3.],dtype=torch.float64).expand(9,2),rtol=2e-14,atol=2e-14)
    field=windows[0].current.state
    f=path(torch.tensor(field.values),torch.tensor(field.raw_deltas),units=UNITS,
        parameters=__import__('src.models.native_features',fromlist=['select_feature_parameters']).select_feature_parameters([],scenario_enabled=False,physics_enabled=False))
    torch.testing.assert_close(f.scaled_gradients,torch.tensor([1.,-1.],dtype=torch.float64).expand(1,16,9,2),rtol=2e-14,atol=2e-14)


def test_feature_pooling_shared_vertices_has_independent_volume_weights():
    from src.models.native_features import TorchNativeGauss
    x=np.zeros((2,4,1,6)); x[0,0,0]=[2,3,5,2,3,5]; x[1,0,0]=[0,0,0,1,1,1]
    x[0,1,0]=1/3; x[1,2,0]=1/3
    vertex=np.array([[0,1],[1,2],[4,5],[3,4]])
    size=np.ones((4,4,2))
    size[1]=np.array([1,-1,-1,1])[:,None]*np.array([1,2])[None,:]
    size[2]=np.array([1,1,-1,-1])[:,None]
    size[3]=size[1]*size[2]
    ev=TorchNativeGauss(x,size,vertex,units=UNITS)
    feat=torch.tensor([10.,20.],dtype=torch.float64)[:,None,None].expand(2,16,1)
    pooled=ev.aggregate(feat)[:,0]
    torch.testing.assert_close(pooled,torch.tensor([10.,52/3,20.,10.,52/3,20.],dtype=torch.float64),rtol=2e-14,atol=2e-14)
    first=evaluator()
    f=(first.coordinates[:,:,0]-2)[...,None]
    torch.testing.assert_close(first.aggregate(f)[:,0],torch.tensor([5/14,11/16,11/16,5/14],dtype=torch.float64),rtol=2e-14,atol=2e-14)


def test_required_routes_and_finite_scaled_features_fail_without_input_changes():
    from src.models.native_features import select_feature_parameters
    from copy import deepcopy
    path,p,windows=feature_fixture()
    state=windows[0].current.state
    u=torch.tensor(state.values); d=torch.tensor(state.raw_deltas); original=u.clone()
    with pytest.raises(ValueError,match='history'): path(u,None,units=UNITS,parameters=p)
    with pytest.raises(ValueError,match='full Torch'): path(None,d,units=UNITS,parameters=p)
    rows=deepcopy(list(parameters())); rows[2]['consumer']='unrelated_unknown_consumer'
    with pytest.raises(ValueError,match='unconfirmed'):
        path(u,d,units=UNITS,parameters=select_feature_parameters(rows,scenario_enabled=True,physics_enabled=True))
    with torch.no_grad(): path.scenario_scale.fill_(1e-320)
    with pytest.raises(ValueError,match='scaled scenario'):
        path(u,d,units=UNITS,parameters=p)
    assert torch.equal(u,original)


def test_parameter_group_flags_are_explicit_booleans_and_sources_not_features():
    from src.models.native_features import select_feature_parameters,NativeFeaturePath
    from copy import deepcopy
    with pytest.raises(ValueError,match='boolean'):
        select_feature_parameters(parameters(),scenario_enabled='false',physics_enabled=True)
    path,p,windows=feature_fixture()
    s=windows[0].current.state; u=torch.tensor(s.values); d=torch.tensor(s.raw_deltas)
    rows=deepcopy(list(parameters()))
    for r in rows: r['provenance']['source']='/different/run/file/path'
    other=select_feature_parameters(rows,scenario_enabled=True,physics_enabled=True)
    a=path(u,d,units=UNITS,parameters=p); b=path(u,d,units=UNITS,parameters=other)
    assert torch.equal(a.scenario,b.scenario) and torch.equal(a.magnetic,b.magnetic)
    stat=path.statistics(); stat['geometry']='different-native-mesh'
    with pytest.raises(ValueError,match='geometry'): NativeFeaturePath(evaluator(),stat)
    with pytest.raises(ValueError,match='training'): NativeFeaturePath.fit(evaluator(),[],scenario_enabled=True,physics_enabled=True)


def test_constant_field_and_eta_parameter_autograd():
    ev=evaluator()
    u=torch.zeros((9,4,1,4),dtype=torch.float64)
    u[:,0]=7.
    g=ev(u,units=UNITS)
    torch.testing.assert_close(g.values,torch.full_like(g.values,7.),rtol=0,atol=3e-14)
    torch.testing.assert_close(g.gradients,torch.zeros_like(g.gradients),rtol=0,atol=3e-14)
    from copy import deepcopy
    from src.models.native_features import select_feature_parameters
    path,_,windows=feature_fixture()
    rows=deepcopy(list(parameters()))
    eta=torch.tensor(rows[2]['value'],dtype=torch.float64,requires_grad=True)
    rows[2]['value']=eta
    p=select_feature_parameters(rows,scenario_enabled=True,physics_enabled=True)
    state=windows[0].current.state
    f=path(torch.tensor(state.values),torch.tensor(state.raw_deltas),units=UNITS,parameters=p)
    f.magnetic.square().sum().backward()
    assert eta.grad is not None and torch.isfinite(eta.grad) and eta.grad.abs()>0
