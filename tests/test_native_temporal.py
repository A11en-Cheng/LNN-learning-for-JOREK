"""Untrained temporal structure diagnostics, never coupled validation."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from tests.test_native_field_step import make_step


def test_official_cfc_same_weights_memory_and_network_time():
    from ncps.torch import CfCCell
    from src.models.native_temporal import CfCTimeCore
    torch.manual_seed(12)
    core=CfCTimeCore(8,backbone_units=11,time_scale_ju=2.)
    official=CfCCell(8,8,mode='default',backbone_activation='lecun_tanh',
        backbone_units=11,backbone_layers=1,backbone_dropout=0.).double()
    official.load_state_dict(core.cell.state_dict())
    x=torch.randn(9,8,dtype=torch.float64); h=torch.randn_like(x)
    expected,_=official(x,h,.25)
    actual,memory=core(x,h,dt_ju=.5)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert torch.equal(actual,memory)
    assert core.checkpoint_configuration['library_version']=='1.0.1'
    assert not torch.equal(core(x,h,dt_ju=1.)[0],actual)
    with pytest.raises(ValueError): core(x,None,dt_ju=.5)


def temporal_step():
    from src.models.native_temporal import CfCTimeCore
    return make_step(temporal_core=CfCTimeCore(8,backbone_units=11,time_scale_ju=2.))


def test_full_step_memory_bptt_and_zero_time_are_distinct_from_zero_increment():
    torch.manual_seed(23)
    model=temporal_step()
    u=torch.randn(9,4,1,9,dtype=torch.float64,requires_grad=True)
    d=torch.randn_like(u,requires_grad=True)
    h=torch.randn(9,8,dtype=u.dtype,requires_grad=True)
    noop=model(u,d,dt_si=0.,previous_dt_si=.5,scheme='Gears',memory=h)
    assert noop.values is u and noop.fluid_deltas is d and noop.memory is h
    assert not noop.increment.any()
    a=model(u,d,dt_si=.25,previous_dt_si=.5,scheme='Gears',memory=h)
    a.values.retain_grad(); a.memory.retain_grad()
    b=model(a.values,a.fluid_deltas,dt_si=.75,previous_dt_si=.25,scheme='Gears',memory=a.memory)
    b.increment.square().sum().backward()
    for tensor in (a.values,a.memory,u,d,h):
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum()>0
    assert all(p.grad is not None and p.grad.abs().sum()>0 for p in model.temporal_core.parameters())
    changed=model(u,d,dt_si=.25,previous_dt_si=.5,scheme='Gears',memory=h+1.)
    assert not torch.equal(a.increment,changed.increment)
    with torch.no_grad(): model.decoder.weight.zero_(); model.decoder.bias.zero_()
    held=model(u,d,dt_si=.25,previous_dt_si=.5,scheme='Gears',memory=h)
    assert torch.equal(held.values,u) and not held.fluid_deltas.any()
    assert not torch.equal(held.memory,h)
    with pytest.raises(ValueError,match='memory'): model(u,d,dt_si=.25,previous_dt_si=.5,scheme='Gears')


def test_window_two_step_graph_and_explicit_truncation():
    from src.models.native_torch_rollout import initialize_field_state,rollout_field_window
    from tests.test_native_windows import sequence,make_windows
    rs=sequence(5)
    # Reuse a genuine declared variable-dt window, with nine nodes for this model.
    from dataclasses import replace
    for r in rs:
        r.state=replace(r.state,units=make_step().units,values=np.repeat(r.state.values,9,axis=-1)[:,:,:,:9],
            raw_deltas=np.repeat(r.state.raw_deltas,9,axis=-1)[:,:,:,:9])
    window=make_windows(rs).windows[0]
    model=temporal_step()
    initial=initialize_field_state(model,window.current.state,previous_dt_si=window.current.previous_dt_si,scheme=window.scheme)
    initial.values.requires_grad_(); initial.raw_deltas.requires_grad_(); initial.memory.requires_grad_()
    result=rollout_field_window(model,window,initial_state=initial,teacher_mask=(False,False))
    first=result.states[0]; first.values.retain_grad(); first.memory.retain_grad()
    result.states[-1].values.square().sum().backward()
    assert first.values.grad.abs().sum()>0 and first.memory.grad.abs().sum()>0
    assert initial.memory.grad.abs().sum()>0 and initial.raw_deltas.grad.abs().sum()>0
    assert result.inputs[1].values is first.values
    truncated=rollout_field_window(model,window,initial_state=initial,teacher_mask=(False,False),detach_before=(1,))
    grads=torch.autograd.grad(truncated.states[-1].values.square().sum(),
        (initial.values,initial.raw_deltas,initial.memory),allow_unused=True)
    assert all(g is None for g in grads)
    assert truncated.inputs[1].time_si==truncated.states[0].time_si
    assert torch.equal(truncated.inputs[1].raw_deltas,truncated.states[0].raw_deltas)


def test_field_checkpoint_new_model_resume_and_bad_memory_are_atomic(tmp_path):
    from src.models.native_torch_rollout import initialize_field_state,advance_field
    from src.models.marker_state import save_model_rollout,load_model_rollout
    from src.models.native_state import NativeFieldState
    model=temporal_step()
    native=NativeFieldState(np.ones((9,4,1,9)),np.full((9,4,1,9),.25),model.units,2.,(0,),4)
    s=initialize_field_state(model,native,previous_dt_si=.5,scheme='Gears')
    a=advance_field(model,s,dt_si=.25)
    expected=advance_field(model,a,dt_si=.75)
    path=tmp_path/'field.pt'; save_model_rollout(path,a,model)
    fresh=temporal_step()
    restored=load_model_rollout(path,fresh)
    actual=advance_field(fresh,restored,dt_si=.75)
    for name in ('values','raw_deltas','memory'):
        assert torch.equal(getattr(actual,name),getattr(expected,name))
    assert actual.time_si==expected.time_si and actual.previous_dt_si==expected.previous_dt_si
    from dataclasses import replace
    before=path.read_bytes()
    with pytest.raises(ValueError,match='memory'):
        save_model_rollout(path,replace(a,memory=None),model)
    assert path.read_bytes()==before
    record=torch.load(path,weights_only=False); record['state'].memory=torch.zeros(9,7)
    bad=tmp_path/'bad.pt';torch.save(record,bad)
    weights=deepcopy(fresh.state_dict())
    with pytest.raises(ValueError,match='memory'): load_model_rollout(bad,fresh)
    assert all(torch.equal(v,fresh.state_dict()[n]) for n,v in weights.items())
    different=make_step(temporal_core=__import__('src.models.native_temporal',fromlist=['CfCTimeCore']).CfCTimeCore(8,time_scale_ju=3.))
    with pytest.raises(ValueError,match='configuration'): load_model_rollout(path,different)


def test_existing_reference_events_commit_and_resume_field_memory(tmp_path):
    from tests.test_native_field_step import reference_fixture,zero_particle_event
    from src.models.marker_state import reference_particle_then_fluid,save_model_rollout,load_model_rollout
    model=temporal_step()
    # Fixture binds native_context after construction, so explicit cold memory first.
    legacy=make_step(); state,_=reference_fixture(legacy)
    from dataclasses import replace
    cfg=deepcopy(state.native_context.configuration);cfg['model']=model.checkpoint_configuration
    state.native_context=replace(state.native_context,configuration=cfg)
    state.field_memory=np.zeros((9,8))
    state.validate_fluid_history()
    def advance(s,m):
        return reference_particle_then_fluid(s,particle_step=zero_particle_event,
            solve_fluid=m.reference_solver(s.native_context),update_shards=lambda shard,*args:shard)[0]
    a=advance(state,model)
    assert np.any(a.field_memory) and not np.any(state.field_memory)
    expected=advance(a,model)
    path=tmp_path/'event.pt';save_model_rollout(path,a,model)
    fresh=temporal_step(); restored=load_model_rollout(path,fresh,native_context=a.native_context)
    actual=advance(restored,fresh)
    assert np.array_equal(actual.fields_current,expected.fields_current)
    assert np.array_equal(actual.fluid_deltas,expected.fluid_deltas)
    assert np.array_equal(actual.field_memory,expected.field_memory)
    with pytest.raises(ValueError,match='memory'):
        advance(replace(state,field_memory=None),model)


def gp_temporal_fixture():
    from tests.test_native_features import feature_fixture
    from src.models.native_field_step import NativeFieldStep,NativeUpdateMapping
    from src.models.native_temporal import CfCTimeCore
    from src.models.stencil_conv import build_fem_stencil_geometry
    path,p,windows=feature_fixture()
    ev=path.evaluator
    geometry=build_fem_stencil_geometry(ev.vertex.numpy().T,ev.node_positions.numpy())
    model=NativeFieldStep(geometry,NativeUpdateMapping(np.arange(1,17).reshape(4,4),
        constrained=np.zeros(4,bool),treat_axis=False,keep_n0_const=False),
        units=ev.units,state_scale=np.ones((9,4)),increment_scale=np.ones((9,4)),
        hidden=8,n_layers=1,n_iters=2,feature_path=path,
        temporal_core=CfCTimeCore(8,backbone_units=11,time_scale_ju=2.))
    return model,p,windows[0]


def test_causal_warmup_teachers_fresh_gradients_and_parameter_history_paths():
    from src.models.native_torch_rollout import rollout_field_window,advance_field
    torch.manual_seed(87)
    model,p,w=gp_temporal_fixture()
    result=rollout_field_window(model,w,teacher_mask=(False,False))
    assert len(result.warmup_states)==w.config.warmup
    assert result.loss_mask==(False,True,True)
    assert torch.equal(result.inputs[0].memory,result.warmup_states[-1].memory)
    assert result.inputs[1].values is result.states[0].values
    predicted=model.feature_path.evaluator(result.inputs[1].values,units=model.units)
    actual=model.feature_path.evaluator(result.states[0].values,units=model.units)
    teacher=model.feature_path.evaluator(torch.tensor(w.targets[0].values),units=model.units)
    assert torch.equal(predicted.gradients,actual.gradients)
    assert not torch.equal(predicted.gradients,teacher.gradients)
    changed=deepcopy(w);changed.targets[0].values.fill(999);changed.targets[0].raw_deltas.fill(888)
    other=rollout_field_window(model,changed,teacher_mask=(False,False))
    assert torch.equal(other.states[-1].values,result.states[-1].values)
    tf=rollout_field_window(model,w,teacher_mask=(False,True))
    assert torch.equal(tf.inputs[1].values,torch.tensor(w.targets[0].values))
    assert torch.equal(tf.inputs[1].raw_deltas,torch.tensor(w.targets[0].raw_deltas))
    assert torch.equal(tf.inputs[1].memory,tf.states[0].memory)
    s=result.inputs[0]
    from dataclasses import replace
    different=advance_field(model,replace(s,raw_deltas=s.raw_deltas+.1),dt_si=w.time_plan_si[0],parameters=p)
    assert not torch.equal(different.values,result.states[0].values)
    from src.models.native_features import select_feature_parameters
    rows=deepcopy(list(p.rows));rows[0]['value']*=2
    q=select_feature_parameters(rows,scenario_enabled=True,physics_enabled=True)
    conditioned=advance_field(model,s,dt_si=w.time_plan_si[0],parameters=q)
    assert not torch.equal(conditioned.memory,result.states[0].memory)
    assert advance_field(model,s,dt_si=0.,parameters=p) is s
    before=s.values.clone()
    for dt in (-1.,float('nan')):
        with pytest.raises(ValueError): advance_field(model,s,dt_si=dt,parameters=p)
    assert torch.equal(s.values,before)
    malformed=deepcopy(w);malformed.targets[0].time_si+=1.
    with pytest.raises(ValueError,match='time'): rollout_field_window(model,malformed,teacher_mask=(False,True))


def test_cfc_hand_gate_oracle_and_si_ju_ownership():
    from src.models.native_temporal import CfCTimeCore
    import math
    core=CfCTimeCore(8,backbone_units=11,time_scale_ju=2.)
    with torch.no_grad():
        for p in core.parameters(): p.zero_()
        core.cell.ff1.bias.fill_(math.atanh(.2));core.cell.ff2.bias.fill_(math.atanh(.8))
        core.cell.time_a.bias.fill_(.4);core.cell.time_b.bias.fill_(-.1)
    x=torch.zeros(9,8,dtype=torch.float64); h=torch.ones_like(x)
    torch.testing.assert_close(core(x,h,dt_ju=.5)[0],torch.full_like(x,.5),rtol=0,atol=2e-16)
    torch.testing.assert_close(core(x,h,dt_ju=1.)[0],torch.full_like(x,.2+.6/(1+math.exp(-.1))),rtol=0,atol=2e-16)
    a=temporal_step()
    from dataclasses import replace
    b=temporal_step();b.load_state_dict(a.state_dict());b.units=replace(b.units,t_norm_si=1e-6)
    u=torch.randn(9,4,1,9,dtype=torch.float64);d=torch.randn_like(u)
    first=a(u,d,dt_si=.5,previous_dt_si=.25,scheme='Gears',memory=h)
    second=b(u,d,dt_si=.5e-6,previous_dt_si=.25e-6,scheme='Gears',memory=h)
    torch.testing.assert_close(first.values,second.values,rtol=0,atol=0)
    assert torch.equal(first.memory,second.memory)


def test_two_real_phase_a_complete_axis_two_step_backward_no_future_inputs(monkeypatch,tmp_path):
    import h5py
    from tools.diagnose_native_temporal import real_fixture
    from src.models.native_torch_rollout import initialize_field_state,rollout_field_window,advance_field
    from src.models.marker_state import save_model_rollout,load_model_rollout
    torch.manual_seed(41)
    for suffix,first,h in [('',4111,.05),('_injInner',2704,5.)]:
        model,w=real_fixture(suffix,first,h)
        assert w.current.feedback is None  # missing projection history remains absent
        s=initialize_field_state(model,w.current.state,previous_dt_si=w.current.previous_dt_si,scheme=w.scheme)
        for a in (s.values,s.raw_deltas,s.memory): a.requires_grad_()
        with monkeypatch.context() as m:
            m.setattr(torch.Tensor,'numpy',lambda *a,**k:pytest.fail('dynamic NumPy'))
            m.setattr(torch.Tensor,'detach',lambda *a,**k:pytest.fail('unregistered truncation'))
            m.setattr(h5py,'File',lambda *a,**k:pytest.fail('dynamic future file read'))
            result=rollout_field_window(model,w,initial_state=s,teacher_mask=(False,False))
            result.states[0].values.retain_grad();result.states[0].memory.retain_grad()
            (result.states[-1].values/model.state_scale).square().mean().backward()
        for a in (s.values,s.raw_deltas,result.states[0].values):
            assert torch.isfinite(a.grad).all() and (a.grad.abs().sum(-1)>0).all()
        for a in (s.memory,result.states[0].memory):
            assert a.grad.abs().sum()>0 and torch.isfinite(a.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.temporal_core.parameters())
        changed=deepcopy(w);changed.targets[0].values.fill(999.)
        again=rollout_field_window(model,changed,initial_state=s,teacher_mask=(False,False))
        assert torch.equal(again.states[-1].values,result.states[-1].values)
        path=tmp_path/(str(first)+'.pt');save_model_rollout(path,result.states[0],model)
        fresh,_=real_fixture(suffix,first,h);restored=load_model_rollout(path,fresh)
        from src.data.native_features import parameters_from_current_input
        out=advance_field(fresh,restored,dt_si=w.time_plan_si[1],
            parameters=parameters_from_current_input(w.current_input(),fresh.feature_path))
        assert torch.equal(out.values,result.states[-1].values)
        assert torch.equal(out.raw_deltas,result.states[-1].raw_deltas)
        assert torch.equal(out.memory,result.states[-1].memory)


def test_noop_arbitrary_bias_clock_and_positive_event_rules():
    from src.models.native_torch_rollout import initialize_field_state,advance_field
    from src.models.native_state import NativeFieldState
    from src.models.marker_state import reference_fluid_event
    from tests.test_native_field_step import reference_fixture
    model=temporal_step()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if 'bias' in name: p.fill_(3.7)
    n=NativeFieldState(np.ones((9,4,1,9)),np.full((9,4,1,9),.125),model.units,7.,(0,),12)
    s=initialize_field_state(model,n,previous_dt_si=.25,scheme='Gears');s.memory.fill_(.4)
    assert advance_field(model,s,dt_si=0.) is s
    assert s.time_si==7. and s.step_index==12 and s.previous_dt_si==.25
    assert torch.all(s.raw_deltas==.125) and torch.all(s.memory==.4)
    with pytest.raises(ValueError,match='memory'):
        model(s.values,s.raw_deltas,dt_si=0.,previous_dt_si=.25,scheme='Gears',memory=s.memory[:,:7])
    with pytest.raises(ValueError):
        model(s.values,s.raw_deltas,dt_si=0.,previous_dt_si=0.,scheme='Gears',memory=s.memory)
    legacy=make_step();event,ctx=reference_fixture(legacy);event.step_plan_si=(0.,)
    before=deepcopy(event)
    with pytest.raises(ValueError):
        reference_fluid_event(event,solve_fluid=legacy.reference_solver(ctx),update_shards=lambda *args:pytest.fail('invalid event'))
    assert event.fluid_time_si==before.fluid_time_si
    assert np.array_equal(event.fluid_deltas,before.fluid_deltas)


def test_bad_source_policy_and_memory_never_reset_physical_state():
    from src.models.native_torch_rollout import rollout_field_window,initialize_field_state
    from dataclasses import replace
    model,p,w=gp_temporal_fixture()
    before=w.current.state.values.copy()
    bad=deepcopy(w);bad.config=replace(bad.config,source_teacher_policy='recorded_full_history')
    with pytest.raises(ValueError,match='field-only'): rollout_field_window(model,bad)
    s=initialize_field_state(model,w.current.state,previous_dt_si=w.current.previous_dt_si,scheme=w.scheme)
    with pytest.raises(ValueError,match='memory'):
        rollout_field_window(model,w,initial_state=replace(s,memory=None),teacher_mask=(False,False))
    with pytest.raises(ValueError,match='truncation'):
        rollout_field_window(model,w,detach_before=(2,))
    np.testing.assert_array_equal(w.current.state.values,before)

@pytest.mark.parametrize('seed',[3,41,73])
def test_positive_time_limit_has_finite_field_and_memory_rates(seed):
    torch.manual_seed(seed)
    model=temporal_step()
    with torch.no_grad(): model.decoder.bias.fill_(.7)
    u=torch.randn(9,4,1,9,dtype=torch.float64)
    d=torch.randn_like(u)
    memory=torch.randn(9,8,dtype=torch.float64)
    changes=[]
    for dt in (1e-4,1e-6,1e-8):
        result=model(u,d,dt_si=dt,previous_dt_si=.5,scheme='Gears',memory=memory)
        changes.append((result.increment.norm().item(),(result.memory-memory).norm().item()))
        assert torch.equal(result.fluid_deltas,result.increment)
    # A continuous finite-rate wrapper contracts both changes with the interval.
    for column in (0,1):
        assert 0 < changes[-1][column] < changes[0][column]*2e-4
        rates=[row[column]/dt for row,dt in zip(changes,(1e-4,1e-6,1e-8))]
        assert max(rates)/min(rates)<1.02
    noop=model(u,d,dt_si=0.,previous_dt_si=.5,scheme='Gears',memory=memory)
    assert noop.values is u and noop.fluid_deltas is d and noop.memory is memory


def test_prescribed_constant_rate_and_memory_candidate_have_independent_units_oracle():
    model=temporal_step()  # reference interval 2 JU; t_norm_si=1 s/JU
    with torch.no_grad():
        for parameter in model.temporal_core.parameters(): parameter.zero_()
        model.decoder.weight.zero_(); model.decoder.bias.fill_(.7)
        model.increment_scale.fill_(3.)
    u=torch.zeros(9,4,1,9,dtype=torch.float64)
    history=torch.full_like(u,4.)
    memory=torch.ones(9,8,dtype=torch.float64)
    # Official zero-weight default cell yields candidate zero. At 1 JU,
    # retained memory exp(-1/2), and native increment = .7 * 3 * 1/2.
    out=model(u,history,dt_si=1.,previous_dt_si=.5,scheme='Gears',memory=memory)
    torch.testing.assert_close(out.memory,torch.full_like(memory,.6065306597126334),rtol=0,atol=2e-16)
    torch.testing.assert_close(out.increment,torch.full_like(u,1.05),rtol=0,atol=3e-16)
    assert torch.equal(out.fluid_deltas,out.increment)
    half=model(u,history,dt_si=.5,previous_dt_si=.5,scheme='Gears',memory=memory)
    assert not torch.equal(out.increment,half.increment)
    second=model(half.values,half.fluid_deltas,dt_si=.5,previous_dt_si=.5,scheme='Gears',memory=half.memory)
    torch.testing.assert_close(second.values,out.values,rtol=0,atol=5e-16)
    torch.testing.assert_close(second.memory,out.memory,rtol=0,atol=5e-16)
    assert not torch.equal(second.fluid_deltas,out.fluid_deltas)  # history is per substep


def test_old_increment_checkpoint_cannot_silently_restore_as_rate(tmp_path):
    from src.models.native_torch_rollout import initialize_field_state
    from src.models.native_state import NativeFieldState
    from src.models.marker_state import save_model_rollout,load_model_rollout
    model=temporal_step()
    native=NativeFieldState(np.ones((9,4,1,9)),np.ones((9,4,1,9)),model.units,2.,(0,),4)
    state=initialize_field_state(model,native,previous_dt_si=.5,scheme='Gears')
    path=tmp_path/'new.pt';save_model_rollout(path,state,model)
    record=torch.load(path,weights_only=False)
    # Configuration is the public checkpoint compatibility boundary.
    assert model.checkpoint_configuration['output_semantics']['decoder']=='dimensionless_rate'
    import hashlib,pickle
    old_configuration=deepcopy(model.checkpoint_configuration)
    old_configuration['native_field_step']='v1'
    old_configuration.pop('output_semantics')
    record['signature']=hashlib.sha256(pickle.dumps(
        {'checkpoint_configuration':old_configuration},protocol=5)).hexdigest()
    old=tmp_path/'old.pt';torch.save(record,old)
    before=deepcopy(model.state_dict())
    with pytest.raises(ValueError,match='configuration'):load_model_rollout(old,model)
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    fresh=temporal_step()
    fresh.time_adapter.reference_time_ju=3.
    before=deepcopy(fresh.state_dict())
    with pytest.raises(ValueError,match='configuration'):load_model_rollout(path,fresh)
    assert all(torch.equal(v,fresh.state_dict()[k]) for k,v in before.items())
