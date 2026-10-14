"""Full-state fidelity only; prescribed updates are not physical closure."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch


def test_full_axes_roundtrip_and_explicit_lossy_observation():
    from src.models.native_state import NativeFieldState, NativeUnits
    u = np.arange(9*4*3*2, dtype=np.float64).reshape(9,4,3,2)
    u[0,0,0,0] = -0.0
    d = np.full_like(u, .125)
    state = NativeFieldState(u, d, NativeUnits(1., 8e19, 1., 5/3),
                             time_si=2., modes=(0,2,2), step_index=7)
    tensor = state.to_node_layout()
    assert tensor.shape == (2,4,3,9)
    assert tensor[1,2,1,4] == u[4,2,1,1]
    restored = state.from_node_layout(tensor, state.to_node_layout(history=True))
    assert restored.values.tobytes() == u.tobytes()
    assert restored.raw_deltas.tobytes() == d.tobytes()
    assert restored.schema_version == state.schema_version
    other = deepcopy(state)
    other.values[:,1:] += 100
    np.testing.assert_array_equal(state.node_dof0(), other.node_dof0())
    assert not np.array_equal(state.values, other.values)
    tensor.zero_()
    assert state.values.tobytes() == u.tobytes()  # adapter owns its copy
    with pytest.raises(ValueError, match='history'):
        NativeFieldState(u, None, state.units, 2., (0,2,2), 7)


def test_read_full_fields_and_feedback_requires_explicit_projection_history(tmp_path):
    from src.data.native_state import read_native_field_state, read_native_feedback_state
    from src.models.native_state import NativeUnits
    units = NativeUnits(.25, 8e19, 1., 5/3)
    u = np.arange(9*4*3*2, dtype=np.float64).reshape(9,4,3,2)
    p = np.arange(5*4*3*2, dtype=np.float64).reshape(5,4,3,2)
    field = tmp_path/'field.h5'
    with h5py.File(field,'w') as f:
        for k,v in dict(values=u,deltas=u*.125,t_now=8.,index_now=17,
                        t_norm=.25,central_density=.8,central_mass=1.,
                        n_order=3,n_tor=3,jorek_model=502).items(): f[k]=v
    a = read_native_field_state(field, units=units, modes=(0,2,2))
    assert a.values.tobytes() == u.tobytes()
    assert a.time_si == 2. and a.step_index == 17
    projection = tmp_path/'projection.h5'
    with h5py.File(projection,'w') as f:
        f['values']=p; f['t_now']=1.75  # particle projection time is SI!
    with pytest.raises(ValueError, match='history'):
        read_native_feedback_state(projection, units=units, modes=(0,2,2))
    b = read_native_feedback_state(projection, units=units, modes=(0,2,2),
        raw_deltas=p*.25, history_provenance='saved_projection_state')
    assert b.time_si == 1.75
    assert b.raw_deltas.tobytes() == (p*.25).tobytes()
    assert b.from_node_layout(b.to_node_layout(),b.to_node_layout(history=True)).values.tobytes() == p.tobytes()
    with pytest.raises(ValueError, match='units'):
        read_native_field_state(field, units=replace(units,t_norm_si=.5), modes=(0,2,2))
    with h5py.File(field,'a') as f: del f['deltas']
    with pytest.raises(ValueError, match='history'):
        read_native_field_state(field, units=units, modes=(0,2,2))


def native_rollout():
    from src.models.native_state import NativeStateContext, NativeUnits
    from src.models.marker_state import ReferenceRollout, MarkerRuntime, MarkerCloud, MarkerClock, ProjectionFeedback
    units = NativeUnits(1.,8e19,1.,5/3)
    context = NativeStateContext(units, (0,2,2), {
        'mesh': {'x': np.array([1.,2.]), 'size': np.array([1.,-1.])},
        'boundary': {'codes': [0,1], 'freeboundary': False},
        'physics': {'species': 'Ne', 'tables': 'fixture-adas-v1'},
        'numerics': {'basis': 'cubic', 'scheme': 'Gears', 'planes': 8},
        'model': {'architecture': 'fixture-full-axis-v1'}})
    u = np.arange(9*4*3*2, dtype=np.float64).reshape(9,4,3,2)/100
    p = np.arange(5*4*3*2,dtype=np.float64).reshape(5,4,3,2)/100
    marker = MarkerRuntime(MarkerCloud.empty(), ProjectionFeedback(p,p*.1,1.),
                           MarkerClock(1.,.05,.1,1.,2.))
    s = ReferenceRollout(marker,u,u-.1,{'radius':np.array([1.])},2.,7,(.5,1.,.25),
        fluid_dt_previous_si=1.,fluid_deltas=np.full_like(u,.125),time_evol_scheme='Gears')
    return s.with_native_context(context), context


def test_reference_rollout_native_views_validate_all_axes_and_clocks():
    state, context = native_rollout()
    assert state.native_field_state().values.shape == (9,4,3,2)
    assert state.native_feedback_state().raw_deltas.shape == (5,4,3,2)
    assert state.native_field_state().units == context.units
    for mutation, match in [
        (lambda s: setattr(s,'fields_current',s.fields_current[:,0]), 'native'),
        (lambda s: setattr(s.marker.feedback,'deltas',None), 'history'),
        (lambda s: setattr(s.marker.feedback,'time_si',1.2), 'time'),
        (lambda s: setattr(s.marker.clock,'field_end_si',3.), 'time'),
        (lambda s: setattr(s,'fields_previous',None), 'previous'),
    ]:
        broken = deepcopy(state); mutation(broken)
        with pytest.raises(ValueError, match=match): broken.validate_fluid_history()


def full_axis_solver(u, shards, feedback, dt, previous_dt, *, fluid_deltas, time_coefficients):
    from src.models.marker_state import FluidSolveResult
    # Independent prescribed recurrence, not a physical nine-field solver.
    delta = fluid_deltas*.5 + dt*.25
    return FluidSolveResult(u+delta, delta)


def particle_event(marker,target,previous,current,shards):
    marker.feedback.deltas = np.full_like(marker.feedback.values,.0625)
    marker.feedback.values += marker.feedback.deltas
    marker.feedback.time_si = marker.clock.sim_time_si = target
    return marker, {'kind':'prescribed-state-test'}


def test_native_model_checkpoint_resume_and_configuration_rejection(tmp_path):
    from src.models.marker_state import save_model_rollout,load_model_rollout,reference_particle_then_fluid
    state, context = native_rollout()
    kwargs = dict(particle_step=particle_event,solve_fluid=full_axis_solver,
                  update_shards=lambda shards,u,t,dt: {'radius':shards['radius']*.5})
    first,_ = reference_particle_then_fluid(state,**kwargs)
    assert first.fluid_deltas[0,0,0,0] == .1875
    module = torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout())
    module.eval(); module[1].train()  # mixed per-module modes must survive
    file = tmp_path/'full.pt'
    save_model_rollout(file,first,module)
    restored = load_model_rollout(file,module,native_context=context)
    assert not restored.marker.clock.startup_pending
    assert not module.training and module[1].training
    direct,_ = reference_particle_then_fluid(first,**kwargs)
    resumed,_ = reference_particle_then_fluid(restored,**kwargs)
    for name in ('fields_current','fields_previous','fluid_deltas'):
        assert getattr(direct,name).tobytes() == getattr(resumed,name).tobytes()
    assert direct.fluid_deltas[0,0,0,0] == .34375
    assert direct.marker.feedback.values.tobytes() == resumed.marker.feedback.values.tobytes()
    assert direct.marker.feedback.deltas.tobytes() == resumed.marker.feedback.deltas.tobytes()
    assert direct.marker.clock == resumed.marker.clock
    assert direct.plan_cursor == resumed.plan_cursor == 2
    before = deepcopy(module.state_dict())
    for group in ('mesh','boundary','physics','numerics','model'):
        wrong = deepcopy(context); wrong.configuration[group]['changed']=True
        with pytest.raises(ValueError,match='configuration'):
            load_model_rollout(file,module,native_context=wrong)
    with pytest.raises(ValueError,match='configuration'):
        load_model_rollout(file,module,native_context=replace(context,units=replace(context.units,t_norm_si=2.)))
    for k,v in before.items(): assert torch.equal(v,module.state_dict()[k])


def test_native_checkpoint_preserves_rng_and_rejects_different_model_semantics(tmp_path):
    from src.models.marker_state import save_model_rollout,load_model_rollout
    state,context=native_rollout()
    module=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.2))
    file=tmp_path/'rng.pt'
    save_model_rollout(file,state,module)
    expected=torch.rand(6)
    load_model_rollout(file,module,native_context=context)
    assert torch.equal(torch.rand(6),expected)
    # Identical state_dict keys/shapes, different non-parametric behavior.
    wrong=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Dropout(.8))
    with pytest.raises(ValueError,match='configuration'):
        load_model_rollout(file,wrong,native_context=context)


@pytest.mark.parametrize('run_name,indices,restart',[
    ('phaseA_dataset_1d22_ang15_massloss_neon_v400',(4109,4110,4111,4112),4111),
    ('phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner',(2702,2703,2704,2705),2704),
])
def test_real_phasea_full_axis_read_and_step_metadata(run_name,indices,restart):
    from src.data.native_state import read_native_field_state,read_native_feedback_state,audit_native_segment,read_native_context
    from src.models.native_state import NativeUnits
    import json
    run=Path('data/EHL-2')/run_name
    events=json.loads(Path('data/restart_events.json').read_text())['runs'][run_name]['restart_events']
    with h5py.File(run/f'jorek{indices[0]:06d}.h5') as f:
        units=NativeUnits(float(f['t_norm'][0]),float(f['central_density'][0])*1e20,
                          float(f['central_mass'][0]),5/3)
    rows=audit_native_segment(run,indices,restart_events=events)
    assert [r['index'] for r in rows] == list(indices)
    assert rows[2]['restart'] and rows[2]['fluid_interval_jorek'] == pytest.approx(0.,abs=1e-7)
    assert rows[1]['projection_phase'] == 'restart_zero_before_startup'
    assert rows[2]['projection_phase'] == 'aligned'
    assert rows[3]['fluid_interval_jorek'] == pytest.approx(.05 if restart==4111 else 5.)
    for r in rows:
        assert r['internal_index'] == r['index']
        assert r['field_shape'] == (9,4,1,4125)
    path=run/f'jorek{indices[-1]:06d}.h5'
    state=read_native_field_state(path,units=units,modes=(0,))
    with h5py.File(path) as f:
        assert state.values.tobytes() == f['values'][()].tobytes()
        assert state.raw_deltas.tobytes() == f['deltas'][()].tobytes()
    context=read_native_context(path,units=units,modes=(0,),configuration={
        'physics':{'species':'Ne','gamma':5/3},'numerics':{'scheme':'Crank-Nicholson'},
        'model':{'architecture':'full-state-read-test'}})
    assert context.configuration['mesh']['size'].shape == (4,4,3986)
    assert context.units == state.units
    # Ordinary real projection snapshots omit raw deltas: require saved history.
    with pytest.raises(ValueError,match='history'):
        read_native_feedback_state(run/f'projections{indices[-1]:06d}.h5',units=units,modes=(0,))


def test_reference_event_failure_is_not_a_partial_commit(tmp_path):
    from src.models.marker_state import reference_particle_then_fluid,save_model_rollout,load_model_rollout
    state,context=native_rollout()
    before=deepcopy(state)
    def broken_shards(shards,u,t,dt):
        shards['radius'][:]=999
        raise RuntimeError('shard update failed')
    with pytest.raises(RuntimeError,match='shard'):
        reference_particle_then_fluid(state,particle_step=particle_event,
            solve_fluid=full_axis_solver,update_shards=broken_shards)
    for name in ('fields_current','fields_previous','fluid_deltas'):
        assert getattr(state,name).tobytes() == getattr(before,name).tobytes()
    assert state.marker.feedback.values.tobytes() == before.marker.feedback.values.tobytes()
    assert state.marker.feedback.deltas.tobytes() == before.marker.feedback.deltas.tobytes()
    assert state.marker.clock == before.marker.clock
    np.testing.assert_array_equal(state.shards['radius'],[1.])
    module=torch.nn.Linear(2,2)
    file=tmp_path/'atomic.pt'
    save_model_rollout(file,state,module)
    committed=file.read_bytes()
    bad=deepcopy(state); bad.h_residual=lambda x:x  # serialization fails after writing begins
    with pytest.raises((AttributeError,TypeError)):
        save_model_rollout(file,bad,module)
    assert file.read_bytes() == committed
    assert sorted(p.name for p in tmp_path.iterdir()) == ['atomic.pt']
    record=torch.load(file,weights_only=False)
    record['module']['weight']=torch.zeros(3,3)  # partial load_state_dict must not leak
    torch.save(record,file)
    old=deepcopy(module.state_dict()); old_rng=torch.get_rng_state().clone()
    with pytest.raises(RuntimeError): load_model_rollout(file,module,native_context=context)
    for k,v in old.items(): assert torch.equal(v,module.state_dict()[k])
    assert torch.equal(old_rng,torch.get_rng_state())


def test_explicit_versions_survive_rollout_and_reject_unknown_state_schema():
    state,_=native_rollout()
    state.native_schema_versions=('native-field-v2-unknown','native-feedback-v1')
    with pytest.raises(ValueError,match='version'):
        state.validate_fluid_history()


def test_feedback_adapter_preserves_history_origin_and_gather_time(tmp_path):
    from src.models.native_state import NativeFeedbackState
    from src.models.marker_state import save_model_rollout,load_model_rollout
    state,context=native_rollout()
    feedback=NativeFeedbackState(state.marker.feedback.values.copy(),state.marker.feedback.deltas.copy(),
        context.units,1.,context.modes,'saved-full-projection-history',gather_time_si=.2)
    state.marker.feedback=feedback.to_projection_feedback()
    file=tmp_path/'metadata.pt'; module=torch.nn.Identity()
    save_model_rollout(file,state,module)
    restored=load_model_rollout(file,module,native_context=context).native_feedback_state()
    assert restored.history_provenance == 'saved-full-projection-history'
    assert restored.gather_time_si == .2
    assert restored.raw_deltas.tobytes() == feedback.raw_deltas.tobytes()


def test_native_restart_pending_requires_explicit_zero_projection_and_matching_scheme():
    state,context=native_rollout()
    state.marker.clock.startup_pending=True
    with pytest.raises(ValueError,match='zero projection'):
        state.validate_fluid_history()
    state.marker.feedback.values.fill(0); state.marker.feedback.deltas.fill(0)
    state.marker.clock.step_rest_time=0
    state.validate_fluid_history()
    state.time_evol_scheme='implicit Euler'
    with pytest.raises(ValueError,match='scheme'):
        state.validate_fluid_history()


def test_model_dtype_mismatch_rejected_before_lossy_loading(tmp_path):
    from src.models.marker_state import save_model_rollout,load_model_rollout
    state,context=native_rollout()
    file=tmp_path/'dtype.pt'
    save_model_rollout(file,state,torch.nn.Linear(2,2,dtype=torch.float64))
    wrong=torch.nn.Linear(2,2,dtype=torch.float32)
    before=deepcopy(wrong.state_dict())
    with pytest.raises(ValueError,match='dtype'):
        load_model_rollout(file,wrong,native_context=context)
    for k,v in before.items(): assert torch.equal(v,wrong.state_dict()[k])


def test_legacy_reference_full_native_arrays_cannot_bypass_projection_history_checks():
    state,_=native_rollout()
    state.native_context=None  # compatibility path still promises native_full arrays
    state.marker.feedback.deltas=None
    with pytest.raises(ValueError,match='projection history'):
        state.validate_fluid_history()
    state.marker.feedback.deltas=np.zeros_like(state.marker.feedback.values)
    state.fields_current=state.fields_current[:,0]
    state.fluid_deltas=state.fluid_deltas[:,0]
    with pytest.raises(ValueError,match='native'):
        state.validate_fluid_history()


def test_real_registered_restart_full_native_checkpoint_is_not_restarted_on_model_load(tmp_path):
    from src.data.marker_checkpoint import initialize_native_checkpoint,reference_rollout_from_checkpoint
    from src.data.native_state import read_native_context
    from src.models.native_state import NativeUnits
    from src.models.marker_state import reference_fluid_event,save_model_rollout,load_model_rollout
    run=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400')
    cp=initialize_native_checkpoint(run/'jorek004110.h5',run/'part04110.h5',event_index=4110,
        times_path=run/'macroscopic_vars.dat',namelist_path=run/'jorek_namelist_spitest',boundary='jorek_restart')
    units=NativeUnits(cp.t_norm,cp.central_density*1e20,cp.provenance['central_mass'],5/3)
    context=read_native_context(run/'jorek004110.h5',units=units,modes=(0,),configuration={
        'physics':{'namelist_sha256':cp.provenance['namelist_sha256'],'species':'Ne'},
        'numerics':{'scheme':cp.provenance['time_evol_scheme'],'control_origin':'external_control'},
        'model':{'architecture':'prescribed-full-state-test'}})
    state=reference_rollout_from_checkpoint(cp,native_context=context,control_plan_si=(.05*cp.t_norm,)*3)
    assert state.marker.clock.startup_pending
    assert not state.native_feedback_state().values.any()
    assert state.native_field_state().raw_deltas.tobytes() == cp.fluid_deltas.tobytes()
    result=reference_fluid_event(state,solve_fluid=full_axis_solver,
                                 update_shards=lambda shards,u,t,dt: shards)
    assert not result.marker.clock.startup_pending
    file=tmp_path/'real.pt'; module=torch.nn.Identity()
    save_model_rollout(file,result,module)
    restored=load_model_rollout(file,module,native_context=context)
    assert not restored.marker.clock.startup_pending
    assert restored.fields_current.tobytes() == result.fields_current.tobytes()
    assert restored.fluid_deltas.tobytes() == result.fluid_deltas.tobytes()
    assert restored.marker.clock == result.marker.clock
    for name in restored.marker.cloud.array_names():
        assert getattr(restored.marker.cloud,name).tobytes() == getattr(result.marker.cloud,name).tobytes()
    assert state.marker.clock.startup_pending  # input not changed by event
