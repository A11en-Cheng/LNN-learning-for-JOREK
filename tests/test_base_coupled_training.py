"""No optimizer: full-event seam, state carry, causal labels and supervision."""
from copy import deepcopy
import numpy as np
import pytest
import torch


def event_fixture(n=4):
    from tests.test_native_joint_source import joint_fixture
    from tests.test_native_context_guard import coupled_context
    from src.models.native_coupling import SmallShardModel
    from src.models.marker_state import ReferenceRollout,MarkerCloud
    model,mesh,source,marker,u,d,k=joint_fixture()
    with torch.no_grad():
        model.field.decoder.weight.mul_(.001);model.field.decoder.bias.mul_(.001)
    shard=SmallShardModel(mesh,width_m=.15,material_density_m3=4.3e28,ablation_model=3)
    model.shard_configuration=deepcopy(shard.configuration)
    context=coupled_context(model,mesh,k['feature_parameters'])
    marker.cloud=MarkerCloud.empty(10,32)
    marker.provenance['fluid_control_origin']='external_control'
    state=ReferenceRollout(marker,u.detach().numpy().copy(),u.detach().numpy().copy(),
        dict(positions=np.array([[2.6,.6,0.]]),velocities=np.array([[.2,0.,0.]]),radius=np.array([1e-5]),
            ablation_atoms_s=np.array([1e12])),1e-9,1,(1e-9,)*n,
        fluid_dt_previous_si=1e-9,fluid_deltas=d.detach().numpy().copy(),time_evol_scheme='Gears',
        native_context=context,field_memory=k['memory'].detach().numpy().copy())
    return model,source,shard,state


def sequence(n=4,burnin=1):
    from src.data.base_coupled import StepControl,RecordedNode,Support,make_sequence
    controls=tuple(StepControl((i+1)*1e-9,1e-9,(i+2)*1e-9,'external_control',f'fixture:{i}') for i in range(n))
    nodes=(RecordedNode(3e-9,{'field':'untouched_true_label'},'fixture:3'),
           RecordedNode(5e-9,{'field':'untouched_true_label'},'fixture:5')) if n==4 else ()
    return make_sequence(run='fixture',task='self_generated',controls=controls,nodes=nodes,
        burnin_steps=burnin,support=(Support('fixture',0.,(n+1)*1e-9,'all_history','fixture'),),
        blocks=(Support('fixture',0.,10e-9,'train','fixture_split'),),
        state_evidence={'complete_state':'saved'},required_state=('complete_state',),
        model_reasons=(),parameters={'ns_radius_min':.05})


def test_k_full_events_equal_chunked_state_and_labels_never_enter_advance(tmp_path):
    from src.training.base_coupled import ReferenceEvents,run_sequence,save_session,load_session
    model,source,shard,state=event_fixture()
    adapter=ReferenceEvents(model,source,shard)
    seq=sequence()
    whole=run_sequence(seq,adapter,state)
    first=run_sequence(seq,adapter,state,stop_cursor=2)
    save_session(tmp_path/'chunk.pt',seq,adapter,first)
    resumed=load_session(tmp_path/'chunk.pt',seq,adapter)
    tail=run_sequence(seq,adapter,session=resumed)
    np.testing.assert_array_equal(whole.state.fields_current,tail.state.fields_current)
    np.testing.assert_array_equal(whole.state.marker.cloud.P_imp,tail.state.marker.cloud.P_imp)
    assert whole.state.marker.clock==tail.state.marker.clock
    assert whole.cursor==4 and whole.state.fluid_index==5
    assert [x['cursor'] for x in whole.supervision]==[2,4]
    changed=deepcopy(seq);changed.nodes[0].targets['field']='different_future_truth'
    other=run_sequence(changed,adapter,state)
    np.testing.assert_array_equal(whole.state.fields_current,other.state.fields_current)
    np.testing.assert_array_equal(state.fields_current,event_fixture()[3].fields_current)


def test_variable_controls_real_node_mask_and_complete_history_isolation():
    from src.data.base_coupled import StepControl,RecordedNode,Support,make_sequence
    controls=(StepControl(1.,.25,1.25,'recorded_schedule_replay','record1'),
              StepControl(1.25,.5,1.75,'recorded_schedule_replay','record2'))
    kw=dict(run='run',task='self_generated',controls=controls,
        nodes=(RecordedNode(1.75,{'field':'true'},'real:7'),),burnin_steps=1,
        support=(Support('run',.9,1.,'cache_history','real:past'),),
        blocks=(Support('run',0.,1.8,'train','split'),),
        state_evidence={'marker':'saved','history':'causal_replayed'},required_state=('marker','history'),
        model_reasons=(),parameters={'ns_radius_min':.01})
    seq=make_sequence(**kw)
    assert seq.qualifications['combined']['eligible']
    assert seq.manifest()['prediction_time_si']==.5
    assert seq.mask_at(1)['field'] is False and seq.mask_at(2)['field'] is True
    bad=make_sequence(**{**kw,'blocks':(Support('run',1.,1.8,'validation','split'),)})
    assert not bad.qualifications['split_isolation']['eligible']
    assert bad.qualifications['data_state']['eligible']
    autonomous=make_sequence(**{**kw,'task':'autonomous'})
    assert not autonomous.qualifications['control_source']['eligible']
    with pytest.raises(ValueError,match='pseudo-label'):
        make_sequence(**{**kw,'nodes':(RecordedNode(1.5,{'field':'right_truth'},'real'),)})


def test_torch_complete_events_recompute_detached_source_but_keep_field_graph_and_cuts(tmp_path):
    from src.training.base_coupled import TorchRecomputedSourceEvents,run_sequence,detach_state,save_session,load_session
    model,source,shard,runtime=event_fixture()
    adapter=TorchRecomputedSourceEvents(model,source,shard)
    state=adapter.initialize(runtime)
    seq=sequence()
    state.values.requires_grad_()
    out=run_sequence(seq,adapter,session=__import__('src.training.base_coupled',fromlist=['SequenceSession']).SequenceSession(state))
    out.state.values.square().sum().backward()
    assert state.values.grad is not None and state.values.grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.temporal_core.parameters())
    assert adapter.gradient_scope=='torch_field_bptt_stop_gradient_numpy_source_shard'
    cut=detach_state(out.state)
    assert not cut.values.requires_grad
    torch.testing.assert_close(cut.values,out.state.values,rtol=0,atol=0)
    np.testing.assert_array_equal(cut.runtime.marker.cloud.P_imp,out.state.runtime.marker.cloud.P_imp)
    first=run_sequence(seq,adapter,adapter.initialize(runtime),stop_cursor=2,detach_before=(2,))
    save_session(tmp_path/'torch.pt',seq,adapter,first)
    restored=load_session(tmp_path/'torch.pt',seq,adapter)
    tail=run_sequence(seq,adapter,session=restored)
    torch.testing.assert_close(tail.state.values,out.state.values,rtol=0,atol=0)
    assert tail.state.runtime.marker.clock==out.state.runtime.marker.clock
    model.zero_grad(set_to_none=True)
    initial=adapter.initialize(runtime);initial.values.requires_grad_()
    truncated=run_sequence(seq,adapter,session=__import__('src.training.base_coupled',fromlist=['SequenceSession']).SequenceSession(initial,detach_before=(2,)))
    assert torch.autograd.grad(truncated.state.values.sum(),initial.values,allow_unused=True)[0] is None


def test_real_node_loss_is_observation_and_partial_teacher_fails_before_advancement():
    from src.training.base_coupled import TorchRecomputedSourceEvents,run_sequence
    from src.training.coupled_metrics import native_supervision
    model,source,shard,runtime=event_fixture()
    adapter=TorchRecomputedSourceEvents(model,source,shard);seq=sequence()
    # Large right-end target is intentionally far from prediction.
    template=dict(values=np.full_like(runtime.fields_current,10.),raw_deltas=np.full_like(runtime.fluid_deltas,3.),
                  units=model.units)
    for node in seq.nodes:node.targets['field']=dict(template,time_si=node.time_si)
    scales=dict(coefficients=np.ones((9,4,1,1)),raw_history=np.ones((9,4,1,1)),gauss_values=np.ones(9),
                compatible_gradients=np.ones((9,2)),provenance='fixture_native_scales')
    def loss(p,n,mask):
        return native_supervision(p['field'],n.targets['field'],evaluator=model.feature_path.evaluator,
            scales=scales,weights=dict(coefficients=1.,raw_history=1.,gauss_values=1.,compatible_gradients=1.))
    plain=run_sequence(seq,adapter,adapter.initialize(runtime))
    observed=run_sequence(seq,adapter,adapter.initialize(runtime),loss_fn=loss)
    torch.testing.assert_close(plain.state.values,observed.state.values,rtol=0,atol=0)
    assert observed.supervision[-1]['loss'].total.requires_grad
    with pytest.raises(ValueError,match='coherent'):
        run_sequence(seq,adapter,adapter.initialize(runtime),teacher_masks={2:dict(field=True,source=False,shard=False)})


def test_teacher_plan_and_full_state_replacement_survive_checkpoint(tmp_path):
    from src.training.base_coupled import ReferenceEvents,run_sequence,save_session,load_session
    model,source,shard,state=event_fixture();adapter=ReferenceEvents(model,source,shard);seq=sequence()
    at=run_sequence(seq,adapter,state,stop_cursor=2)
    masks={2:dict(field=True,source=True,shard=True)}
    bundle={2:dict(state=at.state,status='causal_replayed',provenance='fixture_same_trajectory_cache',
        memory_policy='same_trajectory_cache',adapter_fingerprint=adapter.fingerprint,support_verified=True,
        support=[__import__('src.data.base_coupled',fromlist=['Support']).Support('fixture',0.,3e-9,'cache_full_history','fixture')])}
    direct=run_sequence(seq,adapter,state,teacher_masks=masks,teacher_bundles=bundle)
    partial=run_sequence(seq,adapter,state,stop_cursor=1,teacher_masks=masks,teacher_bundles=bundle)
    save_session(tmp_path/'teacher.pt',seq,adapter,partial)
    restored=load_session(tmp_path/'teacher.pt',seq,adapter)
    tail=run_sequence(seq,adapter,session=restored)
    assert tail.interventions==direct.interventions and len(tail.interventions)==1
    np.testing.assert_array_equal(tail.state.fields_current,direct.state.fields_current)
    bad=deepcopy(bundle)
    bad[2]['support']=[__import__('src.data.base_coupled',fromlist=['Support']).Support('fixture',0.,20e-9,'cache_full_history','fixture_future')]
    with pytest.raises(ValueError,match='support'):
        run_sequence(seq,adapter,state,teacher_masks=masks,teacher_bundles=bad)


def test_actual_record_reader_preserves_internal_times_raw_history_and_missing_identity():
    from pathlib import Path
    from src.data.base_coupled import read_recorded_node
    run=Path('data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner')
    node,metadata=read_recorded_node(run/'jorek002711.h5',gamma=5/3,
                                    projection_path=run/'projections002710.h5')
    assert node.targets['field']['values'].shape==(9,4,1,4125)
    assert node.targets['field']['raw_deltas'].shape==(9,4,1,4125)
    assert node.object_times_si['marker_projection']<node.time_si
    assert 'shard_positions' not in node.targets
    assert 'fixed_shard_identity_not_supplied' in metadata['missing']
    assert metadata['projection_rawD_status']=='unavailable'
    assert metadata['field_index']==2711


def test_actual_variable_base_events_use_cumulative_clock_and_carry_source_shard_state():
    from src.data.base_coupled import StepControl,RecordedNode
    from src.training.base_coupled import ReferenceEvents,run_sequence
    model,source,shard,state=event_fixture()
    state.step_plan_si=(.5e-9,1.5e-9,.75e-9,1.25e-9)
    state.marker.clock.particle_dt_si=.25e-9  # explicit fixture marker resolution, distinct from coupling base
    seq=sequence();controls=[];time=1e-9
    for i,dt in enumerate(state.step_plan_si):
        controls.append(StepControl(time,dt,time+dt,'external_control',f'fixture_variable:{i}'));time+=dt
    seq.controls=tuple(controls)
    seq.nodes=(RecordedNode(3e-9,{'field':'real_node_fixture'},'fixture:3'),
               RecordedNode(5e-9,{'field':'real_node_fixture'},'fixture:5'))
    adapter=ReferenceEvents(model,source,shard)
    direct=run_sequence(seq,adapter,state)
    part=run_sequence(seq,adapter,state,stop_cursor=3)
    resumed=run_sequence(seq,adapter,session=part)
    assert direct.state.fluid_time_si==pytest.approx(5e-9,abs=1e-23)
    assert direct.state.marker.cloud.alive.sum()==4
    np.testing.assert_array_equal(direct.state.marker.feedback.values,resumed.state.marker.feedback.values)
    np.testing.assert_array_equal(direct.state.shards['positions'],resumed.state.shards['positions'])
    assert [x['cursor'] for x in direct.supervision]==[2,4]


def test_burnin_comparison_keeps_prediction_start_and_physical_horizon_fixed():
    from src.data.base_coupled import StepControl,RecordedNode,burnin_comparison_candidates
    controls=(StepControl(0.,1.,1.,'external_control','fixture'),StepControl(1.,.5,1.5,'external_control','fixture'),
        StepControl(1.5,1.5,3.,'external_control','fixture'),StepControl(3.,1.,4.,'external_control','fixture'))
    nodes=tuple(RecordedNode(t,{'field':'fixture_true_node'},'fixture') for t in (1.5,3.,4.))
    rows=burnin_comparison_candidates(controls,nodes,prediction_cursor=3,forecast_steps=1,start_cursors=(0,2,3))
    assert [r['burnin_time_si'] for r in rows]==[3.,1.5,0.]
    assert [r['burnin_record_count'] for r in rows]==[2,1,0]
    assert all(r['prediction_start_si']==3. and r['prediction_time_si']==1. for r in rows)
    assert all(r['initialization_requirement']=='certified_state_or_causal_cache_at_burnin_start' for r in rows)


def test_checkpoint_restores_model_weights_into_fresh_executing_adapter(tmp_path):
    from src.training.base_coupled import ReferenceEvents,run_sequence,save_session,load_session
    model,source,shard,state=event_fixture();adapter=ReferenceEvents(model,source,shard);seq=sequence()
    first=run_sequence(seq,adapter,state,stop_cursor=2)
    save_session(tmp_path/'fresh.pt',seq,adapter,first)
    expected=run_sequence(seq,adapter,session=first)
    fresh,other_source,other_shard,_=event_fixture()
    with torch.no_grad():fresh.field.decoder.weight.add_(.1)
    other=ReferenceEvents(fresh,other_source,other_shard)
    loaded=load_session(tmp_path/'fresh.pt',seq,other)
    actual=run_sequence(seq,other,session=loaded)
    np.testing.assert_array_equal(actual.state.fields_current,expected.state.fields_current)
    for name,value in model.state_dict().items():torch.testing.assert_close(fresh.state_dict()[name],value,rtol=0,atol=0)


def test_recorded_shard_label_clock_comes_from_its_field_not_filename(tmp_path):
    import h5py
    from src.data.base_coupled import read_recorded_node
    path=tmp_path/'jorek999999.h5'
    with h5py.File(path,'w') as f:
        for k,v in dict(t_now=3.,t_norm=2.,central_density=.8,central_mass=1.,index_now=7,tstep=.25).items():f[k]=[v]
        f['values']=np.zeros((9,4,1,4));f['deltas']=np.full((9,4,1,4),.17)
        f['spi_R_arr']=[2.];f['spi_Z_arr']=[3.];f['spi_phi_arr']=[np.pi/2]
    node,metadata=read_recorded_node(path,gamma=5/3,
        shard_identity=dict(ids=['fixed-slot-a'],alive=[True],provenance='explicit_fixture_identity'))
    assert node.object_times_si['shard_positions']==6.
    np.testing.assert_allclose(node.targets['shard_positions']['positions_m'],[[0.,2.,3.]],atol=2e-16)
    assert metadata['field_index']==7
