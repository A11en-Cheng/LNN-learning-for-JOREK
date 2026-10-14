"""Public pilot training seam on the actual small full-event fixture."""
from copy import deepcopy
import numpy as np
import pytest
import torch


def windows():
    from tests.test_base_coupled_training import event_fixture
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from src.data.base_coupled import RecordedNode, StepControl
    from tools.train_native_autonomous_pilot import PilotWindow
    result = []
    for index in range(2):
        model, source, shard, runtime = event_fixture(n=2)
        runtime.shards.update(ids=[0], positions_coordinate_system='RZphi')
        controls = tuple(StepControl((i+1)*1e-9, 1e-9, (i+2)*1e-9,
                                     'external_control', 'fixture') for i in range(2))
        nodes = tuple(RecordedNode(c.end_si, {'field': dict(
            values=runtime.fields_current.copy()+.01*(index+1),
            raw_deltas=runtime.fluid_deltas.copy(), units=model.units, time_si=c.end_si)},
            'independent_fixture_label', {'field': c.end_si}) for c in controls)
        result.append(PilotWindow(f'train-{index}', TorchRecomputedSourceEvents(model, source, shard),
                                  runtime, controls, nodes, split='train', feedback_mode='autonomous'))
    return result


def test_public_update_fits_train_initials_shares_parameters_and_keeps_runtime_immutable():
    from tools.train_native_autonomous_pilot import PilotTrainer
    items = windows()
    before = deepcopy(items[0].runtime)
    trainer = PilotTrainer(items, split_config={'fixture': 'train_only'}, seed=1)
    weights = [p.detach().clone() for p in items[0].adapter.model.parameters()]
    result = trainer.update(0)
    assert result['update'] == 1
    assert result['gradient_norm'] > 0 and np.isfinite(result['gradient_norm'])
    assert len(result['events']) == 2 and len(result['events'][0]['fields']) == 9
    assert result['gradient_scope'] == 'torch_field_bptt_stop_gradient_numpy_source_shard'
    assert any(not torch.equal(a, b) for a, b in zip(weights, items[0].adapter.model.parameters()))
    assert all(a is b for a, b in zip(items[0].adapter.model.parameters(), items[1].adapter.model.parameters()))
    assert items[0].adapter.model.feature_path.evaluator.basis is not items[1].adapter.model.feature_path.evaluator.basis
    np.testing.assert_array_equal(items[0].runtime.fields_current, before.fields_current)
    np.testing.assert_array_equal(items[0].runtime.marker.cloud.x, before.marker.cloud.x)
    assert items[0].runtime.marker.clock == before.marker.clock
    assert trainer.normalization['fit_support'] == ['train-0', 'train-1']
    assert trainer.normalization['fit_objects'] == 'initial_fields_and_raw_history_only'


def test_new_constructor_checkpoint_restores_exact_next_update_and_rng(tmp_path):
    from tools.train_native_autonomous_pilot import PilotTrainer
    trainer=PilotTrainer(windows(),split_config={'fixture':'train_only'})
    trainer.update(0)
    trainer.windows[1].adapter.model.eval()
    path=tmp_path/'update-1.pt'
    trainer.save(path)
    expected=trainer.update(1)
    expected_weights={n:p.detach().clone() for n,p in trainer.windows[0].adapter.model.named_parameters()}
    expected_rng=torch.get_rng_state().clone()
    restored=PilotTrainer(windows(),split_config={'fixture':'train_only'})
    restored.load(path)
    assert [w.adapter.model.training for w in restored.windows]==[True,False]
    actual=restored.update(1)
    assert actual==expected
    assert torch.equal(torch.get_rng_state(),expected_rng)
    for name,p in restored.windows[0].adapter.model.named_parameters():
        torch.testing.assert_close(p,expected_weights[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='budget'):
        restored.update(1)


@pytest.mark.parametrize('damage',['flag','provider','state_evidence','test_split'])
def test_teacher_and_nontrain_guards_are_enforced_by_public_constructor(damage):
    from tools.train_native_autonomous_pilot import PilotTrainer
    items=windows()
    if damage=='flag': items[0].feedback_mode='recorded_psi'
    if damage=='provider': items[0].adapter.equilibrium_provider=lambda fields,time: None
    if damage=='state_evidence': items[0].runtime.shards['psi_teacher_event']={'time':1e-9}
    if damage=='test_split': items[0].split='test'
    with pytest.raises(ValueError,match='teacher|train|feedback'):
        PilotTrainer(items,split_config={})


def test_fixed_evaluation_rng_and_complete_metrics_with_causal_digest_evidence():
    import json
    from tools.train_native_autonomous_pilot import PilotTrainer
    items=windows()
    for w in items:
        for node,control in zip(w.nodes,w.controls):
            node.targets['marker_projection']=np.zeros((5,4,1,4))
            node.object_times_si['marker_projection']=control.start_si
            node.targets['shard_positions']=dict(ids=[0],alive=[False],positions_m=[[2.6,0.,.6]])
            node.object_times_si['shard_positions']=control.end_si
    trainer=PilotTrainer(items,split_config={})
    before=trainer.evaluate()
    torch.rand(11); np.random.random(13)
    saved=torch.get_rng_state().clone()
    after=trainer.evaluate()
    assert after==before
    assert torch.equal(torch.get_rng_state(),saved)
    assert before[0]['evaluation_seed']==1
    assert before[1]['evaluation_seed']==2
    first,second=before[0]['events']
    assert second['input_field_digest']==first['predicted_field_digest']
    assert len(first['metrics']['marker_projection'])==5
    assert first['metrics']['shard_positions']['liveness_mismatch']==[0]
    assert set(first['loss_report']['loss_weights'])=={'coefficients','raw_history','gauss_values','compatible_gradients'}
    json.dumps(before,allow_nan=False)


def test_failed_update_restores_rng_and_does_not_change_weights():
    from tools.train_native_autonomous_pilot import PilotTrainer
    items=windows(); items[0].nodes[1].targets['field']['values'][0,0,0,0]=np.nan
    trainer=PilotTrainer(items,split_config={})
    saved=torch.get_rng_state().clone(); nsaved=np.random.get_state()
    params=[p.detach().clone() for p in items[0].adapter.model.parameters()]
    with pytest.raises(ValueError,match='finite'):
        trainer.update(0)
    assert trainer.cursor==0 and torch.equal(torch.get_rng_state(),saved)
    assert all(torch.equal(a,b) for a,b in zip(params,items[0].adapter.model.parameters()))
    restored=np.random.get_state()
    np.testing.assert_array_equal(restored[1],nsaved[1]); assert restored[2:]==nsaved[2:]


def test_checkpoint_binds_initial_cloud_even_when_controls_are_unchanged(tmp_path):
    from tools.train_native_autonomous_pilot import PilotTrainer
    trainer=PilotTrainer(windows(),split_config={}); path=tmp_path/'initial.pt'
    trainer.save(path)
    changed=windows(); changed[0].runtime.marker.cloud.i_life[0]+=1
    other=PilotTrainer(changed,split_config={})
    with pytest.raises(ValueError,match='checkpoint'):
        other.load(path)


def test_checkpoint_restores_each_runs_numerical_buffers_without_sharing(tmp_path):
    from tools.train_native_autonomous_pilot import PilotTrainer
    trainer=PilotTrainer(windows(),split_config={})
    a=trainer.windows[0].adapter.model.field.stencil_layers[0].term_lambda
    b=trainer.windows[1].adapter.model.field.stencil_layers[0].term_lambda
    with torch.no_grad(): a.fill_(3.); b.fill_(7.)
    path=tmp_path/'buffers.pt'; trainer.save(path)
    fresh=PilotTrainer(windows(),split_config={}); fresh.load(path)
    aa=fresh.windows[0].adapter.model.field.stencil_layers[0].term_lambda
    bb=fresh.windows[1].adapter.model.field.stencil_layers[0].term_lambda
    torch.testing.assert_close(aa,a,rtol=0,atol=0); torch.testing.assert_close(bb,b,rtol=0,atol=0)
    assert aa is not bb


def test_recorded_labels_use_previous_projection_and_true_radius_liveness(tmp_path):
    import h5py
    from tools.train_native_autonomous_pilot import read_pilot_labels
    with h5py.File(tmp_path/'jorek000012.h5','w') as f:
        for k,v in dict(t_now=3.,t_norm=2.,central_density=.8,central_mass=1.,index_now=12,tstep=.5).items():
            f[k]=[v]
        f['values']=np.zeros((9,4,1,4)); f['deltas']=np.ones((9,4,1,4))
        f['spi_radius_arr']=[.01,0.]; f['spi_R_arr']=[2.,3.]
        f['spi_Z_arr']=[.4,.5]; f['spi_phi_arr']=[0.,0.]
    with h5py.File(tmp_path/'projections000011.h5','w') as f:
        f['values']=np.ones((5,4,1,4)); f['t_now']=[5.]
    nodes,metadata=read_pilot_labels(tmp_path,[12],gamma=5/3)
    assert nodes[0].targets['shard_positions']['alive']==[True,False]
    assert nodes[0].targets['shard_positions']['ids']==[0,1]
    assert nodes[0].object_times_si==dict(field=6.,shard_positions=6.,marker_projection=5.)
    assert metadata[0]['projection_source']['path'].endswith('projections000011.h5')


def test_explicit_cpu_device_keeps_shared_parameters_buffers_and_legacy_resume(tmp_path):
    from tools.train_native_autonomous_pilot import PilotTrainer
    trainer=PilotTrainer(windows(),split_config={},device='cpu')
    assert trainer.device==torch.device('cpu')
    a,b=[w.adapter.model for w in trainer.windows]
    assert all(p is q for p,q in zip(a.parameters(),b.parameters()))
    assert all(v.device.type=='cpu' for w in trainer.windows for v in w.adapter.model.buffers())
    expected=trainer.evaluate()
    path=tmp_path/'device.pt';trainer.save(path)
    payload=torch.load(path,weights_only=False)
    assert payload.pop('execution_device')=='cpu'
    legacy=tmp_path/'legacy.pt';torch.save(payload,legacy)
    fresh=PilotTrainer(windows(),split_config={})
    fresh.load(legacy)
    assert fresh.evaluate()==expected and trainer.cursor==fresh.cursor==0


def test_explicit_unavailable_cuda_fails_without_silent_cpu_fallback(monkeypatch):
    from tools.train_native_autonomous_pilot import PilotTrainer
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    with pytest.raises(RuntimeError,match='CUDA unavailable'):
        PilotTrainer([],split_config={},device='cuda')


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA permission/device required; no CPU substitute')
def test_cuda_complete_events_and_cross_device_checkpoint_without_optimizer(tmp_path):
    from tools.train_native_autonomous_pilot import PilotTrainer
    cpu=PilotTrainer(windows(),split_config={})
    cpu_path=tmp_path/'cpu.pt';cpu.save(cpu_path)
    gpu=PilotTrainer(windows(),split_config={},device='cuda')
    gpu.load(cpu_path)
    assert gpu.normalization==cpu.normalization
    a,b=[w.adapter.model for w in gpu.windows]
    assert all(p is q and p.device.type=='cuda' for p,q in zip(a.parameters(),b.parameters()))
    assert all(v.device.type=='cuda' for w in gpu.windows for v in w.adapter.model.buffers())
    assert a.field.stencil_layers[0].term_lambda is not b.field.stencil_layers[0].term_lambda
    assert all(p is q for p,q in zip(gpu.optimizer.param_groups[0]['params'],a.parameters()))
    like=gpu.windows[0].adapter.initialize(gpu.windows[0].runtime)
    assert like.values.device.type==like.memory.device.type=='cuda'
    state=torch.cuda.get_rng_state().clone()
    metrics=gpu.evaluate()
    assert torch.equal(state,torch.cuda.get_rng_state())
    assert all(np.isfinite(row['loss']) and len(row['events'])==2 for row in metrics)
    gpu_path=tmp_path/'gpu.pt';gpu.save(gpu_path)
    restored=PilotTrainer(windows(),split_config={},device='cpu');restored.load(gpu_path)
    for p,q in zip(a.parameters(),restored.windows[0].adapter.model.parameters()):
        torch.testing.assert_close(p.cpu(),q,rtol=0,atol=0)
    assert restored.cursor==gpu.cursor==0
