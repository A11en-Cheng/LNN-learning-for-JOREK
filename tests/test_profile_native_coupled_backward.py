"""Full event/field BPTT seam, no optimizer or real GPU jobs."""
import numpy as np
import torch
import pytest


@pytest.mark.parametrize('profile_ops', [False, True])
def test_two_events_read_truth_only_after_forward_and_backward_without_updates(tmp_path, profile_ops):
    from tools.profile_native_coupled_backward import profile_backward
    from tests.test_native_production_torch_integration import fixture, control, AnalyticEquilibrium
    from src.training.base_coupled import TorchRecomputedSourceEvents
    model, source, shard, runtime = fixture()
    with torch.no_grad():
        model.field.decoder.weight.fill_(1e-4)
    adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=AnalyticEquilibrium())
    calls = []
    hook = model.register_forward_hook(lambda *args: calls.append('forward'))
    initial = runtime.fields_current.copy()
    truth = initial+.03
    def labels(controls):
        assert calls == ['forward', 'forward']
        return tuple(dict(values=truth.copy(), time_si=c.end_si, provenance='fixture_label_only') for c in controls)
    try:
        result = profile_backward(adapter, runtime, (control(0), control(1)), labels,
            tmp_path/'probe', device='cpu', profile_backward_ops=profile_ops, profile_forward=profile_ops,
            fine_components=True)
    finally:
        hook.remove()
    assert result['completed']
    assert result['optimizer_instantiated'] is False and result['optimizer_updates'] == 0
    assert result['model_unchanged'] and result['backward_calls'] == 1
    assert len(result['per_field_loss']) == 2 and len(result['per_field_loss'][0]) == 9
    assert all(row['finite'] and row['nonzero'] for row in result['cross_step_gradients'].values())
    assert result['parameter_gradients'] and all(row['finite'] for row in result['parameter_gradients'].values())
    assert any(row['nonzero'] for row in result['parameter_gradients'].values())
    assert result['gradient_scope'] == 'torch_field_bptt_stop_gradient_numpy_source_shard'
    assert (tmp_path/'probe/result.json').is_file()
    assert sum(r['stage']=='marker_births' for r in result['stages']) == 2
    summary = result['exclusive_timing']
    assert summary['accounting_closed']
    label_seconds = sum(r['seconds'] for r in result['stages'] if r['stage']=='label_read')
    assert summary['total_seconds'] == pytest.approx(result['training_critical_path_seconds']+label_seconds)
    assert sum(summary['categories_seconds'].values()) == pytest.approx(summary['total_seconds'])
    np.testing.assert_array_equal(runtime.fields_current, initial)
    if profile_ops:
        assert (tmp_path/'probe/backward-trace.json').is_file()
        assert (tmp_path/'probe/forward-event-1.prof').is_file()


def test_exclusive_timing_subtracts_direct_children_without_double_counting():
    from tools.profile_native_coupled_backward import exclusive_stage_summary
    def row(i, parent, stage, start, end):
        return dict(id=i, parent_id=parent, stage=stage, started_at_seconds=start,
                    ended_at_seconds=end, seconds=end-start, completed=True)
    rows = [row(0,None,'prefix',0,10), row(1,0,'torch_state_validation',0,4),
            row(2,1,'reference_validation',1,3), row(3,2,'field_solver_prepare',1,2),
            row(4,0,'marker_births',4,7), row(5,None,'backward',10,11),
            row(6,None,'cross_step_autograd_grad',11,15)]
    result = exclusive_stage_summary(rows)
    assert result['categories_seconds'] == dict(event_other=3., state_validation=3.,
        field_prepare=1., marker_births=3., backward=1.)
    assert result['total_seconds'] == 11.
    with pytest.raises(ValueError,match='missing timing parent'):
        exclusive_stage_summary(rows[1:])
    with pytest.raises(ValueError,match='overlapping sibling'):
        exclusive_stage_summary(rows+[row(7,0,'source_advance_inclusive',6,8)])


@pytest.mark.parametrize('deterministic', [False, True])
def test_cli_rejects_invisible_cuda_without_any_cpu_fallback(tmp_path, monkeypatch, deterministic):
    import sys
    import json
    import os
    from tools.profile_native_coupled_backward import main
    argv=['probe', '--run',
        'data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400', '--anchor', '4110',
        '--output-dir', str(tmp_path/'failed')]
    if deterministic:
        argv.append('--deterministic')
    monkeypatch.setattr(sys,'argv',argv)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG',raising=False)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    def unavailable():
        assert torch.are_deterministic_algorithms_enabled() == deterministic
        assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == (':4096:8' if deterministic else None)
        return False
    monkeypatch.setattr(torch.cuda, 'is_available', unavailable)
    previous=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        assert main() == 1
    finally:
        torch.use_deterministic_algorithms(previous,warn_only=warn)
    result = json.loads((tmp_path/'failed/result.json').read_text())
    assert result['optimizer_updates'] == 0 and not result['success']
    assert result['stage'] == 'cuda_initialization'
    assert result['deterministic'] == deterministic


def test_current_curriculum_restores_full_state_and_matches_observe_loss(tmp_path):
    from copy import deepcopy
    from dataclasses import asdict
    from types import SimpleNamespace
    from tests.test_native_autonomous_pilot import windows
    from src.data.base_coupled import RecordedNode, StepControl
    from tools.train_native_autonomous_pilot import _fit_initials, _install_normalization
    from tools.profile_native_coupled_backward import profile_backward, restore_curriculum_window
    from tools.native_memory_burnin import make_prefix, forecast
    from tools.run_native_cuda_curriculum import observe
    from tools.diagnose_native_real_chain import _array_digest
    items = windows(); norm = _fit_initials(items)
    w = items[0]
    w.controls = tuple(StepControl((i+1)*1e-9,1e-9,(i+2)*1e-9,'external_control','fixture') for i in range(4))
    w.runtime.step_plan_si = tuple(c.dt_si for c in w.controls)
    w.nodes = tuple(RecordedNode(c.end_si, {'field':dict(values=w.runtime.fields_current+.02,
        raw_deltas=w.runtime.fluid_deltas.copy(),units=w.adapter.model.units,time_si=c.end_si)},
        'fixture_loss_only', {'field':c.end_si}) for c in w.controls)
    _install_normalization(w,norm)
    payload = dict(schema='native-autonomous-pilot-v1',gradient_scope=w.adapter.gradient_scope,
        windows=[dict(name=w.name,controls=[asdict(c) for c in w.controls],
            configuration=w.adapter.configuration_fingerprint,initial_state_digest=_array_digest(w.runtime))],
        parameters={n:p.detach().clone() for n,p in w.adapter.model.named_parameters()},
        buffers=[{n:b.detach().clone() for n,b in w.adapter.model.named_buffers()}],
        training_modes=[True],normalization=norm)
    restored = deepcopy(w)
    with torch.no_grad():
        next(restored.adapter.model.parameters()).add_(.7)
        restored.adapter.model.field.stencil_layers[0].alpha0.add_(.2)
    proof = restore_curriculum_window(restored,payload,0)
    assert proof['actual_model_digest'] == _array_digest(w.adapter.model.state_dict())
    assert proof['initial_runtime_digest'] == _array_digest(w.runtime)
    assert not proof['optimizer_loaded']
    rng = np.random.get_state()
    prefix = make_prefix(w.adapter,w.runtime,w.controls,2)
    predictions,_ = forecast(w.adapter,prefix,burnin=0,horizon=2,grad=True)
    expected, rows = observe(SimpleNamespace(normalization=norm),w,predictions)
    np.random.set_state(rng)
    forwards=[]
    hook=restored.adapter.model.register_forward_hook(lambda *a:forwards.append(torch.is_grad_enabled()))
    def labels(cs):
        assert forwards == [False,False,True,True]
        assert tuple(cs) == w.controls[2:]
        return w.nodes[2:]
    try:
        result=profile_backward(restored.adapter,restored.runtime,restored.controls,labels,tmp_path/'current',
            curriculum=dict(burnin=0,normalization=norm), capture_endpoints=True)
    finally:
        hook.remove()
    assert result['completed'] and result['model_unchanged']
    assert result['loss'] == float(expected.detach())
    assert [r['loss'] for r in result['loss_reports']] == [r['loss'] for r in rows]
    assert result['curriculum']['full_prefix_support_steps']==2
    assert result['curriculum']['burnin_steps']==0
    names=[r['stage'] for r in result['stages']]
    assert names.count('native_supervision')==2 and names.count('coupled_node_metrics')==2
    assert names.index('backward') < names.index('cross_step_autograd_grad')
    assert not result['eligible_timing_baseline']
    assert len(result['events']) == 4
    assert [e['segment'] for e in result['events']] == ['prefix','prefix','forecast','forecast']
    for i, row in enumerate(result['events'],1):
        assert row['model_digest'] == proof['actual_model_digest']
        arrays=np.load(tmp_path/f'current/event-{i}-arrays.npz',allow_pickle=False)
        assert arrays['cloud.x'].shape == w.runtime.marker.cloud.x.shape
        assert arrays['cloud.P_imp'].shape == w.runtime.marker.cloud.P_imp.shape
        assert 'fields_previous' in arrays and 'native_density_source' in arrays
        if i<=2:
            np.testing.assert_array_equal(arrays['cloud.x'],prefix.states[i].runtime.marker.cloud.x)
            np.testing.assert_array_equal(arrays['values'],prefix.states[i].values.detach().numpy())
        metadata=torch.load(tmp_path/f'current/event-{i}-state.pt',weights_only=False,mmap=True)
        assert metadata['state_digest'] == row['state_digest']
        assert metadata['clocks'] == row['clocks']
        assert _array_digest(metadata['rng']) == row['rng_digest']
        assert row['marker_slots'] == len(w.runtime.marker.cloud.weight)
        arrays.close()


def test_cli_frozen_coupling_import_has_one_class_identity_before_cuda(tmp_path):
    import subprocess
    import sys
    import json
    import hashlib
    from pathlib import Path
    snapshot=Path('src/models/native_coupling.py').resolve()
    code="""
import sys, pathlib, torch
from tools.profile_native_coupled_backward import main
torch.cuda.is_available=lambda:False
assert 'src.models.native_coupling' not in sys.modules
sys.argv=['probe','--reference-coupling-file',sys.argv[1],'--output-dir',sys.argv[2]]
assert main()==1
from src.models.native_coupling import NativeSourceMesh
import src.models.native_coupling as module
assert NativeSourceMesh is module.NativeSourceMesh
assert pathlib.Path(module.__file__).resolve()==pathlib.Path(sys.argv[2]).resolve()
"""
    proc=subprocess.run([sys.executable,'-c',code,str(snapshot),str(tmp_path/'frozen')],capture_output=True,text=True)
    assert proc.returncode==0, proc.stderr
    row=json.loads((tmp_path/'frozen/result.json').read_text())
    assert row['coupling_implementation']['sha256']==hashlib.sha256(snapshot.read_bytes()).hexdigest()
    assert row['coupling_implementation']['kind']=='trusted_local_frozen_reference'
    assert row['stage']=='cuda_initialization'
