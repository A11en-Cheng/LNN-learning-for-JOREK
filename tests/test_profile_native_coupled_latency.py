import json
import random

import numpy as np
import torch
import pytest

from tests.test_native_production_torch_integration import fixture, control, AnalyticEquilibrium
from src.training.base_coupled import TorchRecomputedSourceEvents
from tools.diagnose_native_real_chain import _array_digest


def test_profile_complete_events_preserves_full_state_and_rng(tmp_path):
    from tools.profile_native_coupled_latency import profile_events, rng_state
    model, source, shard, runtime = fixture()
    adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=AnalyticEquilibrium())
    controls = (control(0), control(1))
    random.seed(41); np.random.seed(41); torch.manual_seed(41)
    state = adapter.initialize(runtime)
    with torch.no_grad():
        for c in controls:
            state, _ = adapter.advance(state, c)
    expected = _array_digest(state), _array_digest(rng_state())
    random.seed(41); np.random.seed(41); torch.manual_seed(41)
    result = profile_events(adapter, runtime, controls, tmp_path / 'profiled', capture_arrays=True)
    assert (result['final_state_digest'], result['final_rng_digest']) == expected
    assert len(result['events']) == 2
    from src.models.native_cuda_spatial import NativeCudaSpatial
    assert result['initial_spatial_mesh_binding'] == NativeCudaSpatial(source.mesh, device='cpu').configuration['mesh_binding']
    assert result['model_unchanged']
    assert (tmp_path / 'profiled/event-1.prof').is_file()
    arrays = np.load(tmp_path / 'profiled/event-2-arrays.npz')
    np.testing.assert_array_equal(arrays['values'], state.values.numpy())
    np.testing.assert_array_equal(arrays['raw_deltas'], state.raw_deltas.numpy())
    np.testing.assert_array_equal(arrays['memory'], state.memory.numpy())
    assert 'native_density_source' in arrays
    for name in state.runtime.marker.cloud.array_names():
        np.testing.assert_array_equal(arrays['cloud.'+name], getattr(state.runtime.marker.cloud, name))
    assert arrays['cloud.group'].item() == state.runtime.marker.cloud.group
    np.testing.assert_array_equal(arrays['feedback.values'], state.runtime.marker.feedback.values)
    np.testing.assert_array_equal(arrays['feedback.deltas'], state.runtime.marker.feedback.deltas)
    np.testing.assert_array_equal(arrays['shards.positions'], state.runtime.shards['positions'])
    assert result['events'][-1]['clocks']['fluid']['time_si'] == state.runtime.fluid_time_si
    top = json.loads((tmp_path / 'profiled/event-1-profile.json').read_text())
    assert top['top_cumulative'] and top['top_self']
    random.seed(41); np.random.seed(41); torch.manual_seed(41)
    plain = profile_events(adapter, runtime, controls, tmp_path / 'plain', profile=False)
    assert (plain['final_state_digest'], plain['final_rng_digest']) == expected
    assert plain['events'][-1]['components'] == result['events'][-1]['components']
    assert not list((tmp_path / 'plain').glob('*.prof'))
    with pytest.raises(FileExistsError):
        profile_events(adapter, runtime, controls, tmp_path / 'plain')


def test_complete_event_replay_loads_exact_initial_parameters_buffers_and_rng(tmp_path):
    from tools.profile_native_coupled_latency import profile_events
    model, source, shard, runtime = fixture()
    adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=AnalyticEquilibrium())
    random.seed(81); np.random.seed(81); torch.manual_seed(81)
    first = profile_events(adapter, runtime, (control(0), control(1)), tmp_path / 'first', profile=False)
    snapshot = tmp_path / 'first/initial-model.pt'
    assert snapshot.is_file()
    model2, source2, shard2, runtime2 = fixture()
    with torch.no_grad():
        next(model2.parameters()).add_(.1)
        for name, buffer in model2.named_buffers():
            if name.endswith(('term_lambda', 'alpha0')):
                buffer.add_(.2)
    adapter2 = TorchRecomputedSourceEvents(model2, source2, shard2, equilibrium_provider=AnalyticEquilibrium())
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    replay = profile_events(adapter2, runtime2, (control(0), control(1)), tmp_path / 'replay',
                            profile=False, model_state=snapshot)
    assert replay['initial_model_digest'] == first['initial_model_digest']
    assert replay['final_state_digest'] == first['final_state_digest']
    assert replay['final_rng_digest'] == first['final_rng_digest']


def test_model_snapshot_rejects_changed_source_semantics_before_replay(tmp_path):
    from copy import deepcopy
    from tools.profile_native_coupled_latency import profile_events
    model, source, shard, runtime = fixture()
    adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=AnalyticEquilibrium())
    profile_events(adapter, runtime, (control(0), control(1)), tmp_path/'original', profile=False)
    source.configuration['explicit_test_backend'] = 'changed_semantics'
    model.source_configuration = deepcopy(source.configuration)
    runtime.native_context.configuration['model'] = model.checkpoint_configuration
    with torch.no_grad():
        next(model.parameters()).add_(.3)
    model.train()
    before = _array_digest(model.state_dict())
    from tools.profile_native_coupled_latency import rng_state
    before_rng = _array_digest(rng_state())
    with pytest.raises(ValueError, match='configuration'):
        profile_events(adapter, runtime, (control(0), control(1)), tmp_path/'changed', profile=False,
                       model_state=tmp_path/'original/initial-model.pt')
    assert _array_digest(model.state_dict()) == before
    assert _array_digest(rng_state()) == before_rng
    assert model.training


def test_frozen_joint_field_repeat_probe_records_numeric_outputs_without_mutation(tmp_path):
    from copy import deepcopy
    from tests.test_native_joint_source import joint_fixture
    from tools.profile_native_coupled_latency import field_repeat_probe
    model, _, _, _, u, raw, kwargs = joint_fixture()
    model.eval()
    before = _array_digest(model.state_dict())
    result = field_repeat_probe(model, deepcopy(model), u, raw, kwargs, tmp_path/'field', repeats=2)
    assert result['model_digest'] == before
    assert result['model_unchanged'] and result['inputs_unchanged']
    assert all(r['exact_to_mode_reference'] for r in result['observations'])
    assert (tmp_path/'field/default-original-0.npz').is_file()
    assert result['observations'][0]['metrics']['memory']['relative_l2'] == 0.


def test_explicit_spatial_backend_replay_preserves_weights_and_rejects_other_changes(tmp_path):
    from copy import deepcopy
    from tools.profile_native_coupled_latency import profile_events
    model, source, shard, runtime = fixture()
    adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=AnalyticEquilibrium())
    first = profile_events(adapter, runtime, (control(0), control(1)), tmp_path/'first', profile=False)
    source.spatial_backend = 'batched_v1'
    source.configuration['spatial_backend'] = 'batched_v1'
    model.source_configuration = deepcopy(source.configuration)
    runtime.native_context.configuration['model'] = model.checkpoint_configuration
    snapshot = tmp_path/'first/initial-model.pt'
    replay = profile_events(adapter, runtime, (control(0), control(1)), tmp_path/'allowed',
        profile=False, model_state=snapshot, allow_spatial_backend_change=True)
    assert replay['initial_model_components'] == first['initial_model_components']
    assert replay['initial_model_digest'] == first['initial_model_digest']
    assert replay['model_fingerprint'] != first['model_fingerprint']
    model.source_configuration['diffusion_ju'] = 7.
    with pytest.raises(ValueError, match='configuration'):
        profile_events(adapter, runtime, (control(0), control(1)), tmp_path/'rejected',
            profile=False, model_state=snapshot, allow_spatial_backend_change=True)


@pytest.mark.parametrize('deterministic', [False, True])
def test_cli_sets_and_records_determinism_before_first_cuda_access(tmp_path, monkeypatch, deterministic):
    import os
    import sys
    from tools.profile_native_coupled_latency import main
    output = tmp_path/'attempt'
    argv = ['probe', '--run', 'data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400',
            '--anchor', '4110', '--output-dir', str(output)]
    if deterministic:
        argv.append('--deterministic')
    monkeypatch.setattr(sys, 'argv', argv)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    def unavailable():
        assert torch.are_deterministic_algorithms_enabled() == deterministic
        if deterministic:
            assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'
        return False  # Explicit hardware boundary: never launch a real job in this test.
    monkeypatch.setattr(torch.cuda, 'is_available', unavailable)
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        assert main() == 1
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)
    result = json.loads((output/'result.json').read_text())
    assert result['settings']['deterministic'] == deterministic
    assert result['deterministic'] == deterministic
    assert result['environment']['CUBLAS_WORKSPACE_CONFIG'] == (':4096:8' if deterministic else None)
    assert result['error_type'] == 'RuntimeError'
