import json
import random

import numpy as np
import pytest
import torch

from tools.diagnose_native_real_chain import _array_digest


def test_six_public_events_match_plain_path_without_truth_or_model_mutation(tmp_path):
    from tests.test_native_production_torch_integration import fixture, control, AnalyticEquilibrium
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from tools.benchmark_native_multi_window import measure_windows
    from tools.profile_native_coupled_latency import rng_state
    model, source, shard, runtime = fixture()
    runtime.step_plan_si = (1e-9,)*6
    adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=AnalyticEquilibrium())
    controls = tuple(control(i) for i in range(6))
    random.seed(18); np.random.seed(18); torch.manual_seed(18)
    with torch.no_grad():
        state = adapter.initialize(runtime)
        for c in controls:
            state, _ = adapter.advance(state, c)
    expected = _array_digest(state)
    expected_rng = _array_digest(rng_state())
    random.seed(18); np.random.seed(18); torch.manual_seed(18)
    result = measure_windows(adapter, runtime, controls, tmp_path/'measured')
    assert result['success'] and result['completed_events'] == 6
    assert result['final_state_digest'] == expected
    assert _array_digest(rng_state()) == expected_rng
    assert result['model_unchanged'] and result['initial_runtime_unchanged']
    assert not result['future_truth_feedback'] and not result['optimizer_instantiated']
    assert result['optimizer_updates'] == result['backward_calls'] == 0
    assert result['subsequent_slice_distribution']['count'] == 5
    for i, row in enumerate(result['events'], 1):
        assert row['slice'] == i
        assert row['exclusive_timing']['accounting_closed']
        assert sum(row['exclusive_timing']['categories_seconds'].values()) == pytest.approx(row['seconds'])
    assert sum(r['stage'] == 'marker_births' for r in result['stages']) == 6
    assert 'advance' not in source.__dict__ and 'forward' not in model.__dict__


def test_failure_retains_completed_slice_and_incomplete_timer(tmp_path):
    from tests.test_native_production_torch_integration import fixture, control, AnalyticEquilibrium
    from src.training.base_coupled import TorchRecomputedSourceEvents
    from tools.benchmark_native_multi_window import measure_windows
    model, source, shard, runtime = fixture()
    adapter = TorchRecomputedSourceEvents(model, source, shard, equilibrium_provider=AnalyticEquilibrium())
    with pytest.raises(ValueError, match='control'):
        measure_windows(adapter, runtime, (control(0), control(3)), tmp_path/'failed')
    result = json.loads((tmp_path/'failed/result.json').read_text())
    assert not result['success'] and result['completed_events'] == 1
    assert result['model_unchanged'] and result['initial_runtime_unchanged']
    assert result['stages'][-1]['completed'] is False


def test_scalar_control_extension_never_reads_future_field_arrays(tmp_path, monkeypatch):
    import h5py
    from tools import benchmark_native_multi_window as bench
    run = tmp_path/'fixture'; run.mkdir()
    (tmp_path/'data').mkdir()
    (tmp_path/'data/restart_events.json').write_text(json.dumps({'runs': {'fixture': {'restart_events': [11]}}}))
    def scalars(index, end, dt):
        with h5py.File(run/f'jorek{index:06d}.h5', 'w') as f:
            for k, value in dict(index_now=index, t_now=end, tstep=dt, t_norm=1.).items():
                f[k] = [value]
            # No fields/shards datasets exist. Timing extension must only read scalars.
    scalars(11, 1., 1.)
    scalars(14, 4., 1.)
    scalars(15, 5., 1.)
    monkeypatch.setattr(bench, 'ROOT', tmp_path)
    monkeypatch.setattr(bench, '_initial_pairing', lambda *a: [dict(end_si=2.), dict(end_si=3.)])
    rows = bench.recorded_controls(run, 10, 4, bench.Stages())
    assert len(rows) == 4 and rows[-1]['end_si'] == 5.
    (tmp_path/'data/restart_events.json').write_text(json.dumps({'runs': {'fixture': {'restart_events': [11, 14]}}}))
    with pytest.raises(ValueError, match='restart'):
        bench.recorded_controls(run, 10, 4, bench.Stages())


def test_subsequent_summary_does_not_mix_first_event_cost():
    from tools.benchmark_native_multi_window import distribution
    summary = distribution([1., 2., 3., 4., 5.])
    assert summary == dict(count=5, mean_seconds=3., median_seconds=3., min_seconds=1., max_seconds=5.)
    assert distribution([])['mean_seconds'] is None
