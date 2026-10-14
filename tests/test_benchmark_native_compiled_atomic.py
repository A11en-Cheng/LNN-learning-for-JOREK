"""No real timing jobs: same provider-only fork/control/output contract."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from src.models.native_coupling import NativeMarkerSource, NativeNeonAtomicRates
from src.models.source_module import NeonTables


def compiled_fixture(kernel=None, *, n=4):
    from tests.test_base_coupled_training import event_fixture
    model, old, shard, runtime = event_fixture(n=n)
    axes = np.array([5., 25.]), np.array([-3., 8.])
    table = (*axes, np.full((10, 2, 2), -22.))
    rates = NativeNeonAtomicRates(NeonTables(table, table, table, np.arange(1., 11.)),
        radiation_mode='plt_prb', prb=table)
    if kernel is not None:
        from src.models.native_compiled_atomic import CompiledNeonAtomicRates
        rates = CompiledNeonAtomicRates(rates, kernel=kernel)
    source = NativeMarkerSource(old.mesh, atomic=rates, atomic_version=rates.version,
        potentials_ev=old.potentials_ev, F0=old.F0, diffusion_ju=old.diffusion_ju,
        spatial_backend='batched_v1')
    model.source_configuration = deepcopy(source.configuration)
    runtime.native_context.configuration['model'] = model.checkpoint_configuration
    return model, source, shard, runtime, None


def test_complete_two_event_forks_have_exact_state_and_unique_evidence(tmp_path, monkeypatch):
    import tools.benchmark_native_compiled_atomic as benchmark
    import tools.native_real_chain_builder as builder
    rows = [dict(start_si=(i+1)*1e-9, end_si=(i+2)*1e-9, dt_si=1e-9,
        origin='external_control', source='fixture', phase='ordinary') for i in range(3)]
    monkeypatch.setattr(builder, 'build_real', lambda *args, **kwargs: compiled_fixture(n=3))
    monkeypatch.setattr(benchmark, 'recorded_controls', lambda *args: rows)
    monkeypatch.setattr(benchmark, 'microbenchmark', lambda *args: [{'exact': True}])
    args = SimpleNamespace(device='cpu', threads=1, seed=124, run_index=0,
        output_dir=tmp_path, memory_mib=8192, forecast_events=2, repeats=2)
    result = {}
    benchmark.execute(args, result)
    assert result['success']
    assert len(result['iterations']) == 4
    assert len(result['comparisons']) == 2
    assert all(row['all_components_and_rng_exact'] for row in result['comparisons'])
    assert all(len(row['events']) == 2 for row in result['iterations'])
    assert len(list(tmp_path.glob('progress-*.json'))) == 4


def test_two_step_gradient_and_new_model_checkpoint_resume(tmp_path):
    import torch
    import pytest
    from src.models.native_compiled_atomic import CompiledAtomicInterpolation, build_kernel
    from src.training.base_coupled import TorchRecomputedSourceEvents, run_sequence, save_session, load_session
    from tests.test_base_coupled_training import sequence
    from tools.profile_native_coupled_latency import component_digests
    kernel = CompiledAtomicInterpolation(build_kernel(tmp_path/'atomic.so'))
    model, source, shard, runtime, _ = compiled_fixture(kernel)
    adapter = TorchRecomputedSourceEvents(model, source, shard)
    seq = sequence()
    initial = adapter.initialize(runtime); initial.values.requires_grad_()
    first, _ = adapter.advance(initial, seq.controls[0])
    second, _ = adapter.advance(first, seq.controls[1])
    second.values.square().sum().backward()
    assert initial.values.grad is not None and torch.isfinite(initial.values.grad).all()
    assert initial.values.grad.abs().sum() > 0
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
        for p in model.temporal_core.parameters())
    assert adapter.gradient_scope == 'torch_field_bptt_stop_gradient_numpy_source_shard'
    with torch.no_grad():
        prefix = run_sequence(seq, adapter, adapter.initialize(runtime), stop_cursor=2)
        save_session(tmp_path/'session.pt', seq, adapter, prefix)
        expected = run_sequence(seq, adapter, session=prefix)
        fresh, other_source, other_shard, _, _ = compiled_fixture(kernel)
        with torch.no_grad(): fresh.field.decoder.weight.add_(.1)
        other = TorchRecomputedSourceEvents(fresh, other_source, other_shard)
        loaded = load_session(tmp_path/'session.pt', seq, other)
        actual = run_sequence(seq, other, session=loaded)
    assert component_digests(actual.state) == component_digests(expected.state)
    old, old_source, old_shard, _, _ = compiled_fixture()
    with pytest.raises(ValueError, match='fingerprint'):
        load_session(tmp_path/'session.pt', seq, TorchRecomputedSourceEvents(old, old_source, old_shard))
