"""Instrumentation must replay the same full cached event, not change state."""
from copy import deepcopy
from types import SimpleNamespace

from src.models.native_coupling import NativeMarkerSource


def test_profiler_preserves_complete_state_and_rng(tmp_path, monkeypatch):
    from tests.test_base_coupled_training import event_fixture
    import tools.native_real_chain_builder as builder
    import tools.profile_native_element_function_latency as profiler

    def fixture_builder(*args, **kwargs):
        model, old, shard, runtime = event_fixture(n=2)
        source = NativeMarkerSource(old.mesh, atomic=old.atomic,
            atomic_version=old.configuration['atomic'], potentials_ev=old.potentials_ev,
            F0=old.F0, diffusion_ju=old.diffusion_ju, spatial_backend='batched_v1')
        model.source_configuration = deepcopy(source.configuration)
        runtime.native_context.configuration['model'] = model.checkpoint_configuration
        return model, source, shard, runtime, None

    rows = [dict(start_si=(i+1)*1e-9, end_si=(i+2)*1e-9, dt_si=1e-9,
        origin='external_control', source='fixture', phase='ordinary') for i in range(2)]
    monkeypatch.setattr(builder, 'build_real', fixture_builder)
    monkeypatch.setattr(profiler, 'recorded_controls', lambda *args: rows)
    result = {}
    profiler.execute(SimpleNamespace(device='cpu', threads=1, seed=73, run_index=0,
        output_dir=tmp_path, memory_mib=8192), result)
    assert result['success'] and result['profile_transparency_exact']
    assert result['prefix_unchanged'] and result['weights_unchanged']
    assert len(result['events']) == 2
    assert all(row['exclusive_timing']['accounting_closed'] for row in result['events'])
    assert (tmp_path/'profiled.prof').exists()
    assert (tmp_path/'profiled-profile.json').exists()
    assert not (tmp_path/'boundary_timed.prof').exists()
