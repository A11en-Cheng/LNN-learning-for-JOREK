"""Real source-event seam with CPU-test spatial execution; no CUDA fallback."""
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.models.native_cuda_spatial import NativeCudaSpatial
from src.models.native_coupling import NativeMarkerSource
from src.models.native_source_time import NativeSourceTimeSampler
from src.models.marker_state import MarkerExecutionBudget
from tests.test_native_coupling import native_fixture


def source_for(reference, backend, *, static=True):
    return NativeMarkerSource(reference.mesh, atomic=reference.atomic,
        atomic_version=reference.configuration['atomic'], potentials_ev=reference.potentials_ev,
        F0=reference.F0, diffusion_ju=reference.diffusion_ju,
        time_sampling='jorek_raw_deltas_v1', field_static=static,
        execution_budget=MarkerExecutionBudget(max_slots=2,max_live=2,chunk_size=1,
            max_particle_steps=10,max_seconds=10.,max_rss_bytes=8*1024**3), spatial_backend=backend)


def cpu_test_factory(mesh, *, device):
    assert device == 'cuda'  # Production selection never silently falls back.
    return NativeCudaSpatial(mesh, device='cpu')


@pytest.mark.parametrize('static', [True, False])
def test_complete_source_event_matches_cpu_and_uploads_once_per_time(static):
    _, reference, state, u = native_fixture()
    # Two marker chunks and two substeps expose repeated uploads per chunk.
    for name in state.cloud.array_names():
        getattr(state.cloud, name)[1] = getattr(state.cloud, name)[0]
    state.clock.particle_dt_si = .5e-9
    raw = np.zeros_like(u); raw[4, 0] = .1
    sampler = NativeSourceTimeSampler(u, raw, time_prev_si=0., time_now_si=1e-9,
        tstep_si=1e-9, static=static)
    cpu = source_for(reference, 'batched_v1', static=static)
    with patch('src.models.native_cuda_spatial.NativeCudaSpatial', side_effect=cpu_test_factory):
        candidate = source_for(reference, 'cuda_fp64_v1', static=static)
    expected, _ = cpu.advance(state, 1e-9, u, u, time_sampler=sampler)
    with patch.object(candidate._spatial, 'freeze_fields', wraps=candidate._spatial.freeze_fields) as uploads:
        actual, _ = candidate.advance(state, 1e-9, u, u, time_sampler=sampler)
    keys = [call.args[0] for call in uploads.call_args_list]
    assert keys.count('feedback') == 1
    assert keys.count('field') == (1 if static else 2)
    for name in state.cloud.array_names():
        np.testing.assert_allclose(getattr(actual.cloud, name), getattr(expected.cloud, name),
            rtol=2e-12, atol=1e-25)
    np.testing.assert_allclose(actual.feedback.values, expected.feedback.values, rtol=2e-12, atol=1e-25)
    np.testing.assert_allclose(actual.feedback.deltas, expected.feedback.deltas, rtol=2e-12, atol=1e-25)
    assert actual.clock == expected.clock
    assert not candidate._spatial._fields and not candidate._spatial._in_event
    assert candidate.configuration['spatial_backend'] == 'cuda_fp64_v1'
    assert candidate.configuration != cpu.configuration


def test_failure_clears_snapshots_and_retry_uses_changed_field():
    _, reference, state, u = native_fixture()
    before = deepcopy(state)
    with patch('src.models.native_cuda_spatial.NativeCudaSpatial', side_effect=cpu_test_factory):
        candidate = source_for(reference, 'cuda_fp64_v1')
    def sampler(values):
        return NativeSourceTimeSampler(values, np.zeros_like(values), time_prev_si=0.,
            time_now_si=1e-9, tstep_si=1e-9, static=True)
    def fail(*args):
        raise RuntimeError('atomic failure')
    candidate.atomic = fail
    with pytest.raises(RuntimeError, match='atomic failure'):
        candidate.advance(state, 1e-9, u, u, time_sampler=sampler(u))
    assert not candidate._spatial._fields and not candidate._spatial._in_event
    np.testing.assert_array_equal(state.cloud.P_imp, before.cloud.P_imp)
    assert state.clock == before.clock
    candidate.atomic = reference.atomic
    changed = u.copy(); changed[4, 0] = .75
    actual, _ = candidate.advance(state, 1e-9, changed, changed, time_sampler=sampler(changed))
    assert actual.cloud.P_imp[0, 1] == pytest.approx(.02, abs=1e-14)
    assert not candidate._spatial._fields


def test_cuda_selection_fails_explicitly_without_device():
    _, reference, _, _ = native_fixture()
    with patch('torch.cuda.is_available', return_value=False):
        with pytest.raises(RuntimeError, match='CUDA'):
            source_for(reference, 'cuda_fp64_v1')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real CUDA device required')
def test_real_cuda_two_source_events_preserve_cloud_and_feedback():
    _, reference, state, u = native_fixture()
    cpu = source_for(reference, 'batched_v1')
    cuda = source_for(reference, 'cuda_fp64_v1')
    expected = deepcopy(state); actual = deepcopy(state)
    for event in range(2):
        start = event*1e-9; end = start+1e-9
        sampler = NativeSourceTimeSampler(u, np.zeros_like(u), time_prev_si=start,
            time_now_si=end, tstep_si=1e-9, static=True)
        for candidate in (expected, actual):
            candidate.clock.field_start_si = start; candidate.clock.field_end_si = end
        expected, _ = cpu.advance(expected, end, u, u, time_sampler=sampler)
        actual, _ = cuda.advance(actual, end, u, u, time_sampler=sampler)
        for name in state.cloud.array_names():
            np.testing.assert_allclose(getattr(actual.cloud, name), getattr(expected.cloud, name),
                rtol=2e-12, atol=1e-25)
        for name in ('values', 'deltas'):
            np.testing.assert_allclose(getattr(actual.feedback, name), getattr(expected.feedback, name),
                rtol=2e-12, atol=1e-25)
        assert actual.clock == expected.clock
        assert not cuda._spatial._fields
