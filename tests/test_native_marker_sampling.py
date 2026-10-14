"""Marker-window sampling retains full state and reports bounded evaluation work."""
from copy import deepcopy

import numpy as np
import pytest

from src.models.native_coupling import NativeMarkerSource
from src.models.native_source_time import NativeSourceTimeSampler
from tests.test_native_coupling import native_fixture


def window(*, static, time_sampling='jorek_raw_deltas_v1'):
    mesh, original, state, values = native_fixture()
    state.clock.particle_dt_si = 2.5e-10
    source = NativeMarkerSource(
        mesh, atomic=original.atomic, atomic_version=original.configuration['atomic'],
        potentials_ev=original.potentials_ev, F0=original.F0,
        diffusion_ju=original.diffusion_ju, spatial_backend='batched_v1',
        time_sampling=time_sampling, field_static=static)
    sampler = None
    if time_sampling == 'jorek_raw_deltas_v1':
        sampler = NativeSourceTimeSampler(
            values, np.zeros_like(values), time_prev_si=0., time_now_si=1e-9,
            tstep_si=1e-9, static=static)
    return source, state, values, sampler


def test_static_marker_window_uses_one_snapshot_without_discarded_gradients():
    source, initial, values, sampler = window(static=True)
    before = deepcopy(initial)
    result, diagnostic = source.advance(initial, 1e-9, values, values, time_sampler=sampler)
    work = diagnostic['sampling']
    assert work['field_snapshots'] == 1
    assert work['batches'] == 4
    assert work['field_value_channels'] == [4, 6, 7, 8]
    assert work['field_gradient_channels'] == [0, 1, 7]
    assert work['feedback_value_channels'] == [3, 4]
    assert work['feedback_gradient_channels'] == []
    assert result.feedback.values.shape == (5, 4, 1, source.mesh.n_nodes)
    assert result.cloud.P_imp[0, 1] > 0
    for name in initial.cloud.array_names():
        np.testing.assert_array_equal(getattr(initial.cloud, name), getattr(before.cloud, name))
    assert initial.clock == before.clock


@pytest.mark.parametrize('time_sampling', ['jorek_raw_deltas_v1', 'endpoint_reference'])
def test_time_dependent_marker_window_prepares_each_sampling_time(time_sampling):
    source, initial, values, sampler = window(static=False, time_sampling=time_sampling)
    _, diagnostic = source.advance(initial, 1e-9, values, values, time_sampler=sampler)
    assert diagnostic['sampling']['field_snapshots'] == diagnostic['n_steps'] == 4


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_marker_window_rejects_invalid_unused_field_without_mutating_state(bad):
    source, initial, values, _ = window(static=True)
    before = deepcopy(initial)
    invalid = values.copy()
    invalid[3, 3, 0, -1] = bad  # Not consumed by marker transport, still native state.
    with pytest.raises(ValueError, match='coefficients'):
        source.advance(initial, 1e-9, invalid, values)
    for name in initial.cloud.array_names():
        np.testing.assert_array_equal(getattr(initial.cloud, name), getattr(before.cloud, name))
    np.testing.assert_array_equal(initial.feedback.values, before.feedback.values)
    np.testing.assert_array_equal(initial.feedback.deltas, before.feedback.deltas)
    assert initial.clock == before.clock


def test_marker_snapshots_are_event_local_when_same_field_object_changes():
    source, initial, values, sampler = window(static=True)
    first, _ = source.advance(initial, 1e-9, values, values, time_sampler=sampler)
    first.clock.field_start_si = 1e-9
    first.clock.field_end_si = 2e-9
    unchanged = values.copy()
    values[4, 0] *= 2.
    changed_sampler = NativeSourceTimeSampler(
        values, np.zeros_like(values), time_prev_si=1e-9, time_now_si=2e-9,
        tstep_si=1e-9, static=True)
    actual, diagnostic = source.advance(first, 2e-9, values, values, time_sampler=changed_sampler)
    expected, _ = deepcopy(source).advance(first, 2e-9, values, values, time_sampler=changed_sampler)
    old_sampler = NativeSourceTimeSampler(
        unchanged, np.zeros_like(unchanged), time_prev_si=1e-9, time_now_si=2e-9,
        tstep_si=1e-9, static=True)
    old, _ = source.advance(first, 2e-9, unchanged, unchanged, time_sampler=old_sampler)
    for name in actual.cloud.array_names():
        np.testing.assert_array_equal(getattr(actual.cloud, name), getattr(expected.cloud, name))
    np.testing.assert_array_equal(actual.feedback.values, expected.feedback.values)
    np.testing.assert_array_equal(actual.feedback.deltas, expected.feedback.deltas)
    assert actual.cloud.P_imp[0, 1] > old.cloud.P_imp[0, 1]
    assert diagnostic['sampling']['field_snapshots'] == 1


@pytest.mark.parametrize('static', [False, True])
def test_marker_chunks_share_only_the_current_sampling_snapshot(static):
    from src.models.marker_state import MarkerExecutionBudget
    source, initial, values, sampler = window(static=static)
    for name in initial.cloud.array_names():
        a = getattr(initial.cloud, name)
        a[1] = a[0]
    initial.cloud.i_life[1] = 5
    source.execution_budget = MarkerExecutionBudget(
        max_slots=2, max_live=2, chunk_size=1, max_particle_steps=4,
        max_seconds=30., max_rss_bytes=8*1024**3)
    result, diagnostic = source.advance(initial, 1e-9, values, values, time_sampler=sampler)
    assert diagnostic['sampling']['batches'] == 8
    assert diagnostic['sampling']['field_snapshots'] == (1 if static else 4)
    assert result.cloud.alive.sum() == 2
    np.testing.assert_array_equal(result.cloud.P_imp[0], result.cloud.P_imp[1])
