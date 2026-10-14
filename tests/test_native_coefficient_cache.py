"""Coefficient cache is accepted only through complete public source events."""
from copy import deepcopy

import numpy as np
import pytest

from src.models.native_coefficient_cache_experiment import NativeCoefficientCachedSource
from src.models.native_source_time import NativeSourceTimeSampler
from tests.test_native_locator_events import event_fixture, assert_runtime_exact
from tests.test_native_marker_sampling import window
from tools.benchmark_native_cpu_threads import stable_diagnostics
from tools.diagnose_native_real_chain import _array_digest


@pytest.mark.parametrize('static', [False, True])
@pytest.mark.parametrize('time_sampling', ['endpoint_reference', 'jorek_raw_deltas_v1'])
def test_event_cache_matches_full_native_window_and_dynamic_history(static, time_sampling):
    source, initial, values, sampler = window(static=static, time_sampling=time_sampling)
    if sampler is not None:
        raw = values*.02
        sampler = NativeSourceTimeSampler(values, raw, time_prev_si=0., time_now_si=1e-9,
                                           tstep_si=1e-9, static=static)
    before = deepcopy(initial)
    expected, diag = source.advance(initial, 1e-9, values*.99, values, time_sampler=sampler)
    cached = NativeCoefficientCachedSource(source)
    actual, got = cached.advance(initial, 1e-9, values*.99, values, time_sampler=sampler)
    assert_runtime_exact(actual, expected)
    assert_runtime_exact(initial, before)
    assert _array_digest(stable_diagnostics(diag)) == _array_digest(stable_diagnostics(got))
    assert cached.last_stats['feedback_builds'] == 1
    assert cached.last_stats['field_builds'] == (1 if static and sampler is not None else 4)


@pytest.mark.parametrize('scenario', ['normal', 'crossing', 'lost'])
@pytest.mark.parametrize('layout', ['C', 'F', 'reversed'])
def test_cache_preserves_locator_events_and_field_layouts(scenario, layout):
    source, initial, values = event_fixture(backend='batched_v1', scenario=scenario)
    for name in initial.cloud.array_names():
        setattr(initial.cloud, name, np.repeat(getattr(initial.cloud, name)[:1], 10, axis=0))
    initial.cloud.i_life[:] = np.arange(1, 11)
    initial.cloud.weight[:] *= np.linspace(.8, 1.2, 10)
    if layout == 'F':
        values = np.asfortranarray(values)
        initial.feedback.values = np.asfortranarray(initial.feedback.values)
    elif layout == 'reversed':
        values = values[..., ::-1].copy()[..., ::-1]
        initial.feedback.values = initial.feedback.values[..., ::-1].copy()[..., ::-1]
    before = deepcopy(initial)
    expected, diag = source.advance(initial, 1e-9, values, values)
    actual, got = NativeCoefficientCachedSource(source).advance(initial, 1e-9, values, values)
    assert_runtime_exact(actual, expected)
    assert_runtime_exact(initial, before)
    assert _array_digest(stable_diagnostics(diag)) == _array_digest(stable_diagnostics(got))


def test_same_input_object_changes_after_event_and_failure_retry_are_not_stale():
    source, initial, values, sampler = window(static=True)
    cached = NativeCoefficientCachedSource(source)
    first, _ = cached.advance(initial, 1e-9, values, values, time_sampler=sampler)
    first.clock.field_start_si = 1e-9
    first.clock.field_end_si = 2e-9
    values[4, 0] *= 2.
    first.feedback.values[3, 0] = .4
    first.feedback.values[4, 0] = .1
    first.feedback.deltas[3, 0] = .03
    sampler = NativeSourceTimeSampler(values, np.zeros_like(values), time_prev_si=1e-9,
                                      time_now_si=2e-9, tstep_si=1e-9, static=True)
    before = deepcopy(first)
    atomic = source.atomic
    def fail(te, ne):
        raise RuntimeError('prescribed cache event failure')
    source.atomic = fail
    with pytest.raises(RuntimeError, match='prescribed cache event failure'):
        cached.advance(first, 2e-9, values, values, time_sampler=sampler)
    assert_runtime_exact(first, before)
    source.atomic = atomic
    expected, _ = source.advance(first, 2e-9, values, values, time_sampler=sampler)
    actual, _ = cached.advance(first, 2e-9, values, values, time_sampler=sampler)
    assert_runtime_exact(actual, expected)
    assert cached.last_stats['field_builds'] == cached.last_stats['feedback_builds'] == 1


def test_cache_budget_failure_does_not_commit_and_invalid_unconsumed_field_is_rejected():
    source, initial, values, sampler = window(static=True)
    before = deepcopy(initial)
    with pytest.raises(MemoryError, match='cache budget'):
        NativeCoefficientCachedSource(source, max_cache_bytes=1).advance(
            initial, 1e-9, values, values, time_sampler=sampler)
    assert_runtime_exact(initial, before)
    bad = values.copy()
    bad[3, 3, 0, -1] = np.nan
    with pytest.raises(ValueError, match='coefficients'):
        NativeCoefficientCachedSource(source).advance(initial, 1e-9, bad, values, time_sampler=sampler)
    assert_runtime_exact(initial, before)


@pytest.mark.parametrize('layout', ['C', 'F', 'reversed'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_cache_random_full_cubic_population_is_exact(layout, dtype):
    source, initial, values = event_fixture(backend='batched_v1', scenario='normal')
    initial.clock.particle_dt_si = 2.5e-10
    rng = np.random.default_rng(820)
    values = (values+rng.normal(scale=1e-4, size=values.shape)).astype(dtype)
    if layout == 'F':
        values = np.asfortranarray(values)
    elif layout == 'reversed':
        values = values[..., ::-1].copy()[..., ::-1]
    for name in initial.cloud.array_names():
        setattr(initial.cloud, name, np.repeat(getattr(initial.cloud, name)[:1], 31, axis=0))
    initial.cloud.st[:] = rng.uniform(.1, .9, (31, 2))
    initial.cloud.i_elm[:] = np.arange(31)%len(source.mesh.gauss)+1
    initial.cloud.x[:, :2] = np.array([source.mesh.point(int(e)-1, st, values[:1])[2]
        for e, st in zip(initial.cloud.i_elm, initial.cloud.st)])
    initial.cloud.weight[:] *= rng.uniform(.5, 1.5, 31)
    initial.cloud.i_life[:] = np.arange(1, 32)
    initial.cloud.P_imp[:] = rng.uniform(.01, 1., (31, 11))
    initial.cloud.P_imp[:] /= initial.cloud.P_imp.sum(axis=1)[:, None]
    initial.feedback.values[3, 0] = .02
    initial.feedback.values[4, 0] = .01
    initial.feedback.deltas[3, 0] = .001
    expected, diag = source.advance(initial, 1e-9, values*.99, values)
    actual, got = NativeCoefficientCachedSource(source).advance(initial, 1e-9, values*.99, values)
    assert_runtime_exact(actual, expected)
    assert _array_digest(stable_diagnostics(diag)) == _array_digest(stable_diagnostics(got))


def test_fresh_cache_after_runtime_serialization_preserves_two_event_history():
    import pickle
    source, initial, values, sampler = window(static=True)
    cached = NativeCoefficientCachedSource(source)
    native_first, native_diag = source.advance(initial, 1e-9, values, values, time_sampler=sampler)
    first, first_diag = cached.advance(initial, 1e-9, values, values, time_sampler=sampler)
    assert_runtime_exact(first, native_first)
    assert _array_digest(stable_diagnostics(first_diag)) == _array_digest(stable_diagnostics(native_diag))
    first.clock.field_start_si = 1e-9
    first.clock.field_end_si = 2e-9
    native_first.clock.field_start_si = 1e-9
    native_first.clock.field_end_si = 2e-9
    restored = pickle.loads(pickle.dumps(first))
    values[4, 0] *= 1.01
    sampler = NativeSourceTimeSampler(values, values*.001, time_prev_si=1e-9,
                                      time_now_si=2e-9, tstep_si=1e-9, static=True)
    expected, expected_diag = source.advance(native_first, 2e-9, values, values, time_sampler=sampler)
    continued, continued_diag = cached.advance(first, 2e-9, values, values, time_sampler=sampler)
    resumed, resumed_diag = NativeCoefficientCachedSource(source).advance(
        restored, 2e-9, values, values, time_sampler=sampler)
    assert_runtime_exact(continued, expected)
    assert_runtime_exact(resumed, expected)
    assert _array_digest(stable_diagnostics(continued_diag)) == _array_digest(stable_diagnostics(expected_diag))
    assert _array_digest(stable_diagnostics(resumed_diag)) == _array_digest(stable_diagnostics(expected_diag))
