"""Compiled queries must cross the complete public Ne atomic-rate interface."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from src.models.native_compiled_atomic import CompiledAtomicInterpolation, CompiledNeonAtomicRates, build_kernel
from src.models.native_coupling import NativeNeonAtomicRates
from src.models.source_module import NeonTables
from tests.test_native_atomic_interpolation import _legacy_provider_result, _table


@pytest.fixture(scope='session')
def atomic_kernel(tmp_path_factory):
    return CompiledAtomicInterpolation(build_kernel(tmp_path_factory.mktemp('compiled-atomic')/'kernel.so'))


def random_provider(*, full=True, layout='C'):
    rng = np.random.default_rng(593)
    tables = []
    for k in range(4):
        n = np.array([9., 10.25, 13., 17.])+k/4
        t = np.array([-1., .75, 2., 5., 6.])+k/8
        c = rng.uniform(-30, -10, (10, len(t), len(n)))
        if layout == 'F':
            c = np.asfortranarray(c)
        elif layout == 'strided':
            c = c[:, :, ::-1].copy()[:, :, ::-1]
        tables.append((n, t, c))
    return NativeNeonAtomicRates(NeonTables(*tables[:3], np.arange(1., 11.)),
        radiation_mode='plt_prb' if full else 'plt_reference', prb=tables[3] if full else None)


@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('layout', ['C', 'F', 'strided'])
@pytest.mark.parametrize('count', [1, 7, 257, 2048])
def test_complete_rates_exact_legacy_independent_axes_and_layout(atomic_kernel, full, layout, count):
    reference = random_provider(full=full, layout=layout)
    candidate = CompiledNeonAtomicRates(reference, kernel=atomic_kernel)
    rng = np.random.default_rng(78)
    queries = np.empty((count, 2))
    queries[:, 0] = 10.**rng.uniform(-3, 8, count)
    queries[:, 1] = 10.**rng.uniform(8, 20, count)*1e6
    saved = queries.copy()
    expected = _legacy_provider_result(reference.tables, reference.prb, queries[:, 0], queries[:, 1])
    actual = candidate(queries[:, 0], queries[:, 1])
    for a, b, channels in zip(actual, expected, (10, 10, 11)):
        assert a.shape == (count, channels)
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(queries, saved)
    assert candidate.version != reference.version
    assert candidate.configuration['reference_version'] == reference.version


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_real_all_table_knots_clamps_and_query_precision(atomic_kernel, dtype):
    from src.models.source_module import load_neon_tables
    from src.physics.adas_closure import ADAS_DIR, parse_adf11
    tables = load_neon_tables(); prb = parse_adf11(ADAS_DIR/'prb96_ne.dat')
    reference = NativeNeonAtomicRates(tables, radiation_mode='plt_prb', prb=prb)
    candidate = CompiledNeonAtomicRates(reference, kernel=atomic_kernel)
    logn, logt = [], []
    for n, t, _ in (tables.scd, tables.acd, tables.plt, prb):
        nn, tt = np.meshgrid(n, t)
        logn.extend((nn.ravel(), np.r_[n[0]-3, n[-1]+3]))
        logt.extend((tt.ravel(), np.r_[t[0]-3, t[-1]+3]))
    ne = (10.**np.concatenate(logn)*1e6).astype(dtype)
    te = (10.**np.concatenate(logt)).astype(dtype)
    for a, b in zip(candidate(te, ne), reference(te, ne)):
        np.testing.assert_array_equal(a, b)


def test_units_charge_mapping_and_owned_snapshot(atomic_kernel):
    tables = NeonTables(_table([10, 14], [0, 4]), _table([13, 15], [1, 3]),
        _table([11, 15], [0, 1]), np.arange(1., 11.))
    reference = NativeNeonAtomicRates(tables, radiation_mode='plt_prb', prb=_table([10, 11], [3, 5]))
    candidate = CompiledNeonAtomicRates(reference, kernel=atomic_kernel)
    te, ne = np.array([100.]), np.array([1e18])
    expected = reference(te, ne)
    for a, b in zip(candidate(te, ne), expected):
        np.testing.assert_array_equal(a, b)
    # The compiled provider owns an immutable table snapshot, not external IDs.
    reference.tables.scd[2][:] += 1
    for a, b in zip(candidate(te, ne), expected):
        np.testing.assert_array_equal(a, b)
    assert not np.array_equal(reference(te, ne)[0], expected[0])


@pytest.mark.parametrize('te,ne', [([], []), ([0.], [1e18]), ([10.], [-1.]),
    ([np.nan], [1e18]), ([10.], [np.inf]), ([[10.]], [[1e18]]), ([10., 20.], [1e18])])
def test_invalid_queries_explicitly_fail(atomic_kernel, te, ne):
    candidate = CompiledNeonAtomicRates(random_provider(), kernel=atomic_kernel)
    with pytest.raises(ValueError):
        candidate(np.array(te), np.array(ne))


def test_tables_and_memory_budget_fail_before_query(atomic_kernel):
    reference = random_provider()
    with pytest.raises(MemoryError, match='table.*budget'):
        CompiledNeonAtomicRates(reference, kernel=atomic_kernel, max_table_bytes=1)
    for change in ('axis', 'charges', 'nonfinite'):
        broken = deepcopy(reference)
        if change == 'axis':
            broken.tables.scd[0][1] = broken.tables.scd[0][0]
        elif change == 'charges':
            broken.tables = replace(broken.tables, scd=(*broken.tables.scd[:2], broken.tables.scd[2][:9]))
        else:
            broken.tables.scd[2][0, 0, 0] = np.nan
        with pytest.raises(ValueError, match='table'):
            CompiledNeonAtomicRates(broken, kernel=atomic_kernel)


def test_query_budget_is_explicit_and_no_recompilation(atomic_kernel, tmp_path):
    candidate = CompiledNeonAtomicRates(random_provider(), kernel=atomic_kernel, max_queries=1)
    candidate(np.array([10.]), np.array([1e18]))
    with pytest.raises(MemoryError, match='query.*budget'):
        candidate(np.array([10., 20.]), np.array([1e18, 1e18]))
    destination = tmp_path/'existing.so'
    destination.touch()
    with pytest.raises(FileExistsError):
        build_kernel(destination)


def test_non_fp64_table_arithmetic_is_not_silently_promoted(atomic_kernel):
    # Axis subtraction in the reference follows the stored dtype. Promotion
    # before subtracting, e.g. -.13 to .243, can change an interval width.
    reference = random_provider()
    old = reference.tables.scd
    n = np.array([.11, 3.713, 11.913, 30.], dtype=np.float32)
    t = np.array([-.13, .243, .825, 4.637, 6.13], dtype=np.float32)
    reference.tables = replace(reference.tables, scd=(n, t, old[2]))
    with pytest.raises(ValueError, match='FP64.*table|table.*FP64'):
        CompiledNeonAtomicRates(reference, kernel=atomic_kernel)


def test_knots_neighbours_subnormal_queries_and_concurrent_calls(atomic_kernel):
    from concurrent.futures import ThreadPoolExecutor
    reference = random_provider(); candidate = CompiledNeonAtomicRates(reference, kernel=atomic_kernel)
    density_logs, temperature_logs = [], []
    for n, t, _ in (reference.tables.scd, reference.tables.acd, reference.tables.plt, reference.prb):
        for axis, destination in ((n, density_logs), (t, temperature_logs)):
            destination.extend((np.nextafter(axis, -np.inf), axis, np.nextafter(axis, np.inf)))
    nl = np.resize(np.concatenate(density_logs), 257)
    tl = np.resize(np.concatenate(temperature_logs), 257)
    te, ne = 10.**tl, 10.**nl*1e6
    te[:2] = [np.nextafter(0., 1.), np.finfo(float).max]
    ne[:2] = [np.nextafter(0., 1.), np.finfo(float).max]
    # ne*1e-6 underflow is clipped by the unchanged log-table policy.
    with np.errstate(divide='ignore', under='ignore'):
        expected = reference(te, ne)
        actual = candidate(te, ne)
    for a,b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda _:candidate(te[2:], ne[2:]), range(9)))
    for result in outputs:
        for a,b in zip(result, (x[2:] for x in expected)):
            np.testing.assert_array_equal(a, b)


def test_complete_marker_window_failure_and_resume(atomic_kernel):
    from tests.test_native_marker_sampling import window
    from tests.test_native_locator_events import assert_runtime_exact
    import pickle
    source, initial, fields, sampler = window(static=True)
    reference = random_provider()
    source.atomic = reference
    source.configuration['atomic'] = reference.version
    before = deepcopy(initial)
    expected, ed = source.advance(initial, 1e-9, fields, fields, time_sampler=sampler)
    candidate = CompiledNeonAtomicRates(reference, kernel=atomic_kernel)
    source.atomic = candidate; source.configuration['atomic'] = candidate.version
    actual, ad = source.advance(initial, 1e-9, fields, fields, time_sampler=sampler)
    assert_runtime_exact(actual, expected)
    assert ad['execution']['processed_particle_steps'] == ed['execution']['processed_particle_steps']
    assert_runtime_exact(initial, before)
    actual.clock.field_start_si, actual.clock.field_end_si = 1e-9, 2e-9
    from src.models.native_source_time import NativeSourceTimeSampler
    next_sampler = NativeSourceTimeSampler(fields, fields*.001, time_prev_si=1e-9,
        time_now_si=2e-9, tstep_si=1e-9, static=True)
    resumed = pickle.loads(pickle.dumps(actual))
    a, _ = source.advance(actual, 2e-9, fields, fields, time_sampler=next_sampler)
    b, _ = source.advance(resumed, 2e-9, fields, fields, time_sampler=next_sampler)
    assert_runtime_exact(a, b)
    source.atomic = CompiledNeonAtomicRates(reference, kernel=atomic_kernel, max_queries=1)
    source.configuration['atomic'] = source.atomic.version
    two = deepcopy(initial)
    for name in two.cloud.array_names():
        setattr(two.cloud, name, np.repeat(getattr(two.cloud, name), 2, axis=0))
    before_two = deepcopy(two)
    with pytest.raises(MemoryError, match='query.*budget'):
        source.advance(two, 1e-9, fields, fields, time_sampler=sampler)
    assert_runtime_exact(two, before_two)
    assert_runtime_exact(initial, before)
