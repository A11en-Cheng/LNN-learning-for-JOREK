"""Public ADAS interpolation contracts; timing is not a correctness gate."""

import numpy as np
import pytest

from src.models.native_coupling import NativeNeonAtomicRates
from src.models.source_module import NeonTables


def _table(n, t):
    # Bilinear plane: -30 + log_density + 2*log_temperature + charge/10.
    n, t = np.asarray(n, dtype=float), np.asarray(t, dtype=float)
    c = -30 + n[None, :] + 2*t[:, None]
    return n, t, c[None, :, :] + np.arange(10)[:, None, None]/10


def _legacy_provider_result(tables, prb, te_ev, ne_si):
    """Pre-optimization provider formula, using the unchanged row interpolator.

    This reference deliberately never calls the shared-charge helper or the
    optimized provider. Unit conversion, power placement and charge mapping
    reproduce the old complete provider, not merely table interpolation.
    """
    from src.physics.adas_closure import _bilinear

    ne, te = np.asarray(ne_si), np.asarray(te_ev)
    logn, logt = np.log10(ne*1e-6), np.log10(te)

    def interp(table):
        n, t, coefficients = table
        return np.stack([
            10.**_bilinear(row, n, t, logn, logt) for row in coefficients
        ], axis=1)

    scd = interp(tables.scd)*1e-6
    acd = interp(tables.acd)*1e-6
    radiation = np.zeros((len(te), 11))
    radiation[:, :10] = interp(tables.plt)*(ne*1e-6)[:, None]
    if prb is not None:
        radiation[:, 1:] += interp(prb)*(ne*1e-6)[:, None]
    return scd, acd, radiation


@pytest.mark.parametrize("full_radiation", [False, True])
def test_public_complete_provider_random_unequal_axes_exact_legacy(full_radiation):
    rng = np.random.default_rng(191)
    independent_tables = []
    for k in range(4):
        n = np.array([9., 10.25, 13., 17.]) + k/4
        t = np.array([-1., 0.75, 2., 5., 6.]) + k/8
        independent_tables.append((n, t, rng.uniform(-30, -10, (10, 5, 4))))
    tables = NeonTables(*independent_tables[:3], np.arange(1., 11.))
    prb = independent_tables[3] if full_radiation else None
    provider = NativeNeonAtomicRates(
        tables, radiation_mode="plt_prb" if full_radiation else "plt_reference",
        prb=prb,
    )
    # Strided SI/eV queries exercise interior points and all table clamps.
    queries = np.empty((257, 2))
    queries[:, 0] = 10.**rng.uniform(-2, 8, len(queries))
    queries[:, 1] = 10.**rng.uniform(8, 19, len(queries))*1e6
    te, ne = queries[:, 0], queries[:, 1]
    expected = _legacy_provider_result(tables, prb, te, ne)
    actual = provider(te, ne)
    for got, want, channels in zip(actual, expected, (10, 10, 11)):
        assert got.shape == (257, channels)
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got, want)


def test_public_real_plt_prb_complete_provider_exact_legacy():
    from src.models.source_module import load_neon_tables
    from src.physics.adas_closure import ADAS_DIR, parse_adf11

    tables = load_neon_tables()
    assert tables.source == "adas"
    prb = parse_adf11(ADAS_DIR / "prb96_ne.dat")
    provider = NativeNeonAtomicRates(tables, radiation_mode="plt_prb", prb=prb)
    rng = np.random.default_rng(811)
    all_tables = (tables.scd, tables.acd, tables.plt, prb)
    nmin = min(table[0][0] for table in all_tables)
    nmax = max(table[0][-1] for table in all_tables)
    tmin = min(table[1][0] for table in all_tables)
    tmax = max(table[1][-1] for table in all_tables)
    density_logs = [rng.uniform(nmin-1, nmax+1, 257)]
    temperature_logs = [rng.uniform(tmin-1, tmax+1, 257)]
    for n, t, _ in all_tables:
        # Every real grid knot, followed by exterior corners for each table.
        nn, tt = np.meshgrid(n, t)
        density_logs.extend([nn.ravel(), np.array([n[0]-1, n[-1]+1])])
        temperature_logs.extend([tt.ravel(), np.array([t[0]-1, t[-1]+1])])
    ne = 10.**np.concatenate(density_logs)*1e6
    te = 10.**np.concatenate(temperature_logs)
    expected = _legacy_provider_result(tables, prb, te, ne)
    actual = provider(te, ne)
    for got, want, channels in zip(actual, expected, (10, 10, 11)):
        assert got.shape == (len(ne), channels)
        assert np.isfinite(got).all()
        np.testing.assert_array_equal(got, want)
    # Fully stripped charge radiates through PRB; it must not be left at zero.
    assert np.all(actual[2][:, 10] > 0)


def test_public_rates_independent_axes_units_and_charge_mapping():
    scd = _table([10, 14], [0, 4])
    acd = _table([13, 15], [1, 3])  # density clipped to 13
    plt = _table([11, 15], [0, 1])  # temperature clipped to 1
    prb = _table([10, 11], [3, 5])  # both clipped, independently
    rates = NativeNeonAtomicRates(
        NeonTables(scd, acd, plt, np.arange(1., 11.)),
        radiation_mode="plt_prb", prb=prb,
    )
    s, a, radiation = rates(np.array([100.]), np.array([1e18]))
    q = np.arange(10)/10
    np.testing.assert_allclose(s[0], 10.**(-14+q)*1e-6, rtol=2e-14)
    np.testing.assert_allclose(a[0], 10.**(-13+q)*1e-6, rtol=2e-14)
    expected = np.zeros(11)
    expected[:10] = 10.**(-16+q)*1e12
    expected[1:] += 10.**(-13+q)*1e12
    np.testing.assert_allclose(radiation[0], expected, rtol=2e-14)


def test_charge_table_matches_literal_row_oracle_at_knots_and_clamps():
    from src.models.native_atomic_interpolation import interpolate_charge_table
    from src.physics.adas_closure import _bilinear

    rng = np.random.default_rng(31)
    n = np.array([9., 10.25, 13., 17.])
    t = np.array([-1., 0.75, 2., 5., 6.])
    table = n, t, rng.uniform(-30, -10, (10, len(t), len(n)))
    logn = np.r_[n, 8., 18., rng.uniform(8, 18, 73)]
    logt = np.r_[t, -2., rng.uniform(-2, 7, 73)]
    expected = np.stack([
        10.**_bilinear(row, n, t, logn, logt) for row in table[2]
    ], axis=1)
    actual = interpolate_charge_table(table, logn, logt)
    assert actual.shape == (79, 10)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("size", [0, 1, 64, 2048])
def test_charge_table_uses_current_table_and_queries_without_mutation(size):
    from src.models.native_atomic_interpolation import interpolate_charge_table
    from src.physics.adas_closure import _bilinear

    rng = np.random.default_rng(72)
    # Strided queries also occur when native marker components are sliced.
    queries = rng.uniform(0, 20, (size, 2))
    logn, logt = queries[:, 0], queries[:, 1]
    before = queries.copy()
    tables = [_table([9, 11, 16], [-1, 2, 5]),
              _table([10, 13], [0, 3, 6, 9])]
    for table in tables + tables[:1]:
        original = tuple(a.copy() for a in table)
        n, t, coefficients = table
        expected = np.stack([
            10.**_bilinear(row, n, t, logn, logt) for row in coefficients
        ], axis=1)
        np.testing.assert_array_equal(
            interpolate_charge_table(table, logn, logt), expected)
        for actual, saved in zip(table, original):
            np.testing.assert_array_equal(actual, saved)
    np.testing.assert_array_equal(queries, before)
