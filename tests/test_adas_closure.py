"""Unit tests for the ADAS coronal closure prototype (#0014)."""

from __future__ import annotations

import numpy as np
import pytest

from src.physics.adas_closure import (
    ADAS_DIR,
    coronal_moments,
    load_neon_tables,
    parse_adf11,
)


def test_parse_adf11_shapes():
    ne, te, coeff = parse_adf11(ADAS_DIR / "scd96_ne.dat")
    assert coeff.shape == (10, len(te), len(ne))
    assert coeff.shape[0] == 10  # Neon ionisation states Z=0..9


def test_coronal_moments_monotonic_in_temperature():
    scd, acd = load_neon_tables()
    te_grid = np.linspace(scd[1][0] + 0.1, scd[1][-1], 50)
    z_imp, z_eff = coronal_moments(te_grid, 13.0, scd, acd)
    assert np.all(z_imp > 0)
    assert np.all(z_eff >= z_imp)
    assert z_imp[-1] > z_imp[0]


def test_coronal_moments_are_finite_and_normalised():
    scd, acd = load_neon_tables()
    z_imp, z_eff = coronal_moments(np.full(100, 2.0), 13.0, scd, acd)
    assert np.isfinite(z_imp).all()
    assert z_imp.max() <= 10.0
    assert z_eff.max() <= 100.0
