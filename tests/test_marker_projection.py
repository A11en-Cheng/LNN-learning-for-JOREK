"""Tests for the marker -> projection RHS assembly port."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.marker_projection import (
    basis_2d_cubic,
    density_channel_rhs,
    gather_time,
)


def test_basis_is_a_partition_of_unity():
    rng = np.random.default_rng(0)
    s = rng.uniform(0.0, 1.0, size=64)
    t = rng.uniform(0.0, 1.0, size=64)
    hh = basis_2d_cubic(s, t)
    assert hh.shape == (64, 4, 4)
    # Hermite cubics on [0,1]: the vertex *value* dofs form a partition of unity.
    assert np.allclose(hh[:, :, 0].sum(axis=1), 1.0)
    # At the corners exactly one value dof is active.
    corners = basis_2d_cubic(
        np.array([0.0, 1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0, 1.0])
    )
    assert np.allclose(corners[:, :, 0].sum(axis=1), 1.0)
    assert np.allclose((corners[:, :, 0] > 0.99).sum(axis=1), 1)


def test_single_particle_single_element_assembly():
    size = np.zeros((4, 4, 1), dtype=np.float64)
    size[:, :, 0] = 1.0
    weight = np.array([2.0])
    charge = np.zeros((1, 11))
    charge[0, 3] = 1.0  # Z = 3
    element = np.array([1])
    st = np.array([[0.5, 0.5]])
    z_eff, z_imp, n_imp = density_channel_rhs(
        weight, charge, element, st, size, n_norm=1.0
    )
    hh = basis_2d_cubic(np.array([0.5]), np.array([0.5]))[0]
    assert np.allclose(n_imp[:, :, 0], hh * 2.0)
    assert np.allclose(z_imp[:, :, 0], hh * 2.0 * 3.0)
    assert np.allclose(z_eff[:, :, 0], hh * 2.0 * 9.0)


def test_normalisation_and_invalid_elements_are_skipped():
    size = np.ones((4, 4, 2), dtype=np.float64)
    weight = np.array([1.0, 1.0])
    charge = np.zeros((2, 11))
    charge[:, 0] = 1.0  # neutral: <Z> = 0
    element = np.array([1, 0])  # second particle outside the grid
    st = np.array([[0.25, 0.75], [0.5, 0.5]])
    z_eff, z_imp, n_imp = density_channel_rhs(
        weight, charge, element, st, size, n_norm=2.0
    )
    assert z_imp.max() == pytest.approx(0.0)
    assert z_eff.max() == pytest.approx(0.0)
    assert n_imp[:, :, 1].max() == pytest.approx(0.0)
    assert n_imp[:, :, 0].sum() == pytest.approx(0.5 * basis_2d_cubic(
        np.array([0.25]), np.array([0.75])
    )[0].sum(), rel=1e-9)


def test_gather_time_is_si_seconds():
    assert gather_time(50, 8.17e-9) == pytest.approx(50 * 8.17e-9)
