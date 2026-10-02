"""Tests for the offline reconstruction of JOREK's nonlocal surface table."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.data.nonlocal_table import (
    interpolate_profiles,
    node_volume_weights,
    nonlocal_table_from_frame,
    surface_profiles,
)


def test_node_volume_weights_single_quad():
    # Unit square, R centroid 0.5: each node gets area/4 * R = 0.125.
    vertex = np.array([[1], [2], [3], [4]], dtype=np.int32)
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    weights = node_volume_weights(vertex, pos)
    assert np.allclose(weights, 0.125)
    assert weights.sum() == pytest.approx(0.5)  # area * R_centroid


def test_surface_profiles_average_uniform_field():
    psi_n = np.linspace(0.0, 0.99, 200)
    weights = np.ones_like(psi_n)
    electron_density = np.full_like(psi_n, 3.0)
    temperature = np.full_like(psi_n, 7.0)
    surface_psi_n, surface_te, surface_ne = surface_profiles(
        psi_n, weights, electron_density, temperature
    )
    assert surface_psi_n.shape == (40,)
    assert np.allclose(surface_te, 7.0)
    assert np.allclose(surface_ne, 3.0)


def test_surface_profiles_weight_by_electron_density():
    # Two nodes in the same bin: the <T_e> average is n_e-weighted.
    psi_n = np.array([0.01, 0.02])
    weights = np.array([1.0, 1.0])
    electron_density = np.array([1.0, 3.0])
    temperature = np.array([10.0, 2.0])
    _, surface_te, surface_ne = surface_profiles(
        psi_n, weights, electron_density, temperature
    )
    assert surface_te[0] == pytest.approx((1.0 * 10.0 + 3.0 * 2.0) / 4.0)
    assert surface_ne[0] == pytest.approx(2.0)


def test_interpolation_matches_linear_ramp():
    surface_psi_n = np.arange(40) / 40
    surface_te = 2.0 + surface_psi_n
    surface_ne = np.full(40, 5.0)
    grid_psi_n, grid_te, grid_ne = interpolate_profiles(
        surface_psi_n, surface_te, surface_ne, n_grid=100
    )
    # Points beyond the last populated surface are flat extrapolations.
    covered = grid_psi_n <= surface_psi_n[-1]
    assert np.allclose(grid_te[covered], 2.0 + grid_psi_n[covered])
    assert np.allclose(grid_ne, 5.0)


def _write_synthetic_frame(path, *, psi_axis=2.0, psi_bnd=0.0):
    """Two quads on a 0 <= R <= 2, psi = psi_axis - R (so psi_n = R/2)."""
    pos = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [2.0, 0.0], [2.0, 1.0]]
    )
    vertex = np.array(
        [[1, 2, 3, 4], [2, 5, 6, 3]], dtype=np.int32
    ).T  # [4, 2]
    values = np.zeros((9, 4, 1, pos.shape[0]), dtype=np.float64)
    values[0, 0, 0, :] = psi_axis - pos[:, 0]
    values[4, 0, 0, :] = 4.0  # rho
    values[8, 0, 0, :] = 1.0  # T_e
    with h5py.File(path, "w") as handle:
        handle["values"] = values
        handle["x"] = np.stack(
            [np.tile(pos[:, 0], (4, 1)), np.tile(pos[:, 1], (4, 1))], axis=0
        )[:, :, None, :]
        handle["vertex"] = vertex
        handle["psi_axis_t"] = [psi_axis]
        handle["psi_bnd_t"] = [psi_bnd]


def test_nonlocal_table_from_frame_end_to_end(tmp_path):
    frame = tmp_path / "jorek000100.h5"
    _write_synthetic_frame(frame)
    table = nonlocal_table_from_frame(frame)
    assert len(table) == 1000
    # n_e = rho (no impurity projection available), T_e constant -> surface
    # averages reproduce the uniform fields, T_e in JOREK units.
    assert np.allclose(table.avg_ne.numpy(), 4.0)
    assert np.allclose(table.avg_te.numpy(), 1.0)
    axis_te, axis_ne, inside = table.surface_state(
        np.array([2.0]), psi_axis=2.0, psi_bnd=0.0, central_density=0.8
    )
    assert bool(inside[0])
    assert float(axis_ne[0]) == pytest.approx(4.0 * 1e20 * 0.8, rel=1e-6)
