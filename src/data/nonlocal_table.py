"""Rebuild JOREK's nonlocal-ablation surface averages from a saved frame.

``int3d_new`` (``diagnostics/mod_integrals3D.f90``) fills, on 40 linearly
spaced psi surfaces and then linearly interpolates onto a 1000-point grid:

    n_e       = rho + (m_i/m_imp * Z_imp - 1) * rho_imp
    <n_e>(s)  = sum(n_e dV) / sum(dV)
    <T_e>(s)  = sum(n_e T_e dV) / sum(n_e dV)

where ``Z_imp`` is the marker-projected mean charge
(``projection Z_imp / projection N_imp``) and ``dV`` is the JOREK volume
element.  The on-disk ``nonlocal_array.dat`` is overwritten on every call, so
it is only a snapshot of one (often much later) time; for a specific frame the
table has to be rebuilt from that frame.  This module does exactly that and
returns the :class:`~src.physics.spi_model.NonlocalAblationTable` consumed by
the ablation kernel.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from src.physics.spi_model import NonlocalAblationTable

M_NEON_OVER_M_BG = 1.0 / 20.0
"""``m_i/m_imp`` for a deuterium background and a neon impurity."""


def node_volume_weights(vertex: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Lumped node weights ``sum_e area_e * R_centroid / 4`` [m^3 per rad]."""
    n_nodes = pos.shape[0]
    weights = np.zeros(n_nodes, dtype=np.float64)
    elements = np.asarray(vertex, dtype=np.int64).T - 1  # [n_elements, 4], 1-based
    valid = ((elements >= 0).all(axis=1)) & ((elements < n_nodes).all(axis=1))
    for element in elements[valid]:
        xy = pos[element]
        x, y = xy[:, 0], xy[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        weights[element] += 0.25 * area * float(np.mean(x))
    return weights


def surface_profiles(
    psi_n: np.ndarray,
    weights: np.ndarray,
    electron_density: np.ndarray,
    electron_temperature: np.ndarray,
    *,
    n_surface: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume-averaged ``(<T_e>, <n_e>)`` on the 40 psi surfaces.

    JOREK assigns a Gauss point to bin ``floor(psi_n * n_flux)``; empty bins are
    returned as NaN and are dropped by the interpolation step.
    """
    inside = (psi_n >= 0.0) & (psi_n < 1.0)
    bins = np.minimum((psi_n * n_surface).astype(np.int64), n_surface - 1)
    surface_te = np.full(n_surface, np.nan, dtype=np.float64)
    surface_ne = np.full(n_surface, np.nan, dtype=np.float64)
    for surface in range(n_surface):
        mask = inside & (bins == surface)
        if not mask.any():
            continue
        weight = weights[mask]
        electron = electron_density[mask]
        de = float((electron * weight).sum())
        volume = float(weight.sum())
        if de <= 0.0 or volume <= 0.0:
            continue
        surface_te[surface] = float((electron * electron_temperature[mask] * weight).sum()) / de
        surface_ne[surface] = de / volume
    surface_psi_n = np.arange(n_surface, dtype=np.float64) / n_surface
    return surface_psi_n, surface_te, surface_ne


def interpolate_profiles(
    surface_psi_n: np.ndarray,
    surface_te: np.ndarray,
    surface_ne: np.ndarray,
    *,
    n_grid: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linear interpolation onto the 1000-point grid JOREK writes/reads."""
    valid = np.isfinite(surface_te) & np.isfinite(surface_ne)
    if int(valid.sum()) < 2:
        raise ValueError("nonlocal table reconstruction needs >=2 populated surfaces")
    grid_psi_n = np.arange(n_grid, dtype=np.float64) / n_grid
    grid_te = np.interp(grid_psi_n, surface_psi_n[valid], surface_te[valid])
    grid_ne = np.interp(grid_psi_n, surface_psi_n[valid], surface_ne[valid])
    return grid_psi_n, grid_te, grid_ne


def nonlocal_table_from_frame(
    state_path: str | Path,
    projection_path: str | Path | None = None,
    *,
    n_surface: int = 40,
    n_grid: int = 1000,
    mass_ratio: float = M_NEON_OVER_M_BG,
    density_floor: float = 1e-3,
) -> NonlocalAblationTable:
    """Rebuild the ``nonlocal_array`` table of one frame."""
    state_path = Path(state_path)
    if projection_path is None:
        index = int(state_path.stem.replace("jorek", ""))
        candidate = state_path.parent / f"projections{index:06d}.h5"
        projection_path = candidate if candidate.exists() else None
    with h5py.File(state_path, "r") as handle:
        values = np.asarray(handle["values"][:, 0, 0, :], dtype=np.float64)
        pos = np.asarray(handle["x"][:, 0, 0, :], dtype=np.float64).T
        vertex = np.asarray(handle["vertex"][()])
        psi_axis = float(np.ravel(handle["psi_axis_t"][()])[-1])
        psi_bnd = float(np.ravel(handle["psi_bnd_t"][()])[-1])
    rho = np.clip(values[4], 0.0, None)
    rho_imp = np.clip(values[7], 0.0, None)
    te = np.clip(values[8], 0.0, None)
    z_mean = np.zeros_like(rho)
    if projection_path is not None:
        with h5py.File(projection_path, "r") as handle:
            projection = np.asarray(handle["values"][:5, 0, 0, :], dtype=np.float64)
        z_imp_proj = np.clip(projection[3], 0.0, None)
        n_imp_proj = np.clip(projection[4], 0.0, None)
        z_mean = np.divide(
            z_imp_proj, n_imp_proj, out=np.zeros_like(z_imp_proj), where=n_imp_proj > 1e-30
        )
    electron_density = np.clip(rho + (mass_ratio * z_mean - 1.0) * rho_imp, density_floor, None)
    psi = values[0]
    psi_n = (psi - psi_axis) / (psi_bnd - psi_axis)
    weights = node_volume_weights(vertex, pos)
    surface_psi_n, surface_te, surface_ne = surface_profiles(
        psi_n, weights, electron_density, te, n_surface=n_surface
    )
    grid_psi_n, grid_te, grid_ne = interpolate_profiles(
        surface_psi_n, surface_te, surface_ne, n_grid=n_grid
    )
    return NonlocalAblationTable.from_arrays(grid_psi_n, grid_te, grid_ne)
