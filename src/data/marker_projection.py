"""Port of the JOREK marker -> projection assembly (``use_marker`` branch).

Traced from the JOREK source:

* ``particles/examples/marker_spi_source.f90`` (per-particle channel values and
  the ``HH(l,m) * size(l,m,e)`` accumulation, ~lines 615-650);
* ``particles/diagnostics/mod_project_particles.f90::sample_rhs`` (the generic
  ``HH * size * f * weight`` product);
* ``marker_spi_source.f90`` (~lines 670-676): the accumulated RHS is divided by
  the gather time ``n_steps * timesteps`` (SI seconds) before the projection
  solve; ``rhs_gather_time = n_steps * timesteps`` is set at line ~443.

Per-particle channel definitions (``use_marker``), per particle substep:

| channel | value | normalisation |
|---|---|---|
| 1 dE_ion | ``sum_iZ (P_new(iZ) - P_old(iZ)) * E_ion(iZ) * w * EL_CHG`` [J] | ``t_norm / E_norm`` |
| 2 rad_sink | ``w * Lrad(n_e, T_e) * n_e * timesteps`` [J] | ``t_norm / E_norm`` |
| 3 Z_eff | ``w * sum_iZ iZ**2 * P(iZ)`` | ``1 / n_norm`` |
| 4 Z_imp | ``w * sum_iZ iZ * P(iZ)`` | ``1 / n_norm`` |
| 5 N_imp | ``w`` | ``1 / n_norm`` |

The assembled RHS is then projected with the FEM mass matrix (plus the filter
terms configured in the namelist) and written to ``projections*.h5`` as the
solved node coefficients -- i.e. the recorded files are **not** the raw RHS.
This module ports the marker-side assembly for the three density channels
(which depend only on the marker state); the two energy channels additionally
need the ADAS charge-state update and the radiation function.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

N_VERTEX = 4
N_DOF = 4
N_CHANNELS = 5


def basis_2d_cubic(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Bicubic Hermite basis ``HH[particle, vertex, dof]`` on ``s, t in [0, 1]``.

    Port of ``util/data_extractor/fortran_process_jorek.f90::basisfunctions``
    (vertex/dof ordering identical to JOREK).
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    h = np.empty((s.shape[0], N_VERTEX, N_DOF), dtype=np.float64)
    u = s - 1.0
    w = t - 1.0
    h[:, 0, 0] = u**2 * (1 + 2 * s) * w**2 * (1 + 2 * t)
    h[:, 0, 1] = 3 * u**2 * s * w**2 * (1 + 2 * t)
    h[:, 0, 2] = 3 * u**2 * (1 + 2 * s) * w**2 * t
    h[:, 0, 3] = 9 * u**2 * s * w**2 * t
    h[:, 1, 0] = -(s**2 * (2 * s - 3) * w**2 * (1 + 2 * t))
    h[:, 1, 1] = -3 * u * s**2 * w**2 * (1 + 2 * t)
    h[:, 1, 2] = -3 * s**2 * (2 * s - 3) * w**2 * t
    h[:, 1, 3] = -9 * u * s**2 * w**2 * t
    h[:, 2, 0] = s**2 * (2 * s - 3) * t**2 * (2 * t - 3)
    h[:, 2, 1] = 3 * u * s**2 * t**2 * (2 * t - 3)
    h[:, 2, 2] = 3 * s**2 * (2 * s - 3) * w * t**2
    h[:, 2, 3] = 9 * u * s**2 * w * t**2
    h[:, 3, 0] = -(u**2 * (1 + 2 * s) * t**2 * (2 * t - 3))
    h[:, 3, 1] = -3 * u**2 * s * t**2 * (2 * t - 3)
    h[:, 3, 2] = -3 * u**2 * (1 + 2 * s) * w * t**2
    h[:, 3, 3] = -9 * u**2 * s * w * t**2
    return h


def basis_2d_cubic_derivatives(
    s: np.ndarray, t: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(H, H_s, H_t)`` bicubic Hermite basis and s/t derivatives.

    Port of the same ``basisfunctions`` routine; used to build the projection
    mass matrix (``M_ij,kl = int H_i H_k R dR dZ``).
    """
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    h = basis_2d_cubic(s, t)
    hs = np.empty_like(h)
    ht = np.empty_like(h)
    u = s - 1.0
    w = t - 1.0
    # vertex 1
    hs[:, 0, 0] = 6 * u * s * w**2 * (1 + 2 * t)
    ht[:, 0, 0] = 6 * u**2 * (1 + 2 * s) * w * t
    hs[:, 0, 1] = 3 * u * (-1 + 3 * s) * w**2 * (1 + 2 * t)
    ht[:, 0, 1] = 18 * u**2 * s * w * t
    hs[:, 0, 2] = 18 * u * s * w**2 * t
    ht[:, 0, 2] = 3 * u**2 * (1 + 2 * s) * w * (-1 + 3 * t)
    hs[:, 0, 3] = 9 * u * (-1 + 3 * s) * w**2 * t
    ht[:, 0, 3] = 9 * u**2 * s * w * (-1 + 3 * t)
    # vertex 2
    hs[:, 1, 0] = -6 * u * s * w**2 * (1 + 2 * t)
    ht[:, 1, 0] = -6 * s**2 * (-3 + 2 * s) * w * t
    hs[:, 1, 1] = -3 * s * (-2 + 3 * s) * w**2 * (1 + 2 * t)
    ht[:, 1, 1] = -18 * u * s**2 * w * t
    hs[:, 1, 2] = -18 * u * s * w**2 * t
    ht[:, 1, 2] = 3 * s**2 * (-3 + 2 * s) * (1 - 3 * t) * w
    hs[:, 1, 3] = -9 * s * (-2 + 3 * s) * w**2 * t
    ht[:, 1, 3] = 9 * u * s**2 * (1 - 3 * t) * w
    # vertex 3
    hs[:, 2, 0] = 6 * u * s * t**2 * (-3 + 2 * t)
    ht[:, 2, 0] = 6 * s**2 * (-3 + 2 * s) * w * t
    hs[:, 2, 1] = 3 * s * (-2 + 3 * s) * t**2 * (-3 + 2 * t)
    ht[:, 2, 1] = 18 * u * s**2 * w * t
    hs[:, 2, 2] = 18 * u * s * w * t**2
    ht[:, 2, 2] = 3 * s**2 * (-3 + 2 * s) * t * (-2 + 3 * t)
    hs[:, 2, 3] = 9 * s * (-2 + 3 * s) * w * t**2
    ht[:, 2, 3] = 9 * u * s**2 * t * (-2 + 3 * t)
    # vertex 4
    hs[:, 3, 0] = -6 * u * s * t**2 * (-3 + 2 * t)
    ht[:, 3, 0] = -6 * u**2 * (1 + 2 * s) * w * t
    hs[:, 3, 1] = 3 * (1 - 3 * s) * u * t**2 * (-3 + 2 * t)
    ht[:, 3, 1] = -18 * u**2 * s * w * t
    hs[:, 3, 2] = -18 * u * s * w * t**2
    ht[:, 3, 2] = -3 * u**2 * (1 + 2 * s) * t * (-2 + 3 * t)
    hs[:, 3, 3] = 9 * (1 - 3 * s) * u * w * t**2
    ht[:, 3, 3] = -9 * u**2 * s * t * (-2 + 3 * t)
    return h, hs, ht


def gather_time(n_steps: int, timesteps: float) -> float:
    """JOREK's ``rhs_gather_time`` (SI seconds); the RHS is divided by it."""
    return float(n_steps) * float(timesteps)


def density_channel_rhs(
    weight: np.ndarray,
    charge_distribution: np.ndarray,
    element_index: np.ndarray,
    st: np.ndarray,
    size: np.ndarray,
    n_norm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the marker RHS for ``(Z_eff, Z_imp, N_imp)``.

    ``weight`` ``[P]``, ``charge_distribution`` ``[P, n_Z+1]`` (fractions of the
    charge states ``0..n_Z``), ``element_index`` ``[P]`` (1-based, JOREK
    convention; non-positive entries are skipped), ``st`` ``[P, 2]`` element
    coordinates, ``size`` ``[4, 4, n_elements]`` from the state file.

    Returns one array per channel with JOREK's ``[dof_m, dof_l, element]``
    shape (i.e. exactly the ``feedback_rhs`` layout before the projection
    solve).
    """
    n_elements = size.shape[2]
    out = np.zeros((3, N_VERTEX, N_DOF, n_elements), dtype=np.float64)
    element = np.asarray(element_index, dtype=np.int64).reshape(-1)
    valid = element > 0
    if not valid.any():
        return out[0], out[1], out[2]
    element = element[valid] - 1
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)[valid]
    charge = np.asarray(charge_distribution, dtype=np.float64)[valid]
    st = np.asarray(st, dtype=np.float64)[valid]
    z = np.arange(charge.shape[1], dtype=np.float64)
    mean_z = charge @ z
    mean_z2 = charge @ (z**2)
    hh = basis_2d_cubic(st[:, 0], st[:, 1])  # [P, 4, 4]
    shape = hh * np.moveaxis(size[:, :, element], 2, 0)  # [P, 4, 4]
    for channel, value in enumerate(
        (weight * mean_z2, weight * mean_z, weight)
    ):
        contribution = (shape * (value / n_norm)[:, None, None]).reshape(-1, 16)
        for elem in np.unique(element):
            mask = element == elem
            out[channel].reshape(16, n_elements)[:, elem] += contribution[mask].sum(axis=0)
    return out[0], out[1], out[2]


def density_rhs_from_frame(
    state_path: str | Path,
    part_path: str | Path,
    *,
    n_norm: float,
    group: str = "groups/001",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read one aligned frame pair and assemble the density-channel RHS."""
    with h5py.File(state_path, "r") as handle:
        size = np.asarray(handle["size"][()], dtype=np.float64)
    with h5py.File(part_path, "r") as handle:
        node = handle[group]
        weight = np.asarray(node["weight"][()], dtype=np.float64)
        charge = np.asarray(node["P_imp"][()], dtype=np.float64)
        element = np.asarray(node["i_elm"][()], dtype=np.int64)
        st = np.asarray(node["st"][()], dtype=np.float64)
    return density_channel_rhs(weight, charge, element, st, size, n_norm)
