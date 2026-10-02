"""Phase-1.5.1 gate: check ``M x_recorded`` against the marker-assembled RHS.

The recorded ``projections*.h5`` ``values`` are the *solution* of the FEM
projection system ``M x = b`` (see ``report/marker-projection-recipe.md``).
This script rebuilds the element mass matrix ``M`` (base term; the namelist
filter terms are reported separately) and the marker RHS ``b`` for the three
density channels, then evaluates the element-local residual
``M x_recorded - b / gather_time`` on interior elements.

Example:

    .torch-cuda-env/bin/python tests/check_projection_gate.py \
        --run scan_1d22_ang15_5_massloss_neon_v400 --index 6000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.marker_projection import (
    basis_2d_cubic_derivatives,
    density_channel_rhs,
)

GAUSS_X = np.array([0.0694318442029735, 0.3300094782075720, 0.6699905217924280, 0.9305681557970265])
GAUSS_W = np.array([0.173927422568727, 0.326072577431273, 0.326072577431273, 0.173927422568727])

TWOPI_OVER_N_PLANE = 2.0 * np.pi
"""``assemble_projection_matrix_n0`` copies ``ELM * TWOPI / n_plane`` into the
matrix; the RHS is *not* scaled (the ``!* TWOPI`` in ``marker_spi_source`` is
commented out), so the comparison must include this factor (``n_plane = 1``)."""


def element_mass_matrix(
    coord: np.ndarray,
    size_elem: np.ndarray,
) -> np.ndarray:
    """16x16 base mass matrix (``int H_i H_k size_i size_k R dR dZ``).

    ``coord`` is ``[4, 4, 2]`` (vertex, dof, {R, Z}) from the node data.
    """
    elm = np.zeros((16, 16), dtype=np.float64)
    s = np.repeat(GAUSS_X, 4)
    t = np.tile(GAUSS_X, 4)
    weight = np.repeat(GAUSS_W, 4) * np.tile(GAUSS_W, 4)
    h, h_s, h_t = basis_2d_cubic_derivatives(s, t)  # [16, 4, 4]
    x = np.einsum("gij,ij->g", h, coord[:, :, 0]) * 0.0  # placeholder
    x = np.einsum("gij,ij,ij->g", h, coord[:, :, 0], size_elem)
    y = np.einsum("gij,ij,ij->g", h, coord[:, :, 1], size_elem)
    x_s = np.einsum("gij,ij,ij->g", h_s, coord[:, :, 0], size_elem)
    x_t = np.einsum("gij,ij,ij->g", h_t, coord[:, :, 0], size_elem)
    y_s = np.einsum("gij,ij,ij->g", h_s, coord[:, :, 1], size_elem)
    y_t = np.einsum("gij,ij,ij->g", h_t, coord[:, :, 1], size_elem)
    xjac = x_s * y_t - x_t * y_s
    v = h * size_elem[None, :, :]  # [16, 4, 4]
    for g in range(16):
        vg = v[g].reshape(-1)
        elm += np.outer(vg, vg) * xjac[g] * x[g] * weight[g]
    return elm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--index", type=int, default=6000)
    parser.add_argument("--central-density", type=float, default=0.8)
    parser.add_argument("--n-elements", type=int, default=400, help="interior elements to check")
    args = parser.parse_args()

    run_dir = Path(args.data_root) / args.run
    state = run_dir / f"jorek{args.index:06d}.h5"
    part = run_dir / f"part{args.index:05d}.h5"
    projection = run_dir / f"projections{args.index:06d}.h5"

    with h5py.File(state, "r") as handle:
        values = np.asarray(handle["values"][:, 0, 0, :], dtype=np.float64)
        coord = np.asarray(handle["x"][()], dtype=np.float64)  # [2, 4, 1, nodes]
        size = np.asarray(handle["size"][()], dtype=np.float64)
        vertex = np.asarray(handle["vertex"][()]).astype(np.int64) - 1  # [4, n_elem]
        boundary = np.asarray(handle["boundary"][()]).astype(np.int64)
        tstep = float(np.ravel(handle["tstep"][()])[0])
        t_norm = float(np.ravel(handle["t_norm"][()])[0])
    gather_time = tstep * t_norm

    with h5py.File(part, "r") as handle:
        node = handle["groups/001"]
        weight = np.asarray(node["weight"][()], dtype=np.float64)
        charge = np.asarray(node["P_imp"][()], dtype=np.float64)
        element = np.asarray(node["i_elm"][()], dtype=np.int64)
        st = np.asarray(node["st"][()], dtype=np.float64)
    rhs = density_channel_rhs(
        weight, charge, element, st, size, n_norm=args.central_density * 1e20
    )
    # Density channels already carry ``timesteps`` inside the accumulation
    # (``Z_imp = w <Z> timesteps``), which cancels the ``1/(n_steps
    # timesteps)`` gather normalisation: the projected RHS is the plain
    # ``sum_p HH size w <Z> / n_norm`` (verified against the logfile's
    # "Total generated electrons" = sum_p w<Z>).
    rhs = [channel for channel in rhs]

    with h5py.File(projection, "r") as handle:
        recorded = np.asarray(handle["values"][:5], dtype=np.float64)  # [channel, 4, 1, nodes]

    interior = np.all(boundary[vertex] == 0, axis=0)
    # Pick the interior elements that actually carry source (largest ||b||), not
    # the first ones: far from the shards both x and b are ~0 and the relative
    # residual is meaningless.
    energy = np.sqrt((rhs[0] ** 2 + rhs[1] ** 2 + rhs[2] ** 2).sum(axis=(0, 1)))
    order = np.argsort(np.where(interior, energy, -1.0))[::-1]
    picks = order[: args.n_elements]
    names = ("Z_eff", "Z_imp", "N_imp")
    recorded_channels = (2, 3, 4)
    print(f"run={args.run} index={args.index} gather_time={gather_time:.4e}s")
    print(f"interior elements checked: {len(picks)} / {vertex.shape[1]}")
    for name, rhs_channel, rec_channel in zip(names, rhs, recorded_channels):
        num = den = 0.0
        for elem in picks:
            verts = vertex[:, elem]
            # ``coord[dim, dof, 0, node]`` -> [vertex, dof, dim] for the local
            # element (JOREK's basis index order is (vertex, dof)).
            coord_elem = np.stack(
                [coord[0, :, 0, verts].T, coord[1, :, 0, verts].T], axis=-1
            )
            elm = element_mass_matrix(coord_elem, size[:, :, elem]) * TWOPI_OVER_N_PLANE
            # recorded[channel] is [dof, tor, node]; reorder to [vertex, dof]
            # so the flat index matches ``v.reshape(-1)`` in the mass matrix.
            local_x = recorded[rec_channel, :, 0, verts].T.reshape(-1)
            local_b = rhs_channel[:, :, elem].reshape(-1)
            residual = elm @ local_x - local_b
            num += float(residual @ residual)
            den += float(local_b @ local_b)
        print(f"  {name:6s} ||M x - b|| / ||b|| = {np.sqrt(num / den):.4f}")


if __name__ == "__main__":
    main()
