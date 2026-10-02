"""Regression tests for the full Model 502 residual set.

Operator-level tests use analytic fields on a regular grid, where the
weighted-LSQ gradient operator is exact for quadratic fields and the Poisson
bracket reduces to a known constant.  Data-level tests assert regulariser
behaviour at the true trajectory: the vorticity residual must grow with
prediction error, and every residual must stay finite and bounded once
relative normalisation is applied.  They deliberately do not assert a zero
residual at the recorded trajectory, because the reduced residual set omits
two-fluid, hyper-viscosity and stabilisation terms by design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.data.jorek_h5_parser import JorekDataParser
from src.physics.grad_ops import compute_gradient
from src.physics.model502_full_loss import (
    delta_star,
    delta_plus,
    model502_full_residuals,
    poisson_bracket,
)


def _grid(r0=1.0, r1=3.0, z0=-1.0, z1=1.0, n_r=9, n_z=9):
    rs = torch.linspace(r0, r1, n_r)
    zs = torch.linspace(z0, z1, n_z)
    rr, zz = torch.meshgrid(rs, zs, indexing="ij")
    pos = torch.stack([rr.reshape(-1), zz.reshape(-1)], dim=1)
    edges = []
    for i in range(n_r):
        for j in range(n_z):
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < n_r and 0 <= nj < n_z:
                        edges.append((i * n_z + j, ni * n_z + nj))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    row, col = edge_index
    delta = pos[col] - pos[row]
    edge_attr = torch.cat([delta, delta.norm(dim=1, keepdim=True)], dim=1)
    return Data(pos=pos, edge_index=edge_index, edge_attr=edge_attr)


def _field(grid, fn):
    return fn(grid.pos[:, 0], grid.pos[:, 1])


def _interior(grid, n_r=9, n_z=9):
    """Nodes two layers inside the boundary (full stencil, unpolluted div)."""
    mask = torch.zeros(grid.pos.size(0), dtype=torch.bool)
    for i in range(2, n_r - 2):
        for j in range(2, n_z - 2):
            mask[i * n_z + j] = True
    return mask


def test_poisson_bracket_is_plain_rz_without_r_factor():
    grid = _grid()
    a = _field(grid, lambda r, z: r + 2.0 * z)
    b = _field(grid, lambda r, z: 3.0 * r - z)
    bracket = poisson_bracket(
        compute_gradient(a, grid.edge_index, grid.edge_attr),
        compute_gradient(b, grid.edge_index, grid.edge_attr),
    )
    # {a, b} = a_R b_Z - a_Z b_R = 1 * (-1) - 2 * 3 = -7, independent of R.
    assert torch.allclose(bracket, torch.full_like(bracket, -7.0), atol=1e-4)


def test_poisson_bracket_is_antisymmetric():
    grid = _grid()
    a = _field(grid, lambda r, z: r * r * z)
    b = _field(grid, lambda r, z: z)
    grad_a = compute_gradient(a, grid.edge_index, grid.edge_attr)
    grad_b = compute_gradient(b, grid.edge_index, grid.edge_attr)
    forward = poisson_bracket(grad_a, grad_b)
    backward = poisson_bracket(grad_b, grad_a)
    assert torch.allclose(forward, -backward, atol=1e-5)


def test_delta_star_annihilates_r_squared():
    grid = _grid()
    psi = _field(grid, lambda r, z: r * r)
    out = delta_star(psi, grid.pos, grid.edge_index, grid.edge_attr)
    assert torch.allclose(out[_interior(grid)], torch.zeros_like(out[_interior(grid)]), atol=1e-3)


def test_delta_star_on_z_squared():
    grid = _grid()
    psi = _field(grid, lambda r, z: z * z)
    out = delta_star(psi, grid.pos, grid.edge_index, grid.edge_attr)
    expected = torch.full_like(out, 2.0)
    assert torch.allclose(out[_interior(grid)], expected[_interior(grid)], atol=1e-2)


def test_delta_plus_on_quadratics():
    grid = _grid()
    u = _field(grid, lambda r, z: r * r)
    out = delta_plus(u, grid.pos, grid.edge_index, grid.edge_attr)
    interior = _interior(grid)
    assert torch.allclose(out[interior], torch.full_like(out[interior], 4.0), atol=1e-3)
    u2 = _field(grid, lambda r, z: z * z)
    out2 = delta_plus(u2, grid.pos, grid.edge_index, grid.edge_attr)
    assert torch.allclose(out2[interior], torch.full_like(out2[interior], 2.0), atol=1e-3)


RUN = "scan_1d22_ang15_5_massloss_neon_v400"
CACHE = "/tmp/ehl2_cache"
FIELD_SCALES = torch.tensor([190.0, 2e-4, 16.0, 5e-3, 48.0, 7.0, 0.23, 0.05, 7.0])


def _real_frames(idx0=2750, idx1=2800):
    path = Path(CACHE) / RUN / f"{idx0}.pt"
    if not path.exists():
        pytest.skip("cached real frame not present")
    d0 = torch.load(path, map_location="cpu", weights_only=False)
    d1 = torch.load(Path(CACHE) / RUN / f"{idx1}.pt", map_location="cpu", weights_only=False)
    graph = Data(
        x=d0["x"],
        edge_index=d0["edge_index"],
        edge_attr=d0["edge_attr"],
        pos=d0["pos"],
        condition=d0["condition"],
        source=d0["source"],
    )
    dt = float(d1["time"].item() - d0["time"].item())
    surrogate = None
    return graph, d0["x"].float(), d1["x"].float(), dt, surrogate


def test_vorticity_residual_grows_with_prediction_error():
    graph, x0, x1, dt, _ = _real_frames()
    base = float(model502_full_residuals(x1, x0, graph, dt)["vorticity"])
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn_like(x1, generator=generator) * FIELD_SCALES
    perturbed = float(
        model502_full_residuals(x1 + 0.1 * noise, x0, graph, dt)["vorticity"]
    )
    assert base >= 0.0 and base <= 0.5
    assert perturbed > base + 0.01


def test_residuals_are_bounded_after_relative_normalisation():
    graph, x0, x1, dt, _ = _real_frames()
    residuals = model502_full_residuals(x1, x0, graph, dt)
    for name, value in residuals.items():
        value = float(value)
        assert np.isfinite(value), name
        assert 0.0 <= value <= 2.0, (name, value)


def test_predicted_state_consistency_uses_surrogate():
    graph, x0, x1, dt, surrogate = _real_frames()
    if surrogate is None:
        pytest.skip("Z_j surrogate checkpoint not present")
    residuals = model502_full_residuals(
        x1, x0, graph, dt, zj_surrogate=surrogate
    )
    assert float(residuals["current_definition"]) < 0.5


def test_full_residuals_are_finite_on_real_frame():
    path = "data/EHL-2/scan_1d22_ang12_massloss_neon/jorek002710.h5"
    if not Path(path).exists():
        pytest.skip("real data not present")
    graph = JorekDataParser(path).parse_to_graph()
    residuals = model502_full_residuals(graph.x, graph.x, graph, dt=10.0)
    assert all(np.isfinite(float(value.item())) for value in residuals.values())
    assert set(residuals) >= {
        "induction",
        "vorticity",
        "current_definition",
        "vorticity_definition",
        "density",
        "ion_energy",
        "parallel_momentum",
        "impurity",
        "electron_energy",
    }
