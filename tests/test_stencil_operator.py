"""Phase B0 stencil-conv component tests (``src/models/stencil_conv.py``).

Synthetic meshes pin the FEM assembly contract; the real-frame tests run only
when the EHL-2 data symlink is populated and check the discrete relations
against the recorded ``Z_j`` / ``omega`` channels at the reference convention
(corr 0.994 / 0.997, relL2 0.113 / 0.080).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import stencil_conv as sc
from src.physics import fem_weak
from stencil_fixtures import (
    collapsed_axis_mesh,
    deterministic_state,
    structured_boundary_codes,
    structured_mesh,
)

REAL_FRAME = Path("data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek003000.h5")
real_frame = pytest.mark.skipif(
    not REAL_FRAME.exists(), reason="EHL-2 v400 frame not available in this environment"
)


# ---------------------------------------------------------------------------
# Synthetic geometry / linear algebra
# ---------------------------------------------------------------------------
def test_stiffness_matches_fem_weak_and_annihilates_linear_fields():
    vertex, pos = structured_mesh(5, 5)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)

    stiffness, mass, cluster = fem_weak.build_p1_laplacian(vertex, pos)
    assert np.array_equal(geometry.cluster, cluster)
    assert np.array_equal(geometry.cluster_inverse[cluster], np.arange(pos.shape[0]))
    assert abs(geometry.stiffness - stiffness).max() < 1e-14
    assert np.abs(geometry.mass - mass).max() < 1e-14
    assert geometry.mass.min() > 0

    # Constants are annihilated everywhere (zero row sums).  Linear fields are
    # annihilated by the element integrals; the boundary rows additionally carry
    # the surface flux term integral(N_i du/dn) (JOREK's BC row), so only the
    # interior nodes are checked there.
    constant = torch.ones(pos.shape[0], dtype=torch.float64)
    assert float(geometry.laplacian(constant).abs().max()) < 1e-10
    codes = structured_boundary_codes(5, 5)
    interior = torch.tensor(codes == 0)
    for field in (
        torch.tensor(pos[:, 0]),
        torch.tensor(pos[:, 1]),
        torch.tensor(pos[:, 0] + 3.0 * pos[:, 1] - 2.0),
    ):
        lap = geometry.laplacian(field)
        assert float(lap[interior].abs().max()) < 1e-10


def test_element_kernel_is_symmetric_with_zero_row_sums():
    vertex, pos = structured_mesh(4, 6)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    kernel = geometry.element_kernel[~geometry.degenerate]
    assert kernel.shape[1:] == (4, 4)
    assert np.abs(kernel - kernel.transpose(0, 2, 1)).max() < 1e-14
    assert np.abs(kernel.sum(axis=2)).max() < 1e-12
    assert np.abs(geometry.stiffness - geometry.stiffness.T).max() < 1e-14
    report = sc.geometry_report(geometry)
    assert report["mass_nonpositive"] == 0
    assert report["vertex_in_range"]


def test_stencil_layer_is_deterministic_conservative_and_constant_preserving():
    vertex, pos = structured_mesh(6, 5)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    layer = sc.StencilConv(geometry, dt=0.1, learnable=False)
    state = torch.tensor(deterministic_state(pos.shape[0]))

    first = layer(state)
    second = layer(state)
    assert torch.equal(first, second), "stencil-conv must be deterministic"
    assert torch.isfinite(first).all()

    # Mass conservation in cluster space: sum_i m_i (y_i - x_i) == 0 because
    # A has zero column sums (symmetric, zero row sums).
    mass = torch.tensor(geometry.mass, dtype=torch.float64)
    delta = geometry.to_cluster(first) - geometry.to_cluster(state)
    assert float((mass[:, None] * delta).sum().abs()) < 1e-10

    constant = torch.full((pos.shape[0], 9), 2.5, dtype=torch.float64)
    assert torch.allclose(layer(constant), constant, atol=1e-12)

    # The raw flux exchange is the FEM operator itself, and the layer's
    # stability cap keeps the explicit step inside the Gershgorin bound.
    exchange = layer.flux(state)
    assert torch.isfinite(exchange).all()
    assert layer.stability_limit > 0
    assert layer.dt_effective <= 0.5 * layer.stability_limit + 1e-30


def test_stack_of_sixty_layers_stays_bounded():
    vertex, pos = structured_mesh(5, 5)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    layer = sc.StencilConv(geometry, dt=1.0, learnable=False)
    state = torch.tensor(deterministic_state(pos.shape[0]))
    rolled = sc.StencilConvStack(layer, 60)(state)
    assert torch.isfinite(rolled).all()
    assert float(rolled.abs().max()) < 10.0 * float(state.abs().max())


def test_collapsed_axis_mesh_has_no_metric_singularity():
    vertex, pos = collapsed_axis_mesh(n_arc=8)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    # The eight coincident axis nodes collapse to one cluster dof.
    assert geometry.n_clusters == pos.shape[0] - 7
    assert geometry.mass.min() > 0
    assert np.isfinite(geometry.element_kernel).all()

    constant = torch.full((pos.shape[0], 9), 1.0, dtype=torch.float64)
    layer = sc.StencilConv(geometry, dt=0.5, learnable=False)
    out = layer(constant)
    assert torch.isfinite(out).all()
    assert torch.allclose(out, constant, atol=1e-10)

    # delta_plus divides by R; the clamp must keep R -> 0 finite.
    radius = torch.tensor(np.clip(pos[:, 0], sc.R_MIN, None))
    delta_plus = geometry.delta_plus(constant[:, 1], torch.zeros(pos.shape[0], dtype=torch.float64))
    del radius
    assert torch.isfinite(delta_plus).all()


# ---------------------------------------------------------------------------
# Real-frame checks (skip when the EHL-2 symlink is empty)
# ---------------------------------------------------------------------------
@real_frame
def test_real_frame_geometry_matches_jorek_arrays():
    frame = sc.load_frame_geometry(REAL_FRAME, with_values=True)
    # Same frame's Hermite geometry arrays are present and well formed; ``size``
    # enters the psi_R term through the dof gradient in the relation test.
    assert frame.size.shape == (4, 4, frame.n_elements)
    assert frame.x_dofs.shape == (frame.n_nodes, 4, 2)
    assert np.isfinite(frame.x_dofs).all()
    gradient_map, gradient_vertex = sc.gradient_map_from_frame(frame)
    assert gradient_map.shape == (frame.n_elements, 4, 2, 4)
    assert np.isfinite(gradient_map).all()
    assert np.array_equal(gradient_vertex, frame.vertex)
    geometry = frame.stencil()
    report = sc.geometry_report(geometry)
    assert report["vertex_in_range"]
    assert report["mass_nonpositive"] == 0
    assert report["element_kernel_rowsum_max"] < 1e-10
    assert report["stiffness_symmetry"] < 1e-12

    # Same frame's size/vertex: our assembly is exactly fem_weak's P1 assembly.
    stiffness, mass, cluster = fem_weak.build_p1_laplacian(frame.vertex, frame.pos)
    assert np.array_equal(geometry.cluster, cluster)
    assert abs(geometry.stiffness - stiffness).max() < 1e-12
    assert np.abs(geometry.mass - mass).max() < 1e-14

    adjacency = sc.neighbour_consistency(geometry)
    assert adjacency["available"]
    assert adjacency["mismatch_fraction"] < 0.005
    assert adjacency["matches"] > 0.99 * adjacency["sides_with_noncoincident_corners"]


@real_frame
def test_real_frame_discrete_relations_match_reference_convention():
    frame = sc.load_frame_geometry(REAL_FRAME, with_values=True)
    geometry = frame.stencil()
    values = torch.tensor(frame.node_values, dtype=torch.float64)
    psi_gradient = sc.nodal_gradient(frame, field=0)
    phi_gradient = sc.nodal_gradient(frame, field=1)

    delta_star = geometry.delta_star(values[:, 0], torch.tensor(psi_gradient[:, 0]))
    delta_plus = geometry.delta_plus(values[:, 1], torch.tensor(phi_gradient[:, 0]))
    interior = torch.tensor(frame.boundary_code == 0)

    def stats(prediction, target):
        a = prediction[interior].double()
        b = target[interior].double()
        a = a - a.mean()
        b = b - b.mean()
        corr = float((a * b).mean() / (a.pow(2).mean().sqrt() * b.pow(2).mean().sqrt()))
        rel = float(((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()))
        return corr, rel

    corr_zj, rel_zj = stats(delta_star, values[:, 2])
    corr_om, rel_om = stats(delta_plus, values[:, 3])
    message = (
        f"Z_j: corr={corr_zj:.4f} relL2={rel_zj:.4f}; "
        f"omega: corr={corr_om:.4f} relL2={rel_om:.4f}"
    )
    assert corr_zj > 0.98, message
    assert corr_om > 0.98, message
    assert rel_zj < 0.15, message
    assert rel_om < 0.15, message
