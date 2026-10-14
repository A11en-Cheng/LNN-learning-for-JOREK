"""Phase B0 JOREK boundary-condition layer tests (``src/models/jorek_boundary.py``).

The mapping table is pinned against the JOREK sources
(``mod_boundary_conditions.f90`` / ``mod_boundary_matrix_open.f90`` /
``update_boundary_types.f90``); synthetic meshes exercise every kind of row and
the real-frame tests (skip-if-absent) exercise the actual codes
1/2/5/9/11/15/19/20/21 on 4125 nodes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.model502_schema import FIELD_INDEX
from src.models import jorek_boundary as jb
from src.models import stencil_conv as sc
from stencil_fixtures import deterministic_state, structured_boundary_codes, structured_mesh

REAL_FRAME = Path("data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek003000.h5")
real_frame = pytest.mark.skipif(
    not REAL_FRAME.exists(), reason="EHL-2 v400 frame not available in this environment"
)


def _synthetic_context(*, freeboundary: bool = False, no_mach1_bc: bool = False):
    vertex, pos = structured_mesh(5, 5)
    codes = structured_boundary_codes(5, 5)
    context = jb.build_boundary_context(
        codes, pos, vertex, freeboundary=freeboundary, no_mach1_bc=no_mach1_bc
    )
    return vertex, pos, codes, context


# ---------------------------------------------------------------------------
# Mapping table (JOREK source contract)
# ---------------------------------------------------------------------------
def test_boundary_code_table_pins_jorek_mapping():
    table = {row["code"]: row for row in jb.boundary_code_table()}
    assert set(table) >= {0, 1, 2, 3, 4, 5, 9, 11, 12, 15, 19, 20, 21}
    # mod_boundary_conditions.f90:232-243 / 269-287.
    assert table[2]["apply_dirichlet_all"] and not table[2]["apply_cs"]
    for code in (20, 21):
        assert table[code]["apply_dirichlet_all"] and not table[code]["apply_cs"]
    for code in (1, 3, 4, 5, 9, 11, 15, 19):
        assert table[code]["apply_cs"]
    assert table[1]["field_direction"] == "inward"
    assert table[11]["field_direction"] == "outward"

    psi = FIELD_INDEX["psi"]
    phi = FIELD_INDEX["Phi"]
    zj = FIELD_INDEX["Z_j"]
    omega = FIELD_INDEX["omega"]
    vpar = FIELD_INDEX["v_par"]
    rho = FIELD_INDEX["rho"]
    ti = FIELD_INDEX["T_i"]
    te = FIELD_INDEX["T_e"]
    rho_imp = FIELD_INDEX["rho_imp"]

    def kind(code, field, **kwargs):
        return jb.field_kind_for_code(code, field, **kwargs)

    # psi / Phi / Z_j / omega follow apply_psi_BC / apply_current_BC /
    # apply_dirichlet_1234 (Dirichlet at every BC-coded node).
    for field in (psi, phi, zj, omega):
        for code in (1, 2, 5, 9, 11, 15, 19, 20, 21):
            assert kind(code, field) == jb.DIRICHLET
    # dirichlet_all codes close all nine fields.
    for code in (2, 20, 21):
        for field in range(9):
            assert kind(code, field) == jb.DIRICHLET
    # Mach-1 target codes: v_par sonic, density/temperatures shear flux.
    for code in (1, 5, 9, 11, 15, 19):
        assert kind(code, vpar) == jb.SHEATH_SONIC
        for field in (rho, ti, te, rho_imp):
            assert kind(code, field) == jb.SHEATH_FLUX
    # no_mach1_bc forces apply_dirichlet_all (mod_boundary_conditions.f90:249-250).
    for field in range(9):
        assert kind(1, field, no_mach1_bc=True) == jb.DIRICHLET
    # freeboundary suppresses psi/Z_j Dirichlet (is_freebound).
    assert kind(2, psi, freeboundary=True) == jb.FREE
    assert kind(2, zj, freeboundary=True) == jb.FREE
    assert kind(2, phi, freeboundary=True) == jb.DIRICHLET


def test_synthetic_context_masks_normals_and_one_sided_neighbour():
    vertex, pos, codes, context = _synthetic_context()
    assert context.n_nodes == pos.shape[0]
    assert set(np.unique(context.boundary_code)) == {0, 1, 2, 11, 20}

    centre = pos.mean(axis=0)
    boundary = codes != 0
    outward = np.einsum("ij,ij->i", context.normal[boundary], pos[boundary] - centre)
    assert (outward > 0).all(), "normals must point away from the mesh centre"
    assert (context.boundary_length[boundary] > 0).all()
    assert (codes[context.interior_neighbour[boundary]] == 0).all()

    # bottom row: code 1 -> Phi/Z_j Dirichlet, v_par sonic, T_i sheath flux
    bottom = codes == 1
    assert context.dirichlet_mask[bottom, FIELD_INDEX["Phi"]].all()
    assert context.sonic_mask[bottom, FIELD_INDEX["v_par"]].all()
    assert context.flux_mask[bottom, FIELD_INDEX["T_i"]].all()
    # top row: code 2 -> all nine fields Dirichlet
    top = codes == 2
    assert context.dirichlet_mask[top].all()
    # right column: code 11 (outward target) -> sheath rows as well
    right = codes == 11
    assert context.sonic_mask[right, FIELD_INDEX["v_par"]].all()


# ---------------------------------------------------------------------------
# Row semantics
# ---------------------------------------------------------------------------
def test_dirichlet_hold_and_interior_perturbation_does_not_change_boundary():
    _, pos, _, context = _synthetic_context()
    reference = torch.tensor(deterministic_state(pos.shape[0]))
    perturbed = reference.clone()
    interior = torch.tensor(context.bc_node_mask) == False  # noqa: E712 - explicit
    perturbed[interior] += 1e3

    out = jb.apply_boundary_conditions(perturbed, context, reference=reference)
    mask = torch.tensor(context.dirichlet_mask)
    assert torch.equal(out[mask], reference[mask])


def test_neumann_free_rows_use_one_sided_inward_value():
    vertex, pos, codes, _ = _synthetic_context(freeboundary=True)
    context = jb.build_boundary_context(codes, pos, vertex, freeboundary=True)
    reference = torch.tensor(deterministic_state(pos.shape[0]))
    free = torch.tensor(context.free_mask)
    assert free.any()
    out = jb.apply_boundary_conditions(reference * 3.0, context, reference=reference)
    one_sided = reference.index_select(0, torch.tensor(context.interior_neighbour))
    assert torch.equal(out[free], one_sided[free])


def test_no_mach1_bc_forces_dirichlet_all_nodes():
    _, pos, _, context = _synthetic_context(no_mach1_bc=True)
    boundary = context.bc_node_mask
    assert context.dirichlet_mask[boundary].all()
    assert not context.sonic_mask.any()
    reference = torch.tensor(deterministic_state(pos.shape[0]))
    out = jb.apply_boundary_conditions(reference + 5.0, context, reference=reference)
    assert torch.equal(out[torch.tensor(context.dirichlet_mask)], reference[torch.tensor(context.dirichlet_mask)])


def test_boundary_nan_is_sanitized_before_the_stencil():
    vertex, pos, codes, context = _synthetic_context()
    noisy = torch.tensor(deterministic_state(pos.shape[0]))
    noisy[torch.tensor(context.bc_node_mask)] = float("nan")
    sanitized = jb.sanitize_boundary_inputs(noisy, context)
    assert torch.isfinite(sanitized).all()
    consistent = jb.bc_consistency_report(sanitized, context, reference=sanitized)
    assert consistent["sonic_ok"]


# ---------------------------------------------------------------------------
# Multi-layer rollout and NaN containment
# ---------------------------------------------------------------------------
def test_synthetic_fifty_layers_keep_boundary_consistent():
    vertex, pos, codes, context = _synthetic_context()
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    layer = sc.StencilConv(geometry, dt=1.0, learnable=False, boundary=context)
    state = torch.tensor(deterministic_state(pos.shape[0]))
    rolled = sc.StencilConvStack(layer, 60)(state)

    assert torch.isfinite(rolled).all()
    dirichlet = torch.tensor(context.dirichlet_mask)
    assert torch.allclose(rolled[dirichlet], state[dirichlet], atol=1e-10)
    report = jb.bc_consistency_report(rolled, context, reference=rolled)
    assert report["dirichlet_ok"]
    assert report["sonic_ok"]


def test_boundary_nan_does_not_propagate_inward_synthetic():
    vertex, pos, _, context = _synthetic_context()
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    layer = sc.StencilConv(geometry, dt=1.0, learnable=False, boundary=context)
    state = torch.tensor(deterministic_state(pos.shape[0]))
    broken = state.clone()
    broken[torch.tensor(context.bc_node_mask)] = float("nan")
    rolled = sc.StencilConvStack(layer, 3)(broken)
    assert torch.isfinite(rolled).all()


@real_frame
def test_real_frame_boundary_codes_and_multilayer_rollout():
    frame = sc.load_frame_geometry(REAL_FRAME, with_values=True)
    geometry = frame.stencil()
    gradient = sc.nodal_gradient(frame, field=FIELD_INDEX["psi"])
    context = frame.boundary(psi_gradient=gradient)
    assert context.n_nodes == 4125
    assert context.boundary_code.astype(bool).sum() == 198
    assert set(np.unique(context.boundary_code)) >= {0, 1, 2, 5, 9, 11, 15, 19, 20, 21}
    assert not context.freeboundary
    # psi/Phi/Z_j/omega Dirichlet everywhere; the sheath codes split the rest.
    for field in (0, 1, 2, 3):
        assert context.dirichlet_mask[context.bc_node_mask, field].all()
    assert context.sonic_mask.sum() == 94
    assert context.flux_mask.sum() == 376

    state = torch.tensor(frame.node_values, dtype=torch.float64)
    layer = sc.StencilConv(geometry, dt=0.05, learnable=False, boundary=context)
    rolled = sc.StencilConvStack(layer, 60)(state)
    assert torch.isfinite(rolled).all()
    assert float(rolled.abs().max()) < 1e3

    dirichlet = torch.tensor(context.dirichlet_mask)
    assert torch.equal(rolled[dirichlet], state[dirichlet])
    assert jb.bc_consistency_report(rolled, context, reference=rolled)["sonic_ok"]

    # A NaN at the boundary must be reset before it can enter the interior
    # update; a NaN in the interior must not change the Dirichlet boundary.
    broken = state.clone()
    broken[torch.tensor(context.bc_node_mask)] = float("nan")
    recovered = sc.StencilConvStack(layer, 3)(broken)
    assert torch.isfinite(recovered).all()

    interior_broken = state.clone()
    interior_broken[1000, FIELD_INDEX["psi"]] = float("nan")
    out = layer(interior_broken)
    assert torch.isfinite(out[dirichlet]).all()
