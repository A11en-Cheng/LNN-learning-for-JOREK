"""Phase B2.1 implicit/semi-implicit stencil tests (``src/models/implicit_stencil.py``).

Component-level physical acceptance (ADR-0018 decision 13, roadmap_v2 Phase B2):

* per-term metric weights (parallel ``BigR/BB2`` = ``BigR`` on the
  unit-direction kernel, perpendicular/viscous ``BigR``, ``visco_num``
  unweighted) with the code lines registered per term;
* the exact channel-term map of ``mod_elt_matrix.f90`` including
  ``ZK_perp_num -> T_i/T_e``;
* exact per-term ``lambda_max`` / explicit limits ``dt = 2/(alpha*lambda)``;
* *quasi-steady fidelity*: the physical-dt parallel profile relaxes at the
  correct rate instead of being frozen (the B2 damped-iteration failure mode is
  kept as an explicit comparator), the exact implicit solve matches a dense /
  SciPy direct reference (the "JOREK single step" weak-form system) and the
  field-line average is conserved;
* forward/backward stability, determinism, finite gradients and the solver cost
  record (iterations/time; no performance gate).

Real-frame checks (``jorek003000.h5``) run only when the EHL-2 data is present;
they are frame-level smoke, not a baseline or gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from src.data.model502_schema import FIELD_INDEX, FIELD_NAMES, NUM_STATE_FIELDS
from src.models import implicit_stencil as isd
from src.models import stencil_conv as sc
from src.physics import fem_weak
from stencil_fixtures import (
    deterministic_state,
    fine_structured_mesh,
    uniform_b_hat,
)

REAL_FRAME = Path("data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek003000.h5")
REAL_NAMELIST = Path(
    "data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek_namelist_spitest"
)
real_frame = pytest.mark.skipif(
    not REAL_FRAME.exists(), reason="EHL-2 v400 frame not available in this environment"
)

DT = 0.05


def _synthetic_operators(nx: int = 9, nz: int = 9, *, angle: float = 0.7):
    vertex, pos = fine_structured_mesh(nx, nz)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    operators = isd.build_term_operators(
        geometry, b_hat_nodes=uniform_b_hat(pos.shape[0], angle=angle)
    )
    return vertex, pos, geometry, operators


def _alpha_values(**overrides) -> dict:
    """Full registered alpha table with optional per-term overrides."""

    values = {term: float(isd.DOC_ALPHA_TABLE[term]) for term in isd.TERM_NAMES}
    values.update({term: float(value) for term, value in overrides.items()})
    return values


def _coefficients(**overrides) -> isd.TransportCoefficients:
    return isd.TransportCoefficients(
        values=_alpha_values(**overrides),
        provenance={term: "test" for term in isd.TERM_NAMES},
    )


def _cluster_operator(operators, layer, channel: int, dt: float):
    """Dense ``I + dt * sum_t alpha[t,c] A_t/m`` of one channel (no BC rows)."""

    alpha = layer._implicit_alpha().detach().numpy()
    matrix = np.eye(operators.n_clusters)
    for index, term in enumerate(layer.term_names):
        value = float(alpha[index, channel])
        if value == 0.0:
            continue
        matrix = matrix + dt * value * (
            np.asarray(operators.matrix(term).todense()) / operators.mass[:, None]
        )
    return matrix


def _channel_system(layer, channel: int, dt: float):
    """Sparse ``I + dt * sum_t alpha[t,c] A_t/m`` with the Dirichlet rows."""

    operators = layer.operators
    alpha = layer._implicit_alpha().detach().numpy()
    matrix = sp.identity(operators.n_clusters, format="csr")
    for index, term in enumerate(layer.term_names):
        value = float(alpha[index, channel])
        if value == 0.0:
            continue
        matrix = matrix + dt * value * (
            sp.diags(1.0 / operators.mass) @ operators.matrix(term)
        )
    fixed = np.asarray(layer._dirichlet_cluster)[:, channel].copy()
    if fixed.any():
        matrix = sp.diags((~fixed).astype(np.float64)) @ matrix + sp.diags(
            fixed.astype(np.float64)
        )
    return matrix.tocsr(), fixed


def _backward_error(layer, channel: int, dt: float, rhs, solved) -> float:
    """Machine-precision backward error of the implicit system of one channel.

    The stiff system is ill-conditioned by construction (``dt*alpha*lambda`` up
    to 1e12), so the *relative residual* is condition-number limited; the
    backward error is the correct "true solve" statement.
    """

    geometry = layer.geometry
    matrix, fixed = _channel_system(layer, channel, dt)
    b = geometry.to_cluster_numpy(np.asarray(rhs))[:, channel].copy()
    b[fixed] = b[fixed]  # Dirichlet rows hold the reference (= rhs in tests)
    x = geometry.to_cluster_numpy(np.asarray(solved))[:, channel]
    residual = matrix @ x - b
    norm_matrix = abs(matrix).sum(axis=1).max()
    denominator = norm_matrix * np.linalg.norm(x, np.inf) + np.linalg.norm(b, np.inf)
    return float(np.linalg.norm(residual, np.inf) / denominator)


# ---------------------------------------------------------------------------
# Per-term metric weights and channel map (B2.1)
# ---------------------------------------------------------------------------
def test_term_registry_metric_weights_and_alpha_provenance():
    assert set(isd.TERM_NAMES) == set(isd.TERM_METRIC_WEIGHT)
    assert len(isd.TERM_NAMES) == len(set(isd.TERM_NAMES))
    for term in isd.PARALLEL_TERMS:
        assert isd.TERM_METRIC_WEIGHT[term] == "parallel"
    for term in isd.RADIAL_TERMS:
        assert isd.TERM_METRIC_WEIGHT[term] == "radius"
    for term in isd.UNWEIGHTED_TERMS:
        assert isd.TERM_METRIC_WEIGHT[term] == "none"
    for term in isd.TERM_NAMES:
        assert term in isd.DOC_ALPHA_TABLE
        assert "mod_elt_matrix.f90" in isd.ALPHA_PROVENANCE[term]
    # the three JOREK metric classes of docs/jorek-normalization.md section 3.1
    assert isd.TERM_METRIC_WEIGHT["visco_num"] == "none"
    assert isd.TERM_METRIC_WEIGHT["D_par"] == "parallel"
    assert isd.TERM_METRIC_WEIGHT["visco_par"] == "radius"
    assert isd.TERM_METRIC_WEIGHT["ZK_perp_num"] == "radius"


def test_parallel_weight_is_bigR_over_BB2_on_the_B_kernel():
    """``BigR/BB2 * (B.grad N_i)(B.grad N_j) == BigR * (bhat.grad N_i)(bhat.grad N_j)``."""

    vertex, pos = fine_structured_mesh(5, 5)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    b_hat = uniform_b_hat(pos.shape[0])
    magnitude = 2.5  # constant |B| in tesla (the BB2 factor of the weak form)
    b_vector = magnitude * b_hat
    operators_unit = isd.build_term_operators(geometry, b_hat_nodes=b_hat, terms=("D_par",))
    radius = isd.element_metric_weights(geometry, "radius")
    literal = np.zeros((geometry.n_elements, 4, 4))
    for element in range(geometry.n_elements):
        corners = geometry.vertex[:, element]
        coords = pos[corners]
        b_e = b_vector[corners].mean(axis=0)
        # JOREK factor BigR/BB2 with BB2 = |B|^2 (the B-magnitude kernel
        # already carries |B|^2, so the product collapses to BigR)
        weight = float(radius[element]) / float(np.dot(b_e, b_e))
        for triangle in ((0, 1, 2), (0, 2, 3)):
            idx = np.asarray(triangle)
            kernel = fem_weak.triangle_parallel_stiffness(coords[idx], b_e)
            for i in range(3):
                for j in range(3):
                    literal[element, idx[i], idx[j]] += weight * kernel[i, j]
    assert np.abs(literal - operators_unit.element_kernels["D_par"]).max() < 1e-12


def test_radius_weight_scales_the_b0_kernel_by_the_element_radius():
    vertex, pos = fine_structured_mesh(5, 5)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    operators = isd.build_term_operators(geometry, terms=("visco", "visco_num"))
    radii = pos[geometry.vertex, 0].mean(axis=0) / isd.R_REF
    expected = geometry.element_kernel * radii[:, None, None]
    assert np.abs(operators.element_kernels["visco"] - expected).max() < 1e-14
    assert np.array_equal(operators.element_kernels["visco_num"], geometry.element_kernel)
    # weighted/unweighted isotropic spectra differ by the O(R) metric factor
    assert operators.lambda_max("visco") != operators.lambda_max("visco_num")


def test_isotropic_kernels_reuse_b0_and_match_fem_weak():
    _, _, geometry, operators = _synthetic_operators()
    stiffness, mass, _ = fem_weak.build_p1_laplacian(geometry.vertex, geometry.pos)
    assert abs(operators.matrix("visco_num") - stiffness).max() < 1e-12
    assert np.abs(operators.mass - mass).max() < 1e-12
    # the radial weight is element-wise, so the assembled matrix is the B0
    # stiffness with the radius factor applied before scattering
    radii = geometry.pos[geometry.vertex, 0].mean(axis=0) / isd.R_REF
    rows, cols, vals = [], [], []
    for element in range(geometry.n_elements):
        local = geometry.element_kernel[element] * radii[element]
        for i in range(4):
            for j in range(4):
                rows.append(geometry.vertex_cluster[i, element])
                cols.append(geometry.vertex_cluster[j, element])
                vals.append(local[i, j])
    expected = sp.coo_matrix(
        (vals, (rows, cols)), shape=(geometry.n_clusters, geometry.n_clusters)
    ).tocsr()
    assert abs(operators.matrix("visco") - expected).max() < 1e-12


def test_channel_term_map_covers_all_nine_fields_and_matches_jorek():
    mask = isd.channel_term_mask()
    assert mask.shape == (len(isd.TERM_NAMES), NUM_STATE_FIELDS)
    assert (mask.sum(axis=0) > 0).all(), "every field must own at least one transport term"
    for channel, name in enumerate(FIELD_NAMES):
        expected = isd.DEFAULT_CHANNEL_TERMS[name]
        assert {isd.TERM_NAMES[i] for i in np.flatnonzero(mask[:, channel])} == set(expected)
        for term in expected:
            assert term in isd.CHANNEL_TERM_PROVENANCE[name]
    # B2.1 revision: ZK_perp_num was missing in B2 and must map to T_i/T_e
    assert "ZK_perp_num" in isd.DEFAULT_CHANNEL_TERMS["T_i"]
    assert "ZK_perp_num" in isd.DEFAULT_CHANNEL_TERMS["T_e"]
    assert isd.CHANNEL_TERM_PROVENANCE["T_i"]["ZK_perp_num"] == ":1598"
    assert isd.CHANNEL_TERM_PROVENANCE["T_e"]["ZK_perp_num"] == ":1757"
    assert set(isd.DEFAULT_CHANNEL_TERMS["rho"]) == {"D_perp", "D_perp_num", "D_par"}
    assert set(isd.DEFAULT_CHANNEL_TERMS["rho_imp"]) == {
        "D_perp_imp",
        "Dn_perp_num",
        "D_par_imp",
    }
    assert set(isd.DEFAULT_CHANNEL_TERMS["v_par"]) == {
        "visco_par",
        "visco_par_num",
        "visco_par_par",
    }
    assert set(isd.DEFAULT_CHANNEL_TERMS["Phi"]) == {"visco", "visco_num"}
    assert set(isd.DEFAULT_CHANNEL_TERMS["omega"]) == {"visco", "visco_num"}
    assert "ZK_i_par" in isd.DEFAULT_CHANNEL_TERMS["T_i"]
    assert "ZK_e_par" in isd.DEFAULT_CHANNEL_TERMS["T_e"]
    assert FIELD_INDEX["T_e"] == FIELD_NAMES.index("T_e")


def test_explicit_stability_table_uses_the_per_term_weights():
    _, _, _, operators = _synthetic_operators()
    rows = {row.term: row for row in isd.explicit_stability_table(operators)}
    assert set(rows) == set(isd.TERM_NAMES)
    for row in rows.values():
        if row.alpha == 0.0:
            assert row.dt_limit == float("inf")
        else:
            assert row.dt_limit == pytest.approx(
                2.0 / (row.alpha * row.lambda_max), rel=1e-12
            )
        assert row.metric_weight == isd.TERM_METRIC_WEIGHT[row.term]
    # metric classes share their spectra (same kernel), unweighted differs
    assert rows["visco"].lambda_max == pytest.approx(rows["D_perp"].lambda_max, rel=1e-12)
    assert rows["D_par"].lambda_max == pytest.approx(rows["ZK_i_par"].lambda_max, rel=1e-12)
    assert rows["eta"].lambda_max < rows["visco"].lambda_max
    # docs/jorek-normalization.md section 3.2 conclusions, exact spectra
    assert rows["eta"].dt_limit > 1.0 and rows["eta"].stable_at(DT)
    for term in ("visco", "visco_par", "visco_par_num", "D_par", "ZK_i_par", "ZK_e_par"):
        assert rows[term].dt_limit < DT, term
    assert rows["ZK_e_par"].dt_limit < 1e-8 < rows["eta"].dt_limit
    # the physical perpendicular profile diffusion is the critical row
    assert 0.25 * DT < rows["D_perp"].dt_limit < 4.0 * DT
    assert rows["ZK_perp_num"].dt_limit > DT


# ---------------------------------------------------------------------------
# Solver: exactness, stability, gradients, cost record
# ---------------------------------------------------------------------------
def test_direct_solver_matches_the_dense_exact_solution():
    _, _, geometry, operators = _synthetic_operators(nx=5, nz=5)  # 25 nodes
    moderate = {
        term: 1e-2
        for term in isd.TERM_NAMES
    }
    moderate["eta"] = 0.0
    layer = isd.ImplicitStencilLayer(
        geometry,
        operators=operators,
        dt=DT,
        solver="direct",
        alphas=_coefficients(**moderate),
    )
    state = torch.tensor(deterministic_state(geometry.n_nodes), dtype=torch.float64)
    solved = layer.implicit_solve(state, state, dt=DT).detach()
    for channel in (
        FIELD_INDEX["Phi"],
        FIELD_INDEX["rho"],
        FIELD_INDEX["T_i"],
        FIELD_INDEX["T_e"],
    ):
        exact = np.linalg.solve(
            _cluster_operator(operators, layer, channel, DT),
            geometry.to_cluster_numpy(state.numpy())[:, channel],
        )
        got = geometry.to_cluster_numpy(solved.numpy())[:, channel]
        assert np.allclose(got, exact, rtol=1e-10, atol=1e-10), channel


def test_direct_solve_is_a_true_solve_not_a_frozen_mode():
    _, _, geometry, operators = _synthetic_operators()
    layer = isd.ImplicitStencilLayer(
        geometry, operators=operators, dt=DT, solver="direct", alphas=_coefficients()
    )
    state = torch.tensor(deterministic_state(geometry.n_nodes), dtype=torch.float64)
    solved = layer.implicit_solve(state, state, dt=DT).detach()
    for channel in (FIELD_INDEX["T_i"], FIELD_INDEX["T_e"], FIELD_INDEX["rho"]):
        error = _backward_error(
            layer, channel, DT, state.numpy(), solved.detach().numpy()
        )
        assert error < 1e-14, (channel, error)
    # the stiff parallel mode is *solved*: the correction is large
    correction = float((solved - state).norm() / state.norm())
    assert correction > 1e-3


def _parallel_eigen(operators):
    """Generalized eigen-decomposition of the parallel operator (small meshes)."""

    import scipy.linalg as sla

    matrix = np.asarray(operators.matrix("ZK_i_par").todense())
    eigenvalues, eigenvectors = sla.eigh(matrix, np.diag(operators.mass))
    return eigenvalues, eigenvectors


def test_richardson_solver_is_stable_but_frozen_on_the_parallel_profile():
    """The B2 damped iteration is contractive but leaves the k=1 mode frozen.

    For the field-aligned operator the Richardson contraction of a mode with
    generalized eigenvalue ``lambda`` is ``1 - omega*lambda/D`` with
    ``D ~ dt*alpha*lambda_max``; for the physical ``k = 1`` mode
    (``lambda/lambda_max ~ 2e-3``) that is ~0.998 per iteration, i.e. 12
    iterations leave the profile essentially unchanged while the true implicit
    solve relaxes it to ``1/(1 + dt*alpha*lambda) ~ 5e-3``.
    """

    _, _, geometry, direct_layer = _parallel_profile_setup(stiffness=1e-1)
    operators = direct_layer.operators
    eigenvalues, eigenvectors = _parallel_eigen(operators)
    index = int(np.argmax(eigenvalues > 1e-6 * eigenvalues.max()))
    mode = torch.zeros((geometry.n_nodes, NUM_STATE_FIELDS), dtype=torch.float64)
    mode[:, FIELD_INDEX["T_i"]] = torch.tensor(eigenvectors[:, index])
    mass = torch.tensor(operators.node_mass(), dtype=torch.float64)
    norm = float(torch.sqrt((mass * mode[:, FIELD_INDEX["T_i"]] ** 2).sum()))

    richardson = isd.ImplicitStencilLayer(
        geometry,
        operators=operators,
        dt=DT,
        solver="richardson",
        n_iters=12,
        explicit_terms=(),
        implicit_terms=("ZK_i_par",),
        channel_terms={"T_i": ("ZK_i_par",)},
        alphas=_coefficients(ZK_i_par=1e-1),
    )
    assert float(richardson.contraction_factor().detach().max()) < 1.0
    frozen = richardson.implicit_solve(mode, mode, dt=DT).detach()
    frozen_ratio = float(
        torch.sqrt((mass * frozen[:, FIELD_INDEX["T_i"]] ** 2).sum())
    ) / norm
    exact = direct_layer.implicit_solve(mode, mode, dt=DT).detach()
    exact_ratio = float(
        torch.sqrt((mass * exact[:, FIELD_INDEX["T_i"]] ** 2).sum())
    ) / norm
    expected = 1.0 / (1.0 + DT * 1e-1 * float(eigenvalues[index]))
    assert exact_ratio == pytest.approx(expected, rel=1e-8)
    assert exact_ratio < 0.1
    assert frozen_ratio > 100.0 * exact_ratio


def test_cg_solver_is_stable_reports_iterations_and_stays_finite():
    _, _, geometry, operators = _synthetic_operators()
    layer = isd.ImplicitStencilLayer(
        geometry,
        operators=operators,
        dt=DT,
        solver="cg",
        max_cg_iters=200,
        cg_tol=1e-8,
        alphas=_coefficients(),
    )
    state = torch.tensor(deterministic_state(geometry.n_nodes), dtype=torch.float64)
    solved = layer.implicit_solve(state, state, dt=DT)
    assert torch.isfinite(solved).all()
    stats = layer.last_solve_stats
    assert stats["solver"] == "cg"
    assert 1 <= stats["iterations"] <= 200
    assert stats["residual_relative"] is not None
    # a second call must stay bounded as well (no transient growth)
    second = layer.implicit_solve(solved, solved, dt=DT)
    assert float(second.detach().abs().max()) <= 1.5 * float(state.abs().max())


def test_solver_determinism_finite_gradients_and_cost_record():
    _, _, geometry, operators = _synthetic_operators()
    state = torch.tensor(deterministic_state(geometry.n_nodes), dtype=torch.float64)
    for solver in ("direct", "cg", "richardson"):
        layer = isd.ImplicitStencilLayer(
            geometry,
            operators=operators,
            dt=DT,
            solver=solver,
            n_iters=8,
            max_cg_iters=50,
            alphas=_coefficients(),
        )
        first = layer(state)
        second = layer(state)
        assert torch.equal(first, second), solver
        loss = (first ** 2).mean()
        for name, parameter in layer.named_parameters():
            gradient = torch.autograd.grad(
                loss, parameter, retain_graph=True, allow_unused=True
            )[0]
            assert gradient is not None, (solver, name)
            assert torch.isfinite(gradient).all(), (solver, name)
        stats = layer.last_solve_stats
        assert stats["solver"] == solver
        assert stats["n_solves"] >= 1
        assert stats["time_s"] >= 0.0
        if solver == "direct":
            assert stats["iterations"] is None
            assert stats["matrix_nnz"] > 0
        else:
            assert stats["iterations"] >= 1


def test_explicit_step_with_physical_alpha_explodes_on_a_stiff_term():
    """B0's alpha=1 cap is not the physical cap (ADR-0018 decision 13)."""

    _, _, geometry, operators = _synthetic_operators()
    layer = sc.StencilConv(geometry, dt=DT, learnable=False)
    with torch.no_grad():
        layer.gain.fill_(isd.DOC_ALPHA_TABLE["ZK_e_par"])
    state = torch.tensor(deterministic_state(geometry.n_nodes), dtype=torch.float64)
    out = layer(state)
    assert torch.isfinite(out).all()
    growth = float(out.abs().max() / state.abs().max())
    assert growth > 1e3, growth


def test_explicit_cap_layer_is_a_safety_component_not_a_propagator():
    """The explicit layer is built from the weighted kernels and caps ``dt``."""

    _, pos, geometry, _ = _synthetic_operators()
    layer = isd.ExplicitStencilLayer(
        geometry, b_hat_nodes=uniform_b_hat(pos.shape[0]), dt=DT
    )
    assert layer.dt_capped
    assert layer.dt_effective < DT * 1e-6
    assert layer.spectral_radius > 0.0
    state = torch.tensor(deterministic_state(geometry.n_nodes), dtype=torch.float64)
    out = layer(state)
    assert torch.isfinite(out).all()
    assert float(out.abs().max()) <= float(state.abs().max()) * 1.05


def test_legacy_metric_weight_jorek_matches_the_per_term_parallel_kernel():
    vertex, pos = fine_structured_mesh(5, 5)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    b_hat = uniform_b_hat(pos.shape[0])
    legacy = isd.build_term_operators(
        geometry, b_hat_nodes=b_hat, terms=("D_par",), metric_weight="jorek"
    )
    default = isd.build_term_operators(
        geometry, b_hat_nodes=b_hat, terms=("D_par",)
    )
    assert np.allclose(
        legacy.element_kernels["D_par"], default.element_kernels["D_par"], rtol=0, atol=1e-14
    )


# ---------------------------------------------------------------------------
# Quasi-steady fidelity along the field line (B2.1 core acceptance)
# ---------------------------------------------------------------------------
def _parallel_profile_setup(nx: int = 5, nz: int = 33, *, stiffness: float = 1e2):
    """Fine mesh whose uniform b_hat points along Z; T_i only takes ZK_i_par."""

    vertex, pos = fine_structured_mesh(nx, nz, spacing=5e-4)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    operators = isd.build_term_operators(
        geometry, b_hat_nodes=uniform_b_hat(pos.shape[0], angle=np.pi / 2.0)
    )
    coefficients = _coefficients(ZK_i_par=stiffness)
    layer = isd.ImplicitStencilLayer(
        geometry,
        operators=operators,
        dt=DT,
        solver="direct",
        explicit_terms=(),
        implicit_terms=("ZK_i_par",),
        channel_terms={"T_i": ("ZK_i_par",)},
        alphas=coefficients,
    )
    return vertex, pos, geometry, layer


def _parallel_mode(pos: np.ndarray, k: int) -> torch.Tensor:
    state = torch.zeros((pos.shape[0], NUM_STATE_FIELDS), dtype=torch.float64)
    z = torch.tensor(pos[:, 1], dtype=torch.float64)
    z0, z1 = float(z.min()), float(z.max())
    state[:, FIELD_INDEX["T_i"]] = torch.cos(k * np.pi * (z - z0) / (z1 - z0))
    return state


def test_parallel_profile_relaxes_at_the_correct_rate():
    """Moderate alpha: the discrete implicit-Euler decay matches the dense solve.

    The initial profile is a genuine eigenvector of the field-aligned operator,
    so the one-step decay factor is exactly ``1/(1 + dt*alpha*lambda)`` and the
    layer must reproduce it - the "frozen" ~1 factor of the B2 damped iteration
    is *not* acceptable here.
    """

    _, pos, geometry, layer = _parallel_profile_setup(stiffness=1e-1)
    state = _parallel_mode(pos, k=1)
    channel = FIELD_INDEX["T_i"]
    solved = layer.implicit_solve(state, state, dt=DT).detach()
    exact = np.linalg.solve(
        _cluster_operator(layer.operators, layer, channel, DT),
        geometry.to_cluster_numpy(state.numpy())[:, channel],
    )
    got = geometry.to_cluster_numpy(solved.numpy())[:, channel]
    assert np.allclose(got, exact, rtol=1e-10, atol=1e-10)
    # eigenvalue-resolved rate: use the k=1 eigenvector of the same operator
    eigenvalues, eigenvectors = _parallel_eigen(layer.operators)
    index = int(np.argmax(eigenvalues > 1e-6 * eigenvalues.max()))
    mode = torch.zeros((geometry.n_nodes, NUM_STATE_FIELDS), dtype=torch.float64)
    mode[:, channel] = torch.tensor(eigenvectors[:, index])
    mass = torch.tensor(layer.operators.node_mass(), dtype=torch.float64)
    mode_norm = float(torch.sqrt((mass * mode[:, channel] ** 2).sum()))
    relaxed = layer.implicit_solve(mode, mode, dt=DT).detach()
    decay = float(torch.sqrt((mass * relaxed[:, channel] ** 2).sum())) / mode_norm
    expected = 1.0 / (1.0 + DT * layer.coefficients.alpha("ZK_i_par") * float(eigenvalues[index]))
    assert decay == pytest.approx(expected, rel=1e-8)
    assert decay < 0.1  # not frozen


def test_stiff_parallel_profile_projects_to_the_null_space_not_frozen():
    """Stiff alpha: the physical-dt solve projects onto the field-aligned subspace.

    The stiff implicit solve of ``(I + dt*alpha*A/m) x = x0`` maps
    ``x0 -> P x0`` (the orthogonal projection onto the near-null space of the
    field-aligned operator, i.e. functions constant along the field line),
    *not* to zero and *not* to x0.  The damped iteration instead leaves x0.
    """

    _, pos, geometry, layer = _parallel_profile_setup(stiffness=1e4)
    eigenvalues, eigenvectors = _parallel_eigen(layer.operators)
    keep = eigenvalues < 1e-6 * max(float(eigenvalues.max()), 1.0)
    projector = eigenvectors[:, keep]
    mass = layer.operators.mass
    state = _parallel_mode(pos, k=1)
    channel = FIELD_INDEX["T_i"]
    solved = layer.implicit_solve(state, state, dt=DT).detach()
    x0 = state[:, channel].numpy()
    projected = projector @ (projector.T @ (mass * x0))
    got = geometry.to_cluster_numpy(solved.numpy())[:, channel]
    assert np.linalg.norm(got - projected) / np.linalg.norm(x0) < 1e-3
    # the perturbation (x0 - P x0) is damped away, the projection survives
    assert np.linalg.norm(got) > 0.5 * np.linalg.norm(projected)
    # the quasi-steady projection conserves the field-line average
    mass_t = torch.tensor(geometry.node_mass(), dtype=torch.float64)
    average_in = float((mass_t * state[:, channel]).sum() / mass_t.sum())
    average_out = float((mass_t * solved[:, channel]).sum() / mass_t.sum())
    # the zero-row-sum of A conserves the mass-weighted mean up to the
    # condition-number-limited backward error of the stiff solve
    scale = float(np.std(x0))
    assert abs(average_out - average_in) < 1e-8 * scale
    # the direct solver removes the perturbation (x0 - P x0); the damped
    # iteration's per-eigenvector "frozen vs solved" contrast is asserted in
    # test_richardson_solver_is_stable_but_frozen_on_the_parallel_profile
    perturbation = float(np.linalg.norm(x0 - projected))
    assert perturbation > 0.0


def test_jorek_single_step_reference_matches_scipy_direct_solve():
    """The B2.1 solve equals the weak-form implicit system JOREK assembles."""

    vertex, pos = fine_structured_mesh(7, 7)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    operators = isd.build_term_operators(geometry, b_hat_nodes=uniform_b_hat(pos.shape[0]))
    layer = isd.ImplicitStencilLayer(
        geometry, operators=operators, dt=DT, solver="direct", alphas=_coefficients()
    )
    state = torch.tensor(deterministic_state(geometry.n_nodes), dtype=torch.float64)
    solved = layer.implicit_solve(state, state, dt=DT).detach().numpy()
    alpha = layer.effective_alpha().detach().numpy()
    for channel, name in (
        (FIELD_INDEX["rho"], "rho"),
        (FIELD_INDEX["T_i"], "T_i"),
        (FIELD_INDEX["T_e"], "T_e"),
    ):
        matrix = sp.identity(geometry.n_clusters, format="csr")
        for index, term in enumerate(layer.term_names):
            value = float(alpha[index, channel])
            if value:
                matrix = matrix + DT * value * (
                    sp.diags(1.0 / operators.mass) @ operators.matrix(term)
                )
        reference = spla.spsolve(
            matrix.tocsc(), geometry.to_cluster_numpy(state.numpy())[:, channel]
        )
        got = geometry.to_cluster_numpy(solved)[:, channel]
        assert np.allclose(got, reference, rtol=1e-10, atol=1e-10), name


# ---------------------------------------------------------------------------
# Real frame (frame-level smoke; not a baseline or gate)
# ---------------------------------------------------------------------------
def _real_frame_setup():
    import h5py

    frame = sc.load_frame_geometry(REAL_FRAME, with_values=True)
    geometry = frame.stencil()
    psi_gradient = sc.nodal_gradient(frame, field=0)
    with h5py.File(REAL_FRAME, "r") as handle:
        f0 = float(np.asarray(handle["F0"][()]).reshape(-1)[0])
    b_hat = isd.b_hat_from_gradients(psi_gradient, frame.pos[:, 0], f0)
    operators = isd.build_term_operators(geometry, b_hat_nodes=b_hat)
    return frame, geometry, operators


@real_frame
def test_real_frame_term_lambda_and_explicit_table():
    frame, geometry, operators = _real_frame_setup()
    interior = frame.boundary_code == 0
    rows = {
        row.term: row
        for row in isd.explicit_stability_table(operators, interior_mask=interior)
    }
    for row in rows.values():
        assert np.isfinite(row.lambda_max) and row.lambda_max > 0.0
        if row.alpha == 0.0:
            assert row.dt_limit == float("inf")
        else:
            assert row.dt_limit == pytest.approx(
                2.0 / (row.alpha * row.lambda_max), rel=1e-12
            )
        assert row.lambda_max <= operators.gershgorin_bound(row.term) * (1 + 1e-9)
    # per-term weights: radial and parallel spectra differ from the unweighted one
    assert rows["visco"].lambda_max > rows["eta"].lambda_max
    assert rows["ZK_i_par"].lambda_max != rows["eta"].lambda_max
    # docs/jorek-normalization.md section 3.2 conclusions with exact spectra
    assert rows["eta"].dt_limit > DT and rows["D_perp"].dt_limit > DT
    for term in ("visco", "visco_par", "visco_par_num", "D_par", "ZK_i_par", "ZK_e_par"):
        assert rows[term].dt_limit < DT, term
    assert rows["ZK_e_par"].dt_limit < 1e-10
    assert rows["ZK_e_par"].provenance.startswith("namelist jorek_namelist_spitest:457")


@real_frame
def test_real_frame_quasi_steady_solve_and_rollout_smoke():
    frame, geometry, operators = _real_frame_setup()
    psi_gradient = sc.nodal_gradient(frame, field=0)
    context = frame.boundary(psi_gradient=psi_gradient)
    layer = isd.ImplicitStencilLayer(
        geometry, operators=operators, dt=DT, solver="direct", boundary=context
    )
    state = torch.tensor(frame.node_values, dtype=torch.float64)
    solved = layer.implicit_solve(state, state, dt=DT)
    # exact solve at the physical dt: machine-level backward error even though
    # the stiff system is ill-conditioned (dt*alpha*lambda ~ 1e12)
    for channel in (FIELD_INDEX["T_i"], FIELD_INDEX["T_e"], FIELD_INDEX["rho"]):
        error = _backward_error(
            layer, channel, DT, state.numpy(), solved.detach().numpy()
        )
        # double-precision limit of the ill-conditioned stiff system
        assert error < 1e-10, (channel, error)
    for channel in (FIELD_INDEX["T_i"], FIELD_INDEX["T_e"]):
        assert float((solved - state).detach()[:, channel].abs().max()) > 0.0
    rolled = state
    for _ in range(4):
        rolled = layer(rolled)
    assert torch.isfinite(rolled).all()
    assert float(rolled.detach().abs().max()) < 1e3
    dirichlet = torch.tensor(context.dirichlet_mask)
    assert torch.equal(rolled[dirichlet], state[dirichlet])


@real_frame
def test_real_frame_transport_coefficients_from_value_chain():
    coefficients = isd.TransportCoefficients.from_h5_and_namelist(
        str(REAL_FRAME), str(REAL_NAMELIST) if REAL_NAMELIST.exists() else None
    )
    for term in isd.TERM_NAMES:
        assert np.isfinite(coefficients.alpha(term))
        assert coefficients.alpha(term) >= 0.0
        assert term in coefficients.provenance
    assert coefficients.alpha("eta") == pytest.approx(1.443e-10, rel=1e-6)
    assert "parameter_channel" in coefficients.provenance["eta"]
    if REAL_NAMELIST.exists():
        # the production namelist carries the numerical/numerical-profile terms
        assert coefficients.alpha("visco_num") == pytest.approx(9.0e-4, rel=1e-7)
        assert coefficients.alpha("visco_par_num") == pytest.approx(1.0e-4, rel=1e-7)
        assert coefficients.alpha("D_perp_num") == pytest.approx(1.0e-12, rel=1e-7)
        assert coefficients.alpha("ZK_perp_num") == pytest.approx(1.0e-12, rel=1e-7)
        assert coefficients.alpha("D_par_imp") == pytest.approx(5.0e-2, rel=1e-10)
