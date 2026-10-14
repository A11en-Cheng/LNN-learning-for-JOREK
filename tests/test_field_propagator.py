"""Phase B2 nine-field propagator tests (``src/models/field_propagator.py``).

Acceptance (ADR-0018 decisions 5/6/7/11/12/13, roadmap_v2 Phase B2):

* nine fields on one contract; shallow FiLM with identity initialisation and a
  structural zero condition-Jacobian for the deep stencil layers, plus the
  behavioral group-permutation response-shape invariant;
* physical ``dt = 0.05`` JOREK time multi-layer rollout: bounded, no NaN,
  BC-consistent, deterministic, differentiable;
* boundary-code fail-fast policy and namelist flag registration
  (``use_simple_bnd_types`` / ``no_mach1_bc``);
* LTC closed-form time integration: ``dt = 0`` identity, positive rates, split
  invariance;
* discrete relations re-checked at the B0 convention.

Real-frame tests are frame-level smoke only (no baseline/gate).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models import field_propagator as fp
from src.models import implicit_stencil as isd
from src.models import jorek_boundary as jb
from src.models import stencil_conv as sc
from stencil_fixtures import (
    confirmed_boundary_codes,
    deterministic_state,
    dirichlet_only_codes,
    fine_structured_mesh,
    uniform_b_hat,
)

REAL_FRAME = Path("data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek003000.h5")
REAL_NAMELIST = Path("data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek_namelist_2d")
real_frame = pytest.mark.skipif(
    not REAL_FRAME.exists(), reason="EHL-2 v400 frame not available in this environment"
)

DT = 0.05
GROUPS = (
    "pellet_mass",
    "pellet_mass",
    "pellet_velocity",
    "pellet_angle",
    "pellet_species",
    "background_plasma",
    "excluded_transport",
    "boundary_flags",
)
CONDITION = np.array([1e22, 0.5, 300.0, 12.0, 10.0, 5e-7, 1.443e-10, 1.0])


def _mesh(nx: int = 9, nz: int = 9, *, codes=None, freeboundary: bool = False):
    vertex, pos = fine_structured_mesh(nx, nz)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    codes = confirmed_boundary_codes(nx, nz) if codes is None else codes
    context = jb.build_boundary_context(codes, pos, vertex, freeboundary=freeboundary)
    operators = isd.build_term_operators(geometry, b_hat_nodes=uniform_b_hat(pos.shape[0]))
    return vertex, pos, geometry, context, operators


def _propagator(*, codes=None, freeboundary: bool = False, **kwargs):
    vertex, pos, geometry, context, operators = _mesh(
        codes=codes, freeboundary=freeboundary
    )
    torch.manual_seed(0)
    return fp.FieldPropagator(
        geometry,
        boundary=context,
        operators=operators,
        condition_dim=len(CONDITION),
        condition_groups=GROUPS,
        dt=DT,
        **kwargs,
    )


def _perturb_film(propagator: fp.FieldPropagator, seed: int = 0) -> None:
    """Make the FiLM head non-trivial without changing the architecture."""

    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed)
        propagator.film.net[-1].weight.normal_(0, 0.3, generator=generator)
        propagator.film.net[-1].bias.normal_(0, 0.1, generator=generator)


def _shape(vector: torch.Tensor) -> torch.Tensor:
    flat = vector.reshape(-1)
    flat = flat - flat.mean()
    return flat / flat.norm()


# ---------------------------------------------------------------------------
# FiLM: identity init, shallow-only depth, group policy
# ---------------------------------------------------------------------------
def test_film_identity_initialisation_and_scope():
    propagator = _propagator(n_layers=2)
    gamma, beta = propagator.film_parameters(torch.tensor(CONDITION))
    assert float(gamma.detach()) == 1.0 and float(beta.detach()) == 0.0
    state = torch.tensor(deterministic_state(propagator.geometry.n_nodes), dtype=torch.float64)
    hidden = propagator.encoder(state)
    assert torch.equal(propagator.encode(state), hidden)
    assert torch.equal(propagator.encode(state, condition=torch.tensor(CONDITION)), hidden)
    report = propagator.condition_depth_report()
    assert report["shared_parameters"] == []
    assert report["shared_with_condition_head"] is False


def test_film_condition_actually_changes_the_response():
    propagator = _propagator(n_layers=1)
    _perturb_film(propagator)
    state = torch.tensor(deterministic_state(propagator.geometry.n_nodes), dtype=torch.float64)
    first = propagator(state, condition=torch.tensor(CONDITION), condition_groups=GROUPS)
    scrambled = fp.scramble_group(CONDITION, GROUPS, "pellet_mass")
    second = propagator(state, condition=torch.tensor(scrambled), condition_groups=GROUPS)
    assert not torch.allclose(first, second)


def test_film_group_policy_rejects_excluded_and_unknown_groups():
    eligible = [group != "excluded_transport" for group in GROUPS]
    fp.validate_condition_groups(GROUPS, eligible)  # accepted
    with pytest.raises(ValueError, match="excluded from FiLM"):
        fp.validate_condition_groups(GROUPS, [True] * len(GROUPS))
    with pytest.raises(ValueError, match="unknown condition group"):
        fp.validate_condition_groups(("not_a_group",), [True])
    with pytest.raises(ValueError, match="excluded from FiLM"):
        _propagator(condition_film_eligible=[True] * len(GROUPS))


# ---------------------------------------------------------------------------
# FiLM: deep Jacobian == 0 and local response shape invariance
# ---------------------------------------------------------------------------
def test_deep_stencil_has_zero_condition_jacobian():
    propagator = _propagator(n_layers=2)
    _perturb_film(propagator)
    state = torch.tensor(deterministic_state(propagator.geometry.n_nodes), dtype=torch.float64)
    probe = propagator.encode(state)
    out_a = propagator.run_stencil_stack(probe.clone())
    out_b = propagator.run_stencil_stack(probe.clone())
    assert torch.equal(out_a, out_b), "the deep stencil path must not see the condition"

    loss = propagator.run_stencil_stack(probe).sum()
    gradients = torch.autograd.grad(
        loss, list(propagator.film.parameters()), allow_unused=True
    )
    assert all(gradient is None for gradient in gradients), (
        "the deep stencil output must have no autograd edge to the condition head"
    )


def test_deep_stencil_local_jacobian_shape_is_condition_invariant():
    """The deep local Jacobian is an affine, condition-independent, local operator."""

    # The local Jacobian is an interior operator property; the Mach-1/sheath BC
    # rows carry a sqrt(T) nonlinearity, so use the pure local stencil path.
    propagator = _propagator(n_layers=1, codes=dirichlet_only_codes())
    base = torch.tensor(deterministic_state(propagator.geometry.n_nodes), dtype=torch.float64)

    def jacobian(probe: torch.Tensor) -> torch.Tensor:
        point = probe.detach().clone().requires_grad_(True)

        def function(x):
            return propagator.run_stencil_stack(x)

        return torch.autograd.functional.jacobian(function, point).reshape(
            point.numel(), point.numel()
        )

    first = jacobian(0.7 * base)
    second = jacobian(1.3 * base)
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    # affine deep path: the local operator does not depend on the input or the
    # condition (the condition only rescales the shallow encoding).
    assert torch.allclose(first, second, rtol=1e-9, atol=1e-9)
    assert float((_shape(first) - _shape(second)).abs().max()) < 1e-9
    # the local coupling pattern is sparse: no direct node-to-node coupling
    # outside the FEM 1-ring (positive entries only along the stencil pattern)
    nonzero = (first.abs() > 1e-12 * first.abs().max()).float().mean()
    assert float(nonzero) < 0.2


def test_group_permutation_keeps_local_response_shape():
    propagator = _propagator(n_layers=2, codes=dirichlet_only_codes())
    _perturb_film(propagator)
    nx = 9
    geometry = propagator.geometry
    base = torch.tensor(deterministic_state(geometry.n_nodes), dtype=torch.float64)
    probe = base.clone()
    node = 4 * nx + 4
    perturbed_channel = 4  # rho
    probe[node, perturbed_channel] += 1e-3
    neighbours = [node - 1, node + 1, node - nx, node + nx]

    responses = {}
    gammas = {}
    for key, condition in (
        ("base", CONDITION),
        ("permuted", fp.scramble_group(CONDITION, GROUPS, "pellet_mass")),
    ):
        condition_tensor = torch.tensor(condition)
        gamma, _ = propagator.film_parameters(condition_tensor, GROUPS)
        gammas[key] = float(gamma.detach())
        plain = propagator(
            base, condition=condition_tensor, condition_groups=GROUPS
        ).detach()
        bumped = propagator(
            probe, condition=condition_tensor, condition_groups=GROUPS
        ).detach()
        responses[key] = (bumped - plain)[neighbours].reshape(-1)

    assert float(responses["base"].norm()) > 0.0
    assert float(responses["permuted"].norm()) > 0.0
    assert gammas["permuted"] != pytest.approx(gammas["base"], rel=1e-6)
    cosine = float((_shape(responses["base"]) * _shape(responses["permuted"])).sum())
    assert cosine > 0.999, cosine
    # the condition may only change the overall gain (dissipation/driving)
    gain = float(responses["permuted"].norm() / responses["base"].norm())
    assert gain != pytest.approx(1.0, rel=1e-3), gain


# ---------------------------------------------------------------------------
# Physical dt rollout, BC, determinism, gradients
# ---------------------------------------------------------------------------
def test_physical_dt_multilayer_rollout_is_bounded_and_bc_consistent():
    propagator = _propagator(n_layers=2, n_iters=12)
    _perturb_film(propagator)
    state = torch.tensor(deterministic_state(propagator.geometry.n_nodes), dtype=torch.float64)
    condition = torch.tensor(CONDITION)
    rolled = state
    with torch.no_grad():
        for _ in range(20):
            rolled = propagator(rolled, condition=condition, condition_groups=GROUPS)
    assert torch.isfinite(rolled).all()
    assert float(rolled.abs().max()) <= float(state.abs().max()) * 2.0

    context = propagator.boundary
    dirichlet = torch.tensor(context.dirichlet_mask)
    assert torch.equal(rolled[dirichlet], state[dirichlet])
    report = jb.bc_consistency_report(rolled.detach(), context, reference=rolled.detach())
    assert report["dirichlet_ok"] and report["sonic_ok"]

    broken = state.clone()
    broken[torch.tensor(context.bc_node_mask)] = float("nan")
    recovered = propagator(broken, condition=condition, condition_groups=GROUPS)
    assert torch.isfinite(recovered).all()


def test_propagator_is_deterministic_and_differentiable():
    propagator = _propagator(n_layers=2, n_iters=8)
    _perturb_film(propagator)
    state = torch.tensor(deterministic_state(propagator.geometry.n_nodes), dtype=torch.float64)
    condition = torch.tensor(CONDITION)
    first = propagator(state, condition=condition, condition_groups=GROUPS)
    second = propagator(state, condition=condition, condition_groups=GROUPS)
    assert torch.equal(first, second)

    loss = (first**2).mean()
    loss.backward()
    with_grad = 0
    for name, parameter in propagator.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            assert name.startswith("source_proj"), name
            continue
        assert torch.isfinite(parameter.grad).all(), name
        with_grad += 1
    assert with_grad >= 10


def test_parameter_and_condition_reports():
    propagator = _propagator(n_layers=2, n_iters=8, hidden=9)
    report = propagator.parameter_report()
    assert report["total_parameters"] == sum(p.numel() for p in propagator.parameters())
    assert report["film_scope"] == "shallow_encoder_only"
    assert report["condition_dim"] == len(CONDITION)
    assert report["film_eligible_dim"] == len(CONDITION) - 1  # excluded_transport
    assert report["film_identity_init"] is True
    assert report["n_layers"] == 2
    assert report["boundary_report"]["unknown_codes"] == []
    assert report["boundary_report"]["pending_codes"] == []
    for layer in propagator.stencil_layers:
        layer_report = layer.parameter_report()
        assert layer_report["n_iters"] == 8
        assert layer_report["alpha_scale_init_identity"]
        assert layer_report["contraction_factor_max"] < 1.0
        assert layer_report["solver"] == "direct"
        assert layer_report["preconditioner"] == "spectral"
        assert layer_report["last_solve_stats"]["solver"] == "direct"
        assert layer_report["metric_weight"]["D_par"] == "parallel"
        assert layer_report["metric_weight"]["visco_num"] == "none"
        assert layer_report["metric_weight"]["ZK_perp_num"] == "radius"
    assert report["solver"] == "direct"


def test_propagator_solver_selection_and_cost_record():
    for solver in ("direct", "cg"):
        propagator = _propagator(n_layers=1, n_iters=6, solver=solver, max_cg_iters=30)
        state = torch.tensor(deterministic_state(propagator.geometry.n_nodes), dtype=torch.float64)
        out = propagator(state)
        assert torch.isfinite(out).all()
        assert propagator.parameter_report()["solver"] == solver
        stats = propagator.stencil_layers[0].last_solve_stats
        assert stats["solver"] == solver
        assert stats["n_solves"] >= 1
        assert stats["time_s"] >= 0.0
        if solver == "direct":
            assert stats["iterations"] is None
            assert stats["matrix_nnz"] > 0
        else:
            assert stats["iterations"] >= 1


def test_set_dt_updates_every_layer():
    propagator = _propagator(n_layers=2)
    propagator.set_dt(0.1)
    assert propagator.dt == 0.1
    assert all(layer.dt == 0.1 for layer in propagator.stencil_layers)


# ---------------------------------------------------------------------------
# LTC component
# ---------------------------------------------------------------------------
def test_ltc_field_step_identity_stability_and_semigroup():
    step = fp.LTCFieldStep(9, dt_reference=DT)
    hidden = torch.tensor(deterministic_state(4), dtype=torch.float64)
    features = hidden.clone()
    assert torch.equal(step(hidden, 0.0), hidden)
    assert (step.rate() > 0).all()
    decay = step.decay(DT).detach()
    assert float(decay.max()) <= 1.0 and float(decay.min()) > 0.0

    # identity target => the component is an identity at init (up to rounding)
    assert torch.allclose(step(hidden, DT, features), hidden, rtol=1e-15, atol=1e-15)
    # split invariance with a non-trivial target and frozen features
    with torch.no_grad():
        step.target.weight.copy_(torch.eye(9) * 0.5)
    split = step(step(hidden, 0.02, features), 0.03, features)
    full = step(hidden, DT, features)
    assert torch.allclose(split, full, rtol=1e-14, atol=1e-14)
    # extreme decay parameters still keep the rate positive / bounded
    with torch.no_grad():
        step.rate_raw.fill_(-100.0)
    assert (step.rate() > 0).all()
    with torch.no_grad():
        step.rate_raw.fill_(100.0)
    assert torch.isfinite(step(hidden, DT, features)).all()


# ---------------------------------------------------------------------------
# Boundary policy and namelist flags
# ---------------------------------------------------------------------------
def test_boundary_code_policy_is_fail_fast():
    codes = confirmed_boundary_codes()
    report = fp.check_boundary_codes(codes)
    assert report["pending_codes"] == [] and report["unknown_codes"] == []
    with_pending = codes.copy()
    with_pending[0] = 20
    with pytest.raises(ValueError, match="pending JOREK-side confirmation"):
        fp.check_boundary_codes(with_pending)
    acknowledged = fp.check_boundary_codes(with_pending, acknowledge_pending_codes=True)
    assert acknowledged["pending_codes"] == [20]
    assert acknowledged["acknowledged_pending_codes"] is True
    unknown = codes.copy()
    unknown[0] = 42
    with pytest.raises(ValueError, match="unregistered boundary code"):
        fp.check_boundary_codes(unknown)


def test_namelist_bc_flags_and_simple_code_folding(tmp_path):
    namelist = tmp_path / "jorek_namelist_test"
    namelist.write_text(
        "&physics_list\n use_simple_bnd_types = .true.\n no_mach1_bc = .t.\n"
        " bc_natural_open = .true.\n/\n",
        encoding="utf-8",
    )
    flags = fp.read_namelist_bc_flags(namelist)
    assert flags["values"]["use_simple_bnd_types"] is True
    assert flags["values"]["no_mach1_bc"] is True
    assert flags["values"]["bc_natural_open"] is True
    assert flags["provenance"]["no_mach1_bc"] == "namelist"
    assert fp.read_namelist_bc_flags(None)["provenance"]["no_mach1_bc"] == "default"

    folded = fp.fold_simple_boundary_codes(np.array([11, 15, 19, 12, 20, 21, 1, 0]))
    assert folded.tolist() == [1, 5, 9, 4, 9, 9, 1, 0]


def test_no_mach1_bc_is_registered_in_the_context():
    vertex, pos = fine_structured_mesh()
    codes = confirmed_boundary_codes()
    context = jb.build_boundary_context(codes, pos, vertex, no_mach1_bc=True)
    assert any("no_mach1_bc=T" in note for note in context.notes)
    assert not context.sonic_mask.any()


def test_namelist_sheath_parameters_are_read_and_derived(tmp_path):
    namelist = tmp_path / "jorek_namelist_sheath"
    namelist.write_text(
        "&physics_list\n"
        " gamma_e_stangeby = 8.0\n gamma_i_stangeby = 8.0\n"
        " density_reflection = 0.05\n imp_reflection = 0.1\n"
        " visco_par_heating = 2.5d-4\n"
        " vpar_smoothing = .true.\n mach_one_bnd_integral = .true.\n"
        " min_sheath_angle = 2.0\n grid_to_wall = .true.\n/\n",
        encoding="utf-8",
    )
    flags = fp.read_namelist_bc_flags(namelist)
    assert flags["values"]["grid_to_wall"] is True
    assert flags["values"]["mach_one_bnd_integral"] is True
    sheath = fp.read_namelist_sheath_parameters(namelist)
    gamma = 5.0 / 3.0
    # model502/initialise_parameters.f90:241-251
    assert sheath.gamma_sheath_e == pytest.approx((gamma - 1.0) * (8.0 - 1.0))
    assert sheath.gamma_sheath_i == pytest.approx(
        (gamma - 1.0) * (8.0 - 1.0 - gamma)
    )
    assert sheath.density_reflection == pytest.approx(0.05)
    assert sheath.imp_reflection == pytest.approx(0.1)
    assert sheath.visco_par_heating == pytest.approx(2.5e-4)
    assert sheath.vpar_smoothing is True
    assert sheath.mach_one_bnd_integral is True
    assert sheath.min_sheath_angle == pytest.approx(2.0)
    assert sheath.c_angle == pytest.approx(np.deg2rad(2.0))
    assert "gamma_sheath_i" in sheath.provenance
    assert "initialise_parameters" in sheath.provenance["gamma_sheath_i"]

    # defaults (preset_parameters.f90:80-94) when nothing is set
    defaults = fp.read_namelist_sheath_parameters(None)
    assert defaults.gamma_sheath_e == pytest.approx(3.0)
    assert defaults.gamma_sheath_i == pytest.approx(-0.111)
    assert defaults.density_reflection == 0.0
    assert defaults.mach_one_bnd_integral is False

    # the context registers the sheath parameters and the corner-pair skip
    vertex, pos = fine_structured_mesh()
    codes = confirmed_boundary_codes()
    context = jb.build_boundary_context(
        codes,
        pos,
        vertex,
        sheath=sheath,
        grid_to_wall=True,
        f0=1.0,
    )
    summary = context.summary()
    assert summary["sheath"]["min_sheath_angle"] == pytest.approx(2.0)
    assert summary["sheath"]["mach_one_bnd_integral"] is True
    assert context.edge_nodes is not None and context.edge_nodes.shape[0] > 0


def test_corner_pair_edges_skip_the_sheath_integration():
    """``construct_matrix_mod.f90:124-130``: corner pairs are cycled (skipped)."""

    vertex, pos = fine_structured_mesh(7, 7)
    # a mesh whose wall is made of corner codes only: every wall edge is a
    # corner pair and must be skipped by the sheath integration
    codes = np.zeros(pos.shape[0], dtype=np.int64)
    codes[:7] = 21
    codes[::7] = 20
    context = jb.build_boundary_context(codes, pos, vertex, grid_to_wall=True)
    assert context.edge_nodes is not None
    assert context.edge_skipped is not None and context.edge_skipped.all()
    assert int(context.summary()["n_skipped_edges"]) == int(context.edge_nodes.shape[0])
    assert context.summary()["n_boundary_edges"] == context.edge_nodes.shape[0]
    # no sheath flux rows take part for corner codes (Dirichlet-all), and a
    # synthetic state is finite after the BC application
    state = torch.tensor(deterministic_state(pos.shape[0]), dtype=torch.float64)
    out = jb.apply_boundary_conditions(
        state, context, reference=state, dt=DT, mass=torch.ones(pos.shape[0])
    )
    assert torch.isfinite(out).all()
    report = jb.bc_consistency_report(out, context, reference=state)
    assert report["dirichlet_ok"]
    # a mixed mesh keeps both skipped and integrated edges
    mixed = confirmed_boundary_codes(7, 7)
    mixed_context = jb.build_boundary_context(mixed, pos, vertex, grid_to_wall=True)
    assert 0 < int(mixed_context.edge_skipped.sum()) < mixed_context.edge_nodes.shape[0]


def test_mach1_target_uses_the_jorek_btot_relation():
    """``v_par = direction*factor*c_s/Btot`` (+clamped poloidal correction)."""

    vertex, pos = fine_structured_mesh(7, 7)
    codes = confirmed_boundary_codes(7, 7)
    # without a flux gradient the poloidal correction is dropped
    plain = jb.build_boundary_context(codes, pos, vertex)
    gradient = np.stack([np.full(pos.shape[0], 0.02), np.zeros(pos.shape[0])], axis=1)
    with_gradient = jb.build_boundary_context(
        codes, pos, vertex, psi_gradient=gradient, f0=2.0
    )
    state = torch.tensor(deterministic_state(pos.shape[0]), dtype=torch.float64)
    # a constant stream function removes the poloidal-flow correction, so the
    # Mach-1 target is exactly direction * c_s / Btot
    from src.data.model502_schema import FIELD_INDEX as FI

    state[:, FI["Phi"]] = 3.0
    out_plain = jb.apply_boundary_conditions(
        state, plain, reference=state, dt=DT, mass=torch.ones(pos.shape[0])
    )
    out_grad = jb.apply_boundary_conditions(
        state, with_gradient, reference=state, dt=DT, mass=torch.ones(pos.shape[0])
    )

    for out, context in ((out_plain, plain), (out_grad, with_gradient)):
        report = jb.bc_consistency_report(out, context, reference=state)
        assert report["sonic_ok"], report
    # the Btot factor makes the sonic target differ from the B0 convention
    # (v_par = direction*c_s) wherever Btot != 1
    temperature = out_grad[:, FI["T_i"]] + out_grad[:, FI["T_e"]]
    cs = torch.sqrt((5.0 / 3.0) * temperature.clamp_min(0.0))
    direction = torch.tensor(with_gradient.direction, dtype=torch.float64)
    gradient_t = torch.tensor(gradient, dtype=torch.float64)
    b_total = torch.sqrt(
        with_gradient.f0 ** 2 + gradient_t[:, 0] ** 2 + gradient_t[:, 1] ** 2
    ) / torch.tensor(with_gradient.radius, dtype=torch.float64)
    expected = direction * cs / b_total
    sonic = torch.tensor(with_gradient.sonic_mask[:, FI["v_par"]])
    assert torch.allclose(out_grad[sonic, FI["v_par"]], expected[sonic], rtol=1e-8, atol=1e-8)
    assert not torch.allclose(
        out_grad[sonic, FI["v_par"]], (direction * cs)[sonic], rtol=1e-3
    )
    # without a flux gradient Btot = 1 and the plain sonic value is used
    plain_target = torch.tensor(plain.direction, dtype=torch.float64) * cs
    plain_sonic = torch.tensor(plain.sonic_mask[:, FI["v_par"]])
    assert torch.allclose(
        out_plain[plain_sonic, FI["v_par"]], plain_target[plain_sonic], rtol=1e-8, atol=1e-8
    )


# ---------------------------------------------------------------------------
# Discrete relations (B0 convention) and real-frame smoke
# ---------------------------------------------------------------------------
def test_propagator_quasi_steady_parallel_profile_is_not_frozen():
    """Full-stack acceptance: the physical-dt field step relaxes the ZK-parallel
    profile at the implicit rate instead of leaving it frozen (ADR-0018
    decision 13, roadmap_v2 Phase B2 quasi-steady row)."""

    from src.models.implicit_stencil import TransportCoefficients
    from src.data.model502_schema import FIELD_INDEX as FI

    vertex, pos = fine_structured_mesh(5, 21, spacing=5e-4)
    geometry = sc.build_fem_stencil_geometry(vertex, pos)
    operators = isd.build_term_operators(
        geometry, b_hat_nodes=uniform_b_hat(pos.shape[0], angle=np.pi / 2.0)
    )
    values = {term: 0.0 for term in isd.TERM_NAMES}
    values["ZK_i_par"] = 1.0
    coefficients = TransportCoefficients(
        values=values, provenance={term: "test" for term in isd.TERM_NAMES}
    )
    propagator = fp.FieldPropagator(
        geometry,
        operators=operators,
        alphas=coefficients,
        channel_terms={"T_i": ("ZK_i_par",)},
        explicit_terms=(),
        implicit_terms=("ZK_i_par",),
        n_layers=1,
        dt=DT,
        solver="direct",
    )
    z = pos[:, 1]
    mode = np.cos(np.pi * (z - z.min()) / (z.max() - z.min()))
    state = torch.zeros((geometry.n_nodes, 9), dtype=torch.float64)
    state[:, FI["T_i"]] = torch.tensor(mode)
    with torch.no_grad():
        out = propagator(state)
    assert torch.isfinite(out).all()
    decay = float(out[:, FI["T_i"]].norm() / state[:, FI["T_i"]].norm())
    assert decay < 0.9
    # and the frozen comparator (the B2 damped iteration) leaves it at ~1
    frozen = fp.FieldPropagator(
        geometry,
        operators=operators,
        alphas=coefficients,
        channel_terms={"T_i": ("ZK_i_par",)},
        explicit_terms=(),
        implicit_terms=("ZK_i_par",),
        n_layers=1,
        dt=DT,
        solver="richardson",
        n_iters=12,
    )
    with torch.no_grad():
        frozen_out = frozen(state)
    frozen_decay = float(frozen_out[:, FI["T_i"]].norm() / state[:, FI["T_i"]].norm())
    assert frozen_decay > 0.9


def test_discrete_relations_reuse_the_b0_convention():
    """B0 convention: weak Laplacian is exact on constants/linear fields."""

    codes = dirichlet_only_codes()
    propagator = _propagator(n_layers=1, codes=codes)
    geometry = propagator.geometry
    interior = torch.tensor(codes == 0)
    radius = torch.tensor(np.clip(geometry.radius, sc.R_MIN, None), dtype=torch.float64)
    psi = torch.tensor(geometry.pos[:, 0], dtype=torch.float64)  # linear in R
    psi_r = torch.ones(geometry.n_nodes, dtype=torch.float64)
    phi = 3.0 * torch.tensor(geometry.pos[:, 1], dtype=torch.float64) - 1.0
    zeros = torch.zeros(geometry.n_nodes, dtype=torch.float64)
    constant = torch.full((geometry.n_nodes,), 2.5, dtype=torch.float64)

    # constants are annihilated exactly; linear fields on interior rows (the
    # boundary rows carry the surface flux term handled by the BC layer)
    assert float(geometry.laplacian(constant).abs().max()) < 1e-8
    assert float(geometry.laplacian(psi)[interior].abs().max()) < 1e-8
    assert float(geometry.laplacian(phi)[interior].abs().max()) < 1e-8
    # the exposed Delta*/Delta+ pair is the coded convention
    delta_star = geometry.delta_star(psi, psi_r)
    delta_plus = geometry.delta_plus(phi, zeros)
    assert torch.allclose(
        delta_star, geometry.laplacian(psi) - psi_r / radius, rtol=0, atol=1e-12
    )
    assert torch.allclose(
        delta_plus, geometry.laplacian(phi) + zeros / radius, rtol=0, atol=1e-12
    )
    assert torch.isfinite(delta_star).all() and torch.isfinite(delta_plus).all()


@real_frame
def test_real_frame_smoke_rollout_registers_pending_codes():
    loaded = fp.build_propagator_from_frame(
        REAL_FRAME,
        namelist_path=REAL_NAMELIST if REAL_NAMELIST.exists() else None,
        acknowledge_pending_codes=True,
        dt=DT,
        n_layers=2,
        n_iters=8,
    )
    assert loaded.boundary_report["acknowledged_pending_codes"] is True
    assert set(loaded.boundary_report["pending_codes"]) <= set(fp.PENDING_BOUNDARY_CODES)
    assert loaded.namelist_flags["values"]["use_simple_bnd_types"] in {True, False}
    state = torch.tensor(loaded.frame.node_values, dtype=torch.float64)
    propagator = loaded.propagator
    rolled = state
    with torch.no_grad():
        for _ in range(3):
            rolled = propagator(rolled)
    assert torch.isfinite(rolled).all()
    assert float(rolled.abs().max()) < 1e3
    dirichlet = torch.tensor(loaded.context.dirichlet_mask)
    assert torch.equal(rolled[dirichlet], state[dirichlet])


@real_frame
def test_real_frame_build_requires_pending_code_acknowledgement():
    with pytest.raises(ValueError, match="pending JOREK-side confirmation"):
        fp.build_propagator_from_frame(
            REAL_FRAME,
            namelist_path=REAL_NAMELIST if REAL_NAMELIST.exists() else None,
            dt=DT,
            n_layers=1,
            n_iters=4,
        )


def test_source_forcing_uses_the_shared_five_channel_interface():
    """B2 consumes the same S object as B1: [nodes, 5], frozen channel order."""

    propagator = _propagator(n_layers=1, n_iters=4)
    state = torch.tensor(deterministic_state(propagator.geometry.n_nodes), dtype=torch.float64)
    n_nodes = propagator.geometry.n_nodes

    assert propagator.source_proj.in_features == 5
    source = torch.zeros(n_nodes, 5, dtype=torch.float64)
    first = propagator(state, source=source)
    second = propagator(state, source=source)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    # Zero-initialised forcing: a source with a different magnitude must change
    # the trajectory once the forcing projection is trained; at initialisation
    # the map is source-neutral (registered).
    with torch.no_grad():
        propagator.source_proj.weight.fill_(1e-3)
    driven = propagator(state, source=source + 1.0)
    assert not torch.equal(driven, first)

    with pytest.raises(ValueError, match="shared channel order"):
        propagator(state, source=torch.zeros(n_nodes, 9, dtype=torch.float64))
    with pytest.raises(ValueError):
        propagator(state, source=torch.full((n_nodes, 5), float("nan"), dtype=torch.float64))
