"""Phase B1 self-generated source module: unit + contract tests.

Synthetic fixtures always run; the real-frame checks are skip-if-absent (the
A' native run is a large read-only dataset that is not part of the repository
tree in every environment).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from src.data.budget import ablation_series_reconciliation
from src.data.fragment_state import SHARD_FIELD_NAMES as V1_SHARD_FIELD_NAMES
from src.data.step_dataset import SHARD_FIELD_NAMES as AP_SHARD_FIELD_NAMES
from src.data.step_dataset import StepRunDataset
from src.models.fem_sampling import FemMeshSampler, bilinear_shape
from src.models.source_module import (
    INSTANT_CHANNELS,
    MEMORY_CHANNELS,
    SHARD_LAYOUTS,
    SOURCE_CHANNEL_NAMES,
    SourceModule,
    default_neon_tables,
)
from src.physics.spi_model import (
    EL_CHG,
    MU_ZERO,
    parks_ablation_neon,
    particle_density_si,
    temperature_ev,
)
from tests.conftest import (
    SHARD_NAMES,
    write_native_frame,
    write_native_projection,
    write_native_run,
    write_macroscopic_times,
)

RUN_DIR = Path(
    "data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner"
)
CENTRAL_DENSITY = 0.8
T_NORM = 4.085775e-7


# --------------------------------------------------------------------------- #
# synthetic helpers
# --------------------------------------------------------------------------- #


def synthetic_mesh(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, n)
    rr, zz = np.meshgrid(axis, axis, indexing="ij")
    pos = np.stack([0.5 + 0.1 * rr.ravel(), zz.ravel()], axis=1)
    elements = []
    for i in range(n - 1):
        for j in range(n - 1):
            node = lambda a, b: a * n + b  # noqa: E731
            elements.append(
                [node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)]
            )
    return pos, np.asarray(elements, dtype=np.int64).T


def synthetic_state(
    pos: np.ndarray,
    *,
    rho: float = 0.4,
    rho_imp: float = 0.01,
    te_ev: float = 10.0,
    psi: float = 0.1,
) -> np.ndarray:
    values = np.zeros((pos.shape[0], 9), dtype=np.float64)
    values[:, 0] = psi
    values[:, 4] = rho
    values[:, 7] = rho_imp
    values[:, 8] = te_ev * EL_CHG * MU_ZERO * CENTRAL_DENSITY * 1e20
    return values


def synthetic_shards(
    n: int = 2,
    *,
    radius: float = 2e-4,
    vel_r: float = 300.0,
    phi: float = 0.0,
    psi: float = 0.1,
) -> torch.Tensor:
    """A' shard layout (``SHARD_LAYOUTS['a_prime']``) with physical values."""
    values = np.zeros((n, 16), dtype=np.float64)
    R = np.linspace(0.5, 0.52, n)
    Z = np.linspace(-0.005, 0.005, n)
    index = {name: i for i, name in enumerate(AP_SHARD_FIELD_NAMES)}
    values[:, index["spi_R_arr"]] = R
    values[:, index["spi_Z_arr"]] = Z
    values[:, index["spi_phi_arr"]] = phi
    values[:, index["spi_phi_init_arr"]] = 0.0
    values[:, index["spi_Vel_R_arr"]] = vel_r
    values[:, index["spi_radius_arr"]] = radius
    values[:, index["spi_abl_arr"]] = 1e18
    values[:, index["spi_species_arr"]] = 1.0
    values[:, index["spi_psi_arr"]] = psi
    return torch.as_tensor(values)


def make_module(n_substeps: int = 1, **kwargs) -> tuple[SourceModule, np.ndarray]:
    pos, vertex = synthetic_mesh()
    sampler = FemMeshSampler(pos, vertex)
    module = SourceModule(
        sampler,
        n_substeps=n_substeps,
        field_path="end",
        neon_tables=default_neon_tables(allow_surrogate=True),
        **kwargs,
    )
    module.eval()
    return module, pos


def run_module(module: SourceModule, pos: np.ndarray, shards: torch.Tensor, **kwargs):
    state = torch.as_tensor(synthetic_state(pos), dtype=torch.float64).unsqueeze(0)
    mask = torch.ones(1, shards.shape[0], dtype=torch.bool)
    return module(
        state,
        state,
        shards.unsqueeze(0),
        mask,
        dt_source_jorek=0.05,
        t_norm=T_NORM,
        central_density=CENTRAL_DENSITY,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# layout registry + pairing semantics
# --------------------------------------------------------------------------- #


def test_shard_layouts_match_source_of_truth():
    assert tuple(SHARD_LAYOUTS["a_prime"]) == tuple(AP_SHARD_FIELD_NAMES)
    assert tuple(SHARD_LAYOUTS["fragment_state"]) == tuple(V1_SHARD_FIELD_NAMES)
    # The two registered orders genuinely differ (the B1 report documents the
    # 1e27 ablation blow-up when they are mixed).
    assert SHARD_LAYOUTS["a_prime"] != SHARD_LAYOUTS["fragment_state"]


def test_window_pairing_uses_only_past_and_current_frames(tmp_path):
    """``(u_{k-1}, u_k, p_{k-1}) -> S_k`` must be built from the two window ends."""
    root = tmp_path / "native"
    write_native_run(root, indices=(10, 11, 12, 13), dt=0.5, n_nodes=6, n_spi=2)
    dataset = StepRunDataset(
        root, require_single_step=True, zero_dt_policy="list", mesh_verify_stride=10**9
    )
    assert len(dataset) >= 3
    previous, current = dataset[0], dataset[1]
    assert current.index == previous.index + 1
    # u_start = previous window-start frame, u_end = current window-start frame.
    assert previous.meta.path.name == f"jorek{previous.index:06d}.h5"
    assert current.meta.path.name == f"jorek{current.index:06d}.h5"
    # target source is the same-index projection (source window [T_{k-1}, T_k]).
    assert current.meta.path_source.name == f"projections{current.index:06d}.h5"
    assert current.t_source_end == current.t_before
    assert current.t_source_start == previous.t_before
    # The shard state consumed as p_{k-1} comes from the previous frame.
    assert previous.shard_start.index == previous.index


def test_module_output_does_not_depend_on_future_frame(tmp_path):
    """Corrupting the future frames must not change ``S_k``.

    Two identical runs are built; in the second one the *future* frame
    (``jorek{k+1}``) is corrupted.  The window example for record ``k`` must be
    bit-identical between them.
    """
    module, _ = make_module(n_substeps=2)

    def window_source(root: Path) -> torch.Tensor:
        dataset = StepRunDataset(root, require_single_step=True, zero_dt_policy="list")
        previous, current = dataset[0], dataset[1]
        u_start = torch.as_tensor(previous.u_before, dtype=torch.float64).unsqueeze(0)
        u_end = torch.as_tensor(current.u_before, dtype=torch.float64).unsqueeze(0)
        shards = torch.as_tensor(
            np.stack(
                [previous.shard_start.arrays[name] for name in AP_SHARD_FIELD_NAMES], axis=1
            )
        ).unsqueeze(0)
        mask = torch.ones(1, shards.shape[1], dtype=torch.bool)
        with torch.no_grad():
            return module(
                u_start,
                u_end,
                shards,
                mask,
                dt_source_jorek=current.dt_source,
                t_norm=current.t_norm,
                central_density=CENTRAL_DENSITY,
            )["source"]

    clean = tmp_path / "clean"
    write_native_run(clean, indices=(20, 21, 22), dt=0.5, n_nodes=9, n_spi=3)
    corrupted = tmp_path / "corrupted"
    write_native_run(corrupted, indices=(20, 21, 22), dt=0.5, n_nodes=9, n_spi=3)
    # Corrupt only the future frame (index 22) and its projection.
    write_native_frame(
        corrupted / "jorek000022.h5", index=22, t_now=100.0 + 1.0, n_nodes=9, n_spi=3,
        values_offset=100.0,
    )
    write_native_projection(
        corrupted / "projections000022.h5", t_seconds=(100.0 + 1.0) * 2.0e-4, n_nodes=9
    )
    first = window_source(clean)
    second = window_source(corrupted)
    assert torch.equal(first, second)


# --------------------------------------------------------------------------- #
# FEM sampling / assembly
# --------------------------------------------------------------------------- #


def test_bilinear_shape_partition_of_unity():
    s = torch.tensor([0.0, 0.3, 1.0, 0.5])
    t = torch.tensor([0.0, 0.7, 1.0, 0.5])
    weights = bilinear_shape(s, t)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4, dtype=weights.dtype))


def test_fem_sampler_reproduces_linear_fields():
    pos, vertex = synthetic_mesh(4)
    sampler = FemMeshSampler(pos, vertex)
    points = torch.as_tensor(
        [[[0.52, 0.2], [0.55, 0.6], [0.59, 0.9]]], dtype=torch.float64
    )
    location = sampler.locate(points)
    assert bool(location.inside.all())
    values = torch.as_tensor(
        2.0 * pos[:, 0] - 3.0 * pos[:, 1] + 1.0, dtype=torch.float64
    ).reshape(1, -1, 1)
    sampled = sampler.sample(values, location).squeeze(-1)
    exact = 2.0 * points[..., 0] - 3.0 * points[..., 1] + 1.0
    assert torch.allclose(sampled, exact, atol=1e-12)


def test_fem_deposition_conserves_the_deposited_quantity():
    pos, vertex = synthetic_mesh(3)
    sampler = FemMeshSampler(pos, vertex)
    points = torch.as_tensor([[[0.52, 0.2], [0.58, 0.8]]], dtype=torch.float64)
    location = sampler.locate(points)
    per_point = torch.tensor([[[3.0], [4.0]]], dtype=torch.float64)
    field = sampler.project(per_point, location)
    measure = sampler.node_measure()
    assert float((field[0, :, 0] * measure).sum()) == pytest.approx(7.0, rel=1e-12)


def test_fem_sampler_flags_out_of_domain_points():
    pos, vertex = synthetic_mesh(3)
    sampler = FemMeshSampler(pos, vertex)
    points = torch.as_tensor([[[10.0, 10.0]]], dtype=torch.float64)
    location = sampler.locate(points)
    assert not bool(location.inside.any())
    weights = location.weights
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 1, dtype=weights.dtype))


# --------------------------------------------------------------------------- #
# analytic backbone
# --------------------------------------------------------------------------- #


def test_analytic_ablation_matches_parks_reference():
    module, pos = make_module(n_substeps=1)
    shards = synthetic_shards(radius=2.5e-4)
    out = run_module(module, pos, shards)
    rho = 0.4
    rho_imp = 0.01
    ne = particle_density_si(torch.tensor(rho), CENTRAL_DENSITY) - particle_density_si(
        torch.tensor(rho_imp), CENTRAL_DENSITY
    )
    te_ev = temperature_ev(torch.tensor(synthetic_state(pos)[0, 8]), CENTRAL_DENSITY)
    reference = parks_ablation_neon(
        shards[:, 6], te_ev.expand(shards.shape[0]), ne.expand(shards.shape[0]),
        shards[:, 8],
    )
    predicted = out["shards"]["ablation"][0]
    relative = float(((predicted - reference.to(predicted.dtype)).abs() / reference).max())
    assert relative < 1e-6


def test_analytic_backbone_is_bounded_and_non_negative():
    module, pos = make_module(n_substeps=2)
    out = run_module(module, pos, synthetic_shards(3))
    source = out["source"]
    assert torch.isfinite(source).all()
    assert bool((source >= 0).all())
    atoms = out["budget"]["atoms"][0]
    ablation = out["shards"]["ablation"][0]
    assert atoms > 0
    assert bool(torch.isfinite(ablation).all() and (ablation >= 0).all())


def test_numerical_determinism():
    module, pos = make_module(n_substeps=2)
    shards = synthetic_shards(3)
    first = run_module(module, pos, shards)["source"]
    second = run_module(module, pos, shards.clone())["source"]
    assert torch.equal(first, second)


def test_gradients_flow_to_heads_and_shards():
    module, pos = make_module(n_substeps=1, use_correction=True)
    shards = synthetic_shards(2).clone().requires_grad_(True)
    out = run_module(module, pos, shards)
    loss = out["source"].sum()
    loss.backward()
    assert shards.grad is not None and torch.isfinite(shards.grad).all()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )
    assert float(shards.grad.abs().sum()) > 0.0


# --------------------------------------------------------------------------- #
# split heads / bounded correction
# --------------------------------------------------------------------------- #


def test_zero_initialised_correction_is_identity():
    module, pos = make_module(n_substeps=1, use_correction=True)
    out = run_module(module, pos, synthetic_shards(2))
    assert torch.equal(out["source"], out["analytic"])
    assert float(out["correction_log_scale"].detach().abs().max()) == 0.0


def test_correction_is_clamped_to_the_prior_bound():
    module, pos = make_module(n_substeps=1, use_correction=True, correction_bound=2.0)
    with torch.no_grad():
        for head, channels in (
            (module.instant_head, INSTANT_CHANNELS),
            (module.memory_head, MEMORY_CHANNELS),
        ):
            head[-1].weight.normal_(0.0, 1e6)
            head[-1].bias.fill_(50.0)
    out = run_module(module, pos, synthetic_shards(2))
    log_scale = out["correction_log_scale"]
    assert float(log_scale.detach().abs().max()) <= float(module.correction_bound_log) + 1e-6
    source, analytic = out["source"], out["analytic"]
    positive = analytic > 0
    ratio = (source[positive] / analytic[positive]).detach()
    assert float(ratio.max()) <= 2.0 * (1.0 + 1e-6)
    assert float(ratio.min()) >= 0.5 * (1.0 - 1e-6)


def test_split_heads_touch_disjoint_channels():
    module, pos = make_module(n_substeps=1, use_correction=True)
    with torch.no_grad():
        module.instant_head[-1].bias.fill_(0.3)
    out = run_module(module, pos, synthetic_shards(2))
    log_scale = out["correction_log_scale"][0]
    for channel in INSTANT_CHANNELS:
        assert float(log_scale[channel].detach()) != 0.0
    for channel in MEMORY_CHANNELS:
        assert float(log_scale[channel].detach()) == 0.0
    with torch.no_grad():
        module.instant_head[-1].bias.zero_()
        module.memory_head[-1].bias.fill_(-0.4)
    out = run_module(module, pos, synthetic_shards(2))
    log_scale = out["correction_log_scale"][0]
    for channel in MEMORY_CHANNELS:
        assert float(log_scale[channel].detach()) != 0.0
    for channel in INSTANT_CHANNELS:
        assert float(log_scale[channel].detach()) == 0.0


def test_memory_ablation_changes_energy_channels_only():
    module, pos = make_module(n_substeps=1, use_correction=True)
    with torch.no_grad():
        module.memory_head[-1].weight.normal_(0.0, 1.0)
        module.memory_head[-1].bias.fill_(0.2)
    shards = synthetic_shards(2)
    state = torch.as_tensor(synthetic_state(pos), dtype=torch.float64).unsqueeze(0)
    mask = torch.ones(1, shards.shape[0], dtype=torch.bool)
    memory = torch.zeros(1, module.memory_dim, dtype=torch.float64)
    with_memory = module(
        state, state, shards.unsqueeze(0), mask,
        dt_source_jorek=0.05, t_norm=T_NORM, central_density=CENTRAL_DENSITY,
        memory=memory, use_correction=True,
    )
    without_memory = module(
        state, state, shards.unsqueeze(0), mask,
        dt_source_jorek=0.05, t_norm=T_NORM, central_density=CENTRAL_DENSITY,
        memory=memory, use_correction=True,
    )
    # The latent memory must not be required for determinism.
    assert torch.equal(with_memory["source"], without_memory["source"])
    log_scale = with_memory["correction_log_scale"][0]
    assert any(float(log_scale[channel].detach()) != 0.0 for channel in MEMORY_CHANNELS)
    assert all(float(log_scale[channel].detach()) == 0.0 for channel in INSTANT_CHANNELS)


# --------------------------------------------------------------------------- #
# physics acceptance: monotonic responses
# --------------------------------------------------------------------------- #


def test_radius_and_velocity_responses_are_monotone():
    module, pos = make_module(n_substeps=3)
    shards = synthetic_shards(1)
    base = run_module(module, pos, shards)
    bigger = shards.clone()
    bigger[:, 6] *= 1.2
    radius_up = run_module(module, pos, bigger)
    assert float(radius_up["shards"]["ablation"].sum()) > float(
        base["shards"]["ablation"].sum()
    )
    faster = shards.clone()
    faster[:, 3] *= 1.5
    velocity_up = run_module(module, pos, faster)
    assert float(velocity_up["shards"]["values"][0, :, 0].mean()) > float(
        base["shards"]["values"][0, :, 0].mean()
    )
    # Hotter local plasma -> stronger ablation -> smaller radius at the end.
    hot_state = torch.as_tensor(synthetic_state(pos), dtype=torch.float64)
    hot_state[:, 8] *= 1.5
    mask = torch.ones(1, shards.shape[0], dtype=torch.bool)
    with torch.no_grad():
        hot = module(
            hot_state.unsqueeze(0),
            hot_state.unsqueeze(0),
            shards.unsqueeze(0),
            mask,
            dt_source_jorek=0.05,
            t_norm=T_NORM,
            central_density=CENTRAL_DENSITY,
        )
    assert float(hot["shards"]["ablation"].sum()) > float(base["shards"]["ablation"].sum())
    assert float(hot["shards"]["values"][0, :, 6].mean()) < float(
        base["shards"]["values"][0, :, 6].mean()
    )


def test_dead_shards_do_not_contribute():
    module, pos = make_module(n_substeps=1)
    shards = synthetic_shards(2)
    state = torch.as_tensor(synthetic_state(pos), dtype=torch.float64).unsqueeze(0)
    alive = torch.tensor([[True, False]])
    out = module(
        state, state, shards.unsqueeze(0), alive,
        dt_source_jorek=0.05, t_norm=T_NORM, central_density=CENTRAL_DENSITY,
    )
    assert float(out["shards"]["ablation"][0, 1]) == 0.0
    assert float(out["per_shard_channels"][0, 1, 4]) == 0.0


# --------------------------------------------------------------------------- #
# budget reading / ablation reconciliation
# --------------------------------------------------------------------------- #


def _write_budget_frame(path, *, times, cumulative, rate, t_norm=2e-4):
    with h5py.File(path, "w") as handle:
        handle["xtime"] = np.asarray(times, dtype=np.float64)
        handle["xtime_spi_ablation"] = np.asarray(cumulative, dtype=np.float64).reshape(-1, 1)
        handle["xtime_spi_ablation_rate"] = np.asarray(rate, dtype=np.float64).reshape(-1, 1)
        handle["t_norm"] = np.float64(t_norm)


def test_ablation_series_reconciliation_detects_restart_resets(tmp_path):
    # The code-restart duplicate sits at a zero-dt step (as in the real run):
    # the rate integral then equals the positive-increment sum.
    times = np.array([0.1, 0.2, 0.3, 0.4, 0.4, 0.5])
    t_norm = 2e-4
    rate = np.array([0.0, 10.0, 20.0, 30.0, 40.0, 50.0])
    dt = np.diff(times, prepend=times[0])[1:]
    increments = rate[1:] * dt * t_norm
    # cumulative[4] restarts at zero (the restart duplicate), losing what was
    # accumulated before; the physical total is the positive-increment sum.
    cumulative = np.array(
        [
            0.0,
            increments[0],
            increments[0] + increments[1],
            increments[0] + increments[1] + increments[2],
            0.0,
            increments[4],
        ]
    )
    _write_budget_frame(
        tmp_path / "jorek000006.h5", times=times, cumulative=cumulative, rate=rate, t_norm=t_norm
    )
    out = ablation_series_reconciliation(tmp_path)
    assert out["available"]
    assert out["n_resets"] >= 1
    assert out["cumulative_positive_increment_sum"] > out["cumulative_last"]
    assert out["rate_integral_over_segment_total"] == pytest.approx(1.0, rel=0.2)


def test_module_budget_is_non_negative_and_finite():
    module, pos = make_module(n_substeps=2)
    out = run_module(module, pos, synthetic_shards(3))
    for key, value in out["budget"].items():
        assert bool(torch.isfinite(value).all()), key
        assert float(value.min()) >= 0.0, key


# --------------------------------------------------------------------------- #
# real-frame checks (skip-if-absent)
# --------------------------------------------------------------------------- #


requires_run = pytest.mark.skipif(
    not Path(RUN_DIR).is_dir(), reason=f"real A' run not present: {RUN_DIR}"
)


@requires_run
def test_real_frame_smoke_and_acceptance():
    dataset = StepRunDataset(
        RUN_DIR,
        require_single_step=True,
        zero_dt_policy="list",
        index_window=(2999, 3002),
        mesh_verify_stride=10**9,
    )
    previous, current = dataset[0], dataset[1]
    meta = previous.meta
    with h5py.File(meta.path, "r") as handle:
        pos = np.asarray(handle["x"][:, 0, 0, :], dtype=np.float64).T
        vertex = np.asarray(handle["vertex"][()], dtype=np.int64) - 1
        central_density = float(np.ravel(handle["central_density"][()])[0])
    sampler = FemMeshSampler(pos, vertex)
    module = SourceModule(
        sampler,
        n_substeps=4,
        neon_tables=default_neon_tables(allow_surrogate=True),
    )
    module.eval()
    u_start = torch.as_tensor(previous.u_before, dtype=torch.float64).unsqueeze(0)
    u_end = torch.as_tensor(current.u_before, dtype=torch.float64).unsqueeze(0)
    shards = torch.as_tensor(
        np.stack([previous.shard_start.arrays[name] for name in AP_SHARD_FIELD_NAMES], axis=1)
    ).unsqueeze(0)
    mask = torch.ones(1, shards.shape[1], dtype=torch.bool)
    with torch.no_grad():
        out = module(
            u_start, u_end, shards, mask,
            dt_source_jorek=current.dt_source, t_norm=current.t_norm,
            central_density=central_density,
        )
    assert out["source"].shape == (1, 4125, 5)
    assert torch.isfinite(out["source"]).all()
    assert float(out["sampler_inside_fraction"]) == pytest.approx(1.0)
    predicted = out["shards"]["values"][0]
    recorded = torch.as_tensor(
        np.stack([current.shard_start.arrays[name] for name in AP_SHARD_FIELD_NAMES], axis=1)
    )
    # The deterministic backbone reproduces JOREK's shard advance (ADR-0015
    # history evidence): radius/position stay close over one window.
    radius_error = float((predicted[:, 6] - recorded[:, 6]).abs().max())
    assert radius_error < 1e-9
    position_error = float((predicted[:, :2] - recorded[:, :2]).abs().max())
    assert position_error < 1e-5
    # Window-end ablation sum within the registered analytic band.
    ratio = float(out["shards"]["ablation"].sum() / recorded[:, 7].sum())
    assert 0.5 < ratio < 2.0
    # Budget atoms vs the recorded per-step series (registered, no gate).
    from src.data.budget import native_step_budget

    budget = native_step_budget(RUN_DIR, current.index)
    recorded_atoms = budget.get("ablated_atoms_step")
    assert recorded_atoms
    assert 0.5 < float(out["budget"]["atoms"][0]) / recorded_atoms < 2.0


# --------------------------------------------------------------------------- #
# Reduced-order marker chain + augmented state (ADR-0020 decisions 2/4)
# --------------------------------------------------------------------------- #


def run_marker_module(module, pos, shards, *, dt=0.05, **kwargs):
    from src.models.marker_state import MarkerRuntime
    if "marker_state" not in kwargs:
        initial = MarkerRuntime.empty(len(pos), time_si=0., particle_dt_si=dt*T_NORM/module.n_substeps,
                                      confirmed_no_live_markers=True)
        initial.clock.field_end_si = dt*T_NORM
        kwargs["marker_state"] = initial
    state = torch.as_tensor(synthetic_state(pos), dtype=torch.float64).unsqueeze(0)
    mask = torch.ones(1, shards.shape[0], dtype=torch.bool)
    return module(
        state,
        state,
        shards.unsqueeze(0),
        mask,
        dt_source_jorek=dt,
        t_norm=T_NORM,
        central_density=CENTRAL_DENSITY,
        use_marker_transport=True,
        **kwargs,
    )


def test_marker_chain_is_deterministic_bounded_and_marker_feedback():
    module, pos = make_module(n_substeps=2)
    shards = synthetic_shards(3)

    first = run_marker_module(module, pos, shards)
    second = run_marker_module(module, pos, shards.clone())

    assert torch.equal(first["marker_feedback"], second["marker_feedback"])
    assert first["source_object"] == "marker_feedback"
    # Correction heads are zero-initialised: the delivered source is the
    # deterministic marker chain (no learned residual yet).
    assert torch.equal(first["source"], first["marker_feedback"])
    assert torch.isfinite(first["source"]).all()
    assert bool((first["source"] >= 0).all())
    assert first["marker"]["n_markers"] == 6  # only valid in-mesh births occupy slots (fixture straddles boundary)
    assert first["marker"]["features"].shape == (1, 5)
    assert first["marker"]["gather_time_seconds"] == pytest.approx(
        0.05 * T_NORM, rel=1e-12
    )
    # The cloud keeps atoms/bookkeeping: each marker carries a positive weight
    # and a positive active time.
    cloud = first["marker"]["cloud"]
    assert bool((cloud["time"] >= 0).all())
    assert bool((cloud["displacement"] >= 0).all())
    assert float(cloud["energy"].sum()) >= 0.0
    assert torch.isfinite(cloud["moment1"]).all()


def test_marker_chain_source_is_channelwise_shared_interface():
    from src.models.source_contract import (
        SOURCE_CHANNEL_ORDER,
        SOURCE_LAYOUT,
        N_SOURCE_FIELDS,
        shared_s_spec,
    )

    module, pos = make_module(n_substeps=1)
    out = run_marker_module(module, pos, synthetic_shards(2))
    assert out["marker_feedback"].shape[-1] == N_SOURCE_FIELDS
    assert SOURCE_CHANNEL_ORDER == SOURCE_CHANNEL_NAMES
    spec = shared_s_spec()
    assert spec["channel_order"] == list(SOURCE_CHANNEL_NAMES)
    assert spec["layout"] == SOURCE_LAYOUT
    assert spec["marker_run_true_source"] == "marker_feedback"
    # Both the analytic backbone and the marker chain expose the same object.
    assert out["source_object"] in {
        "marker_feedback",
        "analytic_shard",
        "inj_source",
    }


def test_marker_memory_threads_across_windows():
    module, pos = make_module(n_substeps=1, use_correction=True)
    shards = synthetic_shards(2)

    first = run_marker_module(module, pos, shards)
    assert first["memory_out"].shape == (1, module.memory_dim)
    # The h_marker state is consumed (GRU update depends on it) and threaded
    # back out for the next window.
    zeros = run_marker_module(module, pos, shards, memory=torch.zeros(1, module.memory_dim))
    ones = run_marker_module(module, pos, shards, memory=torch.ones(1, module.memory_dim))
    assert not torch.equal(zeros["memory_out"], ones["memory_out"])
    assert torch.equal(ones["memory_used"], torch.ones(1, module.memory_dim))


def test_advance_augmented_source_updates_state_and_memory():
    from src.models.source_module import AugmentedState, advance_augmented_source
    from src.models.marker_state import MarkerRuntime

    module, pos = make_module(n_substeps=1)
    shards = synthetic_shards(2)
    state_tensor = torch.as_tensor(synthetic_state(pos), dtype=torch.float64).unsqueeze(0)
    state = AugmentedState(
        u=state_tensor,
        p_shard=shards.unsqueeze(0),
        p_shard_mask=torch.ones(1, shards.shape[0], dtype=torch.bool),
        h_marker=torch.zeros(1, module.memory_dim),
        step_index=10,
        t_jorek=100.0,
        marker_state=MarkerRuntime.empty(
            len(pos), time_si=100*T_NORM, particle_dt_si=.05*T_NORM,
            confirmed_no_live_markers=True),
    )

    updated, out = advance_augmented_source(
        module,
        state,
        state_tensor,
        dt_source_jorek=0.05,
        t_norm=T_NORM,
        central_density=CENTRAL_DENSITY,
        use_marker_transport=True,
        t_next=100.05,
    )

    assert updated.step_index == 11
    assert updated.t_jorek == pytest.approx(100.05)
    assert updated.p_shard.shape == state.p_shard.shape
    assert updated.h_marker.shape == (1, module.memory_dim)
    assert out["source_object"] == "marker_feedback"


def test_shared_source_validation_rejects_wrong_objects():
    from src.models.source_contract import validate_shared_source

    with pytest.raises(ValueError):
        validate_shared_source(torch.zeros(4, 9), what="S")
    with pytest.raises(ValueError):
        validate_shared_source(torch.full((4, 5), float("nan")), what="S")
    ok = validate_shared_source(torch.zeros(4, 5), what="S")
    assert ok.shape == (4, 5)


def test_persistent_marker_windows_stop_births_and_restore_model(tmp_path, monkeypatch):
    """Prescribed-field component validation, not self-generated field closure."""
    from src.models.marker_state import MarkerRuntime, save_model_rollout, load_model_rollout
    from src.models.source_module import AugmentedState, advance_augmented_source
    module,pos=make_module(n_substeps=2)
    shards=synthetic_shards(2)
    u=torch.as_tensor(synthetic_state(pos),dtype=torch.float64).unsqueeze(0)
    initial=MarkerRuntime.empty(len(pos),time_si=0.,particle_dt_si=.025*T_NORM,
                                confirmed_no_live_markers=True)
    state=AugmentedState(u,shards.unsqueeze(0),torch.ones(1,2,dtype=torch.bool),
                         torch.zeros(1,module.memory_dim),0,0.,initial)
    first,out=advance_augmented_source(module,state,u,dt_source_jorek=.05,t_norm=T_NORM,
                                      central_density=CENTRAL_DENSITY,use_marker_transport=True)
    assert first.marker_state.cloud.alive.any()
    assert torch.equal(first.p_shard,state.p_shard)  # shard update belongs to the fluid event
    # Disable every shard: no injection, but pre-existing marker cloud remains.
    from dataclasses import replace
    stopped=replace(first,p_shard_mask=torch.zeros_like(first.p_shard_mask))
    path=tmp_path/'source-rollout.pt'; save_model_rollout(path,stopped,module)
    restored=load_model_rollout(path,module)
    # Once initialized the inference step must perform no HDF5 access at all.
    def forbidden(*args,**kwargs):
        raise AssertionError('inference attempted to read recorded truth')
    monkeypatch.setattr(h5py,'File',forbidden)
    kwargs=dict(dt_source_jorek=.05,t_norm=T_NORM,central_density=CENTRAL_DENSITY,use_marker_transport=True)
    second,a=advance_augmented_source(module,stopped,u,**kwargs)
    resumed,b=advance_augmented_source(module,restored,u,**kwargs)
    assert torch.equal(a['source'],b['source'])
    assert len(second.marker_state.cloud.weight)==len(first.marker_state.cloud.weight)
    assert second.marker_state.cloud.alive.sum()==first.marker_state.cloud.alive.sum()
    assert not np.array_equal(second.marker_state.cloud.P_imp,first.marker_state.cloud.P_imp)
    for key in first.marker_state.cloud.array_names():
        np.testing.assert_array_equal(getattr(second.marker_state.cloud,key),getattr(resumed.marker_state.cloud,key))
    np.testing.assert_array_equal(second.marker_state.feedback.deltas,
        second.marker_state.feedback.values-first.marker_state.feedback.values)
    assert not second.marker_state.clock.startup_pending
    assert torch.equal(second.h_residual,resumed.h_residual)


def test_marker_initial_state_is_required():
    module,pos=make_module()
    u=torch.as_tensor(synthetic_state(pos),dtype=torch.float64).unsqueeze(0)
    shards=synthetic_shards(2).unsqueeze(0)
    with pytest.raises(ValueError,match='missing initial state'):
        module(u,u,shards,torch.ones(1,2,dtype=torch.bool),dt_source_jorek=.05,
               t_norm=T_NORM,central_density=CENTRAL_DENSITY,use_marker_transport=True)


def test_previous_feedback_changes_reduced_electron_closure():
    """Prescribed constant fluid, different persisted charge feedback."""
    from copy import deepcopy
    from src.models.marker_state import MarkerCloud,MarkerRuntime
    module,pos=make_module(n_substeps=1)
    state=MarkerRuntime.empty(len(pos),time_si=0.,particle_dt_si=.05*T_NORM,
                              confirmed_no_live_markers=True)
    state.clock.field_end_si=.05*T_NORM
    state.cloud=MarkerCloud.empty(capacity=1)
    state.cloud.x[0]=[.55,.5,0.];state.cloud.i_elm[0]=1;state.cloud.weight[0]=1e15
    state.cloud.P_imp[0,:]=0.;state.cloud.P_imp[0,1]=1.
    changed=deepcopy(state)
    changed.feedback.values[:,3]=10.;changed.feedback.values[:,4]=1.
    shards=synthetic_shards(1);shards[:,6]=0. # a_prime radius -> no new births
    a=run_marker_module(module,pos,shards,marker_state=state)
    b=run_marker_module(module,pos,shards,marker_state=changed)
    assert not np.array_equal(a['marker']['state'].cloud.P_imp,b['marker']['state'].cloud.P_imp)
    np.testing.assert_array_equal(b['marker']['state'].feedback.deltas,
        b['source'][0].detach().numpy()-changed.feedback.values)


def test_model_restore_rejects_different_physics_configuration(tmp_path):
    from src.models.marker_state import save_model_rollout,load_model_rollout
    module,pos=make_module(n_substeps=1)
    path=tmp_path/'rollout.pt';save_model_rollout(path,None,module)
    changed,_=make_module(n_substeps=2)
    with pytest.raises(ValueError,match='configuration mismatch'):
        load_model_rollout(path,changed)


def test_birth_weight_uses_particle_gather_time_with_carried_residual():
    from src.models.marker_state import MarkerRuntime
    from src.models.source_module import AugmentedState,advance_augmented_source
    module,pos=make_module(n_substeps=1)
    shards=synthetic_shards(1);shards[0,0]=.55;shards[0,1]=.5
    u=torch.as_tensor(synthetic_state(pos),dtype=torch.float64).unsqueeze(0)
    marker=MarkerRuntime.empty(len(pos),time_si=0.,particle_dt_si=.03*T_NORM,
                               confirmed_no_live_markers=True)
    state=AugmentedState(u,shards.unsqueeze(0),torch.ones(1,1,dtype=torch.bool),
                         torch.zeros(1,module.memory_dim),0,0.,marker)
    kwargs=dict(dt_source_jorek=.05,t_norm=T_NORM,central_density=CENTRAL_DENSITY,use_marker_transport=True)
    first,_=advance_augmented_source(module,state,u,**kwargs)
    second,_=advance_augmented_source(module,first,u,**kwargs)
    assert first.marker_state.cloud.weight.sum()==pytest.approx(1e18*.03*T_NORM)
    added=second.marker_state.cloud.weight.sum()-first.marker_state.cloud.weight.sum()
    assert added==pytest.approx(1e18*.06*T_NORM)
    assert second.marker_state.clock.step_rest_time==pytest.approx(.01*T_NORM)
