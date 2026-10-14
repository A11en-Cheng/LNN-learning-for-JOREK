"""Phase B1 single-step training/validation for the self-generated source module.

Contract (ADR-0019 / roadmap_v2 Phase B1): for a record ``k`` the module maps
``(u_{k-1}, u_k, p_{k-1})`` (window ends + window-start shard state) to ``S_k``
(five channels) and evolves the shards to ``p_k``.  Inference never touches a
future frame.  Per ADR-0020 decision 1 the recorded ``projections*.h5`` **are**
the true field source of a marker run (``use_marker=.t.``) for the
energy/radiation/charge channels; the arm name ``analytic_proxy_calibrated`` is
kept only as a historical label for the analytic-to-projection scale fit
(reports written before 2026-10-07 called the projection a "proxy"; see the B1
report erratum).

Pipeline (``--marker-chain`` switches the deterministic main term from the
analytic shard backbone to the reduced-order marker chain, ADR-0020 decision 2):

1. ``cache``  -- one example per consecutive native record pair; the
   deterministic backbone (fragment evolution + analytic channels + FEM
   assembly) runs once and the features/targets are cached;
2. train -- blocked train/val split, per-channel relative-L2 + budget loss on
   the split heads (default: analytic-only delivery; the bounded multiplicative
   correction is the ablation arm, 2-3 seeds);
3. evaluate -- per-channel relative L2, budget closure, shard-level acceptance
   (window-end ablation ratio, radius/position MAE), proxy truncation metrics,
   monotonicity checks.

Example::

    .torch-cuda-env/bin/python tools/train_source_module_b1.py \
        --run-dir data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner \
        --out .scratch/phase-b1-source --stride 8 --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.budget import ablation_series_reconciliation, native_step_budget
from src.data.nonlocal_table import nonlocal_table_from_frame
from src.data.step_dataset import SHARD_FIELD_NAMES, StepRunDataset
from src.models.fem_sampling import FemMeshSampler
from src.models.source_module import (
    E_NORM,
    MARKER_FEATURE_DIM,
    SOURCE_CHANNEL_NAMES,
    SourceModule,
    default_neon_tables,
)

DEFAULT_RUN = "data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner"
EPOCHS = 60


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def shard_array(sample, *, side: str = "start") -> np.ndarray:
    state = sample.shard_start if side == "start" else sample.shard_end
    return np.stack([state.arrays[name] for name in SHARD_FIELD_NAMES], axis=1)


def load_mesh_and_constants(run_dir: Path, frame_path: Path):
    with h5py.File(frame_path, "r") as handle:
        pos = np.asarray(handle["x"][:, 0, 0, :], dtype=np.float64).T
        vertex = np.asarray(handle["vertex"][()], dtype=np.int64) - 1
        central_density = float(np.ravel(handle["central_density"][()])[0])
        central_mass = float(np.ravel(handle["central_mass"][()])[0])
    return pos, vertex, central_density, central_mass


def frame_nonlocal_context(frame_path: Path, projection_path: Path | None):
    """Per-frame non-local ablation table + axis/boundary psi (ADR-0015 style).

    ``int3d_new`` writes a snapshot file; the per-frame rebuild
    (:mod:`src.data.nonlocal_table`) is the registered faithful stand-in and is
    what the historical analytic-baseline evidence used.
    """
    table = nonlocal_table_from_frame(frame_path, projection_path)
    with h5py.File(frame_path, "r") as handle:
        psi_axis = float(np.ravel(handle["psi_axis_t"][()])[-1])
        psi_bnd = float(np.ravel(handle["psi_bnd_t"][()])[-1])
    return table, psi_axis, psi_bnd


def example_indices(dataset: StepRunDataset, *, exclude_radius: int) -> list[tuple[int, int]]:
    """Consecutive record positions ``(p-1, p)`` usable as one window pair."""
    metas = dataset.metas
    zero_dt = set(dataset.report.zero_dt_indices)
    pairs: list[tuple[int, int]] = []
    for position in range(1, len(metas)):
        previous, current = metas[position - 1], metas[position]
        if current.index - previous.index != 1:
            continue
        if current.dt_source <= 0.0 or current.dt_step <= 0.0:
            continue
        # Window chaining (ADR-0019): the previous record's field interval start
        # is exactly the current source window start, and the current source
        # window ends at the current field interval start.  The native loader
        # labels jorek{k} shards with their true time T_k, so p_start =
        # spi(jorek{k-1}) (previous record) and p_end = spi(jorek{k}) (current).
        if abs(previous.t_interval_start - current.t_source_start) > 1e-6:
            raise RuntimeError(
                "source window is not chained to the previous record: "
                f"previous interval start {previous.t_interval_start} != "
                f"current source start {current.t_source_start}"
            )
        if abs(current.t_source_end - current.t_interval_start) > 1e-6:
            raise RuntimeError(
                "source window end does not match the current field interval start: "
                f"{current.t_source_end} != {current.t_interval_start}"
            )
        if any(abs(current.index - index) <= exclude_radius for index in zero_dt):
            continue
        if any(abs(previous.index - index) <= exclude_radius for index in zero_dt):
            continue
        pairs.append((position - 1, position))
    return pairs


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #


def build_cache(args) -> dict:
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    index_file = cache_dir / "index.json"
    if index_file.exists() and not args.rebuild_cache:
        cached = json.loads(index_file.read_text())
        if bool(cached.get("marker_chain", False)) == bool(args.marker_chain):
            return cached

    dataset = StepRunDataset(
        run_dir,
        require_single_step=True,
        zero_dt_policy=args.zero_dt_policy,
        index_window=(args.index_start - 1, args.index_end + 1),
        mesh_verify_stride=100000,
    )
    pairs = example_indices(dataset, exclude_radius=args.exclude_radius)
    if args.stride > 1:
        pairs = pairs[:: args.stride]
    if args.max_examples is not None:
        pairs = pairs[: args.max_examples]
    first_frame = dataset.metas[0].path
    pos, vertex, central_density, central_mass = load_mesh_and_constants(run_dir, first_frame)
    sampler = FemMeshSampler(pos, vertex)
    tables = default_neon_tables(allow_surrogate=True)
    module = SourceModule(
        sampler,
        n_substeps=args.n_substeps,
        use_memory=True,
        use_correction=False,
        use_marker_transport=args.marker_chain,
        neon_tables=tables,
        shard_layout="a_prime",
    )
    module.eval()

    records: list[dict] = []
    started = time.time()
    for number, (p_prev, p_cur) in enumerate(pairs):
        previous = dataset[p_prev]
        current = dataset[p_cur]
        u_start = torch.as_tensor(previous.u_before, dtype=torch.float64).unsqueeze(0)
        u_end = torch.as_tensor(current.u_before, dtype=torch.float64).unsqueeze(0)
        p_start = torch.as_tensor(shard_array(previous), dtype=torch.float64).unsqueeze(0)
        mask = torch.ones(1, p_start.shape[1], dtype=torch.bool)
        nonlocal_kwargs: dict = {}
        if args.nonlocal_mode == "frame":
            table, psi_axis, psi_bnd = frame_nonlocal_context(
                current.meta.path, current.meta.path_source
            )
            nonlocal_kwargs = {
                "nonlocal_tables": [table],
                "psi_axis": torch.tensor([psi_axis], dtype=torch.float64),
                "psi_bnd": torch.tensor([psi_bnd], dtype=torch.float64),
            }
        with torch.no_grad():
            result = module(
                u_start,
                u_end,
                p_start,
                mask,
                dt_source_jorek=current.dt_source,
                t_norm=current.t_norm,
                central_density=central_density,
                central_mass=central_mass,
                use_marker_transport=args.marker_chain,
                **nonlocal_kwargs,
            )
        budget = native_step_budget(run_dir, current.index)
        record = {
            "index": int(current.index),
            "dt_source": float(current.dt_source),
            "dt_step": float(current.dt_step),
            "t_norm": float(current.t_norm),
            "shards": shard_array(previous).astype(np.float32),
            "alive": mask[0].numpy().astype(bool),
            "extra": torch.cat(
                [
                    result["sampler_features"]["local_mean"],
                    result["sampler_features"]["inside_fraction"],
                ],
                dim=-1,
            )[0].numpy().astype(np.float32),
            "analytic": result["analytic"][0].numpy().astype(np.float32),
            "target": np.asarray(current.source, dtype=np.float32),
            "marker_source": (
                None
                if result["marker_feedback"] is None
                else result["marker_feedback"][0].numpy().astype(np.float32)
            ),
            "marker_features": (
                None
                if result["marker"] is None
                else result["marker"]["features"][0].numpy().astype(np.float32)
            ),
            "source_object": result["source_object"],
            "atoms": float(result["budget"]["atoms"][0]),
            "radiation_energy": float(result["budget"]["radiation_energy"][0]),
            "ionization_energy": float(result["budget"]["ionization_energy"][0]),
            "predicted_abl_sum": float(result["shards"]["ablation"][0].sum()),
            "recorded_abl_sum": float(shard_array(current)[:, 7].sum()),
            "radius_start_sum": float(shard_array(previous)[:, 6].sum()),
            "radius_end_pred_sum": float(result["shards"]["values"][0, :, 6].sum()),
            "radius_end_rec_sum": float(shard_array(current)[:, 6].sum()),
            "recorded_atoms_step": budget.get("ablated_atoms_step"),
            "recorded_ionization_step": budget.get("ionization_energy_step"),
            "recorded_radiation_step": budget.get("radiated_energy_step"),
            "recorded_electrons_step": budget.get("electrons_injected_step"),
        }
        torch.save(record, cache_dir / f"example_{number:05d}.pt")
        records.append({"file": f"example_{number:05d}.pt", "index": record["index"]})
        if number % 50 == 0:
            print(
                f"[cache] {number}/{len(pairs)} index={current.index} "
                f"elapsed={time.time() - started:.1f}s",
                flush=True,
            )
    meta = {
        "run_dir": str(run_dir),
        "index_start": args.index_start,
        "index_end": args.index_end,
        "stride": args.stride,
        "n_substeps": args.n_substeps,
        "exclude_radius": args.exclude_radius,
        "central_density": central_density,
        "central_mass": central_mass,
        "n_examples": len(records),
        "zero_dt_indices": list(dataset.report.zero_dt_indices),
        "gaps": list(dataset.report.gaps),
        "records": records,
        "source_object": (
            "marker_feedback (recorded projections = true field source for the "
            "energy/radiation/charge channels in this marker run; ADR-0020)"
        ),
        "shard_layout": "a_prime",
        "marker_chain": bool(args.marker_chain),
        "nonlocal": args.nonlocal_mode,
        "window_inputs": (
            "u_start=jorek{k-1}, u_end=jorek{k}, p_start=spi(jorek{k-1}) "
            "(window start T_{k-1}), target=projections{k} (window [T_{k-1},T_k])"
        ),
        "budget_reconciliation": ablation_series_reconciliation(run_dir),
        "table_source": tables.source,
    }
    index_file.write_text(json.dumps(meta, indent=1))
    return meta


def load_cache(cache_dir: Path, records: list[dict]) -> list[dict]:
    return [torch.load(cache_dir / entry["file"], weights_only=False) for entry in records]


# --------------------------------------------------------------------------- #
# split / training
# --------------------------------------------------------------------------- #


def blocked_split(records: list[dict], *, val_blocks: int) -> tuple[list[int], list[int]]:
    """Contiguous blocked split: the last ``val_blocks`` of ``n_blocks`` are val."""
    n = len(records)
    n_blocks = max(4, val_blocks + 1)
    edges = [round(i * n / n_blocks) for i in range(n_blocks + 1)]
    train = [i for b in range(n_blocks - val_blocks) for i in range(edges[b], edges[b + 1])]
    val = [i for b in range(n_blocks - val_blocks, n_blocks) for i in range(edges[b], edges[b + 1])]
    return train, val


def stack(records: list[dict], indices: list[int], key: str) -> torch.Tensor:
    values = [records[i][key] for i in indices]
    if isinstance(values[0], np.ndarray):
        return torch.as_tensor(np.stack(values))
    return torch.as_tensor(values)


def feature_batch(
    module: SourceModule,
    records: list[dict],
    indices: list[int],
    base_scaled: torch.Tensor | None = None,
    *,
    marker_features: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pooled per-sample features (must match ``SourceModule.forward``).

    The feature list is ``[shard embedding, local field means, inside
    fraction, log1p(base summary / source_norm), marker-cloud summary]``.  The
    marker block is five zeros when the reduced-order marker chain is disabled.
    """
    shards = stack(records, indices, "shards").float()
    alive = stack(records, indices, "alive").bool()
    mean = module.shard_mean.to(torch.float32)
    std = module.shard_std.to(torch.float32)
    shard_norm = (torch.nan_to_num(shards, nan=0.0) - mean) / std
    shard_norm = torch.where(alive.unsqueeze(-1), shard_norm, torch.zeros_like(shard_norm))
    embedding = module.encoder(shard_norm, alive)
    extra = stack(records, indices, "extra").float()
    if base_scaled is None:
        analytic = stack(records, indices, "analytic").float()
        base_scaled = analytic * module.channel_scale.to(torch.float32).view(1, 1, 5)
    source_feature = torch.log1p(
        base_scaled.abs().mean(dim=1) / module.source_norm.to(torch.float32).view(1, 5)
    )
    if marker_features is None:
        marker_features = torch.zeros(
            len(indices), MARKER_FEATURE_DIM, dtype=embedding.dtype
        )
    return torch.cat(
        [embedding, extra, source_feature, marker_features.float()], dim=-1
    )


def record_base(records: list[dict], indices: list[int], *, marker_chain: bool) -> torch.Tensor:
    """Base source per record: marker feedback (ADR-0020) or analytic backbone."""
    key = "marker_source" if marker_chain else "analytic"
    return stack(records, indices, key).float()


def record_marker_features(
    records: list[dict], indices: list[int], *, marker_chain: bool
) -> torch.Tensor | None:
    if not marker_chain:
        return None
    return stack(records, indices, "marker_features").float()


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    numerator = (prediction - target).norm(dim=1)
    denominator = target.norm(dim=1).clamp_min(1e-30)
    return (numerator / denominator).mean()


def outlier_clamped_relative_l2(
    prediction: torch.Tensor, target: torch.Tensor, *, maximum: float = 10.0
) -> torch.Tensor:
    """Per-example relative L2 with a clamp on the worst outliers.

    A handful of windows sit right at injection where the recorded proxy is
    ~0 while the shard-deposition source is not (or vice versa); the raw mean is
    then dominated by those windows.  The clamp keeps the training signal
    finite and stable; the reported metric stays the unclamped per-example
    relative L2 (median over the split).
    """
    numerator = (prediction - target).norm(dim=1)
    denominator = target.norm(dim=1).clamp_min(1e-30)
    return (numerator / denominator).clamp_max(maximum).mean()


def robust_relative_error(
    prediction: torch.Tensor, target: torch.Tensor, *, floor_fraction: float = 0.1
) -> torch.Tensor:
    """Relative error with a batch-median floor on the denominator.

    The recorded per-step budget series can be (near) zero for individual
    windows; the raw ratio then dominates the mean.  The floor keeps the
    training signal bounded while the *reported* metric stays the raw median
    relative error (robust by construction).
    """
    scale = target.abs().median().clamp_min(1e-30)
    denominator = target.abs().clamp_min(floor_fraction * scale)
    return ((prediction - target).abs() / denominator).clamp_max(10.0).mean()


def budget_terms(
    prediction: torch.Tensor, target: torch.Tensor, measure: torch.Tensor, channels: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for channel in channels:
        pred = (prediction[..., channel] * measure).sum(dim=1)
        true = (target[..., channel] * measure).sum(dim=1)
        out[SOURCE_CHANNEL_NAMES[channel]] = ((pred - true).abs() / true.abs().clamp_min(1e-30)).mean()
    return out


def budget_supervision(
    records: list[dict], train_indices: list[int]
) -> dict:
    """Registered unit factors between the module's physical totals and the
    recorded per-step budget series (``xtime_rad_power*tstep`` etc.).

    The atom budget needs no factor (both sides are atoms); the energy series
    are in JOREK energy units and their SI conversion is only partly
    documented, so the factor is fitted on the train split and registered as a
    proxy-conversion constant (never used as a physics claim)."""
    atoms = [
        (records[i]["atoms"], records[i]["recorded_atoms_step"])
        for i in train_indices
        if records[i].get("recorded_atoms_step")
    ]
    radiation = [
        (records[i]["radiation_energy"], records[i]["recorded_radiation_step"])
        for i in train_indices
        if records[i].get("recorded_radiation_step")
    ]
    ionization = [
        (records[i]["ionization_energy"], records[i]["recorded_ionization_step"])
        for i in train_indices
        if records[i].get("recorded_ionization_step")
    ]

    def median_ratio(pairs):
        ratios = [a / b for a, b in pairs if b not in (None, 0.0) and a is not None]
        return float(np.median(ratios)) if ratios else None

    return {
        "atoms_ratio_median": median_ratio(atoms),
        "radiation_unit_factor_median": median_ratio(radiation),
        "ionization_unit_factor_median": median_ratio(ionization),
        "note": (
            "atom budget is unit-free; energy/radiation factors are registered "
            "proxy conversions fitted on the train split only"
        ),
    }


def field_budget(
    source: torch.Tensor,
    measure: torch.Tensor,
    *,
    n_norm: float,
    energy_factor: float,
) -> dict[str, torch.Tensor]:
    """Physical totals of the *assembled field* (converted back to SI units).

    ``source[..., 4]`` is the ``N_imp`` channel in JOREK density units
    (``/ n_norm``), ``source[..., 0..1]`` are energy densities in
    ``E_norm / t_norm`` units; multiplying by the nodal measure and the
    conversion factors returns atoms / joules, so the budget terms are unit
    consistent with the recorded ``xtime*`` series.
    """
    return {
        "atoms": (source[..., 4] * measure).sum(dim=1) * n_norm,
        "ionization": (source[..., 0] * measure).sum(dim=1) * energy_factor,
        "radiation": (source[..., 1] * measure).sum(dim=1) * energy_factor,
    }


def shard_stats(records: list[dict], indices: list[int]) -> dict[str, list[float]]:
    """Mean/std of the alive shard fields over the train split (feature norm)."""
    values = []
    for index in indices:
        shards = np.asarray(records[index]["shards"], dtype=np.float64)
        alive = np.asarray(records[index]["alive"], dtype=bool)
        values.append(shards[alive])
    stacked = np.concatenate(values, axis=0)
    stacked = np.where(np.isfinite(stacked), stacked, 0.0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std = np.where(std > 0.0, std, 1.0)
    return {"mean": mean.tolist(), "std": std.tolist()}


def backbone_fidelity_probe(
    dataset: StepRunDataset,
    sampler: FemMeshSampler,
    *,
    central_density: float,
    central_mass: float,
    stride: int,
    limit: int,
    nonlocal_mode: str = "frame",
) -> dict:
    """Separate the *model* from the *evolution* (registered, no gate).

    Three quantities are compared with the recorded window-end ablation rate
    ``sum(spi_abl_arr(jorek{k}))``:

    * ``same_frame``: the Parks/NGS model evaluated with the window-end frame
      ``u_k`` at the recorded window-end shard state (the ADR-0015 (history)
      style check: does the analytic law reproduce the recorded rate?);
    * ``module_<path>``: the module's own last-sub-step ablation after evolving
      ``p_{k-1}`` across the window with ``field_path in {start, linear, end}``.
    """
    from src.physics.spi_model import parks_ablation_neon, particle_density_si, temperature_ev

    modules = {}
    for path in ("start", "linear", "end"):
        module = SourceModule(
            sampler,
            n_substeps=4,
            field_path=path,
            neon_tables=default_neon_tables(allow_surrogate=True),
        )
        module.eval()
        modules[path] = module
    positions = list(range(1, len(dataset), max(1, stride)))[:limit]
    rows: list[dict] = []
    for position in positions:
        previous, current = dataset[position - 1], dataset[position]
        u_start = torch.as_tensor(previous.u_before, dtype=torch.float64).unsqueeze(0)
        u_end = torch.as_tensor(current.u_before, dtype=torch.float64).unsqueeze(0)
        p_start = torch.as_tensor(shard_array(previous), dtype=torch.float64).unsqueeze(0)
        p_end_recorded = shard_array(current)
        mask = torch.ones(1, p_start.shape[1], dtype=torch.bool)
        recorded = float(p_end_recorded[:, 7].sum())
        if recorded <= 0.0:
            continue
        nonlocal_kwargs: dict = {}
        if nonlocal_mode == "frame":
            table, psi_axis, psi_bnd = frame_nonlocal_context(
                current.meta.path, current.meta.path_source
            )
            nonlocal_kwargs = {
                "nonlocal_tables": [table],
                "psi_axis": torch.tensor([psi_axis], dtype=torch.float64),
                "psi_bnd": torch.tensor([psi_bnd], dtype=torch.float64),
            }
        # same-frame model check at the recorded window-end state.
        with torch.no_grad():
            location = sampler.locate(
                torch.as_tensor(p_end_recorded[:, :2], dtype=torch.float64).reshape(1, -1, 2)
            )
            sampled = sampler.sample(u_end, location)
            ne = (
                particle_density_si(sampled[..., 4], central_density)
                - particle_density_si(sampled[..., 7], central_density)
            ).clamp_min(0.0)
            te_ev = temperature_ev(sampled[..., 8], central_density)
            radius = torch.as_tensor(p_end_recorded[:, 6], dtype=torch.float64)
            species = torch.as_tensor(p_end_recorded[:, 8], dtype=torch.float64)
            model_tensor = parks_ablation_neon(radius, te_ev, ne, species)
            if nonlocal_mode == "frame":
                psi_index = modules["linear"].shard_index["psi"]
                te_nl, ne_nl, inside = table.surface_state(
                    torch.as_tensor(p_end_recorded[:, psi_index], dtype=torch.float64),
                    psi_axis,
                    psi_bnd,
                    central_density,
                )
                ablation_nl = parks_ablation_neon(radius, te_nl, ne_nl, species)
                model_tensor = torch.where(
                    inside, torch.maximum(model_tensor, ablation_nl), model_tensor
                )
            model = float(model_tensor.sum())
            row = {"index": int(current.index), "same_frame_ratio": model / recorded}
            for path, module in modules.items():
                result = module(
                    u_start,
                    u_end,
                    p_start,
                    mask,
                    dt_source_jorek=current.dt_source,
                    t_norm=current.t_norm,
                    central_density=central_density,
                    central_mass=central_mass,
                    **nonlocal_kwargs,
                )
                row[f"module_{path}_ratio"] = (
                    float(result["shards"]["ablation"].sum()) / recorded
                )
        rows.append(row)
    if not rows:
        return {"available": False}
    # Phase split by index thirds (early / mid / late).
    indices = np.asarray([row["index"] for row in rows])
    edges = np.quantile(indices, [0.0, 1 / 3, 2 / 3, 1.0])
    out: dict = {"n": len(rows), "phase_edges": edges.tolist(), "phases": {}}
    for name, (low, high) in zip(
        ("early", "mid", "late"), zip(edges[:-1], edges[1:])
    ):
        selected = [
            row
            for row in rows
            if low <= row["index"] <= high
        ]
        if not selected:
            continue
        out["phases"][name] = {
            "n": len(selected),
            **{
                key: float(np.median([row[key] for row in selected]))
                for key in (
                    "same_frame_ratio",
                    "module_start_ratio",
                    "module_linear_ratio",
                    "module_end_ratio",
                )
            },
        }
    return out


def fit_channel_scale(
    records: list[dict], train_indices: list[int], measure: torch.Tensor
) -> dict:
    """Registered proxy calibration (train split only).

    Two estimators per channel are recorded; the *least-squares projection
    coefficient* ``c = <A, T>_m / <A, A>_m`` is used for the delivered
    calibration because it minimises the L2 error, while the median integral
    ratio ``<T,_m>/<A,_m>`` is registered as the (much larger) scale the proxy
    would need if the support matched.
    """
    scales = []
    integral_ratios = []
    for channel in range(5):
        numerator = 0.0
        denominator = 0.0
        ratios = []
        for index in train_indices:
            analytic = torch.as_tensor(records[index]["analytic"], dtype=torch.float64)[:, channel]
            target = torch.as_tensor(records[index]["target"], dtype=torch.float64)[:, channel]
            numerator += float((analytic * target * measure).sum())
            denominator += float((analytic * analytic * measure).sum())
            analytic_mass = float((analytic * measure).sum())
            target_mass = float((target * measure).sum())
            if analytic_mass > 0.0 and target_mass > 0.0:
                ratios.append(target_mass / analytic_mass)
        scales.append(float(numerator / denominator) if denominator > 0.0 else 0.0)
        integral_ratios.append(float(np.median(ratios)) if ratios else None)
    return {
        "scale": scales,
        "estimator": "least_squares <A,T>_m / <A,A>_m (train split only)",
        "median_integral_ratio": integral_ratios,
        "n_train_examples": len(train_indices),
    }


def evaluate(
    module: SourceModule,
    records: list[dict],
    indices: list[int],
    measure: torch.Tensor,
    *,
    use_correction: bool,
    use_memory: bool = True,
    unit_factors: dict | None = None,
    n_norm: float = 1.0,
    energy_factor: float = 1.0,
    marker_chain: bool = False,
) -> dict:
    module.eval()
    with torch.no_grad():
        analytic = stack(records, indices, "analytic").float()
        target = stack(records, indices, "target").float()
        base = record_base(records, indices, marker_chain=marker_chain)
        base_scaled = (
            base
            if marker_chain
            else base * module.channel_scale.to(torch.float32).view(1, 1, 5)
        )
        features = feature_batch(
            module,
            records,
            indices,
            base_scaled,
            marker_features=record_marker_features(
                records, indices, marker_chain=marker_chain
            ),
        )
        heads = module.apply_heads(
            base,
            features,
            use_correction=use_correction,
            use_memory=use_memory,
            apply_calibration=not marker_chain,
        )
        source = heads["source"]
        analytic_scaled = heads["analytic"]
        out: dict = {
            "n_examples": len(indices),
            "rel_l2_channel": {
                name: float(relative_l2(source[..., c], target[..., c]))
                for c, name in enumerate(SOURCE_CHANNEL_NAMES)
            },
            "rel_l2_mean": float(relative_l2(source, target)),
            "budget_rel_error": {
                name: float(value)
                for name, value in budget_terms(
                    source, target, measure, (1, 4)
                ).items()
            },
            "log_scale_abs_max": float(heads["log_scale"].abs().max()),
            "log_scale_abs_mean": float(heads["log_scale"].abs().mean()),
        }
        recorded_atoms = torch.as_tensor(
            [records[i].get("recorded_atoms_step") or 0.0 for i in indices],
            dtype=torch.float32,
        )
        recorded_radiation = torch.as_tensor(
            [records[i].get("recorded_radiation_step") or 0.0 for i in indices],
            dtype=torch.float32,
        )
        radiation_factor = float(
            (unit_factors or {}).get("radiation_unit_factor_median") or 1.0
        )
        totals = field_budget(source, measure, n_norm=n_norm, energy_factor=energy_factor)
        atoms_pred = totals["atoms"]
        rad_pred = totals["radiation"] / radiation_factor
        atoms_mask = recorded_atoms > 0
        rad_mask = recorded_radiation.abs() > 0
        out["budget_vs_recorded"] = {
            "atoms_rel_error_median": float(
                (
                    (atoms_pred[atoms_mask] - recorded_atoms[atoms_mask]).abs()
                    / recorded_atoms[atoms_mask]
                ).median()
            )
            if bool(atoms_mask.any())
            else None,
            "atoms_ratio_median": float(
                (atoms_pred[atoms_mask] / recorded_atoms[atoms_mask]).median()
            )
            if bool(atoms_mask.any())
            else None,
            "radiation_rel_error_median": float(
                (
                    (rad_pred[rad_mask] - recorded_radiation[rad_mask]).abs()
                    / recorded_radiation[rad_mask].abs()
                ).median()
            )
            if bool(rad_mask.any())
            else None,
        }
        # support mismatch: target mass on nodes where the analytic field is 0.
        support = {}
        for channel, name in enumerate(SOURCE_CHANNEL_NAMES):
            analytic_mass = (analytic_scaled[..., channel] * measure).sum(dim=1)
            zero_mask = analytic_scaled[..., channel] <= 0.0
            target_mass = (target[..., channel] * measure).sum(dim=1)
            zero_target = (target[..., channel] * measure * zero_mask).sum(dim=1)
            support[name] = {
                "scale_ratio_median": float(
                    (target_mass / analytic_mass.clamp_min(1e-30)).median()
                ),
                "correlation": float(
                    torch.corrcoef(
                        torch.stack(
                            [
                                analytic_scaled[..., channel].flatten(),
                                target[..., channel].flatten(),
                            ]
                        )
                    )[0, 1]
                ),
                "target_mass_on_zero_analytic_nodes": float(
                    (zero_target / target_mass.clamp_min(1e-30)).median()
                ),
            }
        out["proxy_truncation"] = support
    return out


def shard_acceptance(records: list[dict], indices: list[int]) -> dict:
    """Physical acceptance statistics from the cached backbone pass."""
    def median(values: list[float]) -> float | None:
        values = [v for v in values if v is not None and math.isfinite(v)]
        return float(np.median(values)) if values else None

    ratios = []
    radius_errors = []
    atoms_ratios = []
    monotonicity = []
    for index in indices:
        record = records[index]
        if record["recorded_abl_sum"] > 0:
            ratios.append(record["predicted_abl_sum"] / record["recorded_abl_sum"])
        radius_errors.append(
            abs(record["radius_end_pred_sum"] - record["radius_end_rec_sum"])
            / max(abs(record["radius_start_sum"]), 1e-30)
        )
        recorded_atoms = record.get("recorded_atoms_step")
        if recorded_atoms:
            atoms_ratios.append(record["atoms"] / recorded_atoms)
    return {
        "abl_sum_ratio_median": median(ratios),
        "abl_sum_ratio_p10": float(np.percentile(ratios, 10)) if ratios else None,
        "abl_sum_ratio_p90": float(np.percentile(ratios, 90)) if ratios else None,
        "radius_relative_error_median": median(radius_errors),
        "module_atoms_over_recorded_step_median": median(atoms_ratios),
        "n_examples": len(indices),
    }


def train_arm(
    module: SourceModule,
    records: list[dict],
    train_indices: list[int],
    val_indices: list[int],
    measure: torch.Tensor,
    *,
    seed: int,
    epochs: int,
    lr: float,
    chunk: int,
    use_memory: bool,
    unit_factors: dict | None = None,
    n_norm: float = 1.0,
    energy_factor: float = 1.0,
    proxy_weight: float = 1.0,
    budget_weight: float = 1.0,
    marker_chain: bool = False,
) -> dict:
    """Fit the split heads with truncated BPTT over contiguous chunks."""
    torch.manual_seed(seed)
    for layer in module.encoder.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1.0)
            torch.nn.init.zeros_(layer.bias)
    for head in (module.instant_head, module.memory_head):
        for layer in head.modules():
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.1)
                torch.nn.init.zeros_(layer.bias)
        with torch.no_grad():
            head[-1].weight.zero_()
            head[-1].bias.zero_()
    module.memory_cell.reset_parameters()
    for parameter in module.parameters():
        parameter.requires_grad_(True)
    optimizer = torch.optim.Adam(
        [p for p in module.parameters() if p.requires_grad], lr=lr
    )
    base_all = record_base(records, train_indices, marker_chain=marker_chain)
    marker_features_all = record_marker_features(
        records, train_indices, marker_chain=marker_chain
    )
    target_all = stack(records, train_indices, "target").float()
    recorded_atoms = torch.as_tensor(
        [records[i].get("recorded_atoms_step") or 0.0 for i in train_indices],
        dtype=torch.float32,
    )
    recorded_radiation = torch.as_tensor(
        [records[i].get("recorded_radiation_step") or 0.0 for i in train_indices],
        dtype=torch.float32,
    )
    radiation_factor = float((unit_factors or {}).get("radiation_unit_factor_median") or 1.0)
    chunks = [list(range(i, min(i + chunk, len(train_indices)))) for i in range(0, len(train_indices), chunk)]
    history = []
    for epoch in range(epochs):
        module.train()
        epoch_loss = 0.0
        for positions in chunks:
            optimizer.zero_grad()
            features = feature_batch(
                module,
                records,
                [train_indices[i] for i in positions],
                (
                    base_all[positions]
                    if marker_chain
                    else base_all[positions]
                    * module.channel_scale.to(torch.float32).view(1, 1, 5)
                ),
                marker_features=(
                    None
                    if marker_features_all is None
                    else marker_features_all[positions]
                ),
            )
            base = base_all[positions]
            target = target_all[positions]
            # Cross-step h_marker: thread the memory (and the previous source
            # summary) through consecutive windows in the chunk; the state is
            # reset at chunk boundaries (truncated BPTT).
            memory = None
            prev_source = None
            sources: list[torch.Tensor] = []
            for slot in range(len(positions)):
                heads = module.apply_heads(
                    base[slot : slot + 1],
                    features[slot : slot + 1],
                    memory=None if memory is None else memory.view(1, -1),
                    prev_source=prev_source,
                    use_correction=True,
                    use_memory=use_memory,
                    apply_calibration=not marker_chain,
                )
                sources.append(heads["source"])
                if use_memory:
                    memory = heads["memory_out"].reshape(-1)
                prev_source = heads["source"].mean(dim=1)
            source = torch.cat(sources, dim=0)
            loss = proxy_weight * outlier_clamped_relative_l2(source, target)
            # Budget supervision against the recorded per-step series (the
            # physics-structured acceptance): atoms are unit-free; the
            # radiation series enters through its registered train-only factor.
            totals = field_budget(source, measure, n_norm=n_norm, energy_factor=energy_factor)
            atoms_pred = totals["atoms"]
            atoms_true = recorded_atoms[positions]
            mask_atoms = atoms_true > 0.1 * atoms_true[atoms_true > 0].median().clamp_min(1e-30)
            if bool(mask_atoms.any()):
                loss = loss + budget_weight * robust_relative_error(
                    atoms_pred[mask_atoms], atoms_true[mask_atoms]
                )
            rad_pred = totals["radiation"] / radiation_factor
            rad_true = recorded_radiation[positions]
            rad_scale = rad_true[rad_true.abs() > 0]
            rad_floor = (
                0.1 * rad_scale.abs().median() if rad_scale.numel() else torch.tensor(0.0)
            )
            mask_rad = rad_true.abs() > rad_floor
            if bool(mask_rad.any()):
                loss = loss + budget_weight * robust_relative_error(
                    rad_pred[mask_rad], rad_true[mask_rad]
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in module.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            epoch_loss += float(loss)
        history.append(epoch_loss / max(len(chunks), 1))
    metrics = evaluate(
        module,
        records,
        val_indices,
        measure,
        use_correction=True,
        use_memory=use_memory,
        unit_factors=unit_factors,
        n_norm=n_norm,
        energy_factor=energy_factor,
        marker_chain=marker_chain,
    )
    metrics["seed"] = seed
    metrics["train_loss_first"] = history[0]
    metrics["train_loss_last"] = history[-1]
    metrics["epochs"] = epochs
    return metrics


def monotonicity_checks(
    module: SourceModule,
    dataset: StepRunDataset,
    central_density: float,
    central_mass: float,
    *,
    record_index: int,
) -> dict:
    """Controlled local perturbations on a real window (real-data acceptance).

    Three responses must be monotone for the delivered analytic backbone:

    * larger shard radius -> larger ablation rate (window-end sum);
    * larger radial velocity -> larger ``R`` at the window end;
    * larger local temperature -> larger ablation rate.
    """
    module.eval()
    positions = [p for p, meta in enumerate(dataset.metas) if meta.index == record_index]
    if not positions:
        return {"available": False, "reason": f"index {record_index} not in dataset"}
    position = positions[0]
    if position == 0:
        return {"available": False, "reason": "no predecessor record"}
    previous, current = dataset[position - 1], dataset[position]
    u_start = torch.as_tensor(previous.u_before, dtype=torch.float64).unsqueeze(0)
    u_end = torch.as_tensor(current.u_before, dtype=torch.float64).unsqueeze(0)
    p_start = torch.as_tensor(shard_array(previous), dtype=torch.float64).unsqueeze(0)
    mask = torch.ones(1, p_start.shape[1], dtype=torch.bool)

    def run(shards: torch.Tensor, u_end_local: torch.Tensor | None = None) -> tuple[float, float]:
        with torch.no_grad():
            result = module(
                u_start,
                u_end if u_end_local is None else u_end_local,
                shards,
                mask,
                dt_source_jorek=current.dt_source,
                t_norm=current.t_norm,
                central_density=central_density,
                central_mass=central_mass,
            )
        return (
            float(result["shards"]["ablation"][0].sum()),
            float(result["shards"]["values"][0, :, module.shard_index["R"]].mean()),
        )

    base_ablation, base_R = run(p_start)
    radius_up = p_start.clone()
    radius_up[..., module.shard_index["radius"]] *= 1.1
    radius_ablation, _ = run(radius_up)
    velocity_up = p_start.clone()
    velocity_up[..., module.shard_index["Vel_R"]] *= 1.1
    _, velocity_R = run(velocity_up)
    te_up = p_start.clone()
    u_end_hot = u_end.clone()
    u_end_hot[..., 8] *= 1.1  # T_e channel
    hot_ablation, _ = run(te_up, u_end_hot)
    return {
        "available": True,
        "index": int(record_index),
        "dt_source": float(current.dt_source),
        "ablation_sum_base": base_ablation,
        "ablation_sum_radius_up": radius_ablation,
        "ablation_sum_te_up": hot_ablation,
        "R_mean_base": base_R,
        "R_mean_velocity_up": velocity_R,
        "radius_monotone": bool(radius_ablation > base_ablation),
        "temperature_monotone": bool(hot_ablation > base_ablation),
        "velocity_monotone": bool(velocity_R > base_R),
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN)
    parser.add_argument("--out", default=".scratch/phase-b1-source")
    parser.add_argument("--index-start", type=int, default=2705)
    parser.add_argument("--index-end", type=int, default=11103)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--n-substeps", type=int, default=4)
    parser.add_argument("--exclude-radius", type=int, default=2)
    parser.add_argument("--val-blocks", type=int, default=2)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--proxy-weight", type=float, default=0.1)
    parser.add_argument("--budget-weight", type=float, default=1.0)
    parser.add_argument("--chunk", type=int, default=6)
    parser.add_argument(
        "--zero-dt-policy",
        choices=("registered", "error", "list"),
        default="registered",
        help=(
            "code-restart handling: 'registered' (default; the v400 events are "
            "listed in data/restart_events.json), 'error' (strict regenerated "
            "data) or 'list' (legacy diagnostics)"
        ),
    )
    parser.add_argument(
        "--marker-chain",
        action="store_true",
        help=(
            "use the reduced-order marker chain (birth -> ADAS charge states -> "
            "field-aligned transport -> FEM gather average) as the deterministic "
            "main term instead of the analytic shard backbone (ADR-0020)"
        ),
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-stride", type=int, default=60)
    parser.add_argument("--probe-limit", type=int, default=40)
    parser.add_argument("--nonlocal", dest="nonlocal_mode", choices=("frame", "off"), default="frame")
    args = parser.parse_args()

    meta = build_cache(args)
    cache_dir = Path(args.out) / "cache"
    records = load_cache(cache_dir, meta["records"])
    train_indices, val_indices = blocked_split(records, val_blocks=args.val_blocks)

    run_dir = Path(args.run_dir)
    dataset = StepRunDataset(
        run_dir,
        require_single_step=True,
        zero_dt_policy=args.zero_dt_policy,
        index_window=(args.index_start - 1, args.index_end + 1),
        mesh_verify_stride=100000,
    )
    first_frame = dataset.metas[0].path
    pos, vertex, central_density, central_mass = load_mesh_and_constants(run_dir, first_frame)
    sampler = FemMeshSampler(pos, vertex)
    measure = sampler.node_measure().float()
    if args.probe_only:
        probe = backbone_fidelity_probe(
            dataset,
            sampler,
            central_density=central_density,
            central_mass=central_mass,
            stride=args.probe_stride,
            limit=args.probe_limit,
            nonlocal_mode=args.nonlocal_mode,
        )
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "probe.json").write_text(json.dumps(probe, indent=1))
        print(json.dumps(probe, indent=1))
        return
    module = SourceModule(
        sampler,
        n_substeps=args.n_substeps,
        use_memory=True,
        use_correction=False,
        use_marker_transport=args.marker_chain,
        neon_tables=default_neon_tables(allow_surrogate=True),
    )
    module.eval()

    calibration = fit_channel_scale(records, train_indices, measure)
    scale_np = np.asarray(calibration["scale"], dtype=np.float64)
    source_norm = [
        float(
            np.median(
                [
                    float(np.abs(records[i]["analytic"][:, c]).max())
                    for i in train_indices
                ]
            )
        )
        or 1.0
        for c in range(5)
    ]
    module.set_source_norm(torch.tensor(source_norm, dtype=torch.float32))
    stats = shard_stats(records, train_indices)
    unit_factors = budget_supervision(records, train_indices)
    n_norm = float(meta["central_density"]) * 1e20
    energy_factor = float(E_NORM / records[train_indices[0]]["t_norm"])
    # Registered proxy-calibrated copy (fit to the *projection* object only).
    proxy_module = SourceModule(
        sampler,
        n_substeps=args.n_substeps,
        neon_tables=default_neon_tables(allow_surrogate=True),
    )
    proxy_module.eval()
    proxy_module.set_channel_scale(torch.tensor(calibration["scale"], dtype=torch.float32))
    proxy_module.set_source_norm(torch.tensor(source_norm, dtype=torch.float32))
    for target_module in (module, proxy_module):
        target_module.set_stats(
            state_mean=torch.zeros(9),
            state_std=torch.ones(9),
            shard_mean=torch.tensor(stats["mean"], dtype=torch.float32),
            shard_std=torch.tensor(stats["std"], dtype=torch.float32),
        )

    results: dict = {
        "run_dir": str(run_dir),
        "cache": meta,
        "calibration": calibration,
        "source_norm": source_norm,
        "budget_unit_factors": unit_factors,
        "unit_conversion": {
            "n_norm": n_norm,
            "energy_factor_E_NORM_over_t_norm": energy_factor,
        },
        "split": {
            "n_train": len(train_indices),
            "n_val": len(val_indices),
            "val_blocks": args.val_blocks,
            "train_index_range": [records[train_indices[0]]["index"], records[train_indices[-1]]["index"]],
            "val_index_range": [records[val_indices[0]]["index"], records[val_indices[-1]]["index"]],
        },
    }
    # Uncalibrated analytic-only (raw JOREK-derived unit conversion, no fit).
    raw_module = SourceModule(
        sampler,
        n_substeps=args.n_substeps,
        neon_tables=default_neon_tables(allow_surrogate=True),
    )
    raw_module.eval()
    raw_module.set_stats(
        state_mean=torch.zeros(9),
        state_std=torch.ones(9),
        shard_mean=torch.tensor(stats["mean"], dtype=torch.float32),
        shard_std=torch.tensor(stats["std"], dtype=torch.float32),
    )
    results["arms"] = {}
    results["arms"]["analytic_raw"] = evaluate(
        raw_module,
        records,
        val_indices,
        measure,
        use_correction=False,
        unit_factors=unit_factors,
        n_norm=n_norm,
        energy_factor=energy_factor,
        marker_chain=False,
    )
    results["arms"]["analytic_proxy_calibrated"] = evaluate(
        proxy_module,
        records,
        val_indices,
        measure,
        use_correction=False,
        unit_factors=unit_factors,
        n_norm=n_norm,
        energy_factor=energy_factor,
        marker_chain=False,
    )
    if args.marker_chain:
        # Reduced-order marker chain as the deterministic main term (ADR-0020):
        # correction heads zero-initialised, so this registers the chain itself.
        results["arms"]["marker_chain_raw"] = evaluate(
            module,
            records,
            val_indices,
            measure,
            use_correction=False,
            use_memory=True,
            unit_factors=unit_factors,
            n_norm=n_norm,
            energy_factor=energy_factor,
            marker_chain=True,
        )
    results["shard_acceptance"] = shard_acceptance(records, val_indices)
    results["monotonicity"] = monotonicity_checks(
        module,
        dataset,
        central_density,
        central_mass,
        record_index=records[val_indices[len(val_indices) // 2]]["index"],
    )

    if not args.skip_training:
        seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
        for name, use_memory in (("correction_memory", True), ("correction_nomemory", False)):
            arms = []
            for seed in seeds:
                arms.append(
                    train_arm(
                        module,
                        records,
                        train_indices,
                        val_indices,
                        measure,
                        seed=seed,
                        epochs=args.epochs,
                        lr=args.lr,
                        chunk=args.chunk,
                        use_memory=use_memory,
                        unit_factors=unit_factors,
                        n_norm=n_norm,
                        energy_factor=energy_factor,
                        proxy_weight=args.proxy_weight,
                        budget_weight=args.budget_weight,
                        marker_chain=args.marker_chain,
                    )
                )
            results["arms"][name] = {
                "seeds": arms,
                "rel_l2_mean_median": float(np.median([a["rel_l2_mean"] for a in arms])),
                "rel_l2_mean_min": float(np.min([a["rel_l2_mean"] for a in arms])),
                "rel_l2_mean_max": float(np.max([a["rel_l2_mean"] for a in arms])),
                "rel_l2_channel_median": {
                    name_c: float(np.median([a["rel_l2_channel"][name_c] for a in arms]))
                    for name_c in SOURCE_CHANNEL_NAMES
                },
                "budget_rel_error_median": {
                    name_c: float(np.median([a["budget_rel_error"][name_c] for a in arms]))
                    for name_c in ("N_imp", "rad_sink")
                },
                "budget_vs_recorded_median": {
                    key: float(
                        np.median(
                            [
                                a["budget_vs_recorded"][key]
                                for a in arms
                                if a["budget_vs_recorded"][key] is not None
                            ]
                        )
                    )
                    for key in ("atoms_rel_error_median", "atoms_ratio_median", "radiation_rel_error_median")
                },
            }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=1))
    print(json.dumps(
        {
            "split": results["split"],
            "arms": {
                key: {
                    k: v
                    for k, v in value.items()
                    if k in {"rel_l2_mean", "rel_l2_channel_median", "rel_l2_mean_median"}
                }
                for key, value in results["arms"].items()
            },
            "shard_acceptance": results["shard_acceptance"],
        },
        indent=1,
    ))
    print(f"[b1] wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
