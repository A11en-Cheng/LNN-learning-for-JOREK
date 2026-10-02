"""Validate the offline nonlocal ablation branch against recorded shard data.

JOREK freezes ``nonlocal_array.dat`` at the first SPI step (the
``nl_avg_Te(1) == 0`` guard in ``core/mod_jorek_timestepping.f90``) and then,
for every shard inside the LCFS, looks up the surface-averaged ``T_e``/``n_e``
at the shard psi and takes ``max(local, nonlocal)``
(``models/pellet_module.f90``, ablation model 3 + Ne branch).

This script evaluates both variants against the recorded per-shard
``spi_abl_arr`` on saved frames: the local-only rate (what the Phase-2 backbone
used so far) and the local+nonlocal rate.  ``nonlocal_table`` inputs are the
shard psi, the frame's ``psi_axis_t``/``psi_bnd_t`` and ``central_density``.

Example:

    .torch-cuda-env/bin/python tests/check_nonlocal_ablation.py \
        --runs scan_1d22_ang15_5_massloss_neon_v400 scan_1d21_ang12_massloss_neon \
        --cache-dir .scratch/n/frames --frames 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.data_manifest import aligned_frames
from src.data.interp_dataset import RunFrameStore
from src.data.nonlocal_table import nonlocal_table_from_frame
from src.models.field_sampling import knn_weights, sample_nodes
from src.physics.spi_model import (
    SHARD_ABL,
    SHARD_PSI,
    SHARD_RADIUS,
    SHARD_SPECIES,
    NonlocalAblationTable,
    parks_ablation_neon,
    particle_density_si,
    temperature_ev,
)


def _metrics(predicted: torch.Tensor, recorded: torch.Tensor) -> dict:
    both = (predicted > 0) & (recorded > 0)
    ratio = (predicted[both] / recorded[both]).double().numpy()
    log_pred = np.log(predicted[both].double().numpy())
    log_rec = np.log(recorded[both].double().numpy())
    correlation = (
        float(np.corrcoef(log_pred, log_rec)[0, 1]) if log_pred.size > 2 else float("nan")
    )
    return {
        "n": int(both.sum()),
        "median_ratio": float(np.median(ratio)) if ratio.size else float("nan"),
        "log_corr": correlation,
        "within_2x": float(np.mean((ratio > 0.5) & (ratio < 2.0))) if ratio.size else float("nan"),
        "sum_ratio": float(predicted.sum() / recorded.sum().clamp_min(1e-30)),
    }


def _frame_psi_bounds(path: Path) -> tuple[float, float]:
    with h5py.File(path, "r") as handle:
        psi_axis = float(np.ravel(handle["psi_axis_t"][()])[-1])
        psi_bnd = float(np.ravel(handle["psi_bnd_t"][()])[-1])
    return psi_axis, psi_bnd


def _evaluate_run(
    run_dir: Path,
    store: RunFrameStore,
    frozen_table: NonlocalAblationTable,
    *,
    n_frames: int,
    k: int,
    central_density: float,
    table_from_frame: bool,
) -> dict:
    positions = [
        position
        for position in range(len(store))
        if store.shard_values[position] is not None
        and bool(store.shard_alive[position].any())
        and float(store.shard_values[position][:, SHARD_ABL].max()) > 0.0
    ]
    if not positions:
        raise SystemExit(f"{run_dir.name}: no post-injection frames with shards")
    if len(positions) > n_frames:
        picks = np.linspace(0, len(positions) - 1, n_frames).round().astype(int)
        positions = [positions[i] for i in dict.fromkeys(picks.tolist())]
    frames = aligned_frames(run_dir)
    accumulated = {"local": [], "nonlocal": [], "recorded": []}
    raise_share = []
    for position in positions:
        values = store.shard_values[position]
        alive = store.shard_alive[position]
        alive_idx = torch.nonzero(alive, as_tuple=False).squeeze(-1)
        shards = values[alive_idx]
        psi_axis, psi_bnd = _frame_psi_bounds(frames[position].state_path)
        idx, weights = knn_weights(store.topology.pos, shards[:, :2], k=k)
        u = store.u[position]
        local_rho = sample_nodes(u[:, 4:5], idx, weights).squeeze(-1)
        local_rho_imp = sample_nodes(u[:, 7:8], idx, weights).squeeze(-1)
        local_te = sample_nodes(u[:, 8:9], idx, weights).squeeze(-1)
        ne_si = (
            particle_density_si(local_rho, central_density)
            - particle_density_si(local_rho_imp, central_density)
        ).clamp_min(0.0)
        te_ev = temperature_ev(local_te, central_density)
        radius = shards[:, SHARD_RADIUS].clamp_min(0.0)
        species = shards[:, SHARD_SPECIES]
        local = parks_ablation_neon(radius, te_ev, ne_si, species)
        table = (
            nonlocal_table_from_frame(frames[position].state_path)
            if table_from_frame
            else frozen_table
        )
        te_nl, ne_nl, inside = table.surface_state(
            shards[:, SHARD_PSI], psi_axis, psi_bnd, central_density
        )
        nonlocal_rate = parks_ablation_neon(radius, te_nl, ne_nl, species)
        combined = torch.where(inside, torch.maximum(local, nonlocal_rate), local)
        recorded = shards[:, SHARD_ABL]
        accumulated["local"].append(local)
        accumulated["nonlocal"].append(combined)
        accumulated["recorded"].append(recorded)
        raise_share.append(float((combined > local * 1.001).double().mean()))
    local_all = torch.cat(accumulated["local"])
    nonlocal_all = torch.cat(accumulated["nonlocal"])
    recorded_all = torch.cat(accumulated["recorded"])
    summary = {
        "run": run_dir.name,
        "frames": len(positions),
        "shards": int(recorded_all.numel()),
        "raise_share": float(np.mean(raise_share)),
        "local": _metrics(local_all, recorded_all),
        "nonlocal": _metrics(nonlocal_all, recorded_all),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=["scan_1d22_ang15_5_massloss_neon_v400", "scan_1d21_ang12_massloss_neon"],
    )
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--frames", type=int, default=12, help="frames sampled per run")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--central-density", type=float, default=0.8)
    parser.add_argument(
        "--table",
        choices=("frame", "file"),
        default="frame",
        help="rebuild the surface averages from each frame, or use the on-disk snapshot",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    for run in args.runs:
        run_dir = data_root / run
        table_path = run_dir / "nonlocal_array.dat"
        if not table_path.exists():
            print(f"skip {run}: no nonlocal_array.dat")
            continue
        table = NonlocalAblationTable.from_file(table_path)
        store = RunFrameStore(run_dir, cache_dir=args.cache_dir).load()
        summary = _evaluate_run(
            run_dir,
            store,
            table,
            n_frames=args.frames,
            k=args.k,
            central_density=args.central_density,
            table_from_frame=args.table == "frame",
        )
        print(f"== {summary['run']} [{args.table}] ({summary['frames']} frames, "
              f"{summary['shards']} shard-frames, nonlocal raised "
              f"{summary['raise_share']:.1%} of shards) ==")
        for label, key in (("local only", "local"), ("local+nonlocal", "nonlocal")):
            row = summary[key]
            print(
                f"   {label:14s} median ratio={row['median_ratio']:6.3f} "
                f"log-corr={row['log_corr']:6.3f} within-2x={row['within_2x']:6.1%} "
                f"sum(rate) ratio={row['sum_ratio']:6.3f}"
            )


if __name__ == "__main__":
    main()
