"""Residual-floor distribution on recorded frames (Phase-4 lambda calibration).

For every supervised interval of the evaluation windows this evaluates the
ten-equation residual set on the **recorded** trajectory (u_k -> u_{k+1}) with
the frame's own coefficients (mapped from its parameter channel) and the
recorded projections as teacher source.  The per-channel distribution
(median/p10/p90) is the floor that a physics loss cannot beat.

Conventions follow the trainer exactly (``tests/train_cfc_interpolation.py``):
the coefficient vector of the **interval-start frame** ``k`` is used, together
with the recorded projection of frame ``k + 1`` as the teacher source.

Example:

    ./.torch-cuda-env/bin/python tests/check_residual_floor_stats.py --inner val --max-intervals 120
    ./.torch-cuda-env/bin/python tests/check_residual_floor_stats.py --runs all --max-intervals 400
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import (
    InterpolationWindowDataset,
    RunFrameStore,
    collate_windows,
    compute_transform_stats,
    to_device,
)
from src.physics.model502_full_loss import model502_full_residuals
from tests.train_cfc_interpolation import _condition_from_frame  # reuse the mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument(
        "--runs",
        default=None,
        help="comma-separated run names, or 'all' to pool every run's windows; "
        "default: the fold held-out run",
    )
    parser.add_argument("--inner", default="val")
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--max-intervals", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=".scratch/n/residual_floor_stats.json")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(r for r in manifest["folds"] if r["held_out_run"] == args.held_out)
    if args.runs in (None, ""):
        runs = [args.held_out]
    elif args.runs == "all":
        runs = [record["name"] for record in manifest["runs"]]
    else:
        runs = [name.strip() for name in args.runs.split(",") if name.strip()]
    stats = compute_transform_stats(
        {
            name: RunFrameStore(
                Path(manifest["data_root"]) / name, cache_dir=args.cache_dir
            ).load()
            for name in fold["train_runs"]
        },
        runs=fold["train_runs"],
    )
    dataset = InterpolationWindowDataset(
        manifest, manifest["data_root"], runs=runs, inner=args.inner,
        stats=stats, cache_dir=args.cache_dir, with_shards=False,
    )
    device = torch.device(args.device)
    mean = torch.tensor(stats["field_mean"], device=device)
    std = torch.tensor(stats["field_std"], device=device)
    source_stats = (
        torch.tensor(stats["source_mean"], device=device),
        torch.tensor(stats["source_std"], device=device).clamp_min(1e-12),
    )
    records: dict[str, list[float]] = {}
    per_run: dict[str, dict[str, list[float]]] = {}
    interval_dt: dict[str, list[float]] = {}
    runs_seen: set[str] = set()
    intervals = 0
    started = time.perf_counter()
    for index in range(min(len(dataset), args.max_windows)):
        item = dataset[index]
        batch = to_device(collate_windows([item]), device)
        u = batch["u"][0]
        dt = batch["dt"][0]
        source = batch["source"][0]
        available = batch["source_available"][0]
        condition = item["condition"]
        topology = batch["topology"]
        burn_in = int((~batch["loss_mask"][0]).sum().item())
        runs_seen.add(item["run"])
        for k in range(burn_in, u.shape[0] - 1):
            if not bool(available[k + 1]):
                continue
            graph = Data(
                x=u[k],
                pos=topology.pos.to(device).double(),
                edge_index=topology.edge_index.to(device),
                edge_attr=topology.edge_attr.to(device).double(),
                condition=_condition_from_frame(condition[k].numpy()).to(device),
                source=source[k + 1].to(device).double(),
            )
            residuals = model502_full_residuals(
                u[k + 1].double(), u[k].double(), graph, float(dt[k]),
                stats=(mean, std), source_stats=source_stats,
            )
            for name, value in residuals.items():
                records.setdefault(name, []).append(float(value))
                per_run.setdefault(item["run"], {}).setdefault(name, []).append(float(value))
            interval_dt.setdefault(item["run"], []).append(float(dt[k]))
            intervals += 1
            if intervals >= args.max_intervals:
                break
        if intervals >= args.max_intervals:
            break

    elapsed = time.perf_counter() - started
    print(
        f"runs={sorted(runs_seen)} inner={args.inner} windows={index + 1} "
        f"intervals={intervals} wall={elapsed:.1f}s ({elapsed / max(intervals, 1):.2f}s/interval)"
    )
    print(f"{'residual':24s} {'median':>8s} {'p10':>8s} {'p90':>8s}")
    summary = {}
    for name, values in records.items():
        array = np.asarray(values)
        summary[name] = {
            "count": int(array.size),
            "median": float(np.median(array)),
            "p10": float(np.percentile(array, 10)),
            "p90": float(np.percentile(array, 90)),
            "mean": float(array.mean()),
        }
        print(f"{name:24s} {summary[name]['median']:8.4f} {summary[name]['p10']:8.4f} {summary[name]['p90']:8.4f}")
    print()
    for name in sorted(per_run):
        row = per_run[name]
        median_dt = float(np.median(interval_dt[name]))
        cells = " ".join(
            f"{channel}={np.median(row[channel]):.3f}"
            for channel in ("induction", "current_definition", "vorticity_definition", "vorticity", "density", "impurity")
            if channel in row
        )
        print(f"{name:42s} n={len(row['induction']):3d} dt~{median_dt:8.2f} {cells}")
    payload = {
        "manifest": args.manifest,
        "inner": args.inner,
        "runs": sorted(runs_seen),
        "intervals": intervals,
        "wall_seconds": elapsed,
        "channels": summary,
        "per_run": {
            run: {
                **{
                    channel: {
                        "count": len(values),
                        "median": float(np.median(values)),
                        "p10": float(np.percentile(values, 10)),
                        "p90": float(np.percentile(values, 90)),
                    }
                    for channel, values in sorted(channels.items())
                },
                "dt_median": float(np.median(interval_dt[run])),
            }
            for run, channels in sorted(per_run.items())
        },
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
