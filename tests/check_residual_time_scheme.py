"""How much of the residual floor is *time* discretisation?

The recorded frames satisfy JOREK's own implicit time scheme (Crank-Nicolson over
50 fluid steps per recorded frame), while our residual module differences two
consecutive frames one-sided (backward Euler).  This script separates the two
sources of floor by evaluating the **same spatial operators** on the recorded
trajectory with two different time differences:

* ``backward``  - ``(u_{k+1} - u_k)/dt_k``      vs ``RHS(u_k)``  (what the
  trainer currently penalises);
* ``central``   - ``(u_{k+1} - u_{k-1})/(t_{k+1} - t_{k-1})`` vs ``RHS(u_k)``
  (second-order in time, i.e. what a trapezoidal scheme would approximate).

If ``central`` is materially below ``backward``, the floor is time-scheme limited
and a theta/Crank-Nicolson residual is worth implementing; if they are close, the
floor is dominated by the spatial operators and changing the time scheme buys
nothing.

The central difference is smuggled through the existing module by asking for
``state_next = u_k + (u_{k+1} - u_{k-1})`` with ``dt = t_{k+1} - t_{k-1}``: the
time-derivative terms then see the central difference while every RHS term is
still evaluated on ``u_k``.

Example:

    ./.torch-cuda-env/bin/python tests/check_residual_time_scheme.py --held-out scan_1d22_ang15_5_massloss_neon_v400
"""

from __future__ import annotations

import argparse
import json
import sys
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
from tests.train_cfc_interpolation import _condition_from_frame

CHANNELS = ("induction", "current_definition", "vorticity_definition")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--inner", default="val")
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--max-intervals", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(r for r in manifest["folds"] if r["held_out_run"] == args.held_out)
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
        manifest, manifest["data_root"], runs=[args.held_out], inner=args.inner,
        stats=stats, cache_dir=args.cache_dir, with_shards=False,
    )
    mean = torch.tensor(stats["field_mean"], device=device)
    std = torch.tensor(stats["field_std"], device=device)
    source_stats = (
        torch.tensor(stats["source_mean"], device=device),
        torch.tensor(stats["source_std"], device=device).clamp_min(1e-12),
    )

    records: dict[str, dict[str, list[float]]] = {
        "backward": {name: [] for name in CHANNELS},
        "central": {name: [] for name in CHANNELS},
    }
    intervals = 0
    for index in range(min(len(dataset), args.max_windows)):
        item = dataset[index]
        batch = to_device(collate_windows([item]), device)
        u = batch["u"][0]
        t = batch["t"][0]
        dt = batch["dt"][0]
        source = batch["source"][0]
        available = batch["source_available"][0]
        condition = item["condition"]
        topology = batch["topology"]
        burn_in = int((~batch["loss_mask"][0]).sum().item())
        for k in range(burn_in, u.shape[0] - 1):
            if not bool(available[k + 1]) or k == 0:
                continue
            coeff = _condition_from_frame(condition[k].numpy()).to(device)

            def residual(state_next: torch.Tensor, dt_value: float):
                graph = Data(
                    x=u[k],
                    pos=topology.pos.to(device).double(),
                    edge_index=topology.edge_index.to(device),
                    edge_attr=topology.edge_attr.to(device).double(),
                    condition=coeff,
                    source=source[k + 1].to(device).double(),
                )
                return model502_full_residuals(
                    state_next.double(), u[k].double(), graph, dt_value,
                    stats=(mean, std), source_stats=source_stats,
                )

            backward = residual(u[k + 1], float(dt[k]))
            span = float(t[k + 1] - t[k - 1])
            central_next = u[k] + (u[k + 1] - u[k - 1])
            central = residual(central_next, span)
            for name in CHANNELS:
                records["backward"][name].append(float(backward[name]))
                records["central"][name].append(float(central[name]))
            intervals += 1
            if intervals >= args.max_intervals:
                break
        if intervals >= args.max_intervals:
            break

    print(f"held-out={args.held_out} inner={args.inner} intervals={intervals}")
    print(f"{'channel':22s} {'backward':>10s} {'central':>10s} {'ratio':>7s}")
    summary = {}
    for name in CHANNELS:
        back = float(np.median(records["backward"][name]))
        cent = float(np.median(records["central"][name]))
        summary[name] = {
            "backward_median": back,
            "central_median": cent,
            "ratio": cent / back if back else float("nan"),
            "count": intervals,
        }
        print(f"{name:22s} {back:10.4f} {cent:10.4f} {summary[name]['ratio']:7.2f}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
