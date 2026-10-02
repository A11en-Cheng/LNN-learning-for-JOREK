"""Phase-4 physics signal: model sub-interval residual vs recorded floor.

For every supervised interval of the held-out windows this reports, per channel,

* ``floor`` - the residual of the **recorded** trajectory (u_k -> u_{k+1}) with
  the interval-start frame's own coefficients and the recorded teacher source.
  This is what a perfect data fit still leaves on the table;
* ``model`` - the residual of the CfC sub-interval trajectory
  (u(t*) -> u(t* + rho*dt)), i.e. exactly the quantity the physics loss
  penalises during training.

The ratio ``model / floor`` is the honest signal: values below 1 mean the
physics term is consistent with the data manifold; values far above 1 mean the
interpolated trajectory violates the discrete equations much more than the
recorded data does.

Example:

    ./.torch-cuda-env/bin/python tests/eval_phase4_physics.py \
        --checkpoint output/cfc_interp/phase4_phys_0.05/checkpoint.pt --device cuda
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
from src.models.cfc_interp import CfCInterpolator
from src.models.cfc_interp import model_kwargs_from_meta
from src.physics.model502_full_loss import model502_full_residuals
from tests.train_cfc_interpolation import PHYSICS_CHANNELS, _condition_from_frame


def _median(values: list[float]) -> float:
    return float(np.median(np.asarray(values))) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-windows", type=int, default=40)
    parser.add_argument("--alpha", type=float, default=0.5, help="t* = alpha * dt")
    parser.add_argument(
        "--rho",
        type=float,
        default=None,
        help="delta = rho * dt; default uses the (*trained*) second substep, "
        "delta = (1 - alpha) * dt, so the reported residual is the quantity the "
        "physics loss actually minimised",
    )
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dt-scale", type=float, default=50.0)
    parser.add_argument("--residual", action="store_true", help="decoder outputs the increment")
    parser.add_argument("--input-deriv", action="store_true", help="encoder takes derivative features")
    parser.add_argument("--input-deriv2", action="store_true", help="encoder also takes Laplacian features")
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
        manifest, manifest["data_root"], runs=[args.held_out], inner="val",
        stats=stats, cache_dir=args.cache_dir, with_shards=False,
        with_lap=args.input_deriv2,
    )
    mean = torch.tensor(stats["field_mean"], device=device)
    std = torch.tensor(stats["field_std"], device=device)
    source_stats = (
        torch.tensor(stats["source_mean"], device=device),
        torch.tensor(stats["source_std"], device=device).clamp_min(1e-12),
    )

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CfCInterpolator(**model_kwargs_from_meta(payload.get("meta") or {}))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    floors: dict[str, list[float]] = {name: [] for name in PHYSICS_CHANNELS}
    models: dict[str, list[float]] = {name: [] for name in PHYSICS_CHANNELS}
    intervals = 0
    with torch.no_grad():
        for index in range(min(len(dataset), args.max_windows)):
            item = dataset[index]
            batch = to_device(collate_windows([item]), device)
            u = batch["u"][0]
            u_deriv = batch["u_deriv"][0] if (args.input_deriv or args.input_deriv2) else None
            u_lap = batch["u_lap"][0] if args.input_deriv2 else None
            dt = batch["dt"][0]
            source = batch["source"][0]
            available = batch["source_available"][0]
            condition = item["condition"]
            topology = batch["topology"]
            burn_in = int((~batch["loss_mask"][0]).sum().item())
            hidden = None
            for k in range(u.shape[0] - 1):
                if not bool(available[k + 1]):
                    continue
                controls = source[k + 1]
                coeff = _condition_from_frame(condition[k].numpy()).to(device)
                if k < burn_in:
                    _, _, hidden = model.step(
                        u[k], hidden, dt[k], controls, None,
                        None if u_deriv is None else u_deriv[k],
                        None if u_lap is None else u_lap[k],
                    )
                    continue
                graph = Data(
                    x=u[k],
                    pos=topology.pos.to(device).double(),
                    edge_index=topology.edge_index.to(device),
                    edge_attr=topology.edge_attr.to(device).double(),
                    condition=coeff,
                    source=controls.to(device).double(),
                )
                floor = model502_full_residuals(
                    u[k + 1].double(), u[k].double(), graph, float(dt[k]),
                    stats=(mean, std), source_stats=source_stats,
                )
                step_deriv = None if u_deriv is None else u_deriv[k]
                latent = model.encode(
                    u[k], step_deriv, None if u_lap is None else u_lap[k]
                )
                mid_latent, mid_hidden = model.advance(
                    latent, hidden, args.alpha * dt[k], controls, None
                )
                u_mid = model.decode(mid_latent)
                if model.residual:
                    u_mid = u[k] + u_mid
                rho = args.rho if args.rho is not None else (1.0 - args.alpha)
                end_latent, hidden = model.advance(
                    mid_latent, mid_hidden, rho * dt[k], controls, None
                )
                u_end = model.decode(end_latent)
                if model.residual:
                    u_end = u[k] + u_end
                graph_mid = Data(
                    x=u_mid,
                    pos=graph.pos,
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr,
                    condition=coeff,
                    source=controls.to(device).double(),
                )
                model_res = model502_full_residuals(
                    u_end.double(), u_mid.double(), graph_mid, float(rho * dt[k]),
                    stats=(mean, std), source_stats=source_stats,
                )
                for name in PHYSICS_CHANNELS:
                    floors[name].append(float(floor[name]))
                    models[name].append(float(model_res[name]))
                intervals += 1

    print(f"checkpoint={args.checkpoint} held-out={args.held_out} intervals={intervals} "
          f"alpha={args.alpha} rho={args.rho}")
    print(f"{'channel':22s} {'floor(med)':>11s} {'model(med)':>11s} {'ratio':>7s} {'model p90':>10s}")
    summary = {}
    for name in PHYSICS_CHANNELS:
        floor_median = _median(floors[name])
        model_median = _median(models[name])
        model_p90 = (
            float(np.percentile(np.asarray(models[name]), 90)) if models[name] else float("nan")
        )
        summary[name] = {
            "floor_median": floor_median,
            "model_median": model_median,
            "model_p90": model_p90,
            "ratio": model_median / floor_median if floor_median else float("nan"),
            "count": len(floors[name]),
        }
        print(f"{name:22s} {floor_median:11.4f} {model_median:11.4f} "
              f"{summary[name]['ratio']:7.2f} {model_p90:10.4f}")
    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "held_out": args.held_out,
                    "intervals": intervals,
                    "alpha": args.alpha,
                    "rho": args.rho,
                    "channels": summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
