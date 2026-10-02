"""Per-field / per-quantity metrics (MAE, RMSE, R2) for the Phase-2 checkpoints.

Example:

    .torch-cuda-env/bin/python tests/eval_phase2_metrics.py \
        --held-out scan_1d22_ang15_5_massloss_neon_v400 \
        --cache-dir .scratch/model502-interp/frames
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import RunFrameStore
from src.models.fragment_dynamics import HybridFragmentDynamics
from src.models.source_head import SourceFieldHead
from src.physics.spi_model import SHARD_ABL, SHARD_RADIUS

SOURCE_NAMES = ["dE_ion", "rad_sink", "Z_eff", "Z_imp", "N_imp"]
DYNAMICS_NAMES = ["R", "Z", "phi", "Vel_R", "Vel_Z", "Vel_RxZ", "radius", "log_abl"]
DYNAMICS_INCREMENT_NAMES = ["dR", "dZ", "dphi", "dVel_R", "dVel_Z", "dVel_RxZ"]


class Accumulator:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.count = np.zeros(len(names))
        self.abs_error = np.zeros(len(names))
        self.sq_error = np.zeros(len(names))
        self.value_sum = np.zeros(len(names))
        self.value_sq = np.zeros(len(names))

    def add(self, index: int, predicted: torch.Tensor, target: torch.Tensor) -> None:
        predicted = predicted.double().cpu().numpy()
        target = target.double().cpu().numpy()
        self.count[index] += predicted.size
        self.abs_error[index] += np.abs(predicted - target).sum()
        self.sq_error[index] += ((predicted - target) ** 2).sum()
        self.value_sum[index] += target.sum()
        self.value_sq[index] += (target**2).sum()

    def report(self, label: str) -> list[dict]:
        rows = []
        for index, name in enumerate(self.names):
            if self.count[index] == 0:
                continue
            mean = self.value_sum[index] / self.count[index]
            variance = max(self.value_sq[index] / self.count[index] - mean**2, 0.0)
            r2 = 1.0 - self.sq_error[index] / (variance * self.count[index]) if variance > 0 else float("nan")
            rows.append(
                {
                    "variant": label,
                    "name": name,
                    "n": int(self.count[index]),
                    "mae": float(self.abs_error[index] / self.count[index]),
                    "rmse": float(math.sqrt(self.sq_error[index] / self.count[index])),
                    "r2": float(r2),
                }
            )
        return rows


def _source_metrics(model, store, config, device):
    model.eval()
    baseline = Accumulator(SOURCE_NAMES)
    hybrid = Accumulator(SOURCE_NAMES)
    with torch.no_grad():
        for position in range(len(store)):
            if store.shard_values[position] is None or not bool(store.source_available[position]):
                continue
            shards = store.shard_values[position].unsqueeze(0).to(device)
            mask = store.shard_alive[position].unsqueeze(0).to(device)
            u_raw = store.u[position].unsqueeze(0).to(device)
            target = store.source[position].to(device)
            out = model(
                u_raw,
                shards,
                mask,
                config["pos"],
                central_density=config["central_density"],
                magnetic_field=config["magnetic_field"],
            )
            for label, use in (("baseline", False), ("hybrid", True)):
                prediction = out["source" if use else "baseline"][0]
                acc = hybrid if use else baseline
                for channel in range(len(SOURCE_NAMES)):
                    acc.add(channel, prediction[:, channel], target[:, channel])
    return baseline.report("baseline") + hybrid.report("hybrid")


def _dynamics_metrics(model, store, config, device):
    model.eval()
    baseline = Accumulator(DYNAMICS_NAMES)
    hybrid = Accumulator(DYNAMICS_NAMES)
    baseline_inc = Accumulator(DYNAMICS_INCREMENT_NAMES)
    hybrid_inc = Accumulator(DYNAMICS_INCREMENT_NAMES)
    with torch.no_grad():
        for position in range(len(store) - 1):
            if store.shard_values[position] is None or store.shard_values[position + 1] is None:
                continue
            current = store.shard_values[position].unsqueeze(0).to(device)
            mask = store.shard_alive[position].unsqueeze(0).to(device)
            target_mask = store.shard_alive[position + 1].unsqueeze(0).to(device)
            target = store.shard_values[position + 1].unsqueeze(0).to(device)
            u_raw = store.u[position].unsqueeze(0).to(device)
            dt = torch.tensor([float(store.time[position + 1] - store.time[position])], device=device)
            t_norm = torch.tensor([float(store.t_norm[position])], device=device)
            nonlocal_tables = (
                [store.nonlocal_table(position)] if config.get("nonlocal", True) else None
            )
            psi_axis = (
                store.psi_axis[position : position + 1].to(device)
                if config.get("nonlocal", True)
                else None
            )
            psi_bnd = (
                store.psi_bnd[position : position + 1].to(device)
                if config.get("nonlocal", True)
                else None
            )
            both = (mask & target_mask)[0]
            alive = mask[0]
            for label, use in (("baseline", False), ("hybrid", True)):
                out = model(
                    u_raw,
                    current,
                    mask,
                    config["pos"],
                    dt_jorek=dt,
                    t_norm=t_norm,
                    central_density=config["central_density"],
                    pellet_density=config["pellet_density"],
                    magnetic_field=config["magnetic_field"],
                    use_correction=use,
                    nonlocal_tables=nonlocal_tables,
                    psi_axis=psi_axis,
                    psi_bnd=psi_bnd,
                )["next"][0]
                acc = hybrid if use else baseline
                acc_inc = hybrid_inc if use else baseline_inc
                # position / velocity R2 on the next-frame values (both-alive shards)
                for index, field in enumerate((0, 1, 2, 4, 5, 6)):
                    if bool(both.any()):
                        acc.add(index, out[both, field], target[0][both, field])
                # radius on all shards alive in the current frame (includes deaths)
                if bool(alive.any()):
                    acc.add(6, out[alive, SHARD_RADIUS], target[0][alive, SHARD_RADIUS])
                positive = alive & (target[0][:, SHARD_ABL] > 0)
                if bool(positive.any()):
                    acc.add(
                        7,
                        torch.log1p(out[positive, SHARD_ABL].clamp_min(0.0)),
                        torch.log1p(target[0][positive, SHARD_ABL]),
                    )
                if bool(both.any()):
                    for index, field in enumerate((0, 1, 2, 4, 5, 6)):
                        acc_inc.add(
                            index,
                            out[both, field] - current[0][both, field],
                            target[0][both, field] - current[0][both, field],
                        )
    return (
        baseline.report("baseline")
        + hybrid.report("hybrid")
        + baseline_inc.report("baseline")
        + hybrid_inc.report("hybrid")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--cache-dir", default=".scratch/model502-interp/frames")
    parser.add_argument("--source-checkpoint", default="output/source_head/v400/checkpoint.pt")
    parser.add_argument("--dynamics-checkpoint", default="output/fragment_dynamics/v400/checkpoint.pt")
    parser.add_argument("--central-density", type=float, default=0.8)
    parser.add_argument("--pellet-density", type=float, default=4.30e8)
    parser.add_argument("--f0", type=float, default=-3.308745132)
    parser.add_argument("--r-geo", type=float, default=1.11)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="disable the nonlocal ablation branch (local Parks branch only)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    store = RunFrameStore(Path(args.data_root) / args.held_out, cache_dir=Path(args.cache_dir)).load()
    pos = store.topology.pos.to(device)
    config = {
        "pos": pos,
        "central_density": args.central_density,
        "pellet_density": args.pellet_density,
        "magnetic_field": abs(args.f0 / args.r_geo),
        "nonlocal": not args.local_only,
    }

    source_path = Path(args.source_checkpoint)
    if source_path.exists():
        model = SourceFieldHead().to(device)
        model.load_state_dict(torch.load(source_path, map_location=device, weights_only=False))
        rows = _source_metrics(
            model, store, {k: v for k, v in config.items() if k != "pellet_density"}, device
        )
        print("== source head: 5 projection channels ==")
        for row in rows:
            print(
                f"{row['variant']:8s} {row['name']:9s} n={row['n']:8d} "
                f"MAE={row['mae']:.4e} RMSE={row['rmse']:.4e} R2={row['r2']:.4f}"
            )
    else:
        print(f"source checkpoint not found: {source_path}")

    dynamics_path = Path(args.dynamics_checkpoint)
    if dynamics_path.exists():
        model = HybridFragmentDynamics().to(device)
        model.load_state_dict(torch.load(dynamics_path, map_location=device, weights_only=False))
        rows = _dynamics_metrics(model, store, config, device)
        print("== fragment dynamics: 8 quantities ==")
        for row in rows:
            print(
                f"{row['variant']:8s} {row['name']:9s} n={row['n']:8d} "
                f"MAE={row['mae']:.4e} RMSE={row['rmse']:.4e} R2={row['r2']:.4f}"
            )
    else:
        print(f"dynamics checkpoint not found: {dynamics_path}")


if __name__ == "__main__":
    main()
