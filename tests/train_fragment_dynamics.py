"""Phase-2.3 trainer: hybrid fragment dynamics (analytic backbone + correction).

Gate: on the held-out run, the hybrid must beat the analytic-only path on the
position / radius / ablation metrics, keep radii non-negative, and stay finite
over one recorded interval.

Example:

    .torch-cuda-env/bin/python tests/train_fragment_dynamics.py \
        --held-out scan_1d22_ang15_5_massloss_neon_v400 \
        --epochs 5 --pairs-per-epoch 300 \
        --cache-dir .scratch/model502-interp/frames \
        --out output/fragment_dynamics/v400
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import RunFrameStore
from src.models.fragment_dynamics import HybridFragmentDynamics
from src.physics.spi_model import SHARD_ABL, SHARD_RADIUS


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_stores(data_root: Path, runs: list[str], cache_dir: Path | None) -> dict:
    return {run: RunFrameStore(data_root / run, cache_dir=cache_dir).load() for run in runs}


def _pairs(stores: dict, runs: list[str]) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for run in runs:
        store = stores[run]
        for position in range(len(store) - 1):
            if store.shard_values[position] is None or store.shard_values[position + 1] is None:
                continue
            alive = store.shard_alive[position]
            if alive is None or not bool(alive.any()):
                continue
            pairs.append((run, position))
    return pairs


def _feature_stats(stores: dict, pairs: list[tuple[str, int]], max_pairs: int = 200) -> dict:
    state_sum = np.zeros(9, dtype=np.float64)
    state_sumsq = np.zeros(9, dtype=np.float64)
    state_count = 0
    shard_sum = np.zeros(16, dtype=np.float64)
    shard_sumsq = np.zeros(16, dtype=np.float64)
    shard_count = 0
    for run, position in pairs[:max_pairs]:
        store = stores[run]
        u = store.u[position].double().numpy()
        state_sum += u.sum(axis=0)
        state_sumsq += np.square(u).sum(axis=0)
        state_count += u.shape[0]
        shards = store.shard_values[position]
        alive = store.shard_alive[position]
        if shards is not None and alive is not None and bool(alive.any()):
            values = np.nan_to_num(shards[alive].double().numpy(), nan=0.0)
            shard_sum += values.sum(axis=0)
            shard_sumsq += np.square(values).sum(axis=0)
            shard_count += values.shape[0]
    state_mean = state_sum / max(state_count, 1)
    state_std = np.sqrt(np.clip(state_sumsq / max(state_count, 1) - state_mean**2, 0.0, None))
    shard_mean = shard_sum / max(shard_count, 1)
    shard_std = np.sqrt(np.clip(shard_sumsq / max(shard_count, 1) - shard_mean**2, 0.0, None))
    return {
        "state_mean": state_mean,
        "state_std": np.where(state_std > 0, state_std, 1.0),
        "shard_mean": shard_mean,
        "shard_std": np.where(shard_std > 0, shard_std, 1.0),
    }


def _pad_shards(values: torch.Tensor, mask: torch.Tensor, max_shards: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    padded = torch.zeros(len(values), max_shards, dim)
    padded_mask = torch.zeros(len(values), max_shards, dtype=torch.bool)
    for row, (value, alive) in enumerate(zip(values, mask)):
        n = int(value.shape[0])
        padded[row, :n] = torch.nan_to_num(value, nan=0.0)
        padded_mask[row, :n] = alive
    return padded, padded_mask


def _batch(stores: dict, pairs: list[tuple[str, int]], device: torch.device) -> dict:
    current = [stores[run].shard_values[position] for run, position in pairs]
    current_mask = [stores[run].shard_alive[position] for run, position in pairs]
    target = [stores[run].shard_values[position + 1] for run, position in pairs]
    target_mask = [stores[run].shard_alive[position + 1] for run, position in pairs]
    max_shards = max(int(value.shape[0]) for value in current)
    shard_dim = int(current[0].shape[1])
    current_padded, current_mask_padded = _pad_shards(current, current_mask, max_shards, shard_dim)
    target_padded, target_mask_padded = _pad_shards(target, target_mask, max_shards, shard_dim)
    u_raw = torch.stack([stores[run].u[position] for run, position in pairs])
    dt = torch.tensor(
        [
            float(stores[run].time[position + 1] - stores[run].time[position])
            for run, position in pairs
        ]
    )
    t_norm = torch.tensor(
        [float(stores[run].t_norm[position]) for run, position in pairs]
    )
    ablation_target = []
    for run, position in pairs:
        cumulative = stores[run].ablation_cumulative
        if cumulative is None:
            ablation_target.append(float("nan"))
        else:
            ablation_target.append(
                float(cumulative[position + 1] - cumulative[position])
            )
    psi_axis = torch.tensor(
        [float(stores[run].psi_axis[position]) for run, position in pairs],
        dtype=torch.float64,
    )
    psi_bnd = torch.tensor(
        [float(stores[run].psi_bnd[position]) for run, position in pairs],
        dtype=torch.float64,
    )
    return {
        "u_raw": u_raw.to(device),
        "current": current_padded.to(device),
        "current_mask": current_mask_padded.to(device),
        "target": target_padded.to(device),
        "target_mask": target_mask_padded.to(device),
        "psi_axis": psi_axis.to(device),
        "psi_bnd": psi_bnd.to(device),
        "tables": [stores[run].nonlocal_table(position) for run, position in pairs],
        "dt": dt.to(device),
        "t_norm": t_norm.to(device),
        "ablation_target": torch.tensor(ablation_target, dtype=torch.float32, device=device),
        "run": pairs[0][0],
    }


def _scales(stores: dict, pairs: list[tuple[str, int]], max_pairs: int = 200) -> dict:
    displacements: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    radii: list[torch.Tensor] = []
    logs: list[torch.Tensor] = []
    atoms: list[torch.Tensor] = []
    for run, position in pairs[:max_pairs]:
        store = stores[run]
        current = store.shard_values[position]
        target = store.shard_values[position + 1]
        mask = store.shard_alive[position] & store.shard_alive[position + 1]
        if bool(mask.any()):
            displacements.append((target[mask, :3] - current[mask, :3]).abs())
            velocities.append(target[mask, 4:7].abs())
        alive = store.shard_alive[position]
        if bool(alive.any()):
            radii.append(target[alive, SHARD_RADIUS].abs())
        positive = store.shard_alive[position + 1] & (target[:, SHARD_ABL] > 0)
        if bool(positive.any()):
            logs.append(torch.log1p(target[positive, SHARD_ABL]))
        cumulative = store.ablation_cumulative
        if cumulative is not None:
            increment = float(cumulative[position + 1] - cumulative[position])
            if increment > 0:
                atoms.append(torch.tensor([float(np.log1p(increment))]))
    def _median(values, fallback, dim=None):
        if not values:
            return fallback
        stacked = torch.cat(values, dim=0)
        result = stacked.median(dim=0).values if dim else stacked.median()
        return result
    position = _median(displacements, torch.ones(3), dim=0)
    velocity = _median(velocities, torch.ones(3) * 100.0, dim=0)
    radius = _median(radii, torch.tensor(1e-4))
    log_ablation = _median(logs, torch.tensor(1.0))
    # Components with (near-)zero motion must not produce degenerate scales:
    # floor each scale at 5% of the largest component / quantity scale.
    def _floor(values: torch.Tensor, minimum: float) -> torch.Tensor:
        floor = max(float(values.max()), minimum) * 0.05
        return values.clamp_min(floor)
    return {
        "position": _floor(position, 1e-6),
        "velocity": _floor(velocity, 1e-3),
        "radius": _floor(radius.reshape(1), 1e-7).reshape(()),
        "log_ablation": _floor(log_ablation.reshape(1), 1e-3).reshape(()),
        "log_atoms": _floor(
            (torch.cat(atoms).median() if atoms else torch.tensor(1.0)).reshape(1), 1e-3
        ).reshape(()),
    }


def _wrap_phi(error: torch.Tensor) -> torch.Tensor:
    wrapped = error.clone()
    wrapped[..., 2] = torch.remainder(wrapped[..., 2] + math.pi, 2 * math.pi) - math.pi
    return wrapped


def _forward(model, batch, scales, config, device, *, use_correction: bool):
    use_nonlocal = bool(config.get("nonlocal", True))
    return model(
        batch["u_raw"],
        batch["current"],
        batch["current_mask"],
        config["pos"],
        dt_jorek=batch["dt"],
        t_norm=batch["t_norm"],
        central_density=config["central_density"],
        pellet_density=config["pellet_density"],
        magnetic_field=config["magnetic_field"],
        use_correction=use_correction,
        nonlocal_tables=batch["tables"] if use_nonlocal else None,
        psi_axis=batch["psi_axis"] if use_nonlocal else None,
        psi_bnd=batch["psi_bnd"] if use_nonlocal else None,
    )


def _loss(model, batch, scales, config, device) -> torch.Tensor:
    out = _forward(model, batch, scales, config, device, use_correction=True)
    predicted = out["next"]
    target = batch["target"]
    both = batch["current_mask"] & batch["target_mask"]
    alive = batch["current_mask"]
    position_error = _wrap_phi(predicted[..., :3] - target[..., :3]) / scales["position"]
    velocity_error = (predicted[..., 4:7] - target[..., 4:7]) / scales["velocity"]
    radius_error = (predicted[..., SHARD_RADIUS] - target[..., SHARD_RADIUS]) / scales["radius"]
    ablation_mask = batch["current_mask"] & (target[..., SHARD_ABL] > 0)
    ablation_error = (
        torch.log1p(predicted[..., SHARD_ABL].clamp_min(0.0))
        - torch.log1p(target[..., SHARD_ABL])
    ) / scales["log_ablation"]
    terms = []
    weights = []
    for piece, weight in (
        (position_error[both], 1.0),
        (velocity_error[both], 0.5),
        (radius_error[alive], 1.0),
        (ablation_error[ablation_mask], 2.0),
    ):
        if piece.numel() == 0:
            continue
        flattened = piece.reshape(-1)
        terms.append(
            weight
            * torch.nn.functional.smooth_l1_loss(
                flattened, torch.zeros_like(flattened), beta=1.0
            )
        )
        weights.append(weight)
    if not terms:
        return torch.zeros((), device=device)
    seconds = (batch["dt"] * batch["t_norm"]).view(-1, 1)
    predicted_atoms = (out["ablation"] * seconds).sum(dim=1)
    target_atoms = batch["ablation_target"]
    valid = torch.isfinite(target_atoms) & (target_atoms > 0)
    if bool(valid.any()):
        integral_error = (
            torch.log1p(predicted_atoms.clamp_min(0.0))
            - torch.log1p(target_atoms)
        )[valid] / scales["log_atoms"]
        terms.append(
            1.0
            * torch.nn.functional.smooth_l1_loss(
                integral_error, torch.zeros_like(integral_error), beta=1.0
            )
        )
        weights.append(1.0)
    return torch.stack(terms).sum() / sum(weights)


@torch.no_grad()
def _evaluate(model, stores, run, scales, config, device) -> dict:
    model.eval()
    store = stores[run]
    metrics = {}
    for label, use_correction in (("baseline", False), ("hybrid", True)):
        position_errors: list[torch.Tensor] = []
        velocity_errors: list[torch.Tensor] = []
        radius_errors: list[torch.Tensor] = []
        analytic_all: list[torch.Tensor] = []
        recorded_all: list[torch.Tensor] = []
        predicted_atoms_total = 0.0
        recorded_atoms_total = 0.0
        interval_errors: list[float] = []
        death_total = death_correct = 0
        for position in range(len(store) - 1):
            if store.shard_values[position] is None or store.shard_values[position + 1] is None:
                continue
            batch = _batch(stores, [(run, position)], device)
            out = _forward(model, batch, scales, config, device, use_correction=use_correction)
            predicted = out["next"]
            target = batch["target"]
            both = batch["current_mask"] & batch["target_mask"]
            alive = batch["current_mask"]
            if bool(both.any()):
                position_errors.append(
                    _wrap_phi(predicted[..., :3] - target[..., :3])[both].abs().cpu()
                )
                velocity_errors.append((predicted[..., 4:7] - target[..., 4:7])[both].abs().cpu())
            if bool(alive.any()):
                radius_errors.append(
                    (predicted[..., SHARD_RADIUS] - target[..., SHARD_RADIUS])[alive].abs().cpu()
                )
                predicted_dead = predicted[..., SHARD_RADIUS] <= 0
                target_dead = target[..., SHARD_RADIUS] <= 0
                death_total += int(alive.sum())
                death_correct += int((predicted_dead[alive] == target_dead[alive]).sum())
            positive = batch["current_mask"] & (target[..., SHARD_ABL] > 0)
            if bool(positive.any()):
                analytic_all.append(predicted[..., SHARD_ABL][positive].detach().cpu())
                recorded_all.append(target[..., SHARD_ABL][positive].detach().cpu())
            recorded_atoms = float(batch["ablation_target"][0])
            if np.isfinite(recorded_atoms) and recorded_atoms > 0:
                seconds = float(batch["dt"][0] * batch["t_norm"][0])
                predicted_atoms = float(out["ablation"][0].sum()) * seconds
                predicted_atoms_total += predicted_atoms
                recorded_atoms_total += recorded_atoms
                interval_errors.append(
                    abs(predicted_atoms - recorded_atoms) / recorded_atoms
                )
        position_error = torch.cat(position_errors).mean(dim=0) if position_errors else torch.zeros(3)
        velocity_error = torch.cat(velocity_errors).mean(dim=0) if velocity_errors else torch.zeros(3)
        radius_error = torch.cat(radius_errors).mean() if radius_errors else torch.tensor(0.0)
        entry = {
            "position_mae": position_error.tolist(),
            "velocity_mae": velocity_error.tolist(),
            "radius_mae": float(radius_error),
            "death_agreement": (death_correct / death_total) if death_total else None,
        }
        if analytic_all:
            predicted_all = torch.cat(analytic_all).double().numpy()
            recorded = torch.cat(recorded_all).double().numpy()
            positive = (predicted_all > 0) & (recorded > 0)
            ratio = np.divide(predicted_all, recorded, out=np.zeros_like(predicted_all), where=recorded > 0)
            entry["ablation"] = {
                "median_ratio": float(np.median(ratio[recorded > 0])),
                "log_corr": float(np.corrcoef(np.log(predicted_all[positive]), np.log(recorded[positive]))[0, 1])
                if int(positive.sum()) > 2
                else None,
                "within_2x": float(np.mean((ratio > 0.5) & (ratio < 2.0))),
            }
        if interval_errors:
            entry["ablation_atoms"] = {
                "n_intervals": len(interval_errors),
                "sum_relative_error": (predicted_atoms_total - recorded_atoms_total)
                / recorded_atoms_total,
                "median_interval_relative_error": float(np.median(interval_errors)),
            }
        metrics[label] = entry
    model.train()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--manifest", default=".scratch/model502-interp/manifest.json")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--pairs-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default="output/fragment_dynamics/run")
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
    parser.add_argument(
        "--analytic-only",
        action="store_true",
        help="skip training: evaluate and record the analytic backbone only (no learned correction)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(
        (record for record in manifest["folds"] if record["held_out_run"] == args.held_out),
        None,
    )
    if fold is None:
        raise SystemExit(f"{args.held_out} is not a run in the manifest")
    train_runs = list(fold["train_runs"])
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    stores = _load_stores(Path(args.data_root), train_runs + [args.held_out], cache_dir)
    pairs = _pairs(stores, train_runs)
    print(f"held_out={args.held_out} train_runs={len(train_runs)} pairs={len(pairs)}")
    scales = _scales(stores, pairs)
    scales = {key: value.to(device) for key, value in scales.items()}
    config = {
        "pos": stores[train_runs[0]].topology.pos.to(device),
        "central_density": args.central_density,
        "pellet_density": args.pellet_density,
        "magnetic_field": abs(args.f0 / args.r_geo),
        "nonlocal": not args.local_only,
    }
    model = HybridFragmentDynamics(hidden=args.hidden, k_neighbours=args.k).to(device)
    feature_stats = _feature_stats(stores, pairs)
    model.set_state_stats(
        torch.tensor(feature_stats["state_mean"], dtype=torch.float32),
        torch.tensor(feature_stats["state_std"], dtype=torch.float32),
    )
    model.set_shard_stats(
        torch.tensor(feature_stats["shard_mean"], dtype=torch.float32),
        torch.tensor(feature_stats["shard_std"], dtype=torch.float32),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.analytic_only:
        metrics = _evaluate(model, stores, args.held_out, scales, config, device)
        payload = {
            "held_out": args.held_out,
            "mode": "analytic-only",
            "config": {key: value for key, value in config.items() if key != "pos"},
            "held_out_metrics": metrics,
            "args": vars(args),
        }
        (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        base = metrics["baseline"]
        print(
            f"analytic-only: radius_mae={base['radius_mae']:.3e} "
            f"ablation median={base['ablation']['median_ratio']:.3f} "
            f"log-corr={base['ablation']['log_corr']:.3f} "
            f"within-2x={base['ablation']['within_2x']:.1%} "
            f"atoms sum error={base['ablation_atoms']['sum_relative_error']:+.1%}"
        )
        print(f"artifacts: {out_dir}")
        return

    rng = random.Random(args.seed)
    history = []
    best = None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(pairs)
        subset = pairs[: args.pairs_per_epoch]
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(subset), args.batch_size):
            chunk = subset[start : start + args.batch_size]
            batch = _batch(stores, chunk, device)
            loss = _loss(model, batch, scales, config, device)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            steps += 1
        metrics = _evaluate(model, stores, args.held_out, scales, config, device)
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(steps, 1),
            "held_out": metrics,
        }
        history.append(record)
        print(
            f"epoch {epoch}: loss={record['train_loss']:.4f} "
            f"baseline radius_mae={metrics['baseline']['radius_mae']:.3e} "
            f"hybrid radius_mae={metrics['hybrid']['radius_mae']:.3e}"
        )
        print("   baseline:", metrics["baseline"])
        print("   hybrid  :", metrics["hybrid"])
        score = metrics["hybrid"]["radius_mae"]
        if best is None or score < best["held_out"]["hybrid"]["radius_mae"]:
            best = record
            torch.save(model.state_dict(), out_dir / "checkpoint.pt")
    summary = _jsonable(
        {
            "held_out": args.held_out,
            "train_runs": train_runs,
            "args": vars(args),
            "scales": {key: value.tolist() for key, value in scales.items()},
            "history": history,
            "best": best,
            "elapsed_seconds": time.time() - started,
        }
    )
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"best epoch {best['epoch']}: hybrid radius_mae={best['held_out']['hybrid']['radius_mae']:.3e} "
          f"(baseline {best['held_out']['baseline']['radius_mae']:.3e})")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
