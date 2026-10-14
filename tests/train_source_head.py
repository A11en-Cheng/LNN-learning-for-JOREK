# [v1 LEGACY — 2026-10-04] Trains the source head with same-index
# (state, shard, projection) triples, which ADR-0019 shows is misaligned.
# Rebuild the dataset pairing before any reuse.
"""Phase-2.2 trainer: fragment-to-projection source head (analytic + learned).

Gate (ADR-0011 / roadmap Phase 2.2): on a held-out run, the hybrid head must
beat the analytic baseline per projection channel (relative L2), with the
ablation-rate comparison reported alongside.

Example:

    .torch-cuda-env/bin/python tests/train_source_head.py \
        --held-out scan_1d22_ang15_5_massloss_neon_v400 \
        --epochs 3 --frames-per-epoch 200 \
        --cache-dir .scratch/model502-interp/frames \
        --out output/source_head/v400
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import RunFrameStore
from src.models.source_head import SourceFieldHead
from src.physics.spi_model import SHARD_ABL

NUM_STATE_FIELDS = 9
NUM_SOURCE_FIELDS = 5


def _source_config(args) -> dict:
    return {
        "central_density": args.central_density,
        "magnetic_field": abs(args.f0 / args.r_geo),
        "magnetic_reduction": False,
    }


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


@torch.no_grad()
def _calibrate_channel_scale(
    model: SourceFieldHead,
    stores: dict[str, RunFrameStore],
    samples: list[tuple[str, int]],
    config: dict,
    device: torch.device,
    *,
    max_samples: int = 100,
) -> torch.Tensor:
    """Least-squares scale from the analytic deposit field to each projection channel."""
    model.eval()
    numerator = torch.zeros(NUM_SOURCE_FIELDS, dtype=torch.float64)
    denominator = torch.zeros(NUM_SOURCE_FIELDS, dtype=torch.float64)
    for run, position in samples[:max_samples]:
        u_raw, shards, mask, target = _batch(stores, [(run, position)], device)
        out = model(u_raw, shards, mask, stores[run].topology.pos.to(device), **config)
        deposit = out["deposit"][0].double().cpu()
        target_cpu = target[0].double().cpu()
        numerator += (deposit.unsqueeze(-1) * target_cpu).sum(dim=0)
        denominator += deposit.square().sum()
    model.train()
    return numerator / denominator.clamp_min(1e-300)


def _load_stores(data_root: Path, runs: list[str], cache_dir: Path | None) -> dict[str, RunFrameStore]:
    stores = {}
    for run in runs:
        stores[run] = RunFrameStore(data_root / run, cache_dir=cache_dir).load()
    return stores


def _sample_index(stores: dict[str, RunFrameStore], runs: list[str]) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    for run in runs:
        store = stores[run]
        for position in range(len(store)):
            if store.shard_values[position] is None:
                continue
            if not bool(store.source_available[position]):
                continue
            samples.append((run, position))
    return samples


def _compute_stats(
    stores: dict[str, RunFrameStore],
    samples: list[tuple[str, int]],
) -> dict:
    state_sum = np.zeros(NUM_STATE_FIELDS, dtype=np.float64)
    state_sumsq = np.zeros(NUM_STATE_FIELDS, dtype=np.float64)
    state_count = 0
    shard_sum = np.zeros(16, dtype=np.float64)
    shard_sumsq = np.zeros(16, dtype=np.float64)
    shard_count = 0
    source_sum = np.zeros(NUM_SOURCE_FIELDS, dtype=np.float64)
    source_sumsq = np.zeros(NUM_SOURCE_FIELDS, dtype=np.float64)
    source_count = 0
    for run, position in samples:
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
        source = store.source[position].double().numpy()
        source_sum += source.sum(axis=0)
        source_sumsq += np.square(source).sum(axis=0)
        source_count += source.shape[0]
    state_mean = state_sum / max(state_count, 1)
    state_std = np.sqrt(np.clip(state_sumsq / max(state_count, 1) - state_mean**2, 0.0, None))
    shard_mean = shard_sum / max(shard_count, 1)
    shard_std = np.sqrt(np.clip(shard_sumsq / max(shard_count, 1) - shard_mean**2, 0.0, None))
    source_mean = source_sum / max(source_count, 1)
    source_std = np.sqrt(np.clip(source_sumsq / max(source_count, 1) - source_mean**2, 0.0, None))
    return {
        "state_mean": state_mean,
        "state_std": np.where(state_std > 0, state_std, 1.0),
        "shard_mean": shard_mean,
        "shard_std": np.where(shard_std > 0, shard_std, 1.0),
        "source_mean": source_mean,
        "source_std": np.where(source_std > 0, source_std, 1.0),
        "counts": {
            "state_values": int(state_count),
            "shard_values": int(shard_count),
            "source_values": int(source_count),
        },
    }


def _batch(
    stores: dict[str, RunFrameStore],
    samples: list[tuple[str, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    shard_sets = [stores[run].shard_values[position] for run, position in samples]
    masks = [stores[run].shard_alive[position] for run, position in samples]
    max_shards = max(int(values.shape[0]) for values in shard_sets)
    feature_dim = int(shard_sets[0].shape[1])
    shard_values = torch.zeros(len(samples), max_shards, feature_dim)
    shard_mask = torch.zeros(len(samples), max_shards, dtype=torch.bool)
    for row, (values, mask) in enumerate(zip(shard_sets, masks)):
        n = int(values.shape[0])
        shard_values[row, :n] = torch.nan_to_num(values, nan=0.0)
        shard_mask[row, :n] = mask
    u_raw = torch.stack([stores[run].u[position] for run, position in samples])
    target = torch.stack([stores[run].source[position] for run, position in samples])
    return (
        u_raw.to(device),
        shard_values.to(device),
        shard_mask.to(device),
        target.to(device),
    )


@torch.no_grad()
def _evaluate(
    model: SourceFieldHead,
    stores: dict[str, RunFrameStore],
    run: str,
    config: dict,
    device: torch.device,
    max_frames: int | None = None,
) -> dict:
    model.eval()
    store = stores[run]
    positions = [
        position
        for position in range(len(store))
        if store.shard_values[position] is not None and bool(store.source_available[position])
    ]
    if max_frames is not None:
        positions = positions[:max_frames]
    baseline_sq = torch.zeros(NUM_SOURCE_FIELDS, dtype=torch.float64)
    hybrid_sq = torch.zeros(NUM_SOURCE_FIELDS, dtype=torch.float64)
    target_sq = torch.zeros(NUM_SOURCE_FIELDS, dtype=torch.float64)
    analytic_all: list[torch.Tensor] = []
    recorded_all: list[torch.Tensor] = []
    for position in positions:
        u_raw, shards, mask, target = _batch(stores, [(run, position)], device)
        out = model(u_raw, shards, mask, store.topology.pos.to(device), **config)
        diff_base = (out["baseline"] - target)[0].double().cpu()
        diff_hybrid = (out["source"] - target)[0].double().cpu()
        target_cpu = target[0].double().cpu()
        baseline_sq += diff_base.square().sum(dim=0)
        hybrid_sq += diff_hybrid.square().sum(dim=0)
        target_sq += target_cpu.square().sum(dim=0)
        alive = mask[0]
        if bool(alive.any()):
            analytic_all.append(out["ablation"][0][alive].double().cpu())
            recorded_all.append(shards[0][alive, SHARD_ABL].double().cpu())
    model.train()
    eps = 1e-30
    baseline_rel = torch.sqrt(baseline_sq / target_sq.clamp_min(eps)).numpy()
    hybrid_rel = torch.sqrt(hybrid_sq / target_sq.clamp_min(eps)).numpy()
    metrics = {
        "run": run,
        "frames": len(positions),
        "baseline_rel_l2": baseline_rel.tolist(),
        "hybrid_rel_l2": hybrid_rel.tolist(),
    }
    if analytic_all:
        analytic = torch.cat(analytic_all).numpy()
        recorded = torch.cat(recorded_all).numpy()
        positive = (analytic > 0) & (recorded > 0)
        ratio = np.divide(analytic, recorded, out=np.zeros_like(analytic), where=recorded > 0)
        metrics["ablation"] = {
            "n_shards": int(analytic.size),
            "median_ratio": float(np.median(ratio[recorded > 0])) if (recorded > 0).any() else None,
            "log_corr": float(np.corrcoef(np.log(analytic[positive]), np.log(recorded[positive]))[0, 1])
            if int(positive.sum()) > 2
            else None,
            "within_2x": float(np.mean((ratio > 0.5) & (ratio < 2.0))),
        }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--manifest", default=".scratch/model502-interp/manifest.json")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--frames-per-epoch", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--out", default="output/source_head/run")
    parser.add_argument("--central-density", type=float, default=0.8)
    parser.add_argument("--f0", type=float, default=-3.308745132)
    parser.add_argument("--r-geo", type=float, default=1.11)
    parser.add_argument("--device", default="cpu")
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

    print(f"held_out={args.held_out} train_runs={len(train_runs)} cache={cache_dir}")
    stores = _load_stores(Path(args.data_root), train_runs + [args.held_out], cache_dir)
    samples = _sample_index(stores, train_runs)
    print(f"training frames with shards+source: {len(samples)}")
    stats = _compute_stats(stores, samples)
    config = _source_config(args)

    model = SourceFieldHead(
        shard_dim=16, hidden=args.hidden, embed_dim=args.embed_dim, k_neighbours=args.k
    ).to(device)
    model.set_state_stats(
        torch.tensor(stats["state_mean"], dtype=torch.float32),
        torch.tensor(stats["state_std"], dtype=torch.float32),
    )
    model.set_shard_stats(
        torch.tensor(stats["shard_mean"], dtype=torch.float32),
        torch.tensor(stats["shard_std"], dtype=torch.float32),
    )
    source_std = torch.tensor(stats["source_std"], dtype=torch.float32, device=device).clamp_min(1e-12)
    channel_scale = _calibrate_channel_scale(model, stores, samples, config, device)
    model.set_channel_scale(channel_scale)
    model.set_source_scale(source_std.cpu())
    print("channel scale (deposit -> projection):", [f"{value:.3e}" for value in channel_scale.tolist()])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best = None
    started = time.time()
    rng = random.Random(args.seed)
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(samples)
        subset = samples[: args.frames_per_epoch]
        epoch_loss = 0.0
        steps = 0
        for start in range(0, len(subset), args.batch_size):
            chunk = subset[start : start + args.batch_size]
            u_raw, shards, mask, target = _batch(stores, chunk, device)
            out = model(u_raw, shards, mask, stores[chunk[0][0]].topology.pos.to(device), **config)
            normalized = (out["source"] - target) / source_std.view(1, 1, -1)
            loss = torch.nn.functional.smooth_l1_loss(
                normalized, torch.zeros_like(normalized), beta=1.0
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
            steps += 1
        metrics = _evaluate(model, stores, args.held_out, config, device)
        mean_baseline = float(np.mean(metrics["baseline_rel_l2"]))
        mean_hybrid = float(np.mean(metrics["hybrid_rel_l2"]))
        record = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(steps, 1),
            "held_out": metrics,
            "mean_baseline_rel_l2": mean_baseline,
            "mean_hybrid_rel_l2": mean_hybrid,
        }
        history.append(record)
        print(
            f"epoch {epoch}: train_loss={record['train_loss']:.4f} "
            f"held_out mean rel-L2 baseline={mean_baseline:.4f} hybrid={mean_hybrid:.4f}"
        )
        print("   baseline:", [round(value, 4) for value in metrics["baseline_rel_l2"]])
        print("   hybrid  :", [round(value, 4) for value in metrics["hybrid_rel_l2"]])
        if metrics.get("ablation"):
            print("   ablation:", metrics["ablation"])
        if best is None or mean_hybrid < best["mean_hybrid_rel_l2"]:
            best = record
            torch.save(model.state_dict(), out_dir / "checkpoint.pt")
    summary = _jsonable({
        "held_out": args.held_out,
        "train_runs": train_runs,
        "args": vars(args),
        "stats_counts": stats["counts"],
        "source_std": stats["source_std"],
        "channel_scale": channel_scale,
        "history": history,
        "best": best,
        "elapsed_seconds": time.time() - started,
    })
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"best hybrid mean rel-L2 = {best['mean_hybrid_rel_l2']:.4f} (baseline {best['mean_baseline_rel_l2']:.4f})")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
