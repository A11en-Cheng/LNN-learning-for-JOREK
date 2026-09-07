"""Long cross-scenario baseline training with periodic fixed-val evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames
from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.loss import JorekLoss
from src.physics.model502_physics import model502_residuals
from src.training.trainer import JorekTrainer
from tests.train_baseline import (
    load_condition_stats,
    load_source_stats,
    load_stats,
    normalize_graph,
)
from tests.train_baseline_diagnostics import _load_cached, _preload_times


def _compute_channel_weights(train_windows, train_frames, args, n_sample=150):
    """Inverse-signal per-field weights + T_i/Z_j boost, clamped to [0.5, 3]."""
    rng = random.Random(7)
    sm, ss = load_stats(args.manifest)
    cm, cs = load_condition_stats(args.manifest)
    um, us = load_source_stats(args.manifest)
    sig = torch.zeros(9)
    n = 0
    for run, start in rng.sample(train_windows, min(n_sample, len(train_windows))):
        frames = train_frames[run]
        g0 = normalize_graph(_load_cached(run, frames[start].index, args.cache), sm, ss, cm, cs, um, us)
        g1 = normalize_graph(_load_cached(run, frames[start + 1].index, args.cache), sm, ss, cm, cs, um, us)
        sig += (g1.x - g0.x).square().mean(dim=0)
        n += 1
    sig = (sig / n).sqrt()
    inv = 1.0 / (sig + 1e-6)
    inv = inv / inv.mean()
    boost = torch.ones(9)
    boost[5] *= 1.5  # T_i
    weights = (inv * boost).clamp(0.5, 3.0)
    weights[2] = 0.0  # Z_j derived from psi
    weights[3] = 0.0  # omega derived from u
    return weights


def physics_train_step(trainer, encoder, processor, decoder, criterion, batches, horizons, device, lambda_phys):
    from src.models.loss import compute_spatial_weights
    trainer.optimizer.zero_grad()
    total = torch.zeros((), device=device)
    state = None
    pair_count = 0
    for t, batch in enumerate(batches):
        for k in horizons:
            if t + k >= len(batches):
                continue
            target = batches[t + k]
            dt = float((target.time[0] - batch.time[0]).item())
            latent = encoder(batch.x, batch.edge_index, batch.pos, condition=batch.condition, batch=batch.batch)
            latent, state = processor(latent, batch.batch, state=state, time_steps=dt / trainer.dt_ref, controls=getattr(batch, "source", None))
            delta, p = decoder(latent, batch.edge_index, batch.batch, batch.pos)
            pred = batch.x + delta * (dt / trainer.dt_ref)
            w = compute_spatial_weights(batch.x[:, 0], batch.batch, 10.0, 5.0)
            loss, _ = criterion(pred, target.x, p, torch.zeros(1, 1, device=device), node_weights=w)
            phys = sum(model502_residuals(pred, batch.x, batch, dt).values())
            total = total + loss + lambda_phys * phys
            pair_count += 1
    total = total / max(pair_count, 1)
    total.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), trainer.clip_value)
    torch.nn.utils.clip_grad_norm_(processor.parameters(), trainer.clip_value)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), trainer.clip_value)
    trainer.optimizer.step()
    return float(total.item())


def _monotonic_windows(run_frames, times_by_run, window):
    windows = []
    for run, frames in run_frames.items():
        times = times_by_run[run]
        for start in range(len(frames) - window + 1):
            dts = [
                times[frames[start + i + 1].index] - times[frames[start + i].index]
                for i in range(window - 1)
            ]
            if all(dt > 0 for dt in dts):
                windows.append((run, start))
    return windows


def _eval_loss(windows, run_frames, models, stats, args, device, sample=40):
    encoder, processor, decoder = models
    sm, ss, cm, cs, um, us = stats
    rng = random.Random(999)
    picked = rng.sample(windows, min(sample, len(windows)))
    losses, by_run = [], {}
    encoder.eval(); processor.eval(); decoder.eval()
    with torch.no_grad():
        for run, start in picked:
            frames = run_frames[run]
            graphs = [
                normalize_graph(_load_cached(run, frames[start + off].index, args.cache), sm, ss, cm, cs, um, us)
                for off in range(args.window)
            ]
            batches = [Batch.from_data_list([g]).to(device) for g in graphs]
            loss, _, channels = trainer_step(
                encoder, processor, decoder, batches, device, args, return_channels=True
            )
            losses.append(loss)
            by_run.setdefault(run, []).append(loss)
    encoder.train(); processor.train(); decoder.train()
    return sum(losses) / len(losses), {k: sum(v) / len(v) for k, v in by_run.items()}


def trainer_step(encoder, processor, decoder, batches, device, args, return_channels=False):
    from src.models.loss import compute_spatial_weights
    criterion = JorekLoss(lambda_reg=1.0, lambda_cls=0.05, channel_weights=args.channel_weights)
    total = torch.zeros((), device=device)
    channel_sums = {}
    pair_count = 0
    state = None
    for t, batch in enumerate(batches):
        for k in args.horizons:
            if t + k >= len(batches):
                continue
            target = batches[t + k]
            dt = float((target.time[0] - batch.time[0]).item())
            latent = encoder(batch.x, batch.edge_index, batch.pos, condition=batch.condition, batch=batch.batch)
            latent, state = processor(latent, batch.batch, state=state, time_steps=dt / args.dt_ref, controls=getattr(batch, "source", None))
            delta, p = decoder(latent, batch.edge_index, batch.batch, batch.pos)
            pred = batch.x + delta * (dt / args.dt_ref)
            w = compute_spatial_weights(batch.x[:, 0], batch.batch, 10.0, 5.0)
            loss, diag = criterion(pred, target.x, p, torch.zeros(1, 1, device=device), node_weights=w)
            total = total + loss
            pair_count += 1
            for key, value in diag.items():
                if isinstance(key, str) and key.startswith("loss_") and key not in {"loss_reg", "loss_cls", "loss_total"}:
                    channel_sums[key] = channel_sums.get(key, 0.0) + value
    if pair_count == 0:
        raise ValueError("empty window")
    if return_channels:
        return float(total.item()) / pair_count, pair_count, {k: v / pair_count for k, v in channel_sums.items()}
    return float(total.item()) / pair_count, pair_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=".scratch/model502-baseline/manifest.json")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-phys", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", default="/tmp/ehl2_cache")
    parser.add_argument("--output", default="output/model502_baseline/baseline_long")
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    train_runs = [r["name"] for r in manifest["runs"] if r["split"] == "train"]
    val_runs = [r["name"] for r in manifest["runs"] if r["split"] == "val"]
    train_frames = {n: aligned_frames(Path("data/EHL-2") / n) for n in train_runs}
    val_frames = {n: aligned_frames(Path("data/EHL-2") / n) for n in val_runs}
    train_times = _preload_times(train_frames, args.cache)
    val_times = _preload_times(val_frames, args.cache)
    train_windows = _monotonic_windows(train_frames, train_times, args.window)
    val_windows = _monotonic_windows(val_frames, val_times, args.window)
    all_dts = [
        train_times[r][train_frames[r][s + i + 1].index] - train_times[r][train_frames[r][s + i].index]
        for r, s in train_windows for i in range(args.window - 1)
    ]
    args.dt_ref = float(sorted(all_dts)[len(all_dts) // 2])
    args.channel_weights = _compute_channel_weights(train_windows, train_frames, args)
    print(f"windows train={len(train_windows)} val={len(val_windows)} dt_ref={args.dt_ref:.3f}")
    print("channel_weights:", [round(float(w), 2) for w in args.channel_weights])

    sm, ss = load_stats(args.manifest)
    cm, cs = load_condition_stats(args.manifest)
    um, us = load_source_stats(args.manifest)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = JorekGNNEncoder(input_dim=9, hidden_dim=64, latent_dim=32, num_layers=4, edge_dim=2, checkpoint=True, condition_dim=cm.numel()).to(device)
    processor = JorekLNNProcessor(latent_dim=32, hidden_size=64).to(device)
    decoder = JorekGNNDecoder(latent_dim=32, hidden_dim=64, output_dim=9, num_layers=4, edge_dim=2, checkpoint=True).to(device)
    trainer = JorekTrainer(
        encoder, processor, decoder, None,
        JorekLoss(lambda_reg=1.0, lambda_cls=0.05, channel_weights=args.channel_weights),
        lr=args.lr, clip_value=1.0, dt_ref=args.dt_ref,
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train_log, val_log = [], []
    stats = (sm, ss, cm, cs, um, us)
    for step in range(args.steps):
        run, start = rng.choice(train_windows)
        frames = train_frames[run]
        graphs = [normalize_graph(_load_cached(run, frames[start + off].index, args.cache), sm, ss, cm, cs, um, us) for off in range(args.window)]
        batches = [Batch.from_data_list([g]).to(device) for g in graphs]
        loss = physics_train_step(
            trainer, encoder, processor, decoder, trainer.criterion,
            batches, args.horizons, device, args.lambda_phys,
        )
        train_log.append(loss)
        if (step + 1) % args.eval_every == 0:
            vloss, by_run = _eval_loss(val_windows, val_frames, (encoder, processor, decoder), stats, args, device)
            val_log.append((step + 1, vloss, by_run))
            print(f"step {step+1} train {loss:.2e} val {vloss:.2e}", flush=True)
        if (step + 1) % args.save_every == 0:
            (out / "total_loss.csv").write_text("\n".join(f"{i},{v}" for i, v in enumerate(train_log)))
            (out / "val_loss.json").write_text(json.dumps(val_log, indent=2))
            torch.save({
                "encoder": encoder.state_dict(), "processor": processor.state_dict(), "decoder": decoder.state_dict(),
                "stats": {"mean": sm.tolist(), "std": ss.tolist()},
                "condition_stats": {"mean": cm.tolist(), "std": cs.tolist()},
                "source_stats": {"mean": um.tolist(), "std": us.tolist()},
                "meta": {"step": step + 1, "dt_ref": args.dt_ref},
            }, out / "checkpoint.pt")
    (out / "total_loss.csv").write_text("\n".join(f"{i},{v}" for i, v in enumerate(train_log)))
    (out / "val_loss.json").write_text(json.dumps(val_log, indent=2))
    torch.save({"encoder": encoder.state_dict(), "processor": processor.state_dict(), "decoder": decoder.state_dict(), "meta": {"step": args.steps, "dt_ref": args.dt_ref}}, out / "checkpoint.pt")
    print("done", out)


if __name__ == "__main__":
    main()
