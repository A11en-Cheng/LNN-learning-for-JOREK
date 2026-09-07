"""Scale-separated baseline training: bin windows by dt magnitude and train per bin."""

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
from src.training.trainer import JorekTrainer
from tests.train_baseline import (
    load_condition_stats,
    load_source_stats,
    load_stats,
    normalize_graph,
)
from tests.train_baseline_diagnostics import _load_cached, _preload_times


def _bin_of(dt):
    return int(math.floor(math.log10(max(dt, 1e-12))))


def _collect_windows(run_frames, times_by_run, window, cache):
    windows = []
    for run, frames in run_frames.items():
        times = times_by_run[run]
        for start in range(len(frames) - window + 1):
            dts = [
                times[frames[start + i + 1].index] - times[frames[start + i].index]
                for i in range(window - 1)
            ]
            if all(dt > 0 for dt in dts):
                windows.append((run, start, sum(dts) / (window - 1)))
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=".scratch/model502-baseline/manifest.json")
    parser.add_argument("--bin", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache", default="/tmp/ehl2_cache")
    parser.add_argument("--output", default="output/model502_baseline/baseline_scale")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    train_runs = [r["name"] for r in manifest["runs"] if r["split"] == "train"]
    run_frames = {n: aligned_frames(Path("data/EHL-2") / n) for n in train_runs}
    times_by_run = _preload_times(run_frames, args.cache)
    windows = _collect_windows(run_frames, times_by_run, args.window, args.cache)
    bin_windows = [(r, s, dt) for r, s, dt in windows if _bin_of(dt) == args.bin]
    if not bin_windows:
        raise SystemExit(f"no windows in bin {args.bin}")
    dt_ref = float(
        sorted(dt for _, _, dt in bin_windows)[len(bin_windows) // 2]
    )
    print(f"bin {args.bin}: {len(bin_windows)} windows, dt_ref={dt_ref:.3f}")

    sm, ss = load_stats(args.manifest)
    cm, cs = load_condition_stats(args.manifest)
    um, us = load_source_stats(args.manifest)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    encoder = JorekGNNEncoder(
        input_dim=9, hidden_dim=64, latent_dim=32, num_layers=4, edge_dim=2,
        checkpoint=True, condition_dim=cm.numel(),
    ).to(device)
    processor = JorekLNNProcessor(latent_dim=32, hidden_size=64).to(device)
    decoder = JorekGNNDecoder(
        latent_dim=32, hidden_dim=64, output_dim=9, num_layers=4, edge_dim=2, checkpoint=True
    ).to(device)
    trainer = JorekTrainer(
        encoder, processor, decoder, None, JorekLoss(lambda_reg=1.0, lambda_cls=0.05),
        lr=args.lr, clip_value=1.0, dt_ref=dt_ref,
    )
    losses = []
    for _ in range(args.steps):
        run, start, _ = rng.choice(bin_windows)
        frames = run_frames[run]
        graphs = [
            normalize_graph(
                _load_cached(run, frames[start + off].index, args.cache),
                sm, ss, cm, cs, um, us,
            )
            for off in range(args.window)
        ]
        batches = [Batch.from_data_list([g]).to(device) for g in graphs]
        loss, _ = trainer.train_step_multihorizon(
            batches, torch.zeros(1, 1, device=device), horizons=args.horizons
        )
        losses.append(loss)
    out = Path(args.output) / f"bin{args.bin}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "total_loss.csv").write_text("\n".join(f"{i},{v}" for i, v in enumerate(losses)))
    torch.save(
        {
            "encoder": encoder.state_dict(), "processor": processor.state_dict(),
            "decoder": decoder.state_dict(),
            "stats": {"mean": sm.tolist(), "std": ss.tolist()},
            "condition_stats": {"mean": cm.tolist(), "std": cs.tolist()},
            "source_stats": {"mean": um.tolist(), "std": us.tolist()},
            "meta": {"bin": args.bin, "dt_ref": dt_ref},
        },
        out / "checkpoint.pt",
    )
    print("done", out)


if __name__ == "__main__":
    main()

