"""Longer baseline training with per-field loss logging and structural visuals."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch, Data

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames
from src.data.jorek_h5_parser import JorekDataParser
from src.data.model502_schema import (
    FIELD_NAMES,
    destandardize,
    inverse_transform_state,
    standardize,
    transform_state,
)
from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.loss import JorekLoss
from src.training.trainer import JorekTrainer
from tests.train_baseline import (
    _window_is_monotonic,
    _estimate_dt_ref_jorek,
    load_condition_stats,
    load_source_stats,
    load_stats,
    normalize_graph,
)


def _load_cached(run, index, cache_root):
    d = torch.load(f"{cache_root}/{run}/{index}.pt", map_location="cpu")
    return Data(
        x=d["x"], edge_index=d["edge_index"], edge_attr=d["edge_attr"], pos=d["pos"],
        time=d["time"], t_norm=d["t_norm"], condition=d["condition"], source=d["source"],
    )


def _window_monotonic_cached(frames, start, window, run, cache_root):
    times = [
        _load_cached(run, frames[start + off].index, cache_root).time.item()
        for off in range(window)
    ]
    return all(times[i] < times[i + 1] for i in range(window - 1))


def _preload_times(run_frames, cache_root):
    times_by_run = {}
    for name, frames in run_frames.items():
        times = {}
        for frame in frames:
            d = torch.load(f"{cache_root}/{name}/{frame.index}.pt", map_location="cpu")
            times[frame.index] = d["time"].item()
        times_by_run[name] = times
    return times_by_run


def _train(args):
    manifest = json.loads(Path(args.manifest).read_text())
    train_runs = [r["name"] for r in manifest["runs"] if r["split"] == "train"]
    run_frames = {name: aligned_frames(Path("data/EHL-2") / name) for name in train_runs}
    times_by_run = _preload_times(run_frames, args.cache)
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
    dt_ref = _estimate_dt_ref_jorek(run_frames, train_runs, args.seed)
    trainer = JorekTrainer(
        encoder, processor, decoder, None, JorekLoss(lambda_reg=1.0, lambda_cls=0.05),
        lr=args.lr, clip_value=1.0, dt_ref=dt_ref,
    )

    total_log, channel_log = [], {f"loss_{name}": [] for name in FIELD_NAMES}
    for step in range(args.steps):
        name = rng.choice(train_runs)
        frames = run_frames[name]
        for _ in range(20):
            start = rng.randrange(len(frames) - args.window + 1)
            if all(
                times_by_run[name][frames[start + i].index]
                < times_by_run[name][frames[start + i + 1].index]
                for i in range(args.window - 1)
            ):
                break
        graphs = [
            normalize_graph(
                _load_cached(name, frames[start + off].index, args.cache),
                sm, ss, cm, cs, um, us,
            )
            for off in range(args.window)
        ]
        batches = [Batch.from_data_list([g]).to(device) for g in graphs]
        loss, count, channels = trainer.train_step_multihorizon(
            batches, torch.zeros(1, 1, device=device), horizons=args.horizons,
            return_channels=True,
        )
        total_log.append(loss)
        for key in channel_log:
            channel_log[key].append(channels.get(key, float("nan")))

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "total_loss.csv").write_text("\n".join(f"{i},{v}" for i, v in enumerate(total_log)))
    (out / "channel_loss.csv").write_text(
        ",".join(["step"] + list(channel_log)) + "\n"
        + "\n".join(
            ",".join([str(i)] + [str(channel_log[k][i]) for k in channel_log])
            for i in range(len(total_log))
        )
    )
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "processor": processor.state_dict(),
            "decoder": decoder.state_dict(),
            "stats": {"mean": sm.tolist(), "std": ss.tolist()},
            "condition_stats": {"mean": cm.tolist(), "std": cs.tolist()},
            "source_stats": {"mean": um.tolist(), "std": us.tolist()},
        },
        out / "checkpoint.pt",
    )
    return encoder, processor, decoder, sm, ss, cm, cs, um, us, dt_ref


def _structural_visual(models, stats, val_run, val_index, output, dt_ref, cache_root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    encoder, processor, decoder = models
    sm, ss, cm, cs, um, us = stats
    device = next(encoder.parameters()).device
    frames = aligned_frames(Path("data/EHL-2") / val_run)
    graph = _load_cached(val_run, frames[val_index].index, cache_root)
    next_graph = _load_cached(val_run, frames[val_index + 1].index, cache_root)
    dt_jorek = float((next_graph.time[0] - graph.time[0]).item())
    cond = ((graph.condition - cm) / torch.where(cs > 0, cs, torch.ones_like(cs))).to(device)
    source = ((graph.source - um) / torch.where(us > 0, us, torch.ones_like(us))).to(device)
    x_norm = standardize(transform_state(graph.x), sm, ss, name="state")
    batch = Batch.from_data_list([__import__("torch_geometric.data", fromlist=["Data"]).Data(
        x=x_norm.to(device), edge_index=graph.edge_index.to(device), edge_attr=graph.edge_attr.to(device),
        pos=graph.pos.to(device), time=graph.time.to(device), t_norm=graph.t_norm.to(device),
        condition=cond, source=source,
    )])
    with torch.no_grad():
        latent = encoder(batch.x, batch.edge_index, batch.pos, condition=batch.condition, batch=batch.batch)
        latent, state = processor(latent, batch.batch, state=None, time_steps=dt_jorek / dt_ref)
        delta, _ = decoder(latent, batch.edge_index, batch.batch, batch.pos)
    pred = inverse_transform_state(destandardize(x_norm + delta * (dt_jorek / dt_ref), sm, ss, name="s")).cpu()
    truth = inverse_transform_state(destandardize(x_norm, sm, ss, name="s")).cpu()
    pos = graph.pos.numpy()
    fields = [("psi", 0), ("rho", 4), ("Te", 8)]
    fig, axs = plt.subplots(3, 3, figsize=(13, 11))
    for row, (name, idx) in enumerate(fields):
        p = pred[:, idx].numpy(); t = truth[:, idx].numpy(); e = np.abs(p - t)
        for col, (data, title, cmap) in enumerate(
            [(t, "GT", "viridis"), (p, "pred", "viridis"), (e, "|err|", "inferno")]
        ):
            ax = axs[row, col]
            triangles = getattr(graph, "triangles", None)
            if triangles is not None and triangles.numel():
                tri = mtri.Triangulation(pos[:, 0], pos[:, 1], triangles.numpy())
                im = ax.tricontourf(tri, data, 40, cmap=cmap)
            else:
                im = ax.scatter(pos[:, 0], pos[:, 1], c=data, s=1, cmap=cmap)
            ax.set_title(f"{name} {title}")
            fig.colorbar(im, ax=ax)
    fig.suptitle(f"structural comparison @ {val_run}:{val_index}")
    fig.tight_layout()
    fig.savefig(output, dpi=100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=".scratch/model502-baseline/manifest.json")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="output/model502_baseline/baseline_diag")
    parser.add_argument("--val-run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--val-index", type=int, default=5)
    parser.add_argument("--cache", default="/tmp/ehl2_cache")
    args = parser.parse_args()
    encoder, processor, decoder, sm, ss, cm, cs, um, us, dt_ref = _train(args)
    _structural_visual(
        (encoder, processor, decoder),
        (sm, ss, cm, cs, um, us),
        args.val_run, args.val_index,
        Path(args.output) / "structural.png",
        dt_ref,
        args.cache,
    )
    print("done", args.output)


if __name__ == "__main__":
    main()
