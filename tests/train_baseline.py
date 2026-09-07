"""Short train-split baseline for the #0009 rollout benchmark."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames
from src.data.jorek_h5_parser import JorekDataParser
from src.data.model502_schema import standardize, transform_state
from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.loss import JorekLoss
from src.training.trainer import JorekTrainer


def load_stats(manifest_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    manifest = json.loads(Path(manifest_path).read_text())
    fields = manifest["stats"]["fields"]
    from src.data.model502_schema import FIELD_NAMES

    mean = torch.tensor([fields[name]["mean"] for name in FIELD_NAMES], dtype=torch.float32)
    std = torch.tensor([fields[name]["std"] for name in FIELD_NAMES], dtype=torch.float32)
    return mean, std


def load_condition_stats(manifest_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    manifest = json.loads(Path(manifest_path).read_text())
    condition = manifest["stats"]["condition"]
    mean = torch.tensor(condition["mean"], dtype=torch.float32)
    std = torch.tensor(condition["std"], dtype=torch.float32)
    return mean, std


def normalize_graph(graph, mean, std, cond_mean=None, cond_std=None) -> Data:
    x_norm = standardize(transform_state(graph.x), mean, std, name="state")
    condition = None
    if hasattr(graph, "condition"):
        condition = graph.condition.clone()
        if cond_mean is not None:
            safe_std = torch.where(cond_std > 0, cond_std, torch.ones_like(cond_std))
            condition = (condition - cond_mean) / safe_std
    return Data(
        x=x_norm,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        pos=graph.pos,
        time=graph.time,
        t_norm=graph.t_norm,
        condition=condition,
    )


def _window_is_monotonic(frames, start: int, length: int) -> bool:
    times = []
    for offset in range(length):
        with h5py.File(frames[start + offset].state_path, "r") as handle:
            times.append(float(handle["t_now"][()].item()))
    return all(times[index] < times[index + 1] for index in range(length - 1))


def train_baseline(
    *,
    manifest_path,
    steps,
    window_len,
    horizons,
    lr,
    seed,
    device,
    max_runs,
    output_dir,
) -> dict:
    manifest = json.loads(Path(manifest_path).read_text())
    train_runs = [run["name"] for run in manifest["runs"] if run["split"] == "train"][:max_runs]
    mean, std = load_stats(manifest_path)
    cond_mean, cond_std = load_condition_stats(manifest_path)
    run_frames = {name: aligned_frames(Path("data/EHL-2") / name) for name in train_runs}
    rng = random.Random(seed)
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    encoder = JorekGNNEncoder(
        input_dim=9, hidden_dim=64, latent_dim=32, num_layers=4, edge_dim=2,
        checkpoint=True, condition_dim=cond_mean.numel(),
    ).to(device)
    processor = JorekLNNProcessor(latent_dim=32, hidden_size=64, num_layers=1).to(device)
    decoder = JorekGNNDecoder(
        latent_dim=32, hidden_dim=64, output_dim=9, num_layers=4, edge_dim=2, checkpoint=True
    ).to(device)
    criterion = JorekLoss(lambda_reg=1.0, lambda_cls=0.05)
    trainer = JorekTrainer(encoder, processor, decoder, None, criterion, lr=lr, clip_value=1.0)

    losses = []
    for step in range(steps):
        name = rng.choice(train_runs)
        frames = run_frames[name]
        for _ in range(20):
            start = rng.randrange(len(frames) - window_len + 1)
            if _window_is_monotonic(frames, start, window_len):
                break
        else:
            raise RuntimeError(f"no monotonic window found in run {name}")
        graphs = [
            normalize_graph(
                JorekDataParser(str(frames[start + offset].state_path)).parse_to_graph(),
                mean,
                std,
                cond_mean,
                cond_std,
            )
            for offset in range(window_len)
        ]
        batches = [Batch.from_data_list([graph]).to(device) for graph in graphs]
        loss, _ = trainer.train_step_multihorizon(
            batches, torch.zeros(1, 1, device=device), horizons=horizons
        )
        losses.append(loss)
        if not math.isfinite(loss):
            break

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "encoder": encoder.state_dict(),
        "processor": processor.state_dict(),
        "decoder": decoder.state_dict(),
        "stats": {"mean": mean.tolist(), "std": std.tolist()},
        "condition_stats": {"mean": cond_mean.tolist(), "std": cond_std.tolist()},
        "meta": {
            "seed": seed,
            "steps": len(losses),
            "window_len": window_len,
            "horizons": list(horizons),
            "dataset_id": manifest["dataset_id"],
            "train_runs": train_runs,
            "final_loss": float(losses[-1]) if losses else None,
        },
    }
    torch.save(checkpoint, output / "baseline_checkpoint.pt")
    (output / "train_loss.csv").write_text(
        "\n".join(f"{index},{value}" for index, value in enumerate(losses))
    )
    return checkpoint["meta"]


def main():
    parser = argparse.ArgumentParser(description="Train the #0009 v0 baseline")
    parser.add_argument("--manifest", default=".scratch/model502-baseline/manifest.json")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--window-len", type=int, default=5)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-runs", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="output/model502_baseline/baseline_v0")
    args = parser.parse_args()

    meta = train_baseline(
        manifest_path=args.manifest,
        steps=args.steps,
        window_len=args.window_len,
        horizons=tuple(args.horizons),
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        max_runs=args.max_runs,
        output_dir=args.output_dir,
    )
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
