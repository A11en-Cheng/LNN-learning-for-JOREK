"""Free-run rollout vs persistence benchmark for the v0 baseline (#0009)."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch_geometric.data import Batch, Data
import h5py

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames
from src.data.dataset import step_dt_jorek
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


def evaluate_rollout(
    *,
    checkpoint,
    run_name,
    warmup,
    horizon,
    data_root="data/EHL-2",
    device="cpu",
) -> dict:
    """Free-run the checkpoint against persistence on one held-out run."""
    device = torch.device(device)
    frames = aligned_frames(Path(data_root) / run_name)
    if len(frames) < warmup + horizon + 1:
        raise ValueError("run too short for the requested warmup/horizon")
    start = 0
    for candidate in range(len(frames) - warmup - horizon):
        times = []
        for offset in range(warmup + horizon + 1):
            with h5py.File(frames[candidate + offset].state_path, "r") as handle:
                times.append(float(handle["t_now"][()].item()))
        if all(times[index] < times[index + 1] for index in range(len(times) - 1)):
            start = candidate
            break
    else:
        raise RuntimeError(f"no monotonic segment found in run {run_name}")
    frames = frames[start:]
    mean = torch.tensor(checkpoint["stats"]["mean"], dtype=torch.float32, device=device)
    std = torch.tensor(checkpoint["stats"]["std"], dtype=torch.float32, device=device)

    encoder = JorekGNNEncoder(
        input_dim=9, hidden_dim=64, latent_dim=32, num_layers=4, edge_dim=2,
        condition_dim=len(checkpoint["condition_stats"]["mean"]),
    ).to(device)
    processor = JorekLNNProcessor(latent_dim=32, hidden_size=64, num_layers=1).to(device)
    decoder = JorekGNNDecoder(
        latent_dim=32, hidden_dim=64, output_dim=9, num_layers=4, edge_dim=2
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    processor.load_state_dict(checkpoint["processor"])
    decoder.load_state_dict(checkpoint["decoder"])
    encoder.eval(); processor.eval(); decoder.eval()

    norm_frames = []
    cond_mean = torch.tensor(checkpoint["condition_stats"]["mean"], dtype=torch.float32, device=device)
    cond_std = torch.tensor(checkpoint["condition_stats"]["std"], dtype=torch.float32, device=device)
    for index in range(warmup + horizon + 1):
        graph = JorekDataParser(str(frames[index].state_path)).parse_to_graph()
        condition = None
        if hasattr(graph, "condition"):
            condition = graph.condition.to(device)
            safe_std = torch.where(cond_std > 0, cond_std, torch.ones_like(cond_std))
            condition = (condition - cond_mean) / safe_std
        norm_frames.append(
            Batch.from_data_list(
                [Data(
                    x=standardize(transform_state(graph.x), mean, std, name="state"),
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr,
                    pos=graph.pos,
                    time=graph.time,
                    t_norm=graph.t_norm,
                    condition=condition,
                )]
            ).to(device)
        )

    def frame_dt(index: int) -> float:
        return float(
            step_dt_jorek(norm_frames[index], norm_frames[index + 1]).mean().item()
        )

    state = None
    with torch.no_grad():
        for index in range(warmup):
            batch = norm_frames[index]
            latent = encoder(
                batch.x, batch.edge_index, batch.pos,
                condition=getattr(batch, "condition", None), batch=batch.batch,
            )
            _, state = processor(latent, batch.batch, state=state, time_steps=frame_dt(index))

        model_rmse = {name: [] for name in FIELD_NAMES}
        persistence_rmse = {name: [] for name in FIELD_NAMES}
        drift = []
        current = norm_frames[warmup]
        initial_physical = inverse_transform_state(
            destandardize(current.x, mean, std, name="state")
        )
        mass_0 = float(torch.sum(initial_physical[:, 4] * current.pos[:, 0]).item())
        for h in range(horizon):
            target = norm_frames[warmup + h + 1]
            latent = encoder(
                current.x, current.edge_index, current.pos,
                condition=getattr(current, "condition", None), batch=current.batch,
            )
            next_latent, state = processor(
                latent, current.batch, state=state, time_steps=frame_dt(warmup + h)
            )
            delta, _ = decoder(next_latent, current.edge_index, current.batch, current.pos)
            dt_step = frame_dt(warmup + h)
            pred = current.x + delta * dt_step
            for field_index, name in enumerate(FIELD_NAMES):
                model_rmse[name].append(
                    float((pred[:, field_index] - target.x[:, field_index]).square().mean().sqrt().item())
                )
                persistence_rmse[name].append(
                    float(
                        (current.x[:, field_index] - target.x[:, field_index])
                        .square().mean().sqrt().item()
                    )
                )
            # physical mass drift of rho (R-weighted) from the warmup-end state
            rho_physical = inverse_transform_state(
                destandardize(pred, mean, std, name="state")
            )[:, 4]
            weights = current.pos[:, 0]  # R coordinate
            mass_pred = float(torch.sum(rho_physical * weights).item())
            drift.append((mass_pred - mass_0) / (mass_0 + 1e-12))
            current = Data(
                x=pred.clone(),
                edge_index=current.edge_index,
                edge_attr=current.edge_attr,
                pos=current.pos,
                batch=current.batch,
                time=target.time,
                t_norm=target.t_norm,
                condition=getattr(target, "condition", None),
            )

    mean_model = [sum(model_rmse[name]) / len(model_rmse[name]) for name in FIELD_NAMES]
    mean_persistence = [sum(persistence_rmse[name]) / len(persistence_rmse[name]) for name in FIELD_NAMES]
    all_values = [v for series in model_rmse.values() for v in series]
    no_nan = all(math.isfinite(value) for value in all_values)
    drift_threshold = 0.05  # pre-registered relative drift threshold (raw units)
    relative_drift = [float(d) for d in drift]
    improved_horizons = [
        h
        for h in range(horizon)
        if sum(model_rmse[name][h] for name in FIELD_NAMES)
        < sum(persistence_rmse[name][h] for name in FIELD_NAMES)
    ]
    return {
        "run": run_name,
        "warmup": warmup,
        "horizon": horizon,
        "mean_rmse_model": {name: value for name, value in zip(FIELD_NAMES, mean_model)},
        "mean_rmse_persistence": {name: value for name, value in zip(FIELD_NAMES, mean_persistence)},
        "model_beats_persistence_horizons": improved_horizons,
        "no_nan": no_nan,
        "drift_threshold_registered": drift_threshold,
        "relative_drift_series": relative_drift,
        "drift_within_threshold": all(abs(value) <= drift_threshold for value in relative_drift),
        "gate_passed": bool(improved_horizons) and no_nan,
    }


def main():
    parser = argparse.ArgumentParser(description="v0 baseline rollout benchmark (#0009)")
    parser.add_argument("--checkpoint", default="output/model502_baseline/baseline_v0/baseline_checkpoint.pt")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="output/model502_baseline/baseline_v0/benchmark.json")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    result = evaluate_rollout(
        checkpoint=checkpoint,
        run_name=args.run,
        warmup=args.warmup,
        horizon=args.horizon,
        device=args.device,
    )
    result["checkpoint"] = args.checkpoint
    result["dataset_id"] = checkpoint["meta"]["dataset_id"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print("GATE:", "PASS" if result["gate_passed"] else "FAIL")


if __name__ == "__main__":
    main()
