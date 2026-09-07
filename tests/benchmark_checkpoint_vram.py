"""Peak-VRAM benchmark for the checkpointing gate (#0008)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.jorek_h5_parser import JorekDataParser
from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor


def _nvidia_smi_summary():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception as error:  # pragma: no cover - environment dependent
        return f"nvidia-smi unavailable: {error}"


def _benchmark(sequence, encoder, processor, decoder, epochs=3):
    encoder.train(); processor.train(); decoder.train()
    target_p = torch.zeros(len(sequence[0].batch.unique()), 1, device=sequence[0].x.device)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    total = torch.zeros((), device=sequence[0].x.device)
    for _ in range(epochs):
        state = None
        for batch in sequence:
            latent = encoder(batch.x, batch.edge_index, batch.pos)
            latent, state = processor(latent, batch.batch, state=state)
            delta, p_disrupt = decoder(latent, batch.edge_index, batch.batch, batch.pos)
            total = total + delta.square().mean() + p_disrupt.square().mean()
    total.backward()
    elapsed = time.perf_counter() - start
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Checkpointing VRAM benchmark (#0008)")
    parser.add_argument("--data-dir", default="data/EHL-2/scan_1d22_ang12_massloss_neon")
    parser.add_argument("--first-index", type=int, default=2710)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", default="output/model502_baseline/checkpoint_vram/report.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = []
    for offset in range(args.batch):
        path = Path(args.data_dir) / f"jorek{args.first_index + offset * 10:06d}.h5"
        frames.append(JorekDataParser(str(path)).parse_to_graph())
    sequence = [Batch.from_data_list([graph.clone() for graph in frames]).to(device)]

    results = {"device": str(device), "nvidia_smi_before": _nvidia_smi_summary()}
    for enabled in (False, True):
        torch.manual_seed(0)
        encoder = JorekGNNEncoder(
            input_dim=9, hidden_dim=args.hidden_dim, latent_dim=args.latent_dim,
            num_layers=args.num_layers, edge_dim=2, checkpoint=enabled,
        ).to(device)
        processor = JorekLNNProcessor(
            latent_dim=args.latent_dim, hidden_size=args.hidden_dim, num_layers=1
        ).to(device)
        decoder = JorekGNNDecoder(
            latent_dim=args.latent_dim, hidden_dim=args.hidden_dim, output_dim=9,
            num_layers=args.num_layers, edge_dim=2, checkpoint=enabled,
        ).to(device)
        results[f"checkpoint_{str(enabled).lower()}"] = _benchmark(
            sequence, encoder, processor, decoder, epochs=args.epochs
        )
        del encoder, processor, decoder
        torch.cuda.empty_cache()
    results["nvidia_smi_after"] = _nvidia_smi_summary()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

