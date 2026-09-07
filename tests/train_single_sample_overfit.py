"""Single-sample overfit gate (#0007).

Trains the monolithic 9-field model on one short sequence of real frames with
a fixed seed, then checks the three gate criteria: regression loss drops by at
least 99%, no NaN during training, and a reloaded checkpoint reproduces the
in-memory model output to 1e-6.  The classification term is disabled for the
capacity gate (a constant target label only adds a non-vanishing floor), and
the physics term is off, exactly as ADR-0002 specifies.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.data import Batch

from src.data.dataset import step_dt_seconds
from src.data.jorek_h5_parser import JorekDataParser
from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.loss import JorekLoss
from src.training.trainer import JorekTrainer


def graph_like(offset: float = 0.0, nodes: int = 3):
    """Small synthetic frame used by the fast harness test."""
    from torch_geometric.data import Data

    return Data(
        x=torch.randn(nodes, 9) + offset,
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
        pos=torch.randn(nodes, 2),
        time=torch.tensor([offset]),
        t_norm=torch.tensor([1.0]),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_models(encoder, processor, decoder, hidden_dim, latent_dim, num_layers):
    encoder = encoder(input_dim=9, hidden_dim=hidden_dim, latent_dim=latent_dim, num_layers=num_layers, edge_dim=2)
    processor = processor(latent_dim=latent_dim, hidden_size=hidden_dim, num_layers=1)
    decoder = decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=9, num_layers=num_layers, edge_dim=2)
    return encoder, processor, decoder


def _eval_first_frame(models, sequence, dt):
    encoder, processor, decoder = models
    with torch.no_grad():
        latent = encoder(sequence[0].x, sequence[0].edge_index, sequence[0].pos)
        next_latent, _ = processor(latent, sequence[0].batch, state=None, time_steps=dt)
        delta, p_disrupt = decoder(
            next_latent, sequence[0].edge_index, sequence[0].batch, sequence[0].pos
        )
    return delta, p_disrupt


def run_single_sample_overfit(
    *,
    sequence,
    model_factories,
    hidden_dim=64,
    latent_dim=32,
    num_layers=4,
    epochs=1500,
    lr=5e-4,
    seed=0,
    device="cpu",
    output_dir=None,
) -> dict:
    """Run the overfit gate and return the acceptance summary."""
    _seed_everything(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    encoder_cls, processor_cls, decoder_cls = model_factories
    encoder, processor, decoder = _build_models(
        encoder_cls, processor_cls, decoder_cls, hidden_dim, latent_dim, num_layers
    )
    encoder = encoder.to(device)
    processor = processor.to(device)
    decoder = decoder.to(device)

    sequence = [
        Batch.from_data_list(
            [graph.clone()], exclude_keys=[]
        ).to(device)
        for graph in sequence
    ]
    criterion = JorekLoss(lambda_reg=1.0, lambda_cls=0.0, lambda_phys=0.0)
    trainer = JorekTrainer(
        encoder, processor, decoder, None, criterion, lr=lr, clip_value=1.0
    )
    target_p = torch.zeros(1, 1, device=device)
    dt = float(step_dt_seconds(sequence[0], sequence[1]).mean().item())

    losses = []
    for _ in range(epochs):
        loss, _ = trainer.train_step_multihorizon(sequence, target_p, horizons=(1, 2))
        losses.append(loss)
        if not math.isfinite(loss):
            break
    encoder.eval()
    processor.eval()
    decoder.eval()

    delta_in, p_in = _eval_first_frame((encoder, processor, decoder), sequence, dt)
    checkpoint = {
        "encoder": encoder.state_dict(),
        "processor": processor.state_dict(),
        "decoder": decoder.state_dict(),
    }

    encoder2, processor2, decoder2 = _build_models(
        encoder_cls, processor_cls, decoder_cls, hidden_dim, latent_dim, num_layers
    )
    encoder2.load_state_dict(checkpoint["encoder"])
    processor2.load_state_dict(checkpoint["processor"])
    decoder2.load_state_dict(checkpoint["decoder"])
    encoder2 = encoder2.to(device)
    processor2 = processor2.to(device)
    decoder2 = decoder2.to(device)
    encoder2.eval()
    processor2.eval()
    decoder2.eval()
    delta_re, p_re = _eval_first_frame((encoder2, processor2, decoder2), sequence, dt)

    reload_diff = max(
        float((delta_in - delta_re).abs().max().item()),
        float((p_in - p_re).abs().max().item()),
    )
    initial = losses[0]
    final = losses[-1]
    drop_ratio = (initial - final) / initial if initial > 0 else 0.0
    summary = {
        "seed": seed,
        "epochs": epochs,
        "hidden_dim": hidden_dim,
        "latent_dim": latent_dim,
        "num_layers": num_layers,
        "dt_seconds": dt,
        "initial_loss": float(initial),
        "final_loss": float(final),
        "loss_drop_ratio": float(drop_ratio),
        "reload_max_abs_diff": float(reload_diff),
        "no_nan": all(math.isfinite(value) for value in losses),
        "loss_drop_99_percent": drop_ratio >= 0.99,
        "reload_within_1e-6": reload_diff <= 1e-6,
        "gate_passed": (
            all(math.isfinite(value) for value in losses)
            and drop_ratio >= 0.99
            and reload_diff <= 1e-6
        ),
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, output / "single_sample_checkpoint.pt")
        (output / "loss_curve.csv").write_text(
            "\n".join(f"{index},{value}" for index, value in enumerate(losses))
        )
        (output / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _load_real_sequence(data_dir, first_index, seq_len=3):
    frames = []
    for step in range(seq_len):
        index = first_index + step * 10
        path = Path(data_dir) / f"jorek{index:06d}.h5"
        frames.append(JorekDataParser(str(path)).parse_to_graph())
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Model 502 single-sample overfit gate (#0007)")
    parser.add_argument("--data-dir", default="data/EHL-2/scan_1d22_ang12_massloss_neon")
    parser.add_argument("--first-index", type=int, default=2710)
    parser.add_argument("--seq-len", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="output/model502_baseline/single_sample_overfit")
    args = parser.parse_args()

    sequence = _load_real_sequence(args.data_dir, args.first_index, args.seq_len)
    summary = run_single_sample_overfit(
        sequence=sequence,
        model_factories=(JorekGNNEncoder, JorekLNNProcessor, JorekGNNDecoder),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("GATE:", "PASS" if summary["gate_passed"] else "FAIL")


if __name__ == "__main__":
    main()
