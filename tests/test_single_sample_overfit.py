"""Fast harness test for the #0007 overfit gate (tiny synthetic sequence)."""

from __future__ import annotations

from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from tests.train_single_sample_overfit import graph_like, run_single_sample_overfit


def _frame(offset: float):
    return graph_like(offset)


def test_tiny_overfit_gate_passes_and_reloads_exactly(tmp_path):
    sequence = [_frame(float(index)) for index in range(3)]
    summary = run_single_sample_overfit(
        sequence=sequence,
        model_factories=(JorekGNNEncoder, JorekLNNProcessor, JorekGNNDecoder),
        hidden_dim=8,
        latent_dim=4,
        num_layers=2,
        epochs=300,
        lr=1e-2,
        seed=1,
        device="cpu",
        output_dir=tmp_path,
    )

    assert summary["no_nan"]
    assert summary["loss_drop_99_percent"]
    assert summary["reload_within_1e-6"]
    assert summary["gate_passed"]
    assert (tmp_path / "single_sample_checkpoint.pt").exists()
    assert (tmp_path / "loss_curve.csv").exists()
    assert (tmp_path / "manifest.json").exists()
