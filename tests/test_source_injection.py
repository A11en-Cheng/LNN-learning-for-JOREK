"""Source-injection behaviour: five-field marker projections as LNN controls."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.data.model502_schema import (
    NUM_MARKER_PROJECTION_FIELDS,
    destandardize,
    standardize,
)
from src.data.jorek_h5_parser import JorekDataParser
from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.training.trainer import JorekTrainer


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, pos, condition=None, batch=None):
        return x[:, :4] * (self.dummy + 1.0)


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, latent, edge_index, batch, pos):
        return torch.zeros(pos.shape[0], 9) * (self.dummy + 1.0), torch.zeros(1, 1)


class _RecordingProcessor(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, latent_x, batch_index=None, state=None, time_steps=None, controls=None):
        self.calls.append(controls)
        return latent_x * (self.dummy + 1.0), None


class _Criterion(nn.Module):
    def forward(self, pred, target, pred_p, target_p, node_weights=None, **kwargs):
        return pred.square().mean(), {}


def _graph():
    return Data(
        x=torch.randn(3, 9),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        pos=torch.randn(3, 2),
        time=torch.tensor([0.0]),
        t_norm=torch.tensor([1.0]),
        source=torch.randn(3, NUM_MARKER_PROJECTION_FIELDS),
    )


def test_processor_accepts_and_uses_five_field_controls():
    processor = JorekLNNProcessor(latent_dim=4, hidden_size=6)
    x = torch.randn(3, 4)

    with torch.no_grad():
        out_zero = processor(x, controls=torch.zeros(3, NUM_MARKER_PROJECTION_FIELDS))[0]
        out_ones = processor(x, controls=torch.ones(3, NUM_MARKER_PROJECTION_FIELDS))[0]

    assert not torch.allclose(out_zero, out_ones)


def test_processor_rejects_three_field_controls():
    processor = JorekLNNProcessor(latent_dim=4, hidden_size=6)

    with __import__("pytest").raises(ValueError, match="channels"):
        processor(torch.randn(3, 4), controls=torch.randn(3, 3))


def test_trainer_feeds_batch_source_as_controls_by_default():
    processor = _RecordingProcessor()
    trainer = JorekTrainer(
        _Encoder(), processor, _Decoder(), None, _Criterion(), lr=1e-3
    )
    graph = _graph()
    batch = Batch.from_data_list([graph])

    trainer.train_step(batch, torch.zeros(3, 9), torch.zeros(1, 1), dt=1.0)

    assert processor.calls[0] is not None
    assert processor.calls[0].shape == (3, NUM_MARKER_PROJECTION_FIELDS)


def test_source_stats_round_trip():
    mean = torch.tensor([0.5, -1.0, 2.0, 0.0, 3.0])
    std = torch.tensor([0.2, 1.5, 0.5, 1.0, 2.0])
    value = torch.randn(3, NUM_MARKER_PROJECTION_FIELDS)

    restored = destandardize(
        standardize(value, mean, std, name="source"), mean, std, name="source"
    )

    assert torch.allclose(restored, value)


def test_real_strong_source_forward_backward_is_finite():
    from pathlib import Path

    frame = Path("data/EHL-2/scan_1d22_ang12_massloss_neon/jorek002750.h5")
    if not frame.exists():
        __import__("pytest").skip("real EHL-2 data not present")
    graph = JorekDataParser(str(frame), require_source=True).parse_to_graph()
    assert graph.source is not None

    encoder = JorekGNNEncoder(input_dim=9, hidden_dim=16, latent_dim=8, num_layers=2, edge_dim=2)
    processor = JorekLNNProcessor(latent_dim=8, hidden_size=12)
    decoder = JorekGNNDecoder(latent_dim=8, hidden_dim=16, output_dim=9, num_layers=2, edge_dim=2)
    batch = Batch.from_data_list([graph])

    latent = encoder(batch.x, batch.edge_index, batch.pos)
    next_latent, _ = processor(latent, batch.batch, controls=graph.source)
    delta, p_disrupt = decoder(next_latent, batch.edge_index, batch.batch, batch.pos)
    loss = delta.square().mean() + p_disrupt.square().mean()
    loss.backward()

    assert torch.isfinite(delta).all()
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for module in (encoder, processor, decoder)
        for param in module.parameters()
    )
