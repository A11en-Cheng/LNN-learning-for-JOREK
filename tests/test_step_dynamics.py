"""Non-uniform dt, hidden-state continuity and multi-horizon skip sampling."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from src.data.dataset import step_dt_seconds
from src.data.jorek_h5_parser import JorekDataParser
from src.models.lnn_processor import JorekLNNProcessor
from src.models.loss import JorekLoss
from src.training.trainer import JorekTrainer


def _graph(time_value: float, t_norm: float = 1.0) -> Data:
    return Data(
        x=torch.ones(3, 9),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        pos=torch.zeros(3, 2),
        time=torch.tensor([time_value]),
        t_norm=torch.tensor([t_norm]),
    )


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index, pos):
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
        self.calls.append((state, time_steps))
        next_state = (
            torch.randn(latent_x.shape[0], latent_x.shape[1], requires_grad=True)
            if state is None
            else state
        )
        return latent_x * (self.dummy + 1.0), next_state


def _make_trainer(processor=None):
    encoder = _Encoder()
    processor = processor if processor is not None else _RecordingProcessor()
    decoder = _Decoder()
    criterion = JorekLoss()
    return JorekTrainer(encoder, processor, decoder, None, criterion, lr=1e-3, clip_value=1.0)


def test_step_dt_uses_t_norm_for_physical_seconds():
    batch_a = Batch.from_data_list([_graph(10.0, 2.0), _graph(20.0, 2.0)])
    batch_b = Batch.from_data_list([_graph(12.0, 2.0), _graph(24.0, 2.0)])

    dt = step_dt_seconds(batch_a, batch_b)

    assert torch.allclose(dt, torch.tensor([4.0, 8.0]))


def test_step_dt_rejects_non_monotonic_time():
    batch_a = Batch.from_data_list([_graph(10.0)])
    batch_b = Batch.from_data_list([_graph(9.0)])

    with __import__("pytest").raises(ValueError, match="monotonic"):
        step_dt_seconds(batch_a, batch_b)


def test_trainer_carries_hidden_state_across_steps():
    processor = _RecordingProcessor()
    trainer = _make_trainer(processor)
    batch = Batch.from_data_list([_graph(0.0)])
    target = torch.zeros(3, 9)
    target_p = torch.zeros(1, 1)

    trainer.train_step(batch, target, target_p, dt=1.5, time_steps=1.5)
    first_state = trainer.hidden_state
    trainer.train_step(
        batch, target, target_p, dt=2.5, time_steps=2.5, detach_state_on_entry=False
    )

    assert processor.calls[0][0] is None
    assert processor.calls[1][0] is first_state
    assert processor.calls[1][1] == 2.5


def test_trainer_reset_and_detach_state():
    trainer = _make_trainer()
    batch = Batch.from_data_list([_graph(0.0)])
    trainer.train_step(batch, torch.zeros(3, 9), torch.zeros(1, 1), dt=1.0)
    assert trainer.hidden_state is not None

    trainer.detach_state()
    assert not trainer.hidden_state.requires_grad

    trainer.reset_state()
    assert trainer.hidden_state is None


def test_multihorizon_training_uses_summed_dt_per_skip():
    processor = _RecordingProcessor()
    trainer = _make_trainer(processor)
    sequence = [
        Batch.from_data_list([_graph(float(t))]) for t in range(5)
    ]
    target_p = torch.zeros(1, 1)

    loss, count = trainer.train_step_multihorizon(sequence, target_p, horizons=(1, 2, 4))

    assert count == 8
    assert __import__("math").isfinite(loss)
    time_steps = [call[1] for call in processor.calls]
    # t=0 pairs: k=1 -> dt 1, k=2 -> dt 2, k=4 -> dt 4
    assert time_steps[:3] == [1.0, 2.0, 4.0]
    # t=1 pairs: k=1 -> dt 1, k=2 -> dt 2 (k=4 falls outside the window)
    assert time_steps[3:5] == [1.0, 2.0]
    # t=2 pairs: k=1, k=2; t=3: k=1
    assert time_steps[5:8] == [1.0, 2.0, 1.0]


def test_processor_rollout_is_deterministic_with_carried_state():
    processor = JorekLNNProcessor(latent_dim=4, hidden_size=6, num_layers=1)
    x1 = torch.randn(3, 4)
    x2 = torch.randn(3, 4)

    out1, state1 = processor(x1, state=None, time_steps=torch.tensor(0.5))
    out2, state2 = processor(x2, state=state1, time_steps=torch.tensor(0.5))
    again1, again_state1 = processor(x1, state=None, time_steps=torch.tensor(0.5))
    again2, _ = processor(x2, state=again_state1, time_steps=torch.tensor(0.5))

    assert torch.allclose(out1, again1)
    assert torch.allclose(out2, again2)
    assert state2 is not None


def test_parser_exposes_t_norm(tmp_path):
    import h5py
    import numpy as np

    path = tmp_path / "jorek00001.h5"
    with h5py.File(path, "w") as f:
        f["n_var"] = 9
        f["n_tor"] = 1
        f["n_dim"] = 2
        f["jorek_model"] = 502
        f["t_now"] = np.array([10.0])
        f["t_norm"] = np.array([2.5])
        f["boundary"] = np.zeros(3, dtype=np.int64)
        f["contain_node"] = np.array([[1, 2], [2, 3]], dtype=np.int32).T
        f["x"] = np.zeros((2, 4, 1, 3))
        f["values"] = np.zeros((9, 4, 1, 3))

    graph = JorekDataParser(str(path)).parse_to_graph()

    assert graph.t_norm.item() == 2.5
    assert graph.time.item() == 10.0


def test_real_frames_multihorizon_forward_backward_finite():
    from pathlib import Path

    from src.models.gnn_decoder import JorekGNNDecoder
    from src.models.gnn_encoder import JorekGNNEncoder

    run_dir = Path("data/EHL-2/scan_1d22_ang12_massloss_neon")
    if not run_dir.exists():
        __import__("pytest").skip("real EHL-2 data not present")
    frames = [
        JorekDataParser(str(run_dir / f"jorek{index:06d}.h5")).parse_to_graph()
        for index in (2710, 2720, 2730, 2740)
    ]
    sequence = [Batch.from_data_list([graph]) for graph in frames]

    encoder = JorekGNNEncoder(input_dim=9, hidden_dim=16, latent_dim=8, num_layers=2, edge_dim=2)
    processor = JorekLNNProcessor(latent_dim=8, hidden_size=12, num_layers=1)
    decoder = JorekGNNDecoder(latent_dim=8, hidden_dim=16, output_dim=9, num_layers=2, edge_dim=2)
    trainer = JorekTrainer(encoder, processor, decoder, None, JorekLoss(), lr=1e-3)

    loss, count = trainer.train_step_multihorizon(sequence, torch.zeros(1, 1), horizons=(1, 2))

    assert count == 5
    assert __import__("math").isfinite(loss)
    assert all(
        param.grad is None or torch.isfinite(param.grad).all()
        for module in (encoder, processor, decoder)
        for param in module.parameters()
    )
