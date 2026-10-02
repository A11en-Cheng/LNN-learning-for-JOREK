"""Tests for the hybrid fragment dynamics module."""

from __future__ import annotations

import torch

from src.models.fragment_dynamics import HybridFragmentDynamics


def _inputs(batch: int = 2, nodes: int = 7, shards: int = 4):
    u_raw = torch.ones(batch, nodes, 9) * 0.5
    u_raw[..., 4] = 0.8
    u_raw[..., 7] = 0.01
    u_raw[..., 8] = 0.03
    pos = torch.stack(
        [torch.linspace(1.0, 2.0, nodes), torch.linspace(-0.5, 0.5, nodes)], dim=1
    )
    shard_values = torch.zeros(batch, shards, 16)
    shard_values[..., 0] = torch.linspace(1.2, 1.8, shards)
    shard_values[..., 4] = -300.0
    shard_values[..., 7] = 1e-4
    shard_values[..., 9] = 1.0
    mask = torch.ones(batch, shards, dtype=torch.bool)
    mask[1, -1] = False
    return u_raw, shard_values, mask, pos


def _forward(model, u_raw, shards, mask, pos, *, use_correction: bool):
    return model(
        u_raw,
        shards,
        mask,
        pos,
        dt_jorek=torch.tensor([10.0], dtype=torch.float32).expand(u_raw.shape[0]),
        t_norm=4.085775e-07,
        central_density=0.8,
        pellet_density=4.3e8,
        use_correction=use_correction,
    )


def test_zero_initialised_correction_matches_analytic_path():
    model = HybridFragmentDynamics()
    u_raw, shards, mask, pos = _inputs()
    corrected = _forward(model, u_raw, shards, mask, pos, use_correction=True)
    analytic = _forward(model, u_raw, shards, mask, pos, use_correction=False)
    assert torch.allclose(corrected["next"], analytic["next"], atol=1e-6)
    assert torch.allclose(corrected["ablation"], analytic["ablation"], atol=1e-6)


def test_positivity_and_dead_mask():
    model = HybridFragmentDynamics()
    u_raw, shards, mask, pos = _inputs()
    out = _forward(model, u_raw, shards, mask, pos, use_correction=True)
    assert (out["next"][..., 7] >= 0).all()
    assert (out["alive"][1, -1] == False).item()  # noqa: E712
    assert torch.allclose(out["next"][1, -1], torch.zeros(16), atol=1e-6)


def test_gradients_flow_through_the_correction_head():
    model = HybridFragmentDynamics()
    u_raw, shards, mask, pos = _inputs()
    out = _forward(model, u_raw, shards, mask, pos, use_correction=True)
    out["next"].sum().backward()
    assert any(parameter.grad is not None for parameter in model.correction.parameters())
