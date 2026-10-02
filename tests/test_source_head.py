"""Tests for the fragment-to-projection source head."""

from __future__ import annotations

import torch

from src.models.source_head import SourceFieldHead


def _inputs(batch: int = 2, nodes: int = 7, shards: int = 4):
    torch.manual_seed(0)
    u_raw = torch.ones(batch, nodes, 9) * 0.5
    u_raw[..., 4] = 0.8  # rho
    u_raw[..., 7] = 0.01  # rho_imp
    u_raw[..., 8] = 0.03  # T_e (~1.9 keV)
    pos = torch.stack(
        [torch.linspace(1.0, 2.0, nodes), torch.linspace(-0.5, 0.5, nodes)], dim=1
    )
    shard_values = torch.zeros(batch, shards, 16)
    shard_values[..., 0] = torch.linspace(1.2, 1.8, shards)
    shard_values[..., 1] = 0.0
    shard_values[..., 7] = 1e-4
    shard_values[..., 9] = 1.0
    mask = torch.ones(batch, shards, dtype=torch.bool)
    mask[1, -1] = False
    return u_raw, shard_values, mask, pos


def test_initial_output_equals_analytic_baseline():
    head = SourceFieldHead()
    u_raw, shards, mask, pos = _inputs()
    out = head(u_raw, shards, mask, pos, central_density=0.8)
    assert out["source"].shape == (2, 7, 5)
    assert torch.allclose(out["source"], out["baseline"], atol=1e-6)
    assert float(out["baseline"].detach().sum()) > 0.0


def test_all_dead_shards_give_zero_source():
    head = SourceFieldHead()
    u_raw, shards, mask, pos = _inputs()
    mask = torch.zeros_like(mask)
    out = head(u_raw, shards, mask, pos, central_density=0.8)
    assert float(out["source"].detach().abs().sum()) == 0.0
    assert float(out["deposit"].detach().abs().sum()) == 0.0


def test_gradients_flow_to_encoder_and_channel_scale():
    head = SourceFieldHead()
    u_raw, shards, mask, pos = _inputs()
    out = head(u_raw, shards, mask, pos, central_density=0.8)
    out["source"].sum().backward()
    assert head.channel_log_scale.grad is not None
    assert any(parameter.grad is not None for parameter in head.encoder.parameters())
    assert any(parameter.grad is not None for parameter in head.correction.parameters())
