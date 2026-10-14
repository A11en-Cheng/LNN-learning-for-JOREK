"""Processor tests: per-graph time steps must reach every node's CfC cell."""

from __future__ import annotations

import pytest
import torch

from src.models.lnn_processor import JorekLNNProcessor


@pytest.mark.skipif(
    not __import__("importlib").util.find_spec("ncps"),
    reason="ncps not installed",
)
def test_per_graph_timespans_are_applied_per_node():
    torch.manual_seed(0)
    latent_dim = 8
    processor = JorekLNNProcessor(latent_dim=latent_dim, hidden_size=16)
    processor.eval()

    num_nodes = 10
    latent = torch.randn(2 * num_nodes, latent_dim)
    batch = torch.cat(
        [
            torch.zeros(num_nodes, dtype=torch.long),
            torch.ones(num_nodes, dtype=torch.long),
        ]
    )
    with torch.no_grad():
        out_a, _ = processor(
            latent, batch, time_steps=torch.tensor([0.5, 0.5])
        )
        out_b, _ = processor(
            latent, batch, time_steps=torch.tensor([0.5, 2.0])
        )

    # Changing graph 1's dt must not change graph 0's outputs...
    assert torch.allclose(out_a[:num_nodes], out_b[:num_nodes], atol=1e-6)
    # ...and must change graph 1's outputs (a per-node timespan, not a mean).
    assert not torch.allclose(out_a[num_nodes:], out_b[num_nodes:], atol=1e-6)
