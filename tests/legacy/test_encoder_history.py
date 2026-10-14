"""Encoder history conditioning: the optional previous-rate input must be
accepted, change the representation, and leave the no-history path intact."""

from __future__ import annotations

import torch

from src.data.model502_schema import NUM_STATE_FIELDS
from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder


def _toy_graph(num_nodes=12):
    torch.manual_seed(0)
    pos = torch.rand(num_nodes, 2)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, min(i + 3, num_nodes)):
            edges.append((i, j))
            edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    row, col = edge_index
    delta = pos[col] - pos[row]
    edge_attr = torch.cat([delta, delta.norm(dim=1, keepdim=True)], dim=1)
    return pos, edge_index, edge_attr


def test_history_branch_is_zero_initialised_and_trainable():
    torch.manual_seed(0)
    encoder = JorekGNNEncoder(
        input_dim=NUM_STATE_FIELDS,
        hidden_dim=16,
        latent_dim=8,
        num_layers=2,
        edge_dim=2,
    )
    encoder.eval()
    pos, edge_index, edge_attr = _toy_graph()
    x = torch.randn(pos.size(0), NUM_STATE_FIELDS)

    with torch.no_grad():
        base = encoder(x, edge_index, pos)
        same = encoder(x, edge_index, pos, history=None)
        history_a = torch.zeros(pos.size(0), NUM_STATE_FIELDS)
        history_b = torch.ones(pos.size(0), NUM_STATE_FIELDS)
        out_a = encoder(x, edge_index, pos, history=history_a)
        out_b = encoder(x, edge_index, pos, history=history_b)

    assert torch.allclose(base, same, atol=1e-6)
    # Zero-initialised residual branch: an untrained (or pre-history
    # checkpoint loaded with strict=False) encoder ignores history entirely.
    assert torch.allclose(base, out_a, atol=1e-6)
    assert torch.allclose(out_a, out_b, atol=1e-6)

    # After one update the branch is live and history changes the output.
    encoder.train()
    loss = encoder(x, edge_index, pos, history=history_b).square().mean()
    loss.backward()
    torch.optim.SGD(encoder.parameters(), lr=0.1).step()
    encoder.eval()
    with torch.no_grad():
        out_a = encoder(x, edge_index, pos, history=history_a)
        out_b = encoder(x, edge_index, pos, history=history_b)
    assert not torch.allclose(out_a, out_b, atol=1e-5)


def test_decoder_prev_latent_branch_is_zero_initialised_and_trainable():
    torch.manual_seed(0)
    decoder = JorekGNNDecoder(
        latent_dim=8, hidden_dim=16, output_dim=NUM_STATE_FIELDS, num_layers=2, edge_dim=2
    )
    decoder.eval()
    pos, edge_index, _ = _toy_graph()
    latent = torch.randn(pos.size(0), 8)
    batch_index = torch.zeros(pos.size(0), dtype=torch.long)
    prev_a = torch.zeros(pos.size(0), 8)
    prev_b = torch.ones(pos.size(0), 8)

    with torch.no_grad():
        base, _ = decoder(latent, edge_index, batch_index, pos)
        out_a, _ = decoder(latent, edge_index, batch_index, pos, prev_latent=prev_a)
        out_b, _ = decoder(latent, edge_index, batch_index, pos, prev_latent=prev_b)
    assert torch.allclose(base, out_a, atol=1e-6)
    assert torch.allclose(out_a, out_b, atol=1e-6)

    decoder.train()
    loss = decoder(latent, edge_index, batch_index, pos, prev_latent=prev_b)[0]
    loss.square().mean().backward()
    torch.optim.SGD(decoder.parameters(), lr=0.1).step()
    decoder.eval()
    with torch.no_grad():
        out_a, _ = decoder(latent, edge_index, batch_index, pos, prev_latent=prev_a)
        out_b, _ = decoder(latent, edge_index, batch_index, pos, prev_latent=prev_b)
    assert not torch.allclose(out_a, out_b, atol=1e-5)
