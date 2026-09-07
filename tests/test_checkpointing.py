"""Gradient-checkpointing numerical equivalence and train/eval semantics."""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from src.models.gnn_decoder import JorekGNNDecoder
from src.models.gnn_encoder import JorekGNNEncoder


def _graph(nodes: int = 10) -> Data:
    generator = torch.Generator().manual_seed(7)
    edges = torch.randint(0, nodes, (2, 4 * nodes), generator=generator)
    return Data(
        x=torch.randn(nodes, 9, generator=generator),
        edge_index=edges,
        pos=torch.randn(nodes, 2, generator=generator),
        batch=torch.zeros(nodes, dtype=torch.long),
    )


def _build_pair(checkpoint: bool):
    torch.manual_seed(3)
    encoder = JorekGNNEncoder(
        input_dim=9, hidden_dim=16, latent_dim=8, num_layers=3, edge_dim=2,
        checkpoint=checkpoint,
    )
    decoder = JorekGNNDecoder(
        latent_dim=8, hidden_dim=16, output_dim=9, num_layers=2, edge_dim=2,
        checkpoint=checkpoint,
    )
    return encoder, decoder


def test_checkpointed_forward_backward_matches_plain():
    graph = _graph()
    plain_encoder, plain_decoder = _build_pair(False)
    ckpt_encoder, ckpt_decoder = _build_pair(True)

    out_plain = _forward_loss(plain_encoder, plain_decoder, graph)
    out_ckpt = _forward_loss(ckpt_encoder, ckpt_decoder, graph)

    assert torch.allclose(out_plain[0], out_ckpt[0], atol=1e-6)
    for plain_param, ckpt_param in zip(plain_encoder.parameters(), ckpt_encoder.parameters()):
        assert torch.allclose(plain_param.grad, ckpt_param.grad, atol=1e-5)
    for plain_param, ckpt_param in zip(plain_decoder.parameters(), ckpt_decoder.parameters()):
        assert torch.allclose(plain_param.grad, ckpt_param.grad, atol=1e-5)


def test_checkpoint_flag_is_inert_in_eval_mode():
    graph = _graph()
    plain_encoder, plain_decoder = _build_pair(False)
    ckpt_encoder, ckpt_decoder = _build_pair(True)
    plain_encoder.eval(); plain_decoder.eval()
    ckpt_encoder.eval(); ckpt_decoder.eval()

    with torch.no_grad():
        plain_latent = plain_encoder(graph.x, graph.edge_index, graph.pos)
        ckpt_latent = ckpt_encoder(graph.x, graph.edge_index, graph.pos)
        plain_delta, plain_p = plain_decoder(plain_latent, graph.edge_index, graph.batch, graph.pos)
        ckpt_delta, ckpt_p = ckpt_decoder(ckpt_latent, graph.edge_index, graph.batch, graph.pos)

    assert torch.allclose(plain_latent, ckpt_latent, atol=1e-6)
    assert torch.allclose(plain_delta, ckpt_delta, atol=1e-6)
    assert torch.allclose(plain_p, ckpt_p, atol=1e-6)


def _forward_loss(encoder, decoder, graph):
    encoder.train()
    decoder.train()
    latent = encoder(graph.x, graph.edge_index, graph.pos)
    delta, p_disrupt = decoder(latent, graph.edge_index, graph.batch, graph.pos)
    loss = delta.square().mean() + p_disrupt.square().mean()
    loss.backward()
    return delta.detach(), p_disrupt.detach()

