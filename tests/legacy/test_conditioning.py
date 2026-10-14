"""Condition-injection behaviour: response vs the no-condition control."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch, Data

from src.data.parameter_channel import CONDITION_DIM
from src.models.gnn_encoder import JorekGNNEncoder


def _graph():
    return Data(
        x=torch.randn(6, 9),
        edge_index=torch.tensor(
            [[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], dtype=torch.long
        ),
        pos=torch.randn(6, 2),
        batch=torch.zeros(6, dtype=torch.long),
    )


def _encoder(condition_dim):
    torch.manual_seed(11)
    return JorekGNNEncoder(
        input_dim=9, hidden_dim=16, latent_dim=8, num_layers=2, edge_dim=2,
        condition_dim=condition_dim,
    )


def test_condition_changes_output_and_no_condition_matches_plain():
    graph = _graph()
    encoder = _encoder(condition_dim=CONDITION_DIM)
    plain = _encoder(condition_dim=0)
    base = torch.zeros(CONDITION_DIM)
    perturbed = torch.zeros(CONDITION_DIM)
    perturbed[4] = 5.0  # ZK_i_par slot in the canonical condition order

    with torch.no_grad():
        out_base = encoder(graph.x, graph.edge_index, graph.pos, condition=base, batch=graph.batch)
        out_perturbed = encoder(
            graph.x, graph.edge_index, graph.pos, condition=perturbed, batch=graph.batch
        )
        out_none = encoder(graph.x, graph.edge_index, graph.pos, condition=None, batch=graph.batch)
        out_plain = plain(graph.x, graph.edge_index, graph.pos)

    assert (out_base - out_perturbed).abs().max().item() > 1e-3
    assert torch.equal(out_none, out_plain)


def test_condition_broadcasts_per_graph_in_batch():
    torch.manual_seed(5)
    encoder = JorekGNNEncoder(
        input_dim=9, hidden_dim=16, latent_dim=8, num_layers=2, edge_dim=2,
        condition_dim=CONDITION_DIM,
    )
    graph_a = _graph()
    graph_b = _graph()
    batch = Batch.from_data_list([graph_a, graph_b])
    batch.condition = torch.stack(
        [torch.zeros(CONDITION_DIM), torch.full((CONDITION_DIM,), 4.0)]
    )

    with torch.no_grad():
        out = encoder(batch.x, batch.edge_index, batch.pos, condition=batch.condition, batch=batch.batch)

    first_nodes = out[:6]
    second_nodes = out[6:]
    assert (first_nodes - second_nodes).abs().max().item() > 1e-3
