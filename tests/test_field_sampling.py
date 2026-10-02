"""Tests for k-NN field sampling and deposition."""

from __future__ import annotations

import torch

from src.models.field_sampling import deposit_points, knn_weights, masked_deposit, sample_nodes


def _grid():
    xs = torch.tensor([0.0, 1.0, 2.0])
    yy = torch.tensor([0.0, 1.0, 2.0])
    grid_y, grid_x = torch.meshgrid(yy, xs, indexing="ij")
    pos = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)
    return pos


def test_sampling_at_a_node_returns_that_nodes_value():
    pos = _grid()
    values = torch.arange(pos.shape[0], dtype=torch.float32).unsqueeze(1)
    idx, weights = knn_weights(pos, pos[4:5], k=1)
    assert int(idx[0, 0]) == 4
    sampled = sample_nodes(values, idx, weights)
    assert float(sampled[0, 0]) == 4.0


def test_inverse_distance_weights_normalised_and_local():
    pos = _grid()
    point = torch.tensor([[0.9, 0.9]])
    idx, weights = knn_weights(pos, point, k=4)
    assert torch.allclose(weights.sum(dim=1), torch.ones(1), atol=1e-5)
    assert idx.shape == (1, 4)


def test_deposition_is_conservative_with_k1():
    pos = _grid()
    points = pos[[0, 4, 8]]
    idx, weights = knn_weights(pos, points, k=1)
    values = torch.tensor([[1.0], [2.0], [3.0]])
    deposited = deposit_points(values, idx, weights, pos.shape[0])
    assert deposited.shape == (pos.shape[0], 1)
    assert float(deposited.sum()) == 6.0
    assert float(deposited[0]) == 1.0
    assert float(deposited[4]) == 2.0


def test_empty_and_masked_deposition():
    pos = _grid()
    idx, weights = knn_weights(pos, torch.empty(0, 2), k=2)
    assert idx.shape == (0, 2)
    assert deposit_points(torch.empty(0, 3), idx, weights, pos.shape[0]).shape == (pos.shape[0], 3)

    points = pos[[0, 1]]
    idx, weights = knn_weights(pos, points, k=1)
    values = torch.tensor([[1.0], [1.0]])
    mask = torch.tensor([True, False])
    deposited = masked_deposit(values, mask, idx, weights, pos.shape[0])
    assert float(deposited.sum()) == 1.0
