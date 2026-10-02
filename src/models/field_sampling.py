"""k-nearest-node sampling and deposition on the poloidal JOREK mesh.

The SPI source model needs two directions:

* **sampling**: read local plasma quantities (rho, rho_imp, T_e, psi) at an
  arbitrary shard position ``(R, Z)``;
* **deposition**: scatter per-shard ablation rates back to the mesh nodes.

Both use inverse-distance weights over the ``k`` nearest nodes.  The JOREK
implementation interpolates on the finite-element basis; the k-NN form is a
cheap differentiable stand-in whose error is absorbed by the learned
correction (documented in the Phase-2 gate).
"""

from __future__ import annotations

import torch


def knn_weights(
    pos: torch.Tensor,
    points: torch.Tensor,
    *,
    k: int = 4,
    eps: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse-distance weights of the ``k`` nearest nodes for each point.

    ``pos`` is ``[N, 2]``, ``points`` is ``[P, 2]``; returns ``idx [P, k]`` and
    normalised ``weights [P, k]``.  Empty point sets return empty tensors.
    """
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be [N, 2], got {tuple(pos.shape)}")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"points must be [P, 2], got {tuple(points.shape)}")
    if points.shape[0] == 0:
        return (
            torch.empty(0, k, dtype=torch.long, device=pos.device),
            torch.empty(0, k, dtype=pos.dtype, device=pos.device),
        )
    k = min(int(k), pos.shape[0])
    distance = torch.cdist(points, pos)
    nearest, idx = torch.topk(distance, k=k, largest=False, dim=1)
    weights = 1.0 / (nearest + eps)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return idx, weights


def sample_nodes(values: torch.Tensor, idx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Sample a nodal field ``values [N, C]`` at points given by ``idx/weights``."""
    if idx.numel() == 0:
        return values.new_zeros(0, values.shape[1])
    gathered = values[idx]  # [P, k, C]
    return (gathered * weights.unsqueeze(-1)).sum(dim=1)


def deposit_points(
    values: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Scatter per-point values ``[P, C]`` to nodes with inverse-distance weights."""
    if values.shape[0] == 0:
        return torch.zeros(num_nodes, values.shape[1], dtype=values.dtype, device=values.device)
    contributions = values.unsqueeze(1) * weights.unsqueeze(-1)  # [P, k, C]
    flat_idx = idx.reshape(-1)
    flat_values = contributions.reshape(-1, values.shape[1])
    out = torch.zeros(num_nodes, values.shape[1], dtype=values.dtype, device=values.device)
    out.index_add_(0, flat_idx, flat_values)
    return out


def masked_deposit(
    values: torch.Tensor,
    mask: torch.Tensor,
    idx: torch.Tensor,
    weights: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Deposit only the masked points (``mask`` is boolean ``[P]``)."""
    return deposit_points(values * mask.to(values.dtype).unsqueeze(-1), idx, weights, num_nodes)


__all__ = ["deposit_points", "knn_weights", "masked_deposit", "sample_nodes"]
