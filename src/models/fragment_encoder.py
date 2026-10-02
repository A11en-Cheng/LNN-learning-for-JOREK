"""Permutation-invariant encoder for variable-size shard sets."""

from __future__ import annotations

import torch
import torch.nn as nn


class FragmentSetEncoder(nn.Module):
    """DeepSets-style encoder: per-shard MLP + masked mean/max pooling."""

    def __init__(self, in_dim: int = 16, hidden: int = 64, out_dim: int = 32) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.per_shard = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.pool = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def encode_shards(self, values: torch.Tensor) -> torch.Tensor:
        """Per-shard embeddings ``[..., S, hidden]``."""
        return self.per_shard(values)

    def forward(self, values: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Pooled embedding ``[..., out_dim]``.

        ``values`` is ``[..., S, in_dim]``; ``mask`` is ``[..., S]`` with True
        for alive shards.  An all-masked set returns a zero embedding.
        """
        if mask is None:
            mask = torch.ones(values.shape[:-1], dtype=torch.bool, device=values.device)
        encoded = self.encode_shards(values)
        mask_float = mask.to(encoded.dtype).unsqueeze(-1)
        denominator = mask_float.sum(dim=-2).clamp_min(1e-6)
        mean = (encoded * mask_float).sum(dim=-2) / denominator
        neg_inf = torch.finfo(encoded.dtype).min
        masked = encoded.masked_fill(~mask.unsqueeze(-1), neg_inf)
        maximum = masked.max(dim=-2).values
        maximum = torch.where(mask.any(dim=-1, keepdim=True), maximum, torch.zeros_like(maximum))
        return self.pool(torch.cat([mean, maximum], dim=-1))


__all__ = ["FragmentSetEncoder"]
