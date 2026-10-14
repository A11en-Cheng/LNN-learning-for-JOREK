# [v1 LEGACY — 2026-10-04] v1 source head trained with same-index
# (u_n, shard_n) -> S_n pairs; ADR-0019 requires the window-start particle state
# p_{n-1}=part{n} plus the field path (u_{n-1}, u_n). Fix the pairing before reuse.
"""Fragment-to-projection source head: analytic baseline + learned correction.

The head consumes one frame worth of node state ``u_raw`` (physical JOREK
units), the shard set ``p`` and the mesh, samples the local plasma fields at the
shard positions, evaluates the analytic JOREK ablation (model 3), deposits the
ablation rate on the mesh, and adds a node-wise learned correction for the five
projection channels.  The correction MLP is zero-initialised, so at
initialisation the head reproduces the analytic baseline exactly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.field_sampling import knn_weights, sample_nodes
from src.models.fragment_encoder import FragmentSetEncoder
from src.physics.spi_model import (
    SHARD_RADIUS,
    SHARD_SPECIES,
    parks_ablation_neon,
    temperature_ev,
    particle_density_si,
)

NUM_STATE_FIELDS = 9
NUM_SOURCE_FIELDS = 5


class SourceFieldHead(nn.Module):
    """Predict the five projection source fields from a frame and its shards."""

    def __init__(
        self,
        *,
        shard_dim: int = 16,
        hidden: int = 64,
        embed_dim: int = 32,
        k_neighbours: int = 4,
    ) -> None:
        super().__init__()
        self.k = k_neighbours
        self.encoder = FragmentSetEncoder(shard_dim, hidden, embed_dim)
        self.correction = nn.Sequential(
            nn.Linear(NUM_STATE_FIELDS + embed_dim + NUM_SOURCE_FIELDS, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, NUM_SOURCE_FIELDS),
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.channel_log_scale = nn.Parameter(torch.zeros(NUM_SOURCE_FIELDS))
        self.register_buffer("state_mean", torch.zeros(NUM_STATE_FIELDS))
        self.register_buffer("state_std", torch.ones(NUM_STATE_FIELDS))
        self.register_buffer("shard_mean", torch.zeros(shard_dim))
        self.register_buffer("shard_std", torch.ones(shard_dim))
        self.register_buffer("source_scale", torch.ones(NUM_SOURCE_FIELDS))

    @torch.no_grad()
    def set_state_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the node-feature standardisation used by the correction MLP."""
        self.state_mean.copy_(torch.as_tensor(mean, dtype=self.state_mean.dtype).view(-1))
        self.state_std.copy_(
            torch.as_tensor(std, dtype=self.state_std.dtype).view(-1).clamp_min(1e-12)
        )

    @torch.no_grad()
    def set_shard_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Set the per-field standardisation used by the fragment encoder."""
        self.shard_mean.copy_(torch.as_tensor(mean, dtype=self.shard_mean.dtype).view(-1))
        self.shard_std.copy_(
            torch.as_tensor(std, dtype=self.shard_std.dtype).view(-1).clamp_min(1e-12)
        )

    @torch.no_grad()
    def set_channel_scale(self, scale: torch.Tensor) -> None:
        """Set the analytic deposit -> projection channel scale (inverse softplus)."""
        scale = torch.as_tensor(scale, dtype=self.channel_log_scale.dtype).view(-1)
        self.channel_log_scale.copy_(torch.log(torch.expm1(scale.clamp_min(1e-30))))

    @torch.no_grad()
    def set_source_scale(self, scale: torch.Tensor) -> None:
        """Set the per-channel magnitude used to normalise the learned correction."""
        self.source_scale.copy_(
            torch.as_tensor(scale, dtype=self.source_scale.dtype).view(-1).clamp_min(1e-30)
        )

    def forward(
        self,
        u_raw: torch.Tensor,
        shard_values: torch.Tensor,
        shard_mask: torch.Tensor,
        pos: torch.Tensor,
        *,
        central_density: float,
        magnetic_field: float = 0.0,
        magnetic_reduction: bool = False,
    ) -> dict:
        """``u_raw``: ``[B, N, 9]``; ``shard_values``: ``[B, S, F]``; ``shard_mask``: ``[B, S]``."""
        if u_raw.ndim != 3 or u_raw.shape[-1] != NUM_STATE_FIELDS:
            raise ValueError(f"u_raw must be [B, N, 9], got {tuple(u_raw.shape)}")
        batch, nodes, _ = u_raw.shape
        shard_points = shard_values[..., :2].reshape(batch * shard_values.shape[-2], 2)
        idx, weights = knn_weights(pos, shard_points, k=self.k)

        flat_mask = shard_mask.reshape(-1)
        radius = shard_values[..., SHARD_RADIUS].reshape(-1)
        species = shard_values[..., SHARD_SPECIES].reshape(-1)
        alive = flat_mask

        def _sample(channel: int) -> torch.Tensor:
            field = u_raw.reshape(batch * nodes, NUM_STATE_FIELDS)[:, channel]
            sampled = sample_nodes(field.unsqueeze(-1), idx, weights).squeeze(-1)
            return sampled

        local_rho = _sample(4)
        local_rho_imp = _sample(7)
        local_te = _sample(8)
        n_si = particle_density_si(local_rho, central_density)
        n_imp_si = particle_density_si(local_rho_imp, central_density)
        ne_si = (n_si - n_imp_si).clamp_min(0.0)
        te_ev = temperature_ev(local_te, central_density)
        b_field = torch.full_like(te_ev, float(magnetic_field))
        ablation = parks_ablation_neon(
            radius.clamp_min(0.0),
            te_ev,
            ne_si,
            species,
            magnetic_field=b_field,
            magnetic_reduction=magnetic_reduction,
        )
        ablation = ablation * alive.to(ablation.dtype)
        # Batched deposition: each sample gets its own node copy so shards from
        # different samples never mix.
        batch_offset = torch.arange(batch, device=u_raw.device).repeat_interleave(
            shard_values.shape[-2]
        )
        flat_idx = idx + (batch_offset * nodes).unsqueeze(-1)
        contributions = ablation.unsqueeze(-1).unsqueeze(1) * weights.unsqueeze(-1)
        flat_deposit = torch.zeros(batch * nodes, 1, dtype=u_raw.dtype, device=u_raw.device)
        flat_deposit.index_add_(0, flat_idx.reshape(-1), contributions.reshape(-1, 1))
        deposit = flat_deposit.reshape(batch, nodes)

        shard_norm = (torch.nan_to_num(shard_values, nan=0.0) - self.shard_mean) / self.shard_std
        embedding = self.encoder(shard_norm, shard_mask)
        u_norm = (u_raw - self.state_mean) / self.state_std
        channel_scale = torch.nn.functional.softplus(self.channel_log_scale)
        baseline = deposit.unsqueeze(-1) * channel_scale
        baseline_norm = baseline / self.source_scale
        node_features = torch.cat(
            [
                u_norm,
                baseline_norm,
                embedding.unsqueeze(1).expand(batch, nodes, embedding.shape[-1]),
            ],
            dim=-1,
        )
        correction = self.correction(node_features) * self.source_scale
        return {
            "source": baseline + correction,
            "baseline": baseline,
            "correction": correction,
            "ablation": (ablation * alive.to(ablation.dtype)).reshape(batch, -1),
            "deposit": deposit,
        }


__all__ = ["SourceFieldHead"]
