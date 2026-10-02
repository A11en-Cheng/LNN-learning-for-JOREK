"""Hybrid fragment dynamics: JOREK analytic backbone + learned corrections.

One ``field -> fragment`` step mirrors ``update_spi``: the shard position is
advanced with the current velocity, the radius is eroded with the ablation
rate, and the velocity receives the PELOTON rocket acceleration.  A per-shard
MLP predicts a bounded multiplicative correction to the ablation rate and an
additive acceleration correction; the correction head is zero-initialised, so
at initialisation the module reproduces the analytic backbone exactly.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from src.models.field_sampling import knn_weights, sample_nodes
from src.physics.spi_model import (
    EL_CHG,
    MU_ZERO,
    NonlocalAblationTable,
    SHARD_ABL,
    SHARD_PHI,
    SHARD_PSI,
    SHARD_R,
    SHARD_RADIUS,
    SHARD_SPECIES,
    SHARD_Z,
    erode_radius,
    parks_ablation_neon,
    particle_density_si,
    peloton_rocket_acceleration,
    temperature_ev,
)

NUM_STATE_FIELDS = 9
SHARD_VEL_R = 4
SHARD_VEL_Z = 5
SHARD_VEL_RXZ = 6
SHARD_PHI_INIT = 3


class HybridFragmentDynamics(nn.Module):
    """Predict the next shard state from the current state and the local fields."""

    def __init__(self, *, hidden: int = 64, k_neighbours: int = 4, shard_dim: int = 16) -> None:
        super().__init__()
        self.k = k_neighbours
        # features: 16 shard fields + rho/rho_imp/T_e/psi + log1p(abl) + 3 accel + dt
        self.correction = nn.Sequential(
            nn.Linear(shard_dim + 4 + 1 + 3 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),  # log ablation scale + 3 acceleration corrections
        )
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.register_buffer("state_mean", torch.zeros(NUM_STATE_FIELDS))
        self.register_buffer("state_std", torch.ones(NUM_STATE_FIELDS))
        self.register_buffer("shard_mean", torch.zeros(shard_dim))
        self.register_buffer("shard_std", torch.ones(shard_dim))
        # Bounded corrections: the analytic backbone is already close on
        # position/velocity, so the learned terms must stay small.
        self.register_buffer("accel_limit", torch.tensor(100.0))
        self.register_buffer("log_ablation_limit", torch.tensor(0.7))

    @torch.no_grad()
    def set_state_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.state_mean.copy_(torch.as_tensor(mean, dtype=self.state_mean.dtype).view(-1))
        self.state_std.copy_(
            torch.as_tensor(std, dtype=self.state_std.dtype).view(-1).clamp_min(1e-12)
        )

    @torch.no_grad()
    def set_shard_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.shard_mean.copy_(torch.as_tensor(mean, dtype=self.shard_mean.dtype).view(-1))
        self.shard_std.copy_(
            torch.as_tensor(std, dtype=self.shard_std.dtype).view(-1).clamp_min(1e-12)
        )

    def forward(
        self,
        u_raw: torch.Tensor,
        shard_values: torch.Tensor,
        shard_mask: torch.Tensor,
        pos: torch.Tensor,
        *,
        dt_jorek: torch.Tensor | float,
        t_norm: float,
        central_density: float,
        pellet_density: float = 4.30e8,
        pellet_density_bg: float | None = None,
        central_mass: float = 1.0,
        magnetic_field: float = 0.0,
        magnetic_reduction: bool = False,
        use_correction: bool = True,
        nonlocal_tables: Sequence[NonlocalAblationTable] | None = None,
        psi_axis: torch.Tensor | None = None,
        psi_bnd: torch.Tensor | None = None,
    ) -> dict:
        """Return the predicted next shard state plus analytic diagnostics."""
        batch, nodes, _ = u_raw.shape
        num_shards = shard_values.shape[-2]
        dt_tensor = torch.as_tensor(
            dt_jorek, dtype=shard_values.dtype, device=shard_values.device
        )
        if dt_tensor.ndim == 0:
            dt_tensor = dt_tensor.expand(batch)
        t_norm_tensor = torch.as_tensor(
            t_norm, dtype=shard_values.dtype, device=shard_values.device
        )
        if t_norm_tensor.ndim == 0:
            t_norm_tensor = t_norm_tensor.expand(batch)
        points = shard_values[..., :2].reshape(batch * num_shards, 2)
        idx, weights = knn_weights(pos, points, k=self.k)
        flat_states = u_raw.reshape(batch * nodes, NUM_STATE_FIELDS)

        def _sample(channel: int) -> torch.Tensor:
            sampled = sample_nodes(flat_states[:, channel : channel + 1], idx, weights)
            return sampled.squeeze(-1).reshape(batch, num_shards)

        local_rho = _sample(4)
        local_rho_imp = _sample(7)
        local_te = _sample(8)
        local_psi = _sample(0)

        radius = shard_values[..., SHARD_RADIUS]
        species = shard_values[..., SHARD_SPECIES]
        n_si = particle_density_si(local_rho, central_density)
        n_imp_si = particle_density_si(local_rho_imp, central_density)
        ne_si = (n_si - n_imp_si).clamp_min(0.0)
        te_ev = temperature_ev(local_te, central_density)
        magnetic = torch.full_like(te_ev, float(magnetic_field))
        ablation_analytic = parks_ablation_neon(
            radius,
            te_ev,
            ne_si,
            species,
            magnetic_field=magnetic,
            magnetic_reduction=magnetic_reduction,
        )
        if nonlocal_tables is not None:
            if psi_axis is None or psi_bnd is None:
                raise ValueError("nonlocal_tables requires psi_axis and psi_bnd per sample")
            psi_shard = shard_values[..., SHARD_PSI]
            te_nl = torch.zeros_like(te_ev)
            ne_nl = torch.zeros_like(ne_si)
            inside = torch.zeros_like(te_ev, dtype=torch.bool)
            for row, table in enumerate(nonlocal_tables):
                row_te, row_ne, row_inside = table.surface_state(
                    psi_shard[row], psi_axis[row], psi_bnd[row], central_density
                )
                te_nl[row] = row_te.to(te_nl.dtype).to(te_nl.device)
                ne_nl[row] = row_ne.to(ne_nl.dtype).to(ne_nl.device)
                inside[row] = row_inside.to(inside.device)
            ablation_nonlocal = parks_ablation_neon(
                radius,
                te_nl,
                ne_nl,
                species,
                magnetic_field=magnetic,
                magnetic_reduction=magnetic_reduction,
            )
            ablation_analytic = torch.where(
                inside, torch.maximum(ablation_analytic, ablation_nonlocal), ablation_analytic
            )
        delta_phi = shard_values[..., SHARD_PHI] - shard_values[..., SHARD_PHI_INIT]
        feature_a_r, feature_a_z, feature_a_rxz = peloton_rocket_acceleration(
            te_ev,
            ne_si,
            radius,
            species,
            delta_phi=delta_phi,
            pellet_density=pellet_density,
            pellet_density_bg=pellet_density_bg,
            central_mass=central_mass,
        )

        if use_correction:
            features = torch.cat(
                [
                    (torch.nan_to_num(shard_values, nan=0.0) - self.shard_mean) / self.shard_std,
                    ((local_rho - self.state_mean[4]) / self.state_std[4]).unsqueeze(-1),
                    ((local_rho_imp - self.state_mean[7]) / self.state_std[7]).unsqueeze(-1),
                    ((local_te - self.state_mean[8]) / self.state_std[8]).unsqueeze(-1),
                    local_psi.unsqueeze(-1),
                    torch.log1p(ablation_analytic).unsqueeze(-1),
                    feature_a_r.unsqueeze(-1),
                    feature_a_z.unsqueeze(-1),
                    feature_a_rxz.unsqueeze(-1),
                    dt_tensor.view(batch, 1).expand(batch, num_shards).unsqueeze(-1),
                ],
                dim=-1,
            )
            corrections = self.correction(features)
            log_ablation_correction = corrections[..., 0].clamp(
                -self.log_ablation_limit, self.log_ablation_limit
            )
            acceleration_correction = torch.tanh(corrections[..., 1:4]) * self.accel_limit
        else:
            log_ablation_correction = torch.zeros_like(ablation_analytic)
            acceleration_correction = (
                torch.zeros_like(feature_a_r).unsqueeze(-1).expand(-1, -1, 3)
            )

        ablation = ablation_analytic * torch.exp(log_ablation_correction)

        seconds = (dt_tensor * t_norm_tensor).view(batch, 1)
        columns = [shard_values[..., index] for index in range(shard_values.shape[-1])]
        if pellet_density_bg is None:
            pellet_density_bg = pellet_density
        density = torch.where(
            species > 0.5,
            torch.full_like(species, float(pellet_density)),
            torch.full_like(species, float(pellet_density_bg)),
        )
        columns[SHARD_RADIUS] = erode_radius(
            columns[SHARD_RADIUS], ablation, density * 1e20, seconds
        )
        # JOREK evaluates the rocket acceleration after the radius erosion.
        a_r, a_z, a_rxz = peloton_rocket_acceleration(
            te_ev,
            ne_si,
            columns[SHARD_RADIUS],
            species,
            delta_phi=delta_phi,
            pellet_density=pellet_density,
            pellet_density_bg=pellet_density_bg,
            central_mass=central_mass,
        )
        a_r = a_r + acceleration_correction[..., 0]
        a_z = a_z + acceleration_correction[..., 1]
        a_rxz = a_rxz + acceleration_correction[..., 2]
        # Frame-granularity integration: JOREK advances the shard in many small
        # internal sub-steps; at the recorded-frame granularity the second-order
        # (trapezoidal) position update is the faithful stand-in.
        columns[SHARD_R] = (
            shard_values[..., SHARD_R]
            + shard_values[..., SHARD_VEL_R] * seconds
            + 0.5 * a_r * seconds**2
        )
        columns[SHARD_Z] = (
            shard_values[..., SHARD_Z]
            + shard_values[..., SHARD_VEL_Z] * seconds
            + 0.5 * a_z * seconds**2
        )
        columns[SHARD_PHI] = (
            shard_values[..., SHARD_PHI]
            + shard_values[..., SHARD_VEL_RXZ] * seconds
            + 0.5 * a_rxz * seconds**2
        )
        columns[SHARD_VEL_R] = shard_values[..., SHARD_VEL_R] + a_r * seconds
        columns[SHARD_VEL_Z] = shard_values[..., SHARD_VEL_Z] + a_z * seconds
        columns[SHARD_VEL_RXZ] = shard_values[..., SHARD_VEL_RXZ] + a_rxz * seconds
        columns[SHARD_ABL] = ablation
        updated = torch.stack(columns, dim=-1)
        dead = updated[..., SHARD_RADIUS] <= 0.0
        alive = shard_mask & ~dead
        updated = torch.where(shard_mask.unsqueeze(-1), updated, torch.zeros_like(updated))
        return {
            "next": updated,
            "alive": alive,
            "ablation": ablation * shard_mask.to(ablation.dtype),
            "ablation_analytic": ablation_analytic,
            "acceleration": torch.stack([a_r, a_z, a_rxz], dim=-1),
            "log_ablation_correction": log_ablation_correction,
        }


__all__ = ["HybridFragmentDynamics"]
