import os
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.data.dataset import collate_temporal_graphs  # noqa: F401 (for type hints / consistency)
from src.data.model502_schema import (
    FIELD_INDEX,
    FIELD_NAMES,
    transform_state,
    validate_background_source_tensor,
)
from src.models.loss import JorekPhysicsLoss, compute_normalized_flux
from src.physics.grad_ops import compute_gradient, compute_divergence
from src.physics.mhd import get_magnetic_field
from src.physics.cocos8 import reconstruct_velocity


CHANNEL_NAMES = list(FIELD_NAMES)


class JorekEvaluator:
    """Standardized benchmark evaluator for HyPhy-GLNN on JOREK sequences.

    This class performs long-horizon autoregressive rollout on held-out
    sequences and computes four categories of metrics:
    1) Accuracy vs. persistence baseline (relative RMSE per channel).
    2) Physics diagnostics (global mass drift, PDE residuals).
    3) Magnetic axis tracking (R_axis, Z_axis trajectory and error).
    4) Performance (wall-clock speed vs. notional JOREK CPU time).
    """

    def __init__(self, model: Any, config: Dict[str, Any], device: torch.device):
        """Args:
        model: Trained wrapper model containing encoder/processor/decoder, or
               an object exposing attributes encoder, processor, decoder with
               the same interfaces as JorekGNNEncoder/JorekLNNProcessor/
               JorekGNNDecoder. The model is expected to operate in the
               *normalized* space and predict residuals Δx.
        config: Dict-like configuration. Recommended keys:
            - 'stats': {'mean': Tensor[9], 'std': Tensor[9]} for Z-score.
            - 'dt': float physical time step for PDE residuals (fallback=1.0).
            - 'seq_len': int maximum rollout length (optional).
            - 'warmup_steps': int LNN warmup steps using teacher forcing.
            - 'rollout_steps': int free-run autoregressive steps.
            - 'vis_dir': directory to save benchmark plots.
            - 'jorek_cpu_hours_per_sequence': float, reference CPU hours.
        device: torch device for evaluation (cuda or cpu).
        """
        self.model = model
        self.device = device
        self.config = dict(config) if config is not None else {}

        # Normalization statistics for (log1p-rho/Te + Z-score) space.
        self.stats: Optional[Dict[str, torch.Tensor]] = self.config.get("stats")
        if self.stats is None and hasattr(model, "stats"):
            self.stats = getattr(model, "stats")

        # Physics-loss helper for consistent denormalization.
        dt = float(self.config.get("dt", 1.0))
        self.physics_loss_helper = JorekPhysicsLoss(
            lambda_data=0.0, lambda_phys=0.0, lambda_bc=0.0,
            stats=self.stats, dt=dt,
        )
        self.dt = dt

        # Rollout configuration
        self.seq_len: Optional[int] = self.config.get("seq_len")
        self.warmup_steps: int = int(self.config.get("warmup_steps", 0))
        self.rollout_steps: Optional[int] = self.config.get("rollout_steps")

        # Performance reference
        self.jorek_cpu_hours_per_sequence: float = float(
            self.config.get("jorek_cpu_hours_per_sequence", 100.0)
        )

        # Output directory for visualizations
        self.vis_dir: str = self.config.get("vis_dir", "output/benchmark")
        os.makedirs(self.vis_dir, exist_ok=True)

        # Containers for aggregated metrics (filled by run_autoregressive_rollout)
        self.metrics: Dict[str, Any] = {}
        self.speedup_factor: Optional[float] = None
        self.avg_wallclock_per_sequence: Optional[float] = None

    # ------------------------------------------------------------------
    # Helper: geometric / volume weights for global mass
    # ------------------------------------------------------------------
    def _get_mass_weights(self, graph_data: Any) -> torch.Tensor:
        """Return per-node geometric weights for mass integration.

        Preference order (if present on graph_data and length == num_nodes):
          1) cell_volume / volume / vol
          2) cell_area / area
          3) generic weights / jacobian

        If none are available, falls back to uniform weights = 1.0.

        This keeps the benchmark backward-compatible while allowing
        more physically faithful total-mass diagnostics when the
        dataset provides geometric factors.
        """
        device = self.device
        pos = getattr(graph_data, "pos", None)
        if pos is None:
            raise AttributeError("graph_data must have 'pos' with node coordinates for mass weights")

        num_nodes = pos.size(0)
        ones = torch.ones(num_nodes, 1, device=device)

        candidate_attrs = [
            "cell_volume",
            "volume",
            "vol",
            "cell_area",
            "area",
            "weights",
            "jacobian",
        ]

        for attr in candidate_attrs:
            if hasattr(graph_data, attr):
                val = getattr(graph_data, attr)
                if val is None:
                    continue
                if not isinstance(val, torch.Tensor):
                    val = torch.as_tensor(val, dtype=torch.float32, device=device)
                else:
                    val = val.to(device)
                val = val.view(-1, 1)
                if val.size(0) == num_nodes:
                    return val

        # Fallback: uniform weights
        return ones

    # ------------------------------------------------------------------
    # Normalization / denormalization helpers
    # ------------------------------------------------------------------
    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized features back to physical space.

        This reuses the exact logic from JorekPhysicsLoss.denormalize:
        1) Reverse Z-score using global mean/std.
        2) Reverse log1p for rho/Te and clamp to positive values.

        Args:
            x: [N, 5] tensor in normalized space.
        Returns:
            x_phys: [N, 5] tensor in physical units.
        """
        if self.physics_loss_helper is None or self.physics_loss_helper.stats is None:
            return x
        return self.physics_loss_helper.denormalize(x)

    # ------------------------------------------------------------------
    # Core rollout
    # ------------------------------------------------------------------
    @torch.no_grad()
    def run_autoregressive_rollout(self, dataloader) -> Dict[str, Any]:
        """Perform autoregressive rollout over all sequences in the dataloader.

        The dataloader is expected to yield temporal sequences in the form
        produced by collate_temporal_graphs: a list "batch_sequence" of length
        T, where each element is a PyG Batch (or single Data) at time t.

        Workflow per sequence:
          1) Preprocess all frames with log1p (rho/Te) + global Z-score.
          2) Warm up the LNN hidden state for `warmup_steps` using
             teacher forcing on ground-truth frames.
          3) Start from the ground-truth state at warmup_steps, then
             autoregressively roll out for up to `rollout_steps` steps,
             always feeding the *predicted* state as the next input.
          4) At each free-run step, compute accuracy, persistence baseline,
             physics metrics, and magnetic axis diagnostics.

        Returns:
            metrics: nested dict with per-time-step aggregated statistics.
        """
        encoder = getattr(self.model, "encoder", None)
        processor = getattr(self.model, "processor", None)
        decoder = getattr(self.model, "decoder", None)
        if encoder is None or processor is None or decoder is None:
            raise AttributeError(
                "model must expose encoder, processor, decoder attributes"
            )

        encoder.eval()
        processor.eval()
        decoder.eval()

        # Per-step aggregators across all sequences
        step_rel_rmse = {ch: {} for ch in CHANNEL_NAMES}      # step_idx -> list[float]
        step_persistence = {ch: {} for ch in CHANNEL_NAMES}   # step_idx -> list[float]
        step_mass_drift = {}                                  # step_idx -> list[float]
        step_pde_mass = {}                                    # step_idx -> list[float]
        step_pde_mom = {}                                     # step_idx -> list[float]
        step_axis_err = {}                                    # step_idx -> list[float]
        step_mass_error = {}                                  # step_idx -> list[float] (pred vs GT)
        # For axis trajectories, store all points (flatten over sequences)
        axis_R_gt_all: List[float] = []
        axis_Z_gt_all: List[float] = []
        axis_R_pred_all: List[float] = []
        axis_Z_pred_all: List[float] = []
        # For per-step average R/Z evolution (GT vs Pred)
        step_axis_R_gt: Dict[int, List[float]] = {}
        step_axis_Z_gt: Dict[int, List[float]] = {}
        step_axis_R_pred: Dict[int, List[float]] = {}
        step_axis_Z_pred: Dict[int, List[float]] = {}

        total_sequences = 0
        wallclock_times: List[float] = []

        for batch_sequence in dataloader:
            # batch_sequence: List[Data or Batch] with temporal order
            if not isinstance(batch_sequence, (list, tuple)):
                # Single sequence given as plain Data/Batch; wrap as length-1
                batch_sequence = [batch_sequence]

            T = len(batch_sequence)
            if T < 2:
                continue

            # Determine rollout horizon for this sequence
            seq_len_cfg = self.seq_len if self.seq_len is not None else T
            max_len = min(seq_len_cfg, T)
            warmup = min(self.warmup_steps, max_len - 1)
            # Free-run steps cannot exceed available frames
            max_rollout = max_len - warmup - 1
            if self.rollout_steps is not None:
                max_rollout = min(max_rollout, int(self.rollout_steps))
            if max_rollout <= 0:
                continue

            # ------------------------------------------------------------------
            # 1) Preprocess: move to device, log-space + Z-score normalization
            # ------------------------------------------------------------------
            processed_sequence = []
            for data in batch_sequence[: max_len]:
                d = data.to(self.device)
                if hasattr(d, "x"):
                    new_x = transform_state(d.x)

                    if self.stats is None:
                        raise RuntimeError(
                            "Global stats (mean/std) must be provided in config['stats'] "
                            "or as model.stats for proper evaluation."
                        )
                    mean = self.stats["mean"].to(self.device)
                    std = self.stats["std"].to(self.device)
                    d.x = (new_x - mean) / std
                processed_sequence.append(d)

            # Cache normalized states for ground truth at each t
            gt_norm_sequence = [d.x for d in processed_sequence]

            # Geometric weights for global mass integration (fixed over time)
            mass_weights = self._get_mass_weights(processed_sequence[0])  # [N, 1]

            # Baseline initial physical state x0 (for persistence baseline)
            x0_norm = gt_norm_sequence[0]
            x0_phys = self.denormalize(x0_norm)

            # Global mass at t=0 (sum rho)
            rho0 = x0_phys[:, FIELD_INDEX["rho"]:FIELD_INDEX["rho"] + 1]
            if rho0.numel() > 0:
                M0 = torch.sum(rho0 * mass_weights).item()
            else:
                M0 = 1.0

            # ------------------------------------------------------------------
            # 2) Warm-up LNN hidden state using teacher forcing
            # ------------------------------------------------------------------
            hidden_state = None
            for t in range(warmup):
                current_data = processed_sequence[t]
                next_data = processed_sequence[t + 1]

                latent_x = encoder(current_data.x, current_data.edge_index, current_data.pos)

                # Time-step for CfC (Δt between current and next frame)
                dt_scalar = None
                if hasattr(current_data, "time") and hasattr(next_data, "time"):
                    t_curr = current_data.time.to(self.device).view(-1)
                    t_next = next_data.time.to(self.device).view(-1)
                    dt_vals = t_next - t_curr
                    dt_scalar = dt_vals.mean()

                _, hidden_state = processor(
                    latent_x,
                    getattr(current_data, "batch", None),
                    state=hidden_state,
                    time_steps=dt_scalar,
                )

            # At the end of warmup, we start from ground-truth state at t=warmup
            current_pred_norm = gt_norm_sequence[warmup]

            # ------------------------------------------------------------------
            # 3) Free-run autoregressive rollout
            # ------------------------------------------------------------------
            t_start_seq = time.perf_counter()

            for k in range(1, max_rollout + 1):
                # Global time indices: previous frame index and current index
                t_prev = warmup + k - 1
                t_curr = warmup + k
                if t_curr >= max_len:
                    break

                current_data = processed_sequence[t_prev]
                next_data_gt = processed_sequence[t_curr]

                # Encode predicted current state (residual evolution in normalized space)
                latent_x = encoder(current_pred_norm, current_data.edge_index, current_data.pos)

                dt_scalar = None
                if hasattr(current_data, "time") and hasattr(next_data_gt, "time"):
                    t_prev_tensor = current_data.time.to(self.device).view(-1)
                    t_curr_tensor = next_data_gt.time.to(self.device).view(-1)
                    dt_vals = t_curr_tensor - t_prev_tensor
                    dt_scalar = dt_vals.mean()

                next_latent_x, hidden_state = processor(
                    latent_x,
                    getattr(current_data, "batch", None),
                    state=hidden_state,
                    time_steps=dt_scalar,
                )

                pred_delta_norm, _ = decoder(
                    next_latent_x,
                    current_data.edge_index,
                    getattr(current_data, "batch", None),
                    current_data.pos,
                )

                next_pred_norm = current_pred_norm + pred_delta_norm

                # Ground-truth normalized state at t_curr
                gt_norm = gt_norm_sequence[t_curr]

                # Denormalize predictions and ground truth to physical units
                x_pred_phys = self.denormalize(next_pred_norm)
                x_gt_phys = self.denormalize(gt_norm)

                # ------------------------------------------------------------------
                # Accuracy metrics (per channel) vs GT and persistence baseline
                # ------------------------------------------------------------------
                step_idx = k  # rollout step index (1-based after warmup)

                for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
                    diff = x_pred_phys[:, ch_idx] - x_gt_phys[:, ch_idx]
                    num = torch.norm(diff.reshape(-1), p=2)
                    denom = torch.norm(x_gt_phys[:, ch_idx].reshape(-1), p=2) + 1e-8
                    rel_rmse_val = (num / denom).item()

                    # Persistence baseline: always keep x0
                    diff_p = x_gt_phys[:, ch_idx] - x0_phys[:, ch_idx]
                    num_p = torch.norm(diff_p.reshape(-1), p=2)
                    denom_p = denom  # same denominator
                    persistence_val = (num_p / (denom_p + 1e-8)).item()

                    step_rel_rmse[ch_name].setdefault(step_idx, []).append(rel_rmse_val)
                    step_persistence[ch_name].setdefault(step_idx, []).append(persistence_val)

                # ------------------------------------------------------------------
                # Global mass conservation: drift of sum(rho)
                # ------------------------------------------------------------------
                rho_pred = x_pred_phys[:, FIELD_INDEX["rho"]:FIELD_INDEX["rho"] + 1]
                rho_gt = x_gt_phys[:, FIELD_INDEX["rho"]:FIELD_INDEX["rho"] + 1]

                if rho_pred.numel() > 0:
                    M_pred = torch.sum(rho_pred * mass_weights).item()
                    M_gt = torch.sum(rho_gt * mass_weights).item()
                else:
                    M_pred = 0.0
                    M_gt = 0.0

                # (1) Drift of predicted total mass vs initial mass M0
                mass_drift = (M_pred - M0) / (M0 + 1e-8)
                step_mass_drift.setdefault(step_idx, []).append(mass_drift)

                # (2) Predicted vs ground-truth total mass error at each step
                mass_error = (M_pred - M_gt) / (M_gt + 1e-8)
                step_mass_error.setdefault(step_idx, []).append(mass_error)

                # ------------------------------------------------------------------
                # Magnetic axis tracking (argmin psi per graph)
                # ------------------------------------------------------------------
                psi_pred = x_pred_phys[:, 0]
                psi_gt = x_gt_phys[:, 0]
                batch_index = getattr(current_data, "batch", None)

                if batch_index is None:
                    # Single-graph sequence
                    idx_pred = torch.argmin(psi_pred)
                    idx_gt = torch.argmin(psi_gt)
                    R = current_data.pos[:, 0]
                    Z = current_data.pos[:, 1]
                    R_pred, Z_pred = R[idx_pred].item(), Z[idx_pred].item()
                    R_gt, Z_gt = R[idx_gt].item(), Z[idx_gt].item()

                    axis_R_gt_all.append(R_gt)
                    axis_Z_gt_all.append(Z_gt)
                    axis_R_pred_all.append(R_pred)
                    axis_Z_pred_all.append(Z_pred)

                    # Per-step collections for R/Z evolution
                    step_axis_R_gt.setdefault(step_idx, []).append(R_gt)
                    step_axis_Z_gt.setdefault(step_idx, []).append(Z_gt)
                    step_axis_R_pred.setdefault(step_idx, []).append(R_pred)
                    step_axis_Z_pred.setdefault(step_idx, []).append(Z_pred)

                    axis_err = ((R_pred - R_gt) ** 2 + (Z_pred - Z_gt) ** 2) ** 0.5
                    step_axis_err.setdefault(step_idx, []).append(float(axis_err))
                else:
                    # Batched graphs: compute axis per graph and aggregate
                    batch_index = batch_index.to(self.device)
                    num_graphs = int(batch_index.max().item()) + 1
                    R_all = current_data.pos[:, 0]
                    Z_all = current_data.pos[:, 1]

                    for g in range(num_graphs):
                        mask_g = batch_index == g
                        if not mask_g.any():
                            continue
                        psi_pred_g = psi_pred[mask_g]
                        psi_gt_g = psi_gt[mask_g]

                        # Axis at argmin psi (standard JOREK convention)
                        local_idx_pred = torch.argmin(psi_pred_g)
                        local_idx_gt = torch.argmin(psi_gt_g)

                        # Map back to global indices
                        global_indices = torch.nonzero(mask_g, as_tuple=False).view(-1)
                        gi_pred = global_indices[local_idx_pred]
                        gi_gt = global_indices[local_idx_gt]

                        R_pred = R_all[gi_pred].item()
                        Z_pred = Z_all[gi_pred].item()
                        R_gt = R_all[gi_gt].item()
                        Z_gt = Z_all[gi_gt].item()

                        axis_R_gt_all.append(R_gt)
                        axis_Z_gt_all.append(Z_gt)
                        axis_R_pred_all.append(R_pred)
                        axis_Z_pred_all.append(Z_pred)

                        # Per-step collections for R/Z evolution
                        step_axis_R_gt.setdefault(step_idx, []).append(R_gt)
                        step_axis_Z_gt.setdefault(step_idx, []).append(Z_gt)
                        step_axis_R_pred.setdefault(step_idx, []).append(R_pred)
                        step_axis_Z_pred.setdefault(step_idx, []).append(Z_pred)

                        axis_err = ((R_pred - R_gt) ** 2 + (Z_pred - Z_gt) ** 2) ** 0.5
                        step_axis_err.setdefault(step_idx, []).append(float(axis_err))

                # ------------------------------------------------------------------
                # PDE residual diagnostics (mass & parallel momentum)
                # ------------------------------------------------------------------
                # curr_state: predicted state at t_prev; next_state: predicted at t_curr
                x_curr_norm = current_pred_norm
                x_next_norm = next_pred_norm

                res_mass_mean, res_mom_mean = self._compute_pde_residuals(
                    x_curr_norm, x_next_norm, current_data, dt_override=dt_scalar
                )
                step_pde_mass.setdefault(step_idx, []).append(res_mass_mean)
                step_pde_mom.setdefault(step_idx, []).append(res_mom_mean)

                # Update current state for next AR step
                current_pred_norm = next_pred_norm

            t_end_seq = time.perf_counter()
            wallclock_times.append(t_end_seq - t_start_seq)
            total_sequences += 1

        # ------------------------------------------------------------------
        # Aggregate metrics across sequences per rollout step
        # ------------------------------------------------------------------
        def _aggregate_step_dict(step_dict: Dict[int, List[float]]) -> Tuple[List[int], List[float]]:
            steps_sorted = sorted(step_dict.keys())
            values_mean = []
            for s in steps_sorted:
                vals = step_dict[s]
                if len(vals) == 0:
                    values_mean.append(float("nan"))
                else:
                    values_mean.append(float(sum(vals) / len(vals)))
            return steps_sorted, values_mean

        metrics: Dict[str, Any] = {}

        # Accuracy metrics per channel
        metrics["steps"] = None  # will be filled from first channel
        metrics["rel_rmse"] = {}
        metrics["persistence"] = {}
        for ch in CHANNEL_NAMES:
            s_rmse, v_rmse = _aggregate_step_dict(step_rel_rmse[ch])
            s_pers, v_pers = _aggregate_step_dict(step_persistence[ch])
            # Ensure shared step axis
            if metrics["steps"] is None:
                metrics["steps"] = s_rmse
            metrics["rel_rmse"][ch] = v_rmse
            metrics["persistence"][ch] = v_pers

        # Physics metrics
        s_mass, v_mass = _aggregate_step_dict(step_mass_drift)
        s_mass_err, v_mass_err = _aggregate_step_dict(step_mass_error)
        s_pde_m, v_pde_m = _aggregate_step_dict(step_pde_mass)
        s_pde_p, v_pde_p = _aggregate_step_dict(step_pde_mom)
        # Use same steps axis as accuracy (assume identical)
        metrics["mass_drift"] = v_mass
        metrics["mass_error_vs_gt"] = v_mass_err
        metrics["pde_res_mass"] = v_pde_m
        metrics["pde_res_momentum"] = v_pde_p

        # Axis error
        s_axis, v_axis = _aggregate_step_dict(step_axis_err)
        metrics["axis_error"] = v_axis
        metrics["axis_R_gt"] = axis_R_gt_all
        metrics["axis_Z_gt"] = axis_Z_gt_all
        metrics["axis_R_pred"] = axis_R_pred_all
        metrics["axis_Z_pred"] = axis_Z_pred_all

        # Axis R/Z evolution per rollout step (average over all sequences/graphs)
        _, axis_R_gt_series = _aggregate_step_dict(step_axis_R_gt)
        _, axis_Z_gt_series = _aggregate_step_dict(step_axis_Z_gt)
        _, axis_R_pred_series = _aggregate_step_dict(step_axis_R_pred)
        _, axis_Z_pred_series = _aggregate_step_dict(step_axis_Z_pred)
        metrics["axis_R_gt_series"] = axis_R_gt_series
        metrics["axis_Z_gt_series"] = axis_Z_gt_series
        metrics["axis_R_pred_series"] = axis_R_pred_series
        metrics["axis_Z_pred_series"] = axis_Z_pred_series

        # Performance metrics
        if total_sequences > 0 and len(wallclock_times) > 0:
            avg_wallclock = sum(wallclock_times) / len(wallclock_times)
        else:
            avg_wallclock = float("nan")

        self.avg_wallclock_per_sequence = avg_wallclock
        # Reference JOREK CPU time per sequence (in seconds)
        ref_cpu_seconds = self.jorek_cpu_hours_per_sequence * 3600.0
        if avg_wallclock > 0.0:
            self.speedup_factor = ref_cpu_seconds / avg_wallclock
        else:
            self.speedup_factor = None

        metrics["avg_wallclock_per_sequence"] = avg_wallclock
        metrics["speedup_factor_vs_jorek_cpu"] = self.speedup_factor
        metrics["total_sequences"] = total_sequences

        self.metrics = metrics
        return metrics

    # ------------------------------------------------------------------
    # PDE residual computation (reusing physics kernel)
    # ------------------------------------------------------------------
    def _compute_pde_residuals(
        self,
        curr_state_norm: torch.Tensor,
        next_state_norm: torch.Tensor,
        graph_data: Any,
        dt_override: Optional[torch.Tensor] = None,
    ) -> Tuple[float, float]:
        """Compute mean dimensionless PDE residuals for a single frame.

        This mirrors the internal construction in JorekPhysicsLoss, but
        operates purely on the predicted states (no data term or BC term)
        and returns the mean absolute residuals for diagnostics.

        Returns:
            (mean_res_mass, mean_res_momentum)
        """
        device = curr_state_norm.device

        # Use per-step dt from data.time when provided, otherwise fall back
        # to the evaluator-level dt from config.
        if dt_override is not None:
            try:
                dt = float(dt_override.detach().cpu().item())
            except Exception:
                dt = float(self.dt)
        else:
            dt = float(self.dt)

        # Denormalize to physical space
        curr_denorm = self.denormalize(curr_state_norm)
        next_denorm = self.denormalize(next_state_norm)

        rho_curr = curr_denorm[:, FIELD_INDEX["rho"]:FIELD_INDEX["rho"] + 1]
        v_par_curr = curr_denorm[:, FIELD_INDEX["v_par"]:FIELD_INDEX["v_par"] + 1]

        psi_next = next_denorm[:, FIELD_INDEX["psi"]:FIELD_INDEX["psi"] + 1]
        Phi_next = next_denorm[:, FIELD_INDEX["Phi"]:FIELD_INDEX["Phi"] + 1]
        rho_next = next_denorm[:, FIELD_INDEX["rho"]:FIELD_INDEX["rho"] + 1]
        t_i_next = next_denorm[:, FIELD_INDEX["T_i"]:FIELD_INDEX["T_i"] + 1]
        t_e_next = next_denorm[:, FIELD_INDEX["T_e"]:FIELD_INDEX["T_e"] + 1]
        v_par_next = next_denorm[:, FIELD_INDEX["v_par"]:FIELD_INDEX["v_par"] + 1]

        R = graph_data.pos[:, 0:1]

        # F0 normalization factor
        F0 = 1.0
        if hasattr(graph_data, "norm_factors") and graph_data.norm_factors is not None:
            nf = graph_data.norm_factors
            if isinstance(nf, dict) and "F0" in nf:
                F0 = nf["F0"]
            elif hasattr(nf, "F0"):
                F0 = nf.F0
        if isinstance(F0, torch.Tensor):
            if hasattr(graph_data, "batch") and graph_data.batch is not None:
                F0 = F0[graph_data.batch].view(-1, 1)
            else:
                F0 = F0[0].item() if F0.numel() > 0 else 1.0

        # Spatial gradients in physical space
        grad_rho = compute_gradient(rho_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_v_par = compute_gradient(v_par_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_Phi = compute_gradient(Phi_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_psi = compute_gradient(psi_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)

        # Reconstruct B and v fields
        B_field = get_magnetic_field(grad_psi, R, F0)
        v_field = reconstruct_velocity(grad_Phi, v_par_next, B_field, R)

        v_R = v_field[:, 0:1]
        v_Z = v_field[:, 1:2]

        # Mass continuity residual
        rho_v_R = rho_next * v_R
        rho_v_Z = rho_next * v_Z
        rho_v_pol = torch.cat([rho_v_R, rho_v_Z], dim=-1)

        div_cart_rho_v = compute_divergence(
            rho_v_pol, graph_data.edge_index, graph_data.edge_attr
        ).unsqueeze(-1)
        div_rho_v = div_cart_rho_v + rho_v_R / (R + 1e-8)

        source_rho = torch.zeros_like(rho_next)
        if hasattr(graph_data, "background_source") and graph_data.background_source is not None:
            validate_background_source_tensor(
                graph_data.background_source, name="benchmark background source"
            )
            source_rho = graph_data.background_source[:, 0:1].to(
                device=device, dtype=rho_next.dtype
            )
        res_mass_raw = (rho_next - rho_curr) / dt + div_rho_v - source_rho
        res_mass = (res_mass_raw * dt) / (torch.abs(rho_curr) + 1e-3)

        # Parallel momentum residual
        v_dot_grad_v_par = v_R * grad_v_par[:, 0:1] + v_Z * grad_v_par[:, 1:2]
        pressure_next = rho_next * (t_i_next + t_e_next)
        grad_pressure = compute_gradient(
            pressure_next, graph_data.edge_index, graph_data.edge_attr
        ).squeeze(1)

        B_R = B_field[:, 0:1]
        B_Z = B_field[:, 1:2]
        B_mag = torch.norm(B_field, p=2, dim=-1, keepdim=True) + 1e-8

        B_dot_grad_pressure = B_R * grad_pressure[:, 0:1] + B_Z * grad_pressure[:, 1:2]
        grad_par_pressure = B_dot_grad_pressure / B_mag

        res_momentum_raw = rho_next * (
            (v_par_next - v_par_curr) / dt + v_dot_grad_v_par
        ) + grad_par_pressure

        v_ref = torch.abs(v_par_curr) + 0.1
        res_momentum = (res_momentum_raw * dt) / (
            torch.abs(rho_curr) * v_ref + 1e-3
        )

        # Core/SOL mask using normalized flux if available
        boundary_tensor = None
        if hasattr(graph_data, "boundary") and graph_data.boundary is not None:
            boundary = graph_data.boundary
            if isinstance(boundary, torch.Tensor):
                boundary_tensor = boundary.to(device).view(-1)
            else:
                boundary_tensor = torch.tensor(boundary, dtype=torch.long, device=device).view(-1)

        internal_mask = None
        if boundary_tensor is not None:
            internal_mask = boundary_tensor == 0

        if internal_mask is not None and internal_mask.any():
            mask_core = internal_mask.clone()

            psi_axis = getattr(graph_data, "psi_axis", None)
            psi_bnd = getattr(graph_data, "psi_bnd", None)
            batch_index = getattr(graph_data, "batch", None)

            if psi_axis is not None and psi_bnd is not None and batch_index is not None:
                psi_field = psi_next  # [N, 1]
                psi_n = compute_normalized_flux(
                    psi_field, batch_index, psi_axis, psi_bnd
                )
                mask_core = (psi_n <= 1.0) & internal_mask
                if not mask_core.any():
                    mask_core = internal_mask
        else:
            mask_core = None

        if mask_core is not None and mask_core.any():
            res_mass_use = res_mass[mask_core]
            res_mom_use = res_momentum[mask_core]
        else:
            res_mass_use = res_mass
            res_mom_use = res_momentum

        mean_res_mass = torch.mean(torch.abs(res_mass_use)).item()
        mean_res_mom = torch.mean(torch.abs(res_mom_use)).item()
        return mean_res_mass, mean_res_mom

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------
    def plot_benchmarks(self, vis_dir: Optional[str] = None) -> None:
        """Generate standard benchmark plots from the last run.

        Plots saved:
          1) rel_rmse_vs_time.png: Relative RMSE vs time step for each
             channel, with persistence baseline.
          2) mass_conservation.png: Global mass drift (%) vs time.
          3) axis_trajectory_RZ.png: 2D R-Z scatter/trajectory of GT vs
             predicted magnetic axis.
          4) pde_residuals.png: Log-scale mean PDE residuals vs time.
        """
        if not self.metrics:
            raise RuntimeError(
                "No metrics found. Please run run_autoregressive_rollout() first."
            )

        if vis_dir is None:
            vis_dir = self.vis_dir
        os.makedirs(vis_dir, exist_ok=True)

        steps = self.metrics.get("steps", [])
        if steps is None:
            steps = list(range(1, len(next(iter(self.metrics["rel_rmse"].values()))) + 1))

        # 1) Error evolution (per channel, with persistence baseline)
        plt.figure(figsize=(10, 8))
        for idx, ch in enumerate(CHANNEL_NAMES):
            rmse_vals = self.metrics["rel_rmse"].get(ch, [])
            pers_vals = self.metrics["persistence"].get(ch, [])
            plt.subplot(3, 2, idx + 1)
            plt.plot(steps, rmse_vals, label=f"Rel RMSE ({ch})")
            plt.plot(steps, pers_vals, "k--", label="Persistence Baseline")
            plt.xlabel("Rollout Step")
            plt.ylabel("Relative RMSE")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
        if len(CHANNEL_NAMES) < 6:
            # Hide unused subplot
            for extra in range(len(CHANNEL_NAMES) + 1, 7):
                plt.subplot(3, 2, extra)
                plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, "rel_rmse_vs_time.png"))
        plt.close()

        # 2) Mass conservation
        mass_drift = self.metrics.get("mass_drift", [])
        mass_error_vs_gt = self.metrics.get("mass_error_vs_gt", [])
        if mass_drift:
            plt.figure(figsize=(6, 4))
            drift_percent = [m * 100.0 for m in mass_drift]
            plt.plot(steps, drift_percent, marker="o", label="Drift vs initial mass")

            if mass_error_vs_gt:
                err_percent = [e * 100.0 for e in mass_error_vs_gt]
                plt.plot(steps, err_percent, marker="s", linestyle="--", label="Pred vs GT mass")

            plt.xlabel("Rollout Step")
            plt.ylabel("Relative Mass Error (%)")
            plt.title("Global Mass Diagnostics")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.savefig(os.path.join(vis_dir, "mass_conservation.png"))
            plt.close()

        # 3) Magnetic axis trajectory in R-Z plane
        axis_R_gt = self.metrics.get("axis_R_gt", [])
        axis_Z_gt = self.metrics.get("axis_Z_gt", [])
        axis_R_pred = self.metrics.get("axis_R_pred", [])
        axis_Z_pred = self.metrics.get("axis_Z_pred", [])

        if axis_R_gt and axis_R_pred:
            plt.figure(figsize=(6, 6))
            plt.scatter(axis_R_gt, axis_Z_gt, c="blue", s=10, label="GT Axis")
            plt.scatter(axis_R_pred, axis_Z_pred, c="red", s=10, label="Pred Axis")
            plt.xlabel("R")
            plt.ylabel("Z")
            plt.title("Magnetic Axis Trajectory (R-Z Plane)")
            plt.axis("equal")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(vis_dir, "axis_trajectory_RZ.png"))
            plt.close()

        # 3b) Magnetic axis R(t), Z(t) evolution (GT vs Pred, per rollout step)
        axis_R_gt_series = self.metrics.get("axis_R_gt_series", [])
        axis_Z_gt_series = self.metrics.get("axis_Z_gt_series", [])
        axis_R_pred_series = self.metrics.get("axis_R_pred_series", [])
        axis_Z_pred_series = self.metrics.get("axis_Z_pred_series", [])

        if axis_R_gt_series and axis_R_pred_series:
            plt.figure(figsize=(10, 4))

            # R(t)
            plt.subplot(1, 2, 1)
            plt.plot(steps, axis_R_gt_series, label="R_axis GT", color="blue")
            plt.plot(steps, axis_R_pred_series, label="R_axis Pred", color="red", linestyle="--")
            plt.xlabel("Rollout Step")
            plt.ylabel("R (m)")
            plt.title("Magnetic Axis R(t)")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)

            # Z(t)
            plt.subplot(1, 2, 2)
            plt.plot(steps, axis_Z_gt_series, label="Z_axis GT", color="blue")
            plt.plot(steps, axis_Z_pred_series, label="Z_axis Pred", color="red", linestyle="--")
            plt.xlabel("Rollout Step")
            plt.ylabel("Z (m)")
            plt.title("Magnetic Axis Z(t)")
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, "axis_evolution_vs_time.png"))
            plt.close()

        # 4) PDE residuals (log-scale)
        res_mass = self.metrics.get("pde_res_mass", [])
        res_mom = self.metrics.get("pde_res_momentum", [])
        if res_mass and res_mom:
            plt.figure(figsize=(6, 4))
            plt.semilogy(steps, res_mass, label="Mass Residual")
            plt.semilogy(steps, res_mom, label="Momentum Residual")
            plt.xlabel("Rollout Step")
            plt.ylabel("Mean |Residual|")
            plt.title("PDE Residuals vs Time (log scale)")
            plt.grid(True, which="both", alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(vis_dir, "pde_residuals.png"))
            plt.close()

        # Optionally print out performance summary
        if self.avg_wallclock_per_sequence is not None and self.speedup_factor is not None:
            print(
                f"[Benchmark] Avg wall-clock per sequence: "
                f"{self.avg_wallclock_per_sequence:.4f} s | "
                f"Speed-up vs {self.jorek_cpu_hours_per_sequence:.1f} h JOREK: "
                f"×{self.speedup_factor:.2e}"
            )
