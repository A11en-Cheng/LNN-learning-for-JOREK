"""Phase-3 trainer: data-only CfC interpolation with random N-substep splits.

Burn-in frames warm the hidden state (no loss), then every supervised interval
is split into ``--sub-steps`` random sub-intervals summing to ``dt`` and must
reproduce the next recorded frame.  ``--sub-steps 2`` is the historical
two-substep sampler (``alpha``, ``1 - alpha``); larger values shorten each cell
step towards the ~50 fluid steps JOREK actually takes per dumped frame
(pitfalls #39).  The recorded projections are fed as *teacher* source controls
(target-frame pairing, see CONTEXT.md) and the mode is stamped in the metrics.

Two knobs must agree with the evaluation script, otherwise the metric measures
a different integrator than the one being optimised:
``--sub-steps`` (supervised interval split, random here / equal in eval) and
``--burn-in-substeps`` (granularity of the hidden-state warm-up).

Example:

    ./.torch-cuda-env/bin/python tests/train_cfc_interpolation.py \
        --manifest .scratch/n/manifest.json --cache-dir .scratch/n/frames \
        --held-out scan_1d22_ang15_5_massloss_neon_v400 \
        --epochs 2 --steps-per-epoch 60 --out output/cfc_interp/smoke
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import (
    InterpolationWindowDataset,
    RunFrameStore,
    collate_windows,
    compute_transform_stats,
    to_device,
)
from src.data.model502_schema import destandardize, inverse_transform_state
from src.models.cfc_interp import (
    CfCInterpolator,
    deterministic_fractions,
    make_deriv_fn,
    sample_split_fractions,
)
from src.physics.grad_ops import build_gradient_operator
from src.physics.grad_ops import compute_gradient
from src.physics.model502_full_loss import model502_full_residuals

PHYSICS_CHANNELS = ("induction", "current_definition", "vorticity_definition")


def _condition_from_frame(frame_condition) -> torch.Tensor:
    """Frame parameter channel -> residual-module 34-entry coefficient layout."""
    condition = torch.zeros(56, dtype=torch.float64)  # CONDITION_DIM (34 residual + 22 appended)
    defaults = {0: 1e-5, 1: 1e-5, 2: 1e-5, 3: 0.0, 4: 1.0, 5: 1.0, 6: 1e-5, 11: 1e-5, 12: 1e-5, 18: 1e-5}
    for index, value in defaults.items():
        condition[index] = value
    values = np.asarray(frame_condition, dtype=np.float64)
    condition[0:6] = torch.as_tensor(values[0:6], dtype=torch.float64)
    condition[6] = float(values[6])
    condition[11] = float(values[6])
    condition[12] = float(values[12])
    condition[18] = float(values[18])
    condition[33] = float(values[33])
    # Transport-closure block (appended to the parameter channel, identity map):
    # ZK_par_max / ZKpar_T_dependent / Ti|Te_min_ZKpar / T_min[+neg] / rho_min[+neg]
    # / ZK_*_par_neg / D_par / D_par_imp.
    if values.size >= 56:
        condition[44:56] = torch.as_tensor(values[44:56], dtype=torch.float64)
    # Entries the residual module reads through the *canonical* channel layout
    # (everything except the per-species profile blocks it remaps above).
    for index in (29, 30, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55):
        if values.size > index:
            condition[index] = float(values[index])
    return condition


def _physics_term(
    residuals_module,
    *,
    u_mid: torch.Tensor,
    u_end: torch.Tensor,
    delta: float,
    pos: torch.Tensor,
    edges: torch.Tensor,
    attrs: torch.Tensor,
    condition: torch.Tensor,
    source: torch.Tensor,
    stats: tuple[torch.Tensor, torch.Tensor],
    source_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    channel_scales: dict[str, float] | None = None,
    channel_weights: dict[str, float] | None = None,
    channel_values: dict[str, list[float]] | None = None,
    detach_denominator: bool = True,
) -> torch.Tensor:
    """Mean physics residual of the CfC sub-interval trajectory.

    ``channel_scales`` divides each channel by its recorded-data floor
    (``.scratch/n/residual_floor_stats.json``) so that the three residuals, whose
    gradients w.r.t. the decoded fields differ by ~65x (pitfalls #27), carry
    comparable weight; the returned value is then in units of "x times the floor
    that a perfect data fit would still show".
    """
    graph = Data(
        x=u_mid.detach(),
        pos=pos.double(),
        edge_index=edges,
        edge_attr=attrs,
        condition=condition,
        source=source.double(),
    )
    residuals = residuals_module(
        u_end.double(), u_mid.double(), graph, delta,
        stats=stats, source_stats=source_stats, detach_denominator=detach_denominator,
    )
    total = torch.zeros((), device=u_mid.device)
    for name in PHYSICS_CHANNELS:
        value = residuals[name].to(u_mid.device)
        scale = 1.0 if not channel_scales else float(channel_scales.get(name, 1.0))
        weight = 1.0 if not channel_weights else float(channel_weights.get(name, 1.0))
        total = total + weight * value / max(scale, 1e-12)
        if channel_values is not None:
            channel_values.setdefault(name, []).append(float(value.detach()))
    return total / len(PHYSICS_CHANNELS)


def _fold_for(manifest: dict, held_out: str) -> dict:
    fold = next((r for r in manifest["folds"] if r["held_out_run"] == held_out), None)
    if fold is None:
        raise SystemExit(f"{held_out} is not a run in the manifest")
    return fold


def _stores(manifest: dict, cache_dir: str | None) -> dict[str, RunFrameStore]:
    root = Path(manifest["data_root"])
    stores = {}
    for record in manifest["runs"]:
        stores[record["name"]] = RunFrameStore(root / record["name"], cache_dir=cache_dir).load()
    return stores


def _window_loss(
    model,
    item,
    device,
    *,
    alpha: float | None,
    sub_steps: int = 2,
    burn_in_substeps: int = 1,
    horizon: int = 1,
) -> tuple[torch.Tensor, int]:
    """Teacher-forced burn-in + one supervised pass over the window.

    ``horizon`` = how many recorded frames ahead the supervised target sits: the
    state at ``t_k`` is propagated over ``dt_total = t_{k+horizon} - t_k`` and the
    data loss is taken at that target frame (one propagation, arbitrary dt).
    """
    batch = to_device(collate_windows([item]), device)
    u = batch["u"][0]
    dt = batch["dt"][0]
    source = batch["source"][0]
    available = batch["source_available"][0]
    burn_in = int((~batch["loss_mask"][0]).sum().item())
    deriv = batch["u_deriv"][0] if getattr(model, "input_deriv", False) else None
    lap_feat = batch["u_lap"][0] if getattr(model, "input_deriv2", False) else None
    hidden = None
    loss = torch.zeros((), device=device)
    count = 0
    for k in range(u.shape[0] - 1):
        target_index = k + horizon
        if target_index >= u.shape[0]:
            continue
        if not bool(available[target_index]):
            continue
        controls = source[target_index]
        if k < burn_in:
            # Burn-in only warms the hidden state; keeping 30 cell calls in the
            # autograd graph costs tens of GB on GPU and adds no gradient signal
            # (the loss starts at the first supervised interval).
            with torch.no_grad():
                _, _, hidden = model.step(
                    u[k], hidden, dt[k], controls, None,
                    None if deriv is None else deriv[k],
                    None if lap_feat is None else lap_feat[k],
                    sub_steps=burn_in_substeps,
                )
            continue
        a = random.random() if alpha is None else alpha
        fractions = (
            [a, 1.0 - a]
            if sub_steps == 2
            else sample_split_fractions(1, sub_steps, device=device, dtype=u.dtype)
        )
        out = model.multi_substep(
            u[k], hidden, fractions, dt[k:target_index].sum(), controls, None,
            None if deriv is None else deriv[k],
            None if lap_feat is None else lap_feat[k],
        )
        hidden = out["hidden"]
        loss = loss + torch.nn.functional.mse_loss(out["prediction"], u[target_index])
        count += 1
    if count == 0:
        return torch.zeros((), device=device, requires_grad=True), 0
    return loss / count, count


def _batch_loss(
    model,
    items: list[dict],
    device,
    *,
    alpha: float | None,
    lambda_h1: float = 0.0,
    lambda_axis: float = 0.0,
    lambda_phys: float = 0.0,
    phys_windows: int = 4,
    phys_rho: float = 0.2,
    stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    source_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
    phys_channel_scales: dict[str, float] | None = None,
    phys_channel_weights: dict[str, float] | None = None,
    phys_channel_values: dict[str, list[float]] | None = None,
    phys_detach_denominator: bool = True,
    loss_weights: torch.Tensor | None = None,
    h1_global: bool = False,
    axis_temperature: float = 0.1,
    axis_scale: float = 0.1,
    sub_steps: int = 2,
    burn_in_substeps: int = 1,
    split_mode: str = "random",
    min_fraction: float = 0.0,
    sub_step_checkpoint: bool = False,
    horizon: int = 1,
    horizon_mode: str = "jump",
) -> tuple[torch.Tensor, int, float, float, float, float]:
    """Batched version of :func:`_window_loss`.

    Every window is an independent graph; ``batch_index`` keeps their hidden
    states separate while the matmuls see ``batch * nodes`` rows, which is what
    makes the GPU useful.  The sub-interval fractions are sampled per window
    when ``alpha`` is ``None`` (``split_mode="random"``).
    """
    batch = to_device(collate_windows(items), device)
    u = batch["u"]                     # [B, T, nodes, 9]
    dt = batch["dt"]                   # [B, T-1]
    source = batch["source"]           # [B, T, nodes, 5]
    available = batch["source_available"]
    if not bool(available[:, 1:].all()):
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, 0, 0.0, 0.0, 0.0, 0.0
    burn_in = int((~batch["loss_mask"][0]).sum().item())
    deriv = None
    if getattr(model, "input_deriv", False):
        deriv = batch["u_deriv"]
    lap = batch["u_lap"] if getattr(model, "input_deriv2", False) else None
    b, t, n, _ = u.shape
    batch_index = torch.arange(b, device=device).repeat_interleave(n)
    # The topology is per window; batch the edges by offsetting every copy so
    # ``compute_gradient`` can be applied to the whole [b * nodes] batch.
    edges_single = batch["topology"].edge_index
    attrs_single = batch["topology"].edge_attr.float()
    edges_per_window = edges_single.shape[1]
    offsets = (
        torch.arange(b, device=device).repeat_interleave(edges_per_window) * n
    ).unsqueeze(0)
    edges = edges_single.repeat(1, b) + offsets
    attrs = attrs_single.repeat(b, 1)
    hidden = None
    loss = torch.zeros((), device=device)
    h1 = torch.zeros((), device=device)
    axis = torch.zeros((), device=device)
    physics = torch.zeros((), device=device)
    pos = batch["topology"].pos.to(device).float()
    count = 0
    if sub_steps == 2:
        # Historical sampler: a single uniform knot brackets the interval.
        alphas = (
            torch.rand(b, device=device, dtype=u.dtype) if alpha is None
            else torch.full((b,), float(alpha), device=device, dtype=u.dtype)
        )
        fractions = torch.stack([alphas, 1.0 - alphas], dim=1)
    else:
        fractions = sample_split_fractions(
            b, sub_steps, device=device, dtype=u.dtype,
            mode=split_mode, min_fraction=min_fraction,
        )
    for k in range(t - 1):
        if k < burn_in:
            controls = source[:, k + 1].reshape(b * n, -1)
            step_deriv = deriv[:, k].reshape(b * n, -1) if deriv is not None else None
            step_lap = lap[:, k].reshape(b * n, -1) if lap is not None else None
            with torch.no_grad():
                _, _, hidden = model.step(
                    u[:, k].reshape(b * n, -1), hidden, dt[:, k], controls, batch_index,
                    step_deriv, step_lap, sub_steps=burn_in_substeps,
                )
            continue
        end = k + horizon
        if end >= t:
            continue
        controls = source[:, end].reshape(b * n, -1)
        step_deriv = deriv[:, k].reshape(b * n, -1) if deriv is not None else None
        step_lap = lap[:, k].reshape(b * n, -1) if lap is not None else None
        # ``dt_span`` = the physical time from the base frame to the target frame
        # (sum of the recorded intervals in between), fed to the cell as the
        # integration span: one propagation over the whole span ("jump").
        dt_span = dt[:, k:end].sum(dim=1)
        if horizon_mode == "rollout":
            if lambda_phys > 0.0:
                raise ValueError("--horizon-mode rollout does not support the physics term")
            prediction = None
            for step_index in range(k, end):
                frame = step_index + 1
                step_controls = source[:, frame].reshape(b * n, -1)
                # Each *recorded* interval has its own (non-uniform) length dt_k,
                # and inside it the CfC chain is split at *random* fractions
                # (never a uniform dt/h grid): the network must be trained on
                # arbitrary interpolation points (ADR-0013).
                step_fractions = sample_split_fractions(
                    b, sub_steps, device=device, dtype=u.dtype,
                    mode=split_mode, min_fraction=min_fraction,
                )
                out = model.multi_substep(
                    u[:, step_index].reshape(b * n, -1), hidden, step_fractions,
                    dt[:, step_index], step_controls, batch_index,
                    deriv[:, step_index].reshape(b * n, -1) if deriv is not None else None,
                    lap[:, step_index].reshape(b * n, -1) if lap is not None else None,
                    checkpoint=sub_step_checkpoint,
                )
                prediction, hidden = out["prediction"], out["hidden"]
            target = u[:, end].reshape(b * n, -1)
            mid_latent = None
            delta_vec = dt_span
        else:
            # The sub-intervals must complete the recorded interval
            # (sum_i f_i * dt = dt); using phys_rho*dt here instead made the trained
            # chain shorter than the target time and disagreed with the evaluation
            # integrator (pitfalls #29).
            out = model.multi_substep(
                u[:, k].reshape(b * n, -1), hidden, fractions, dt_span, controls,
                batch_index, step_deriv, step_lap, checkpoint=sub_step_checkpoint,
            )
            prediction = out["prediction"]
            hidden = out["hidden"]
            mid_latent = out["mid_latent"]
            delta_vec = fractions[..., -1] * dt_span
            target = u[:, end].reshape(b * n, -1)
        base = u[:, k].reshape(b * n, -1)
        if loss_weights is None:
            loss = loss + torch.nn.functional.mse_loss(prediction, target)
        else:
            # Per-field weights 1/E[(delta u_norm)^2]: the objective becomes the
            # field-averaged *increment* R2 instead of the field R2, which
            # otherwise hands ~all the weight to Phi/omega (pitfalls #30).
            loss = loss + (((prediction - target) ** 2) * loss_weights).mean()
        if lambda_h1 > 0.0:
            # Per-channel *relative* gradient error, then average over channels.
            # Summing numerator and denominator over channels instead gave each
            # channel a weight of ||grad_true_c||^2, which spans 365x on this
            # corpus (rho+rho_imp = 86%), so lambda_H1 was really only smoothing
            # the density channels (pitfalls #31).
            per_channel: list[torch.Tensor] = []
            numerator = torch.zeros((), device=device)
            denominator = torch.zeros((), device=device)
            for channel in range(prediction.shape[1]):
                grad_pred = compute_gradient(prediction[:, channel].float(), edges, attrs)
                grad_true = compute_gradient(target[:, channel].float(), edges, attrs)
                per_channel.append(
                    ((grad_pred - grad_true) ** 2).sum()
                    / (grad_true**2).sum().clamp_min(1e-12)
                )
                if h1_global:
                    numerator = numerator + ((grad_pred - grad_true) ** 2).sum()
                    denominator = denominator + (grad_true**2).sum()
            if h1_global:
                h1 = h1 + numerator / denominator.clamp_min(1e-12)
            else:
                h1 = h1 + torch.stack(per_channel).mean()
        if lambda_axis > 0.0:
            # Differentiable magnetic axis: soft-argmax of the psi channel over
            # nodes, compared with the recorded frame's soft axis.
            pred_psi = prediction[:, 0].reshape(b, n)
            true_psi = target[:, 0].reshape(b, n)
            weights_pred = torch.softmax(pred_psi / axis_temperature, dim=1)
            weights_true = torch.softmax(true_psi / axis_temperature, dim=1)
            axis_pred = weights_pred @ pos
            axis_true = weights_true @ pos
            axis = axis + ((axis_pred - axis_true) ** 2).sum(dim=1).mean() / max(axis_scale**2, 1e-12)
        if lambda_phys > 0.0 and stats is not None and phys_windows > 0:
            u_mid = model.decode(mid_latent)
            if model.residual:
                u_mid = base + u_mid
            rows = torch.randperm(b, device="cpu")[: min(phys_windows, b)].tolist()
            term = torch.zeros((), device=device)
            for row in rows:
                condition = _condition_from_frame(items[row]["condition"][k].numpy()).to(device)
                # NOTE: must not shadow the batch tensor ``source`` (pitfalls
                # #26) - doing so made the next k iteration index a [nodes, 5]
                # slice and crash as soon as a window had >1 supervised step.
                phys_source = items[row]["source"][k + 1].to(device).float()
                term = term + _physics_term(
                    model502_full_residuals,
                    u_mid=u_mid[row * n : (row + 1) * n],
                    u_end=prediction[row * n : (row + 1) * n],
                    delta=float(delta_vec[row]),
                    pos=pos,
                    edges=edges_single,
                    attrs=attrs_single,
                    condition=condition,
                    source=phys_source,
                    stats=stats,
                    source_stats=source_stats,
                    channel_scales=phys_channel_scales,
                    channel_weights=phys_channel_weights,
                    channel_values=phys_channel_values,
                    detach_denominator=phys_detach_denominator,
                )
            physics = physics + term / max(len(rows), 1)
        count += 1
    if count == 0:
        zero = torch.zeros((), device=device, requires_grad=True)
        return zero, 0, 0.0, 0.0, 0.0, 0.0
    data_loss = loss / count
    h1_term = h1 / count
    axis_term = axis / count
    physics_term = physics / count
    total = data_loss + lambda_h1 * h1_term + lambda_axis * axis_term + lambda_phys * physics_term
    return (
        total, count,
        float(data_loss.detach()), float(h1_term.detach()),
        float(axis_term.detach()), float(physics_term.detach()),
    )


@torch.no_grad()
def _evaluate(
    model,
    dataset,
    device,
    stats: dict,
    *,
    alphas=(0.25, 0.5, 0.75),
    alpha_fixed: float = 0.5,
    max_windows: int = 12,
    sub_steps: int = 2,
    burn_in_substeps: int = 1,
    horizon: int = 1,
    horizon_mode: str = "jump",
) -> dict:
    """Held-out metrics: normalized MSE, per-field raw R2/relative L2 and the
    alpha-split self-consistency spread (same inputs, different integration
    splits).  The main metric uses a *deterministic* split (equal sub-intervals
    when ``sub_steps > 2``) so repeated evaluations are comparable.

    ``mean_dr2`` is the field-averaged *increment* R2 in raw physical units -
    the same quantity ``tests/eval_phase3_structure.py`` gates on, so model
    selection here cannot drift away from the held-out Gate (pitfalls #29)."""
    model.eval()
    field_mean = torch.tensor(stats["field_mean"], device=device)
    field_std = torch.tensor(stats["field_std"], device=device)
    n_fields = len(stats["field_names"])
    sq_err = torch.zeros(n_fields, dtype=torch.float64, device=device)
    tgt_sum = torch.zeros_like(sq_err)
    tgt_sumsq = torch.zeros_like(sq_err)
    inc_err = torch.zeros_like(sq_err)
    inc_sq = torch.zeros_like(sq_err)
    inc_pred_sq = torch.zeros_like(sq_err)
    # ``prev_sq`` = ||u_pred - u_prev||^2: the other half of the "is it closer to
    # the target frame or to the previous frame?" comparison (closer metric).
    prev_sq = torch.zeros_like(sq_err)
    losses, spreads, nan = [], [], False
    count = 0
    for index in range(min(len(dataset), max_windows)):
        item = dataset[index]
        batch = to_device(collate_windows([item]), device)
        u, u_raw = batch["u"][0], batch["u_raw"][0]
        deriv = batch["u_deriv"][0] if getattr(model, "input_deriv", False) else None
        lap_feat = batch["u_lap"][0] if getattr(model, "input_deriv2", False) else None
        dt, source = batch["dt"][0], batch["source"][0]
        available = batch["source_available"][0]
        burn_in = int((~batch["loss_mask"][0]).sum().item())
        hidden = None
        for k in range(u.shape[0] - 1):
            target_index = k + horizon
            if target_index >= u.shape[0]:
                continue
            if not bool(available[target_index]):
                continue
            if k < burn_in:
                _, _, hidden = model.step(
                    u[k], hidden, dt[k], source[k + 1], None,
                    None if deriv is None else deriv[k],
                    None if lap_feat is None else lap_feat[k],
                    sub_steps=burn_in_substeps,
                )
                continue
            hidden_before = hidden
            if horizon_mode == "rollout":
                # h successive interpolations, one per recorded interval: the
                # total span is sum(delta_i t), the intermediate *recorded*
                # frames are the encoder inputs and the hidden state is chained.
                for step_index in range(k, target_index):
                    frame = step_index + 1
                    out = model.multi_substep(
                        u[step_index], hidden,
                        deterministic_fractions(sub_steps, alpha_fixed),
                        dt[step_index], source[frame], None,
                        None if deriv is None else deriv[step_index],
                        None if lap_feat is None else lap_feat[step_index],
                    )
                    prediction, hidden = out["prediction"], out["hidden"]
            else:
                out = model.multi_substep(
                    u[k], hidden, deterministic_fractions(sub_steps, alpha_fixed),
                    dt[k:target_index].sum(), source[target_index], None,
                    None if deriv is None else deriv[k],
                    None if lap_feat is None else lap_feat[k],
                )
                hidden = out["hidden"]
                prediction = out["prediction"]
            if not torch.isfinite(prediction).all():
                nan = True
            losses.append(float(torch.nn.functional.mse_loss(prediction, u[target_index])))
            raw = inverse_transform_state(
                destandardize(prediction, field_mean, field_std, name="prediction")
            )
            diff = (raw - u_raw[target_index]).double()
            sq_err += (diff**2).sum(dim=0)
            tgt_sum += u_raw[target_index].double().sum(dim=0)
            tgt_sumsq += (u_raw[target_index].double() ** 2).sum(dim=0)
            delta_true = (u_raw[target_index] - u_raw[k]).double()
            delta_pred = (raw - u_raw[k]).double()
            prev_sq += (delta_pred**2).sum(dim=0)
            inc_err += ((delta_pred - delta_true) ** 2).sum(dim=0)
            inc_sq += (delta_true**2).sum(dim=0)
            inc_pred_sq += (delta_pred**2).sum(dim=0)
            count += raw.shape[0]
            if k == burn_in:  # alpha gate on the first supervised interval
                # Use the state *entering* the interval: re-using the already
                # advanced ``hidden`` measured the spread of a different map
                # (found in the 2026-10-01 propagator audit).
                stack = torch.stack(
                    [model.multi_substep(u[k], hidden_before,
                                         deterministic_fractions(sub_steps, a),
                                         dt[k:target_index].sum(),
                                         source[target_index], None,
                                         None if deriv is None else deriv[k],
                                         None if lap_feat is None else lap_feat[k])["prediction"]
                     for a in alphas]
                )
                scale = stack.mean(dim=0).norm().clamp_min(1e-12)
                spreads.append(float((stack - stack.mean(dim=0)).norm() / scale))
    model.train()
    if not losses or count == 0:
        return {"windows": 0}
    mean = tgt_sum / count
    variance = (tgt_sumsq / count - mean**2).clamp_min(0.0)
    r2 = 1.0 - sq_err / (variance * count).clamp_min(1e-30)
    dr2 = 1.0 - inc_err / inc_sq.clamp_min(1e-30)
    closer = sq_err.sqrt() / (sq_err.sqrt() + prev_sq.sqrt()).clamp_min(1e-30)
    return {
        "windows": len(losses),
        "mse": float(np.mean(losses)),
        "alpha_spread_max": float(np.max(spreads)) if spreads else float("nan"),
        "nan": nan,
        "mean_dr2": float(dr2.mean()),
        "mean_closer": float(closer.mean()),
        "closer": {name: float(value) for name, value in zip(stats["field_names"], closer)},
        "dR2": {name: float(value) for name, value in zip(stats["field_names"], dr2)},
        "r2": {name: float(value) for name, value in zip(stats["field_names"], r2)},
    }


def _save(model, path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def _load_channel_scales(spec: str) -> dict[str, float]:
    """Per-channel residual floors used to normalise the physics term.

    ``none`` keeps the raw residual mean (historical behaviour); ``auto`` reads
    the distribution written by ``tests/check_residual_floor_stats.py``.
    """
    if spec in ("", "none"):
        return {}
    path = Path(".scratch/n/residual_floor_stats.json") if spec == "auto" else Path(spec)
    payload = json.loads(path.read_text(encoding="utf-8"))
    channels = payload.get("channels", {})
    scales: dict[str, float] = {}
    for name in PHYSICS_CHANNELS:
        entry = channels.get(name)
        if entry is None:
            raise SystemExit(f"floor file {path} has no entry for channel {name!r}")
        scales[name] = float(entry["median"])
    return scales


def _load_channel_weights(spec: str) -> dict[str, float]:
    """``'induction=8,current_definition=1,vorticity_definition=0.25'`` -> dict."""
    if spec in ("", "none"):
        return {}
    weights: dict[str, float] = {}
    for chunk in spec.split(","):
        name, _, value = chunk.partition("=")
        name = name.strip()
        if name not in PHYSICS_CHANNELS:
            raise SystemExit(f"unknown physics channel {name!r}; expected one of {PHYSICS_CHANNELS}")
        weights[name] = float(value)
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=60, help="batches per epoch")
    parser.add_argument("--batch-size", type=int, default=8, help="windows per batch")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; >0 measured slower (worker IPC > collate savings)",
    )
    parser.add_argument("--prefetch", type=int, default=4, help="batches prefetched per worker")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "cosine"),
        default="constant",
        help="'cosine' decays lr -> --lr-final over the whole run; constant lr 3e-3 "
        "destabilises after ~600 epochs (pitfalls #54)",
    )
    parser.add_argument("--lr-final", type=float, default=None,
                        help="final lr for the cosine schedule (default: lr / 10)")
    parser.add_argument("--lambda-h1", type=float, default=0.0, help="H1 structure loss weight")
    parser.add_argument(
        "--h1-global",
        action="store_true",
        help="legacy H1 aggregation: sum numerator/denominator over channels (weight per "
        "channel = ||grad_true_c||^2, 365x spread) instead of the per-channel relative mean",
    )
    parser.add_argument("--lambda-axis", type=float, default=0.0, help="soft magnetic-axis loss weight")
    parser.add_argument("--lambda-phys", type=float, default=0.0, help="physics residual weight")
    parser.add_argument(
        "--residual",
        action="store_true",
        help="decoder outputs the increment over the interval (u(t+dt) = u(t) + decode): "
        "the recorded fields move only 1-20%% per frame, so the absolute "
        "parameterisation spends its budget on the static part (pitfalls #29)",
    )
    parser.add_argument(
        "--loss-scale",
        choices=("none", "dynamics"),
        default="none",
        help="'dynamics' weights each field by 1/E[(delta u_norm)^2] from the training runs, "
        "so the objective is the field-averaged increment R2.  Plain MSE in z-score space "
        "weights the fields by their spatial spread, which differs by 5000x between "
        "channels and hides the slow ones (pitfalls #30)",
    )
    parser.add_argument(
        "--input-deriv2",
        action="store_true",
        help="also feed the graph-LSQ Laplacian of every field (2nd-order features); "
        "verified against the recorded Z_j/omega channels at corr 0.987-0.997",
    )
    parser.add_argument(
        "--input-deriv",
        action="store_true",
        help="concatenate first-order local-derivative features to the encoder input",
    )
    parser.add_argument(
        "--input-dt",
        action="store_true",
        help="concatenate log1p(dt)/log1p(dt_scale) to the encoder input; without it the "
        "network only meets the interval length inside the cell (pitfalls #48: the "
        "recorded cadence spans 0.5..250 JOREK units and train/eval distributions differ)",
    )
    parser.add_argument(
        "--feedback-deriv",
        action="store_true",
        help="recompute derivative features from the model's own decoded state",
    )
    parser.add_argument("--feedback-clamp", type=float, default=5.0,
                        help="hard clamp (in standardised units) for model derivative features")
    parser.add_argument("--feedback-anneal", action="store_true",
                        help="anneal the model-derivative probability from 0 to 1 over epochs")
    parser.add_argument("--phys-windows", type=int, default=4, help="windows per batch for the physics term")
    parser.add_argument(
        "--phys-rho",
        type=float,
        default=0.2,
        help="deprecated: the physics residual now uses the second substep, delta=(1-alpha)*dt, "
        "so the pair (u_mid, u_end) is the one the data loss already integrates (pitfalls #29)",
    )
    parser.add_argument(
        "--phys-floor",
        default="none",
        help="'none' (raw residual mean), 'auto' (.scratch/n/residual_floor_stats.json) "
        "or a path to a floor JSON: divide each channel by its recorded-data floor so the "
        "three residuals carry comparable weight (pitfalls #27)",
    )
    parser.add_argument(
        "--phys-weights",
        default="none",
        help="per-channel multipliers, e.g. 'induction=8,current_definition=1,"
        "vorticity_definition=0.25'; applied after --phys-floor.  Needed because the "
        "residual gradients differ by orders of magnitude (pitfalls #27/#28)",
    )
    parser.add_argument(
        "--phys-keep-denominator",
        action="store_true",
        help="do NOT detach the relative normalisers: the |sum|/sum|term| ratio then "
        "saturates at 1.0 with zero gradient, which is where the model sits (pitfalls #28)",
    )
    parser.add_argument(
        "--grad-balance",
        type=float,
        default=0.0,
        help="adaptive physics weight: every --grad-balance-every steps measure "
        "||dL_data/dtheta|| and ||dL_phys/dtheta|| and set lambda_phys so the physics "
        "GRADIENT is this fraction of the data gradient (e.g. 0.1).  Value-based "
        "calibration is off by 13x here because the second-derivative residuals "
        "amplify gradients by ~1/h^2 (pitfalls #36)",
    )
    parser.add_argument("--grad-balance-every", type=int, default=10)
    parser.add_argument(
        "--grad-balance-batch",
        type=int,
        default=8,
        help="windows used for the gradient-norm probe (a sub-batch is enough for the "
        "ratio and keeps the extra backward passes off the memory limit)",
    )
    parser.add_argument("--axis-temperature", type=float, default=0.1)
    parser.add_argument("--axis-scale", type=float, default=0.1, help="axis error normalisation [m]")
    parser.add_argument(
        "--sub-steps",
        type=int,
        default=2,
        help="cell calls per recorded frame in the supervised interval.  JOREK takes ~50 "
        "fluid steps per dumped frame, so a chain of S sub-steps shortens each CfC step "
        "from dt to dt/S (pitfalls #39b).  2 keeps the historical random two-substep "
        "sampler",
    )
    parser.add_argument(
        "--burn-in-substeps",
        type=int,
        default=1,
        help="equal sub-intervals used while warming the hidden state on recorded frames; "
        "raise it together with --sub-steps so the warm-up uses the same granularity as "
        "the supervised interval",
    )
    parser.add_argument(
        "--sub-step-split",
        choices=("random", "equal"),
        default="random",
        help="how the supervised interval is split when --sub-steps > 2 (random sorted "
        "knots, or equal sub-intervals)",
    )
    parser.add_argument(
        "--sub-step-min",
        type=float,
        default=0.0,
        help="optional lower bound on a sub-interval fraction (renormalised); 0 disables "
        "the guard and keeps the sampler distribution unchanged",
    )
    parser.add_argument(
        "--sub-step-checkpoint",
        action="store_true",
        help="recompute each sub-step during backward instead of storing it: needed for "
        "long chains (one retained cell call is ~0.7 GB at 32 windows x 4125 nodes)",
    )
    parser.add_argument(
        "--cell",
        choices=("ncps", "ltc"),
        default="ltc",
        help="'ltc' (default) = our own cell: it predicts rate=softplus(...) and "
        "target=tanh(...) and applies the exact solution of dh/dt=-rate*(h-target), so "
        "ts=0 is the identity and splitting an interval is a no-op (identity/semigroup "
        "checked by tests/probe_ltc_semigroup.py).  'ncps' = the old installed CfC cell "
        "+ LSTM mixed memory, kept ONLY for re-loading pre-2026-10-01 checkpoints",
    )
    parser.add_argument("--backbone", type=int, default=256,
                        help="LTC backbone width (replaces the LSTM shortcut capacity)")
    parser.add_argument("--backbone-layers", type=int, default=2)
    parser.add_argument("--state-dependent", action="store_true",
                        help="let rate/target also see h (true LTC gating).  Off by default: "
                        "h-dependent coefficients re-evaluated at a split point break the "
                        "exact semigroup (the split error becomes O(ts^2))")
    parser.add_argument("--target-activation", choices=("tanh", "linear"), default="tanh",
                        help="'tanh' bounds the equilibrium target (state stays in [-1,1] for "
                        "h0=0); 'linear' restores the unbounded dynamic range the legacy LSTM "
                        "shortcut used to provide")
    parser.add_argument("--init-decay", type=float, default=0.5,
                        help="initial per-step relaxation: one ts_reference=0.1 step moves "
                        "the state this far towards the target")
    parser.add_argument("--predict-source", action="store_true",
                        help="add the augmented source head + integral accumulator "
                        "(advance_with_source_integral); off in the pure-data field path")
    parser.add_argument("--latent", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dt-scale", type=float, default=50.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="output/cfc_interp/run")
    parser.add_argument("--overfit", action="store_true", help="single-window gate")
    parser.add_argument("--overfit-steps", type=int, default=150)
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="also write checkpoint_best.pt whenever the inner-val mean increment R2 "
        "(mean_dr2, the same metric the held-out Gate reports) improves",
    )
    parser.add_argument(
        "--save-best-metric",
        choices=("closer", "dR2"),
        default="closer",
        help="inner-val metric used by --save-best: 'closer' (lower is better: mean "
        "||pred-true||/(||pred-true||+||pred-prev||) over fields, the frame-level "
        "similarity metric) or the legacy 'dR2' (higher is better)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="supervised target sits this many recorded frames ahead: the state at t_k is "
        "propagated over dt_total = t_{k+h} - t_k and the data loss is taken at frame k+h "
        "(the manifest must provide windows with at least burn_in + horizon + 1 frames)",
    )
    parser.add_argument(
        "--horizon-mode",
        choices=("jump", "rollout"),
        default="jump",
        help="'jump' = one propagation over the whole span dt_total (default); "
        "'rollout' = h teacher-forced model steps, one per recorded frame",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = _fold_for(manifest, args.held_out)
    stats = compute_transform_stats(
        _stores(manifest, args.cache_dir), runs=fold["train_runs"],
        with_lap=args.input_deriv2,
    )
    field_mean = torch.tensor(stats["field_mean"], device=device)
    field_std = torch.tensor(stats["field_std"], device=device)
    # Projections are z-scored by the dataset; the residual module works in
    # physical units, so hand it the matching de-normalisation (pitfalls #25).
    source_stats = (
        torch.tensor(stats["source_mean"], device=device, dtype=torch.float64),
        torch.tensor(stats["source_std"], device=device, dtype=torch.float64).clamp_min(1e-12),
    )
    phys_channel_scales = _load_channel_scales(args.phys_floor)
    if phys_channel_scales:
        print("physics channel floors: " + ", ".join(
            f"{name}={value:.4f}" for name, value in phys_channel_scales.items()
        ))
    phys_channel_weights = _load_channel_weights(args.phys_weights)
    if phys_channel_weights:
        print("physics channel weights: " + ", ".join(
            f"{name}={value:g}" for name, value in phys_channel_weights.items()
        ))
    loss_weights = None
    if args.loss_scale == "dynamics":
        increment_var = torch.tensor(stats["increment_var"], device=device, dtype=torch.float32)
        loss_weights = 1.0 / increment_var.clamp_min(1e-30)
        loss_weights = loss_weights / loss_weights.mean()
        print("dynamics loss weights: " + ", ".join(
            f"{name}={value:.4g}" for name, value in zip(stats["field_names"], loss_weights.tolist())
        ))
    train_set = InterpolationWindowDataset(
        manifest, manifest["data_root"], fold=fold["fold"], inner="train",
        stats=stats, cache_dir=args.cache_dir, with_shards=False,
        with_lap=args.input_deriv2,
    )
    val_set = InterpolationWindowDataset(
        manifest, manifest["data_root"], runs=[args.held_out], inner="val",
        stats=stats, cache_dir=args.cache_dir, with_shards=False,
        with_lap=args.input_deriv2,
    )
    deriv_fn = None
    if args.feedback_deriv:
        topology = train_set._store(train_set.windows[0]["run"]).topology
        operator = build_gradient_operator(
            topology.edge_index, topology.edge_attr.float(), topology.pos.shape[0]
        )
        operator = (operator[0].to(device), operator[1].to(device))
        deriv_fn = make_deriv_fn(
            operator,
            field_mean,
            field_std,
            torch.tensor(stats["deriv_mean"], device=device),
            torch.tensor(stats["deriv_std"], device=device).clamp_min(1e-12),
            clamp=args.feedback_clamp,
            detach=True,
        )
    model = CfCInterpolator(
        latent_dim=args.latent, hidden_size=args.hidden, dt_scale=args.dt_scale,
        input_deriv=args.input_deriv or args.feedback_deriv or args.input_deriv2,
        input_deriv2=args.input_deriv2, input_dt=args.input_dt,
        feedback_deriv=args.feedback_deriv, deriv_fn=deriv_fn, residual=args.residual,
        cell=args.cell, backbone_units=args.backbone, backbone_layers=args.backbone_layers,
        state_dependent=args.state_dependent, target_activation=args.target_activation,
        init_decay=args.init_decay,
        predict_source=args.predict_source,
    )
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.lr_schedule == "cosine":
        total_steps = max(args.epochs * args.steps_per_epoch, 1)
        eta_min = args.lr_final if args.lr_final is not None else args.lr / 10.0
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=eta_min
        )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.overfit:
        item = train_set[0]
        first = last = float("nan")
        for step in range(args.overfit_steps):
            loss, count = _window_loss(
                model, item, device, alpha=None,
                sub_steps=args.sub_steps, burn_in_substeps=args.burn_in_substeps,
                horizon=args.horizon,
            )
            if count == 0:
                raise SystemExit("overfit window has no supervised interval with teacher source")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if step == 0:
                first = float(loss.detach())
            last = float(loss.detach())
        payload = {"mode": "overfit", "window": item["run"], "steps": args.overfit_steps,
                   "loss_first": first, "loss_last": last}
        (out_dir / "overfit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _save(model, out_dir / "checkpoint.pt", payload)
        print(f"overfit: loss {first:.4e} -> {last:.4e} (window {item['run']})")
        return

    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=list,
        prefetch_factor=args.prefetch if args.num_workers > 0 else None,
        persistent_workers=args.num_workers > 0,
    )

    history = []
    started = time.time()
    best_score = float("-inf")
    best_epoch = -1
    # Adaptive physics weight (see --grad-balance); starts from the CLI value.
    lambda_phys = float(args.lambda_phys)
    last_grad_ratio = float("nan")
    for epoch in range(1, args.epochs + 1):
        if args.feedback_deriv:
            # Curriculum: teacher derivative -> model derivative.
            probability = (epoch - 1) / max(args.epochs - 1, 1) if args.feedback_anneal else 1.0
            model.feedback_prob = probability
        total, h1_total, axis_total, phys_total, steps = 0.0, 0.0, 0.0, 0.0, 0
        epoch_phys_channels: dict[str, list[float]] = {}
        for items in loader:
            if steps >= args.steps_per_epoch:
                break
            loss, count, data_mse, h1_term, axis_term, phys_term = _batch_loss(
                model, list(items), device, alpha=None, lambda_h1=args.lambda_h1,
                lambda_axis=args.lambda_axis, axis_temperature=args.axis_temperature,
                axis_scale=args.axis_scale, lambda_phys=lambda_phys,
                phys_windows=args.phys_windows, phys_rho=args.phys_rho,
                stats=(field_mean, field_std), source_stats=source_stats,
                phys_channel_scales=phys_channel_scales or None,
                phys_channel_weights=phys_channel_weights or None,
                phys_channel_values=epoch_phys_channels,
                phys_detach_denominator=not args.phys_keep_denominator,
                loss_weights=loss_weights,
                h1_global=args.h1_global,
                sub_steps=args.sub_steps,
                burn_in_substeps=args.burn_in_substeps,
                split_mode=args.sub_step_split,
                min_fraction=args.sub_step_min,
                sub_step_checkpoint=args.sub_step_checkpoint,
                horizon=args.horizon,
                horizon_mode=args.horizon_mode,
            )
            if count == 0:
                continue
            if args.grad_balance > 0.0 and lambda_phys > 0.0 and steps % max(args.grad_balance_every, 1) == 0:
                # Gradient-normalised physics weight (pitfalls #36).  The second
                # derivative residuals amplify ||dL_phys/dtheta|| by ~1/h^2, so a
                # value-calibrated lambda is off by ~1e5 and swamps the data term.
                # Two extra backward passes every K steps buy an interpretable
                # and self-adapting weight.
                parameters = [p for p in model.parameters() if p.requires_grad]

                probe_items = list(items)[: max(args.grad_balance_batch, 1)]

                def _grad_norm(stats_args: dict) -> float:
                    term_loss, term_count, *_ = _batch_loss(
                        model, probe_items, device, alpha=None, **stats_args
                    )
                    if term_count == 0:
                        return float("nan")
                    grads = torch.autograd.grad(
                        term_loss, parameters, retain_graph=False, allow_unused=True
                    )
                    return float(
                        torch.sqrt(
                            sum(
                                (g.detach() ** 2).sum()
                                for g in grads
                                if g is not None
                            )
                        )
                    )

                common = dict(
                    axis_temperature=args.axis_temperature, axis_scale=args.axis_scale,
                    phys_windows=len(probe_items), phys_rho=args.phys_rho,
                    stats=(field_mean, field_std), source_stats=source_stats,
                    phys_channel_scales=phys_channel_scales or None,
                    phys_channel_weights=phys_channel_weights or None,
                    phys_detach_denominator=not args.phys_keep_denominator,
                    loss_weights=loss_weights, h1_global=args.h1_global,
                    lambda_h1=0.0, lambda_axis=0.0,
                    sub_steps=args.sub_steps,
                    burn_in_substeps=args.burn_in_substeps,
                    split_mode=args.sub_step_split,
                    min_fraction=args.sub_step_min,
                    sub_step_checkpoint=args.sub_step_checkpoint,
                    horizon=args.horizon,
                    horizon_mode=args.horizon_mode,
                )
                grad_data = _grad_norm({**common, "lambda_phys": 0.0})
                grad_phys = _grad_norm({**common, "lambda_phys": 1.0})
                optimizer.zero_grad(set_to_none=True)
                if grad_phys > 0 and grad_data > 0:
                    target = args.grad_balance * grad_data / grad_phys
                    # EMA so a single noisy batch cannot swing the weight.
                    lambda_phys = 0.5 * lambda_phys + 0.5 * target if lambda_phys > 0 else target
                    lambda_phys = float(min(max(lambda_phys, 0.0), 1e-2))
                    grad_ratio = grad_phys / grad_data
                else:
                    grad_ratio = float("nan")
            else:
                grad_ratio = float("nan")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += data_mse
            h1_total += h1_term
            axis_total += axis_term
            phys_total += phys_term
            last_grad_ratio = grad_ratio
            steps += 1
        metrics = _evaluate(
            model, val_set, device, stats,
            sub_steps=args.sub_steps, burn_in_substeps=args.burn_in_substeps,
            horizon=args.horizon, horizon_mode=args.horizon_mode,
        )
        channel_means = {
            name: float(np.mean(values))
            for name, values in sorted(epoch_phys_channels.items())
            if values
        }
        record = {
            "epoch": epoch,
            "train_mse": total / max(steps, 1),
            "train_h1": h1_total / max(steps, 1),
            "train_axis": axis_total / max(steps, 1),
            "train_phys": phys_total / max(steps, 1),
            "lambda_phys_effective": lambda_phys,
            "grad_ratio_phys_data": last_grad_ratio,
            # Raw (un-normalised) per-channel residuals at the interpolation point.
            "train_phys_channels": channel_means,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val": metrics,
        }
        history.append(record)
        if args.save_best and not metrics.get("nan", False):
            # Selection metric on the inner-val split, NOT the training loss: the
            # data-only baseline keeps improving its val MSE long after the
            # training loss flattens (pitfalls #29).  Default is the frame-level
            # similarity ``closer`` (lower = the prediction sits closer to the
            # true next frame than to the previous one).
            if args.save_best_metric == "closer":
                score = -float(metrics.get("mean_closer", float("nan")))
            else:
                score = float(metrics.get("mean_dr2", float("nan")))
            if not math.isnan(score) and score > best_score:
                best_score, best_epoch = score, epoch
                _save(
                    model, out_dir / "checkpoint_best.pt",
                    {"args": vars(args), "stats": stats, "epoch": epoch,
                     "val": metrics, "history": history},
                )
        print(
            f"epoch {epoch}: train_mse={record['train_mse']:.4e} "
            f"train_h1={record['train_h1']:.4e} train_axis={record['train_axis']:.4e} "
            f"train_phys={record['train_phys']:.4e} lam_phys={lambda_phys:.3e} "
            f"channels={{{', '.join(f'{k}={v:.3f}' for k, v in channel_means.items())}}} "
            f"val={metrics}"
        )

    # checkpoint reload gate: identical predictions for the same inputs.
    checkpoint = out_dir / "checkpoint.pt"
    meta = {"args": vars(args), "stats": stats, "teacher_source": True, "history": history}
    _save(model, checkpoint, meta)
    reloaded = CfCInterpolator(
        latent_dim=args.latent, hidden_size=args.hidden, dt_scale=args.dt_scale,
        input_deriv=args.input_deriv or args.feedback_deriv or args.input_deriv2,
        input_deriv2=args.input_deriv2, input_dt=args.input_dt,
        feedback_deriv=args.feedback_deriv, deriv_fn=deriv_fn, residual=args.residual,
        cell=args.cell, backbone_units=args.backbone, backbone_layers=args.backbone_layers,
        state_dependent=args.state_dependent, target_activation=args.target_activation,
        init_decay=args.init_decay,
        predict_source=args.predict_source,
    )
    reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"])
    reloaded.to(device).eval()
    with torch.no_grad():
        item = val_set[0]
        batch = to_device(collate_windows([item]), device)
        u, dt, source = batch["u"][0], batch["dt"][0], batch["source"][0]
        deriv = batch["u_deriv"][0] if (args.input_deriv or args.input_deriv2) else None
        lap_feat = batch["u_lap"][0] if args.input_deriv2 else None
        burn_in = int((~batch["loss_mask"][0]).sum().item())
        hidden = None
        for k in range(burn_in):
            _, _, hidden = model.step(u[k], hidden, dt[k], source[k + 1], None,
                                      None if deriv is None else deriv[k],
                                      None if lap_feat is None else lap_feat[k])
        out_a = model.two_substep(u[burn_in], hidden, 0.5, dt[burn_in], source[burn_in + 1],
                                  None, None if deriv is None else deriv[burn_in],
                                  None if lap_feat is None else lap_feat[burn_in])
        hidden = None
        for k in range(burn_in):
            _, _, hidden = reloaded.step(u[k], hidden, dt[k], source[k + 1], None,
                                         None if deriv is None else deriv[k],
                                         None if lap_feat is None else lap_feat[k])
        out_b = reloaded.two_substep(u[burn_in], hidden, 0.5, dt[burn_in], source[burn_in + 1],
                                     None, None if deriv is None else deriv[burn_in],
                                     None if lap_feat is None else lap_feat[burn_in])
        reload_exact = bool(torch.equal(out_a["prediction"], out_b["prediction"]))

    meta.update({
        "checkpoint_reload_exact": reload_exact,
        "elapsed_seconds": time.time() - started,
        "held_out": args.held_out,
        "train_windows": len(train_set),
        "val_windows": len(val_set),
        "best_epoch": best_epoch,
        "best_score": None if best_score == float("-inf") else best_score,
        "save_best_metric": args.save_best_metric,
    })
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"checkpoint reload exact: {reload_exact}")
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
