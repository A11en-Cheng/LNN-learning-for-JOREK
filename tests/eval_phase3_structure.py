"""Phase-3 structure metrics (roadmap 3.4): R2 / relative L2 / H1 / axis / budget.

Runs the trained CfC over the held-out windows' supervised interval and reports
per-field pointwise and *structure* metrics, plus the magnetic-axis error
(node of maximal psi) and the relative integral (budget) error.

Example:

    ./.torch-cuda-env/bin/python tests/eval_phase3_structure.py \
        --checkpoint output/cfc_interp/phase3_v3/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import (
    InterpolationWindowDataset,
    RunFrameStore,
    collate_windows,
    compute_transform_stats,
    to_device,
)
from src.data.model502_schema import FIELD_NAMES, destandardize, inverse_transform_state
from src.models.cfc_interp import CfCInterpolator
from src.models.cfc_interp import deterministic_fractions
from src.models.cfc_interp import make_deriv_fn
from src.models.cfc_interp import model_kwargs_from_meta
from src.physics.grad_ops import build_gradient_operator
from src.physics.grad_ops import compute_gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument(
        "--inner",
        choices=("val", "train", "all"),
        default="val",
        help="which window split of the held-out run to evaluate.  'val' is the "
        "historical default, but with short windows that split can collapse to a "
        "single window (manifest_b10: 1 val window out of 65); since the whole run "
        "is excluded from training, 'all' is both valid and far better conditioned",
    )
    parser.add_argument("--checkpoint", default="output/cfc_interp/phase3_v3/checkpoint.pt")
    # Architecture flags default to None: the checkpoint's own meta['args'] is
    # the source of truth (see model_kwargs_from_meta), so an ncps checkpoint
    # from before the 2026-10-01 rewrite is evaluated with the cell it was
    # trained with instead of the new default.
    parser.add_argument("--latent", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--dt-scale", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument(
        "--sub-steps",
        type=int,
        default=2,
        help="cell calls per recorded frame; must match the trainer's --sub-steps. "
        "2 reproduces the historical random two-substep model (evaluated at --alpha), "
        "larger values use equal sub-intervals",
    )
    parser.add_argument(
        "--burn-in-substeps",
        type=int,
        default=1,
        help="equal sub-intervals used during the burn-in; must match the trainer",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="supervised target frame = burn_in + horizon; must match the trainer "
        "(dt fed to the cell is the summed span t_{k+h} - t_k)",
    )
    parser.add_argument(
        "--horizon-mode",
        choices=("jump", "rollout"),
        default="jump",
        help="must match the trainer: 'jump' = one propagation over the whole span; "
        "'rollout' = h successive interpolations, one per recorded interval",
    )
    parser.add_argument("--input-deriv", action="store_true", default=None)
    parser.add_argument("--input-deriv2", action="store_true", default=None,
                        help="also feed Laplacian features")
    parser.add_argument("--input-dt", action="store_true", default=None,
                        help="also feed log1p(dt)/log1p(dt_scale) to the encoder")
    parser.add_argument("--feedback-deriv", action="store_true", default=None)
    parser.add_argument("--residual", action="store_true", default=None,
                        help="decoder outputs the increment")
    parser.add_argument(
        "--cell",
        choices=("ncps", "ltc"),
        default=None,
        help="normally taken from the checkpoint; override to force 'ltc' (closed-form "
        "propagator) or 'ncps' (legacy installed CfC cell + LSTM mixed memory)",
    )
    parser.add_argument("--backbone", type=int, default=None, help="LTC backbone width")
    parser.add_argument("--backbone-layers", type=int, default=None)
    parser.add_argument("--state-dependent", action="store_true", default=None)
    parser.add_argument("--target-activation", choices=("tanh", "linear"), default=None)
    parser.add_argument("--init-decay", type=float, default=None)
    parser.add_argument("--max-windows", type=int, default=45)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", default="", help="also dump the metrics as JSON")
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    overrides = {
        "latent_dim": args.latent, "hidden_size": args.hidden, "dt_scale": args.dt_scale,
        "residual": args.residual, "cell": args.cell, "input_deriv": args.input_deriv or None,
        "input_deriv2": args.input_deriv2, "feedback_deriv": args.feedback_deriv,
        "input_dt": args.input_dt,
        "backbone_units": args.backbone, "backbone_layers": args.backbone_layers,
        "state_dependent": args.state_dependent, "init_decay": args.init_decay,
        "target_activation": args.target_activation,
    }
    kwargs = model_kwargs_from_meta(payload.get("meta") or {}, **overrides)
    with_lap = bool(kwargs["input_deriv2"])
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(r for r in manifest["folds"] if r["held_out_run"] == args.held_out)
    stores = {
        name: RunFrameStore(Path(manifest["data_root"]) / name, cache_dir=args.cache_dir).load()
        for name in fold["train_runs"]
    }
    # must match the trainer: the Laplacian features are standardised with
    # train-split statistics, which only exist when with_lap=True
    stats = compute_transform_stats(
        stores, runs=fold["train_runs"], with_lap=with_lap
    )
    dataset = InterpolationWindowDataset(
        manifest, manifest["data_root"], runs=[args.held_out],
        inner=None if args.inner == "all" else args.inner,
        stats=stats, cache_dir=args.cache_dir, with_shards=False,
        with_lap=with_lap,
    )
    deriv_fn = None
    if kwargs["feedback_deriv"]:
        topology = dataset._store(dataset.windows[0]["run"]).topology
        operator = build_gradient_operator(
            topology.edge_index, topology.edge_attr.float(), topology.pos.shape[0]
        )
        operator = (operator[0].to(device), operator[1].to(device))
        deriv_fn = make_deriv_fn(
            operator,
            torch.tensor(stats["field_mean"], device=device),
            torch.tensor(stats["field_std"], device=device),
            torch.tensor(stats["deriv_mean"], device=device),
            torch.tensor(stats["deriv_std"], device=device).clamp_min(1e-12),
        )
    model = CfCInterpolator(deriv_fn=deriv_fn, **kwargs)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    mean = torch.tensor(stats["field_mean"], device=device)
    std = torch.tensor(stats["field_std"], device=device)

    n_fields = len(FIELD_NAMES)
    # Accumulators must live on the eval device (they used to be CPU-only, which
    # broke every --device cuda run).
    sq = torch.zeros(n_fields, dtype=torch.float64, device=device)
    grad_sq = torch.zeros(n_fields, dtype=torch.float64, device=device)
    tgt_sq = torch.zeros(n_fields, dtype=torch.float64, device=device)
    tgt_grad_sq = torch.zeros(n_fields, dtype=torch.float64, device=device)
    sum_err = torch.zeros(n_fields, dtype=torch.float64, device=device)
    sum_tgt = torch.zeros(n_fields, dtype=torch.float64, device=device)
    inc_err = torch.zeros(n_fields, dtype=torch.float64, device=device)  # ||d_pred - d_true||^2
    inc_sq = torch.zeros(n_fields, dtype=torch.float64, device=device)   # ||d_true||^2
    inc_pred_sq = torch.zeros(n_fields, dtype=torch.float64, device=device)
    inc_cross = torch.zeros(n_fields, dtype=torch.float64, device=device)
    # ``prev_sq`` = ||u_pred - u_prev||^2, the missing half of the user's
    # "is it closer to the next frame or to the previous one?" comparison:
    #   closer = ||u_pred-u_true|| / (||u_pred-u_true|| + ||u_pred-u_prev||)
    # with the *same* normalisation in numerator and denominator, so it is
    # dimensionless and dt-robust (both distances scale with the increment).
    #   closer -> 0   prediction sits on the true next frame
    #   closer -> 0.5 prediction is exactly halfway (ambiguous)
    #   closer -> 1   prediction sits on the previous frame (a pure copy)
    prev_sq = torch.zeros(n_fields, dtype=torch.float64, device=device)
    prev_dist = {name: [] for name in FIELD_NAMES}
    # Same increment accumulators, split by the interval length dt: with the
    # recorded cadence varying over 0.5..250 JOREK units (pitfalls #48) the
    # plain average hides whether the failure lives in the fine or the coarse
    # intervals.
    dt_edges = (3.0, 15.0, 60.0)
    bucket_names = [f"dt<{dt_edges[0]:g}"]
    bucket_names += [f"{dt_edges[i]:g}-{dt_edges[i + 1]:g}" for i in range(len(dt_edges) - 1)]
    bucket_names += [f"dt>={dt_edges[-1]:g}"]
    bucket_inc_err = {name: torch.zeros(n_fields, dtype=torch.float64, device=device)
                      for name in bucket_names}
    bucket_inc_sq = {name: torch.zeros(n_fields, dtype=torch.float64, device=device)
                     for name in bucket_names}
    bucket_sq = {name: torch.zeros(n_fields, dtype=torch.float64, device=device)
                 for name in bucket_names}
    bucket_tgt_sq = {name: torch.zeros(n_fields, dtype=torch.float64, device=device)
                     for name in bucket_names}
    bucket_prev_sq = {name: torch.zeros(n_fields, dtype=torch.float64, device=device)
                      for name in bucket_names}
    bucket_count = {name: 0 for name in bucket_names}
    axis_errors = []
    axis_value_errors: list[float] = []
    used = 0
    node_count = 0
    # per-window increment R2, one list per field: the mean over windows is
    # unbounded (a single fine-dt window can score -500) so the median is the
    # robust headline and the mean is kept only in the JSON.
    window_dr2 = {name: [] for name in FIELD_NAMES}

    for index in range(min(len(dataset), args.max_windows)):
        item = dataset[index]
        batch = to_device(collate_windows([item]), device)
        u, u_raw = batch["u"][0], batch["u_raw"][0]
        dt, source = batch["dt"][0], batch["source"][0]
        deriv = batch["u_deriv"][0] if kwargs["input_deriv"] else None
        lap_feat = batch["u_lap"][0] if kwargs["input_deriv2"] else None
        burn_in = int((~batch["loss_mask"][0]).sum().item())
        target_index = burn_in + args.horizon
        if target_index >= u_raw.shape[0]:
            continue
        dt_value = float(dt[burn_in:target_index].sum())
        bucket_index = int(sum(dt_value >= edge for edge in dt_edges))
        bucket = bucket_names[bucket_index]
        bucket_count[bucket] += 1
        topology = batch["topology"]
        with torch.no_grad():
            hidden = None
            for k in range(burn_in):
                _, _, hidden = model.step(
                    u[k], hidden, dt[k], source[k + 1], None,
                    None if deriv is None else deriv[k],
                    None if lap_feat is None else lap_feat[k],
                    sub_steps=args.burn_in_substeps,
                )
            if args.horizon_mode == "rollout":
                for step_index in range(burn_in, target_index):
                    frame = step_index + 1
                    out = model.multi_substep(
                        u[step_index], hidden,
                        deterministic_fractions(args.sub_steps, args.alpha),
                        dt[step_index], source[frame], None,
                        None if deriv is None else deriv[step_index],
                        None if lap_feat is None else lap_feat[step_index],
                    )
                    prediction, hidden = out["prediction"], out["hidden"]
            else:
                prediction = model.multi_substep(
                    u[burn_in], hidden, deterministic_fractions(args.sub_steps, args.alpha),
                    dt[burn_in:target_index].sum(), source[target_index], None,
                    None if deriv is None else deriv[burn_in],
                    None if lap_feat is None else lap_feat[burn_in],
                )["prediction"]
        raw_pred = inverse_transform_state(
            destandardize(prediction, mean, std, name="prediction")
        ).double()
        raw_true = u_raw[target_index].double()
        for field in range(n_fields):
            diff = raw_pred[:, field] - raw_true[:, field]
            sq[field] += (diff**2).sum()
            tgt_sq[field] += (raw_true[:, field] ** 2).sum()
            sum_err[field] += diff.sum()
            sum_tgt[field] += raw_true[:, field].sum()
            # grad_ops works in float32 (edge features are float32).
            edges = topology.edge_index
            attrs = topology.edge_attr.float()
            grad_pred = compute_gradient(raw_pred[:, field].float(), edges, attrs)
            grad_true = compute_gradient(raw_true[:, field].float(), edges, attrs)
            grad_sq[field] += ((grad_pred - grad_true) ** 2).sum()
            tgt_grad_sq[field] += (grad_true**2).sum()
            # Frame-to-frame *increment*: the field itself is nearly static, so
            # a persistence predictor already scores high R2; dR2 measures how
            # much of the actual change the model reproduces (0 = persistence).
            base = u_raw[burn_in][:, field].double()
            delta_true = raw_true[:, field] - base
            delta_pred = raw_pred[:, field] - base
            inc_err[field] += ((delta_pred - delta_true) ** 2).sum()
            inc_sq[field] += (delta_true**2).sum()
            inc_pred_sq[field] += (delta_pred**2).sum()
            inc_cross[field] += (delta_pred * delta_true).sum()
            bucket_inc_err[bucket][field] += ((delta_pred - delta_true) ** 2).sum()
            bucket_inc_sq[bucket][field] += (delta_true**2).sum()
            bucket_sq[bucket][field] += (diff**2).sum()
            bucket_tgt_sq[bucket][field] += (raw_true[:, field] ** 2).sum()
            prev_diff = raw_pred[:, field] - base
            prev_sq[field] += (prev_diff**2).sum()
            bucket_prev_sq[bucket][field] += (prev_diff**2).sum()
        for field, name in enumerate(FIELD_NAMES):
            base = u_raw[burn_in][:, field].double()
            delta_true = raw_true[:, field] - base
            delta_pred = raw_pred[:, field] - base
            denom = float((delta_true**2).sum())
            if denom > 0.0:
                window_dr2[name].append(
                    1.0 - float(((delta_pred - delta_true) ** 2).sum()) / denom
                )
            d_true = float(((raw_pred[:, field] - raw_true[:, field]) ** 2).sum()) ** 0.5
            d_prev = float((delta_pred**2).sum()) ** 0.5
            if d_true + d_prev > 0.0:
                prev_dist[name].append(d_true / (d_true + d_prev))
        pos = topology.pos.double()
        axis_pred = int(raw_pred[:, 0].argmax())
        axis_true = int(raw_true[:, 0].argmax())
        axis_errors.append(float((pos[axis_pred] - pos[axis_true]).norm()))
        # The hard argmax has no resolution once relL2(psi) << the flatness of
        # the psi top (pitfalls #54): report instead the *psi value at the true
        # axis node*, which is well defined regardless of the peak shape.
        psi_scale = float(raw_true[:, 0].abs().max()) or 1.0
        axis_value_errors.append(
            abs(float(raw_pred[axis_true, 0] - raw_true[axis_true, 0])) / psi_scale
        )
        used += 1
        node_count += int(raw_pred.shape[0])

    print(
        f"held-out={args.held_out} windows={used} alpha={args.alpha} "
        f"sub_steps={args.sub_steps} burn_in_substeps={args.burn_in_substeps}"
    )
    # MSE is reported in the JOREK physical units of each field (mean squared
    # error over nodes and windows), i.e. the quantity R2 is 1 - MSE/Var of.
    # ``loss`` in the training log is the *normalised* objective; both are
    # reported side by side in the sweep summary.
    # Increment metrics are reported with a *resolvability* guard: the model can
    # only be credited with predicting a change if that change is larger than the
    # model's own field-level error (otherwise the true increment sits inside the
    # representation noise floor and any "" skill "" number is meaningless - on
    # fine-dt windows both the ncps and the LTC model score dR2 ~ -40, pitfalls
    # #49).  ``dR2`` is therefore printed as ``n/r`` when unresolved, and the
    # dt-robust companion column ``derr/|u|`` gives the step error in units of the
    # field itself.
    print(f"{'field':9s} {'R2':>8s} {'MSE':>11s} {'relL2':>9s} {'closer':>7s} "
          f"{'derr/|u|':>9s} {'|d|/|u|':>9s} {'dR2(med)':>9s} {'corr':>7s} {'amp':>7s}")
    r2_values, rel_values = [], []
    derr_values = []
    dr2_resolved: list[float] = []
    per_field: dict[str, dict[str, float]] = {}
    for field, name in enumerate(FIELD_NAMES):
        r2 = 1.0 - float(sq[field] / tgt_sq[field].clamp_min(1e-30))
        mse = float(sq[field] / max(node_count, 1))
        rel = float((sq[field] / tgt_sq[field].clamp_min(1e-30)).sqrt())
        h1 = float((grad_sq[field] / tgt_grad_sq[field].clamp_min(1e-30)).sqrt())
        budget = float(sum_err[field] / sum_tgt[field].abs().clamp_min(1e-30))
        d_r2 = 1.0 - float(inc_err[field] / inc_sq[field].clamp_min(1e-30))
        d_scale = float((inc_sq[field] / tgt_sq[field].clamp_min(1e-30)).sqrt())
        d_err_rel = float((inc_err[field] / tgt_sq[field].clamp_min(1e-30)).sqrt())
        d_true_norm = float((sq[field]).sqrt())
        d_prev_norm = float((prev_sq[field]).sqrt())
        closer = d_true_norm / max(d_true_norm + d_prev_norm, 1e-30)
        # flag (not filter): the true step is smaller than the model's own
        # field-level error, so dR2 is dominated by representation error
        below_floor = d_scale < rel
        # dR2 = 2*corr*amp - amp^2 ; reporting corr/amp separately shows whether a
        # field fails by pointing the wrong way or by under/over-shooting.
        amp = float((inc_pred_sq[field] / inc_sq[field].clamp_min(1e-30)).sqrt())
        corr = float(
            inc_cross[field]
            / (inc_pred_sq[field].clamp_min(1e-30) * inc_sq[field].clamp_min(1e-30)).sqrt()
        )
        r2_values.append(r2)
        rel_values.append(rel)
        derr_values.append(d_err_rel)
        if not below_floor:
            dr2_resolved.append(d_r2)
        d_r2_median = float(np.median(window_dr2[name])) if window_dr2[name] else float("nan")
        per_field[name] = {
            "R2": r2, "MSE": mse, "relL2": rel, "H1_rel": h1, "budget_rel": budget,
            "dR2": d_r2, "d_scale": d_scale, "corr": corr, "amp": amp,
            "d_err_rel": d_err_rel, "below_floor": bool(below_floor),
            "dR2_median": d_r2_median,
            "closer": closer,
            "closer_median": float(np.median(prev_dist[name])) if prev_dist[name] else float("nan"),
        }
        marker = "*" if below_floor else " "
        print(f"{name:9s} {r2:8.4f} {mse:11.3e} {rel:9.4f} {closer:7.3f} "
              f"{d_err_rel:9.4f} {d_scale:9.4f} {d_r2_median:9.4f} "
              f"{corr:7.3f} {amp:7.3f}")
    print(
        f"{'MEAN':9s} {float(np.mean(r2_values)):8.4f} {'':11s}"
        f" {float(np.mean(rel_values)):9.4f} "
        f"{float(np.mean([v['closer'] for v in per_field.values()])):7.3f} "
        f"{float(np.mean(derr_values)):9.4f} {'':9s} "
        f"{float(np.mean([v['dR2_median'] for v in per_field.values()])):9.4f}"
        f"   (closer: 0.5 = halfway between the two frames, ->1 = copies the "
        f"previous frame; * per-field flag not used here)"
    )
    axis = np.asarray(axis_errors)
    axis_value = np.asarray(axis_value_errors)
    print(
        f"magnetic-axis error [m]: median={np.median(axis):.3e} "
        f"p90={np.percentile(axis, 90):.3e} max={axis.max():.3e}"
        f"   (hard argmax; unstable on a flat psi top, pitfalls #54)"
    )
    print(
        f"psi at the TRUE axis node: |pred-true|/max|psi| "
        f"median={np.median(axis_value):.3e} p90={np.percentile(axis_value, 90):.3e} "
        f"(<- use this one as the axis/shape gate)"
    )
    print(f"\nper-dt-bucket (dR2 averaged over *resolved* fields only, "
          f"plus the dt-robust step error `derr/|u|`):")
    print(f"{'bucket':10s} {'n':>4s} {'closer':>7s} {'dR2(pool)':>9s} {'derr/|u|':>9s} "
          f"{'psi':>7s} {'Phi':>7s} {'Z_j':>7s} {'omega':>7s} {'T_e':>7s}")
    bucket_rows = {}
    for bucket in bucket_names:
        if bucket_count[bucket] == 0:
            continue
        per_field_bucket = 1.0 - (
            bucket_inc_err[bucket] / bucket_inc_sq[bucket].clamp_min(1e-30)
        )
        bucket_rel = (bucket_sq[bucket] / bucket_tgt_sq[bucket].clamp_min(1e-30)).sqrt()
        bucket_dtrue = (bucket_inc_sq[bucket] / bucket_tgt_sq[bucket].clamp_min(1e-30)).sqrt()
        bucket_derr = (bucket_inc_err[bucket] / bucket_tgt_sq[bucket].clamp_min(1e-30)).sqrt()
        bucket_prev = (bucket_prev_sq[bucket] / bucket_tgt_sq[bucket].clamp_min(1e-30)).sqrt()
        bucket_closer = float(
            (bucket_rel / (bucket_rel + bucket_prev).clamp_min(1e-30)).mean()
        )
        resolved_mask = bucket_dtrue >= bucket_rel
        mean_bucket = float(per_field_bucket[resolved_mask].mean()) if bool(resolved_mask.any()) else float("nan")
        bucket_rows[bucket] = {
            "n": bucket_count[bucket],
            "mean_dR2": mean_bucket,
            "n_resolved": int(resolved_mask.sum().item()),
            "mean_d_err_rel": float(bucket_derr.mean()),
            "median_dR2": float(np.median([
                np.median(window_dr2[name]) for name in FIELD_NAMES if window_dr2[name]
            ])) if any(window_dr2.values()) else float("nan"),
            "closer": bucket_closer,
            **{name: float(per_field_bucket[FIELD_NAMES.index(name)])
               for name in ("psi", "Phi", "Z_j", "omega", "T_e")},
        }
        print(f"{bucket:10s} {bucket_count[bucket]:4d} {bucket_closer:7.3f} "
              f"{mean_bucket:9.4f} {float(bucket_derr.mean()):9.4f} "
              f"{bucket_rows[bucket]['psi']:8.4f} {bucket_rows[bucket]['Phi']:8.4f} "
              f"{bucket_rows[bucket]['Z_j']:8.4f} {bucket_rows[bucket]['omega']:8.4f} "
              f"{bucket_rows[bucket]['T_e']:8.4f}")
    if args.json:
        payload = {
            "checkpoint": args.checkpoint,
            "held_out": args.held_out,
            "windows": used,
            "alpha": args.alpha,
            "sub_steps": args.sub_steps,
            "burn_in_substeps": args.burn_in_substeps,
            "mean_R2": float(np.mean(r2_values)),
            "mean_relL2": float(np.mean(rel_values)),
            "mean_d_err_rel": float(np.mean(derr_values)),
            "mean_closer": float(np.mean([v["closer"] for v in per_field.values()])),
            "mean_closer_median": float(np.nanmean(
                [v["closer_median"] for v in per_field.values()]
            )),
            "mean_dR2_median": float(np.mean(
                [per_field[name]["dR2_median"] for name in FIELD_NAMES]
            )),
            "mean_dR2_resolved": float(np.mean(dr2_resolved)) if dr2_resolved else None,
            "n_dR2_resolved": len(dr2_resolved),
            "axis_error_m": {
                "median": float(np.median(axis)),
                "p90": float(np.percentile(axis, 90)),
                "max": float(axis.max()),
            },
            "axis_value_rel_error": {
                "median": float(np.median(axis_value)),
                "p90": float(np.percentile(axis_value, 90)),
            },
            "fields": per_field,
            "dt_buckets": bucket_rows,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
