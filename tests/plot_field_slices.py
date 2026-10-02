"""Predicted vs recorded field on the poloidal cross-section (one window).

For each of the nine state fields the figure shows three panels (true / predicted
/ difference) drawn on the FEM quad mesh in physical (R, Z) coordinates with an
equal aspect ratio, so shape errors (the part R2 hides) are visible by eye.
The prediction is the decoded field at the *next recorded frame* of the window -
the same object every metric is computed from.

Example:

    MPLCONFIGDIR=/tmp/mpl ./.torch-cuda-env/bin/python tests/plot_field_slices.py \
        --checkpoint output/cfc_interp/ltc_lr3e3_dt/checkpoint_best.pt \
        --window 0 --out .scratch/n/fig/slices_w0.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import PolyCollection

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import (
    InterpolationWindowDataset,
    RunFrameStore,
    collate_windows,
    compute_transform_stats,
    to_device,
)
from src.data.model502_schema import FIELD_NAMES, destandardize, inverse_transform_state
from src.models.cfc_interp import CfCInterpolator, deterministic_fractions
from src.models.cfc_interp import model_kwargs_from_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window", type=int, default=0, help="dataset index in the val split")
    parser.add_argument("--sub-steps", type=int, default=2)
    parser.add_argument("--burn-in-substeps", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--horizon", type=int, default=1,
                        help="target frame = burn_in + horizon (must match the checkpoint)")
    parser.add_argument("--horizon-mode", choices=("jump", "rollout"), default="jump")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=".scratch/n/fig/slices.png")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    kwargs = model_kwargs_from_meta(payload.get("meta") or {})
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(r for r in manifest["folds"] if r["held_out_run"] == args.held_out)
    stores = {
        name: RunFrameStore(Path(manifest["data_root"]) / name, cache_dir=args.cache_dir).load()
        for name in fold["train_runs"]
    }
    stats = compute_transform_stats(
        stores, runs=fold["train_runs"], with_lap=bool(kwargs["input_deriv2"])
    )
    dataset = InterpolationWindowDataset(
        manifest, manifest["data_root"], runs=[args.held_out], inner="val", stats=stats,
        cache_dir=args.cache_dir, with_shards=False, with_lap=bool(kwargs["input_deriv2"]),
    )
    model = CfCInterpolator(**kwargs)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    mean = torch.tensor(stats["field_mean"], device=device)
    std = torch.tensor(stats["field_std"], device=device)

    item = dataset[args.window]
    batch = to_device(collate_windows([item]), device)
    u, u_raw = batch["u"][0], batch["u_raw"][0]
    dt, source = batch["dt"][0], batch["source"][0]
    deriv = batch["u_deriv"][0] if kwargs["input_deriv"] else None
    lap = batch["u_lap"][0] if kwargs["input_deriv2"] else None
    burn_in = int((~batch["loss_mask"][0]).sum().item())
    target_index = burn_in + args.horizon
    with torch.no_grad():
        hidden = None
        for k in range(burn_in):
            _, _, hidden = model.step(
                u[k], hidden, dt[k], source[k + 1], None,
                None if deriv is None else deriv[k],
                None if lap is None else lap[k],
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
                    None if lap is None else lap[step_index],
                )
                prediction, hidden = out["prediction"], out["hidden"]
        else:
            prediction = model.multi_substep(
                u[burn_in], hidden, deterministic_fractions(args.sub_steps, args.alpha),
                dt[burn_in:target_index].sum(), source[target_index], None,
                None if deriv is None else deriv[burn_in],
                None if lap is None else lap[burn_in],
            )["prediction"]
    raw_pred = inverse_transform_state(
        destandardize(prediction, mean, std, name="prediction")
    ).cpu().numpy()
    raw_true = u_raw[target_index].cpu().numpy()
    raw_prev = u_raw[burn_in].cpu().numpy()

    topology = batch["topology"]
    pos = topology.pos.cpu().numpy()
    vertex = topology.vertex.cpu().numpy()          # [4, n_elem], 0-based
    quads = np.transpose(pos[vertex], (1, 0, 2))    # [n_elem, 4, 2]

    figure, axes = plt.subplots(
        len(FIELD_NAMES), 3, figsize=(10.5, 2.2 * len(FIELD_NAMES)), squeeze=False
    )
    for field, name in enumerate(FIELD_NAMES):
        true_values = raw_true[:, field]
        pred_values = raw_pred[:, field]
        diff = pred_values - true_values
        element_true = true_values[vertex].mean(axis=0)
        element_pred = pred_values[vertex].mean(axis=0)
        element_diff = diff[vertex].mean(axis=0)
        vmin, vmax = np.nanmin(element_true), np.nanmax(element_true)
        if vmax - vmin < 1e-30:
            vmax = vmin + 1e-30
        scale = np.nanmax(np.abs(element_diff)) or 1e-30
        for column, (values, title, limits, cmap) in enumerate((
            (element_true, f"{name} (recorded frame)", (vmin, vmax), "viridis"),
            (element_pred, f"{name} (predicted)", (vmin, vmax), "viridis"),
            (element_diff, "pred - true", (-scale, scale), "coolwarm"),
        )):
            axis = axes[field][column]
            collection = PolyCollection(quads, array=values, cmap=cmap)
            collection.set_clim(*limits)
            axis.add_collection(collection)
            axis.autoscale_view()
            axis.set_aspect("equal")
            axis.tick_params(labelsize=6, length=2)
            if field == 0:
                axis.set_title(title, fontsize=9, pad=8)
            if column == 0:
                axis.set_ylabel(name, fontsize=9)
            if field == len(FIELD_NAMES) - 1:
                axis.set_xlabel("R [m]", fontsize=8)
        figure.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(*[element_true.min(), element_true.max()]),
                cmap="viridis",
            ),
            ax=axes[field][1], fraction=0.046,
        )
        figure.colorbar(
            matplotlib.cm.ScalarMappable(
                norm=matplotlib.colors.Normalize(-scale, scale), cmap="coolwarm"
            ),
            ax=axes[field][2], fraction=0.046,
        )
    figure.suptitle(
        f"{item['run']} window {args.window} (dt={float(dt[burn_in]):g}): "
        f"recorded / predicted / difference at the next recorded frame",
        fontsize=11, y=0.997,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.985))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")
    # quick numeric summary so a run can be logged without opening the figure
    print(f"{'field':9s} {'max|pred-true|':>14s} {'std(true)':>10s} {'rel_err':>9s}")
    for field, name in enumerate(FIELD_NAMES):
        diff = raw_pred[:, field] - raw_true[:, field]
        scale = raw_true[:, field].std() or 1e-30
        print(f"{name:9s} {np.abs(diff).max():14.3e} {scale:10.3e} "
              f"{np.linalg.norm(diff) / max(np.linalg.norm(raw_true[:, field]), 1e-30):9.4f}")


if __name__ == "__main__":
    main()
