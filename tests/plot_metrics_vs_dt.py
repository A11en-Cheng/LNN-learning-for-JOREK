"""Metrics vs the recorded interval length dt, per field and averaged.

This is the diagnostic that motivated the metric work: the corpus mixes frame
spacings from 0.5 to 250 JOREK units and the *task* changes with dt, so every
metric has to be read as a function of dt (pitfalls #48-#51).

Panels (x = dt, log scale): ``closer`` (lower better, 0.5 = halfway between the
previous and the next frame), ``relL2`` (field-level relative error) and the
legacy ``skill_pers`` (higher better; shown only for reference - it loses
meaning when the true increment approaches the representation floor).

Also writes a CSV with one row per evaluated window so the numbers can be
re-plotted or aggregated elsewhere.

Example:

    MPLCONFIGDIR=/tmp/mpl ./.torch-cuda-env/bin/python tests/plot_metrics_vs_dt.py \
        --checkpoint output/cfc_interp/ltc_lr3e3_dt/checkpoint_best.pt \
        --held-out scan_1d22_ang15_5_massloss_neon_v400 \
        --out .scratch/n/fig/metrics_vs_dt.png
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from src.models.cfc_interp import CfCInterpolator, deterministic_fractions
from src.models.cfc_interp import model_kwargs_from_meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--sub-steps", type=int, default=2)
    parser.add_argument("--burn-in-substeps", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-windows", type=int, default=45)
    parser.add_argument(
        "--inner", choices=("val", "train", "all"), default="val",
        help="which windows of the held-out run to plot; 'all' is the correct "
        "choice for manifest_h5 (the val split is only 13 of 61 windows)",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=".scratch/n/fig/metrics_vs_dt.png")
    parser.add_argument("--csv", default="")
    parser.add_argument("--dpi", type=int, default=140)
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
        stores, runs=fold["train_runs"], with_lap=bool(kwargs["input_deriv2"]),
        with_condition=bool(kwargs.get("input_condition", False)),
    )
    dataset = InterpolationWindowDataset(
        manifest, manifest["data_root"], runs=[args.held_out],
        inner=None if args.inner == "all" else args.inner, stats=stats,
        cache_dir=args.cache_dir, with_shards=False, with_lap=bool(kwargs["input_deriv2"]),
        with_condition=bool(kwargs.get("input_condition", False)),
    )
    model = CfCInterpolator(**kwargs)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    mean = torch.tensor(stats["field_mean"], device=device)
    std = torch.tensor(stats["field_std"], device=device)

    rows = []
    with torch.no_grad():
        for index in range(min(len(dataset), args.max_windows)):
            item = dataset[index]
            batch = to_device(collate_windows([item]), device)
            u, u_raw = batch["u"][0], batch["u_raw"][0]
            dt, source = batch["dt"][0], batch["source"][0]
            deriv = batch["u_deriv"][0] if kwargs["input_deriv"] else None
            lap = batch["u_lap"][0] if kwargs["input_deriv2"] else None
            condition = (
                batch["condition"][0] if kwargs.get("input_condition", False) else None
            )
            burn_in = int((~batch["loss_mask"][0]).sum().item())
            hidden = None
            for k in range(burn_in):
                _, _, hidden = model.step(
                    u[k], hidden, dt[k], source[k + 1], None,
                    None if deriv is None else deriv[k],
                    None if lap is None else lap[k],
                    sub_steps=args.burn_in_substeps,
                    condition=None if condition is None else condition[k],
                )
            prediction = model.multi_substep(
                u[burn_in], hidden, deterministic_fractions(args.sub_steps, args.alpha),
                dt[burn_in], source[burn_in + 1], None,
                None if deriv is None else deriv[burn_in],
                None if lap is None else lap[burn_in],
                condition=None if condition is None else condition[burn_in],
            )["prediction"]
            raw_pred = inverse_transform_state(
                destandardize(prediction, mean, std, name="prediction")
            ).double()
            raw_true = u_raw[burn_in + 1].double()
            raw_prev = u_raw[burn_in].double()
            row = {"window": index, "run": item["run"], "dt": float(dt[burn_in])}
            closers, rels, skills = [], [], []
            for field, name in enumerate(FIELD_NAMES):
                d_true = float(((raw_pred[:, field] - raw_true[:, field]) ** 2).sum()) ** 0.5
                d_prev = float(((raw_pred[:, field] - raw_prev[:, field]) ** 2).sum()) ** 0.5
                inc_true = float(((raw_true[:, field] - raw_prev[:, field]) ** 2).sum()) ** 0.5
                closer = d_true / max(d_true + d_prev, 1e-30)
                rel = float(((raw_pred[:, field] - raw_true[:, field]) ** 2).sum()) ** 0.5 / max(
                    float((raw_true[:, field] ** 2).sum()) ** 0.5, 1e-30
                )
                skill = 1.0 - (d_true / inc_true) ** 2 if inc_true > 0 else float("nan")
                row[f"closer_{name}"] = closer
                row[f"relL2_{name}"] = rel
                row[f"skill_{name}"] = skill
                closers.append(closer)
                rels.append(rel)
                skills.append(skill)
            row["closer_mean"] = sum(closers) / len(closers)
            row["relL2_mean"] = sum(rels) / len(rels)
            row["skill_mean"] = sum(skills) / len(skills)
            rows.append(row)

    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for field, name in enumerate(FIELD_NAMES):
        axes[0].scatter([r["dt"] for r in rows], [r[f"closer_{name}"] for r in rows],
                        s=18, alpha=0.6, label=name)
        axes[1].scatter([r["dt"] for r in rows], [r[f"relL2_{name}"] for r in rows],
                        s=18, alpha=0.6)
        axes[2].scatter([r["dt"] for r in rows], [r[f"skill_{name}"] for r in rows],
                        s=18, alpha=0.6)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("dt (JOREK units)")
        axis.grid(alpha=0.3)
    axes[0].axhline(0.5, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("closer (lower better)")
    axes[0].set_title("frame similarity vs dt")
    axes[0].legend(fontsize=6, ncol=2)
    axes[1].set_ylabel("relL2")
    axes[1].set_title("field-level relative error vs dt")
    axes[2].axhline(0.0, color="k", lw=0.8, ls="--")
    axes[2].set_ylabel("skill_pers (reference only)")
    axes[2].set_title("legacy dR2 vs dt (loses meaning at small dt)")
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")
    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {csv_path}")
    summary = {
        "windows": len(rows),
        "closer_mean": sum(r["closer_mean"] for r in rows) / max(len(rows), 1),
        "relL2_mean": sum(r["relL2_mean"] for r in rows) / max(len(rows), 1),
        "skill_mean": sum(r["skill_mean"] for r in rows) / max(len(rows), 1),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
