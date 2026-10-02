"""Loss curves + inner-val skill curves for one or more training runs.

Left panel : train_mse (the objective actually optimised) and the inner-val MSE.
Right panel: inner-val ``closer`` (lower is better; the frame-level similarity
             metric) and ``skill_pers`` (legacy dR2, higher is better), with the
             epoch ``--save-best`` picked marked.

Example:

    MPLCONFIGDIR=/tmp/mpl ./.torch-cuda-env/bin/python tests/plot_training_curves.py \
        --run output/cfc_interp/ltc_lr3e3_dt --out .scratch/n/fig/curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _series(history: list[dict], key: str) -> list[float]:
    values = []
    for record in history:
        if key == "train_mse":
            values.append(record.get("train_mse"))
        elif key == "val_mse":
            values.append(record.get("val", {}).get("mse"))
        else:
            values.append(record.get("val", {}).get(key))
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", nargs="+", required=True, help="training output dirs")
    parser.add_argument("--out", default=".scratch/n/fig/curves.png")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for run in args.run:
        metrics = json.loads((Path(run) / "metrics.json").read_text(encoding="utf-8"))
        history = metrics["history"]
        epochs = [r["epoch"] for r in history]
        name = Path(run).name
        axes[0].plot(epochs, _series(history, "train_mse"), label=f"{name} train")
        axes[0].plot(epochs, _series(history, "val_mse"), "--", label=f"{name} val")
        closer = _series(history, "mean_closer")
        if any(v is not None for v in closer):
            axes[1].plot(epochs, closer, label=f"{name} closer")
        axes[1].plot(epochs, _series(history, "mean_dr2"), ":", label=f"{name} skill_pers")
        best_epoch = metrics.get("best_epoch")
        if best_epoch:
            axes[1].axvline(best_epoch, color="grey", lw=0.8, alpha=0.6)

    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("MSE (normalised objective)")
    axes[0].set_title("loss / inner-val MSE")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].axhline(0.5, color="k", lw=0.8, ls="--")
    # A single unstable epoch can drive skill_pers to -O(100) and squash the whole
    # panel; clip the view (the values stay in the JSON) so the trend is readable.
    axes[1].set_ylim(-0.25, 1.05)
    axes[1].text(
        0.02, 0.02, "values outside [-0.25, 1.05] clipped (unstable epochs)",
        transform=axes[1].transAxes, fontsize=6, color="grey",
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("inner-val metric")
    axes[1].set_title("closer (lower better, 0.5 = halfway) / skill_pers (higher better)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
