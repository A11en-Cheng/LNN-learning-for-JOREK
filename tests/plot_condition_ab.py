"""Condition on/off A/B summary: per-field R2/relL2 and per-dt-bucket step error.

Reads the JSON files produced by ``tests/eval_phase3_structure.py`` (all
windows of the held-out run) and writes one comparison figure plus, optionally,
a Markdown table.  Usage:

    ./.torch-cuda-env/bin/python tests/plot_condition_ab.py \
        --eval "cond_best=.scratch/n/eval_allwin_cond_h1_1000_checkpoint_best.json" \
        --eval "nocond_best=.scratch/n/eval_allwin_nocond_h1_1000_checkpoint_best.json" \
        --eval "old_step_s1=.scratch/n/eval_old_step_s1_on_manifest_h5.json" \
        --out .scratch/n/fig/allwin_condition_ab.png \
        --markdown .scratch/n/fig/allwin_condition_ab.md
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

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.model502_schema import FIELD_NAMES


def _load(spec: str) -> tuple[str, dict]:
    if "=" not in spec:
        raise SystemExit(f"--eval expects LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    return label, json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", action="append", required=True,
                        help="LABEL=eval_phase3_structure.json (repeatable)")
    parser.add_argument("--out", default=".scratch/n/fig/condition_ab.png")
    parser.add_argument("--markdown", default="")
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args()

    runs = [_load(spec) for spec in args.eval]
    labels = [label for label, _ in runs]
    buckets = ["dt<3", "3-15", "15-60", "dt>=60"]
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(runs), 2)))

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 4.8))
    width = 0.8 / len(runs)
    x = np.arange(len(FIELD_NAMES))
    for run_index, (label, payload) in enumerate(runs):
        offset = (run_index - (len(runs) - 1) / 2.0) * width
        axes[0].bar(x + offset, [payload["fields"][n]["R2"] for n in FIELD_NAMES],
                    width=width, label=label, color=colors[run_index])
        axes[1].bar(x + offset, [payload["fields"][n]["relL2"] for n in FIELD_NAMES],
                    width=width, label=label, color=colors[run_index])
        bucket_values = [
            payload["dt_buckets"].get(bucket, {}).get("mean_d_err_rel", np.nan)
            for bucket in buckets
        ]
        axes[2].bar(np.arange(len(buckets)) + offset, bucket_values,
                    width=width, label=label, color=colors[run_index])
    counts = [
        runs[0][1]["dt_buckets"].get(bucket, {}).get("n", 0) for bucket in buckets
    ]
    axes[0].axhline(0.9, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel("R2 (higher better)")
    axes[0].set_title("per-field R2 on the held-out run")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("relative L2 (lower better, log)")
    axes[1].set_title("per-field relative error")
    axes[2].set_yscale("log")
    axes[2].set_ylabel("mean d_err/|u| (lower better, log)")
    axes[2].set_title("step error by dt bucket")
    for axis in axes:
        axis.grid(alpha=0.3, axis="y")
        axis.legend(fontsize=8)
    axes[0].set_xticks(x, FIELD_NAMES, rotation=45, ha="right")
    axes[1].set_xticks(x, FIELD_NAMES, rotation=45, ha="right")
    axes[2].set_xticks(
        np.arange(len(buckets)),
        [f"{bucket}\n(n={count})" for bucket, count in zip(buckets, counts)],
    )
    figure.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=args.dpi)
    print(f"wrote {out}")

    if args.markdown:
        lines = ["# Condition on/off A/B (held-out run, all windows)", ""]
        lines.append("| field | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        for name in FIELD_NAMES:
            values = [f"{payload['fields'][name]['R2']:.4f}" for _, payload in runs]
            lines.append(f"| {name} R2 | " + " | ".join(values) + " |")
        lines.append("")
        for name in FIELD_NAMES:
            values = [f"{payload['fields'][name]['relL2']:.4f}" for _, payload in runs]
            lines.append(f"| {name} relL2 | " + " | ".join(values) + " |")
        lines.append("")
        lines.append("| bucket | " + " | ".join(labels) + " |")
        lines.append("|---" * (len(labels) + 1) + "|")
        for bucket in buckets:
            values = []
            for _, payload in runs:
                record = payload["dt_buckets"].get(bucket, {})
                values.append(
                    f"n={record.get('n', 0)}, derr={record.get('mean_d_err_rel', float('nan')):.4f}"
                )
            lines.append(f"| {bucket} | " + " | ".join(values) + " |")
        lines.append("")
        lines.append("| run | mean R² | mean relL2 | mean closer | windows |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, payload in runs:
            lines.append(
                f"| {label} | {payload['mean_R2']:.4f} | {payload['mean_relL2']:.4f} "
                f"| {payload['mean_closer']:.4f} | {payload['windows']} |"
            )
        table = Path(args.markdown)
        table.parent.mkdir(parents=True, exist_ok=True)
        table.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {table}")


if __name__ == "__main__":
    main()
