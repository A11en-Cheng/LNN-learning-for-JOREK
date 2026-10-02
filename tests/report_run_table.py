"""One table with everything we are supposed to look at: loss / R2 / MSE / dR2.

For every run directory (training ``metrics.json``) plus, when present, its
held-out evaluation (``eval*.json`` written by ``eval_phase3_structure.py
--json``) print one row:

  * training side: last ``train_mse`` (the normalised objective actually
    optimised), last inner-val MSE, and the best inner-val ``mean_dr2`` with the
    epoch it was reached at (that checkpoint is what ``--save-best`` keeps);
  * held-out side: mean R2, mean relative L2, mean MSE (per-field columns are
    printed below the table when ``--fields`` is given), mean dR2 and the
    magnetic-axis error.

Example:

    ./.torch-cuda-env/bin/python tests/report_run_table.py \
        output/cfc_interp/ltc_v2_60x20 output/cfc_interp/ltc_h256 --fields
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict | None:
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="training output directories")
    parser.add_argument("--eval-dir", default=".scratch/n",
                        help="where the eval JSONs live (looked up by run name)")
    parser.add_argument("--fields", action="store_true", help="print per-field tables")
    args = parser.parse_args()

    rows = []
    for run in args.runs:
        run_dir = Path(run)
        metrics = _load(run_dir / "metrics.json")
        if metrics is None:
            print(f"!! {run}: no metrics.json")
            continue
        history = metrics.get("history", [])
        train_kwargs = metrics.get("args", {})
        best_epoch, best_dr2 = None, float("-inf")
        for record in history:
            value = record.get("val", {}).get("mean_dr2")
            if value is not None and value > best_dr2:
                best_epoch, best_dr2 = record["epoch"], value
        last = history[-1] if history else {}
        name = run_dir.name
        evals = {}
        for label in ("checkpoint_best", "checkpoint"):
            payload = _load(Path(args.eval_dir) / f"eval_{name}_{label}.json")
            if payload is not None:
                evals[label] = payload
        rows.append({
            "run": name,
            "cell": train_kwargs.get("cell", "ncps"),
            "hidden": train_kwargs.get("hidden"),
            "backbone": train_kwargs.get("backbone"),
            "lr": train_kwargs.get("lr"),
            "epochs": train_kwargs.get("epochs"),
            "steps": train_kwargs.get("steps_per_epoch"),
            "train_mse_last": last.get("train_mse"),
            "val_mse_last": last.get("val", {}).get("mse"),
            "best_epoch": best_epoch,
            "best_val_dr2": None if best_dr2 == float("-inf") else best_dr2,
            "evals": evals,
        })

    print(f"{'run':18s} {'cell':5s} {'hid':>4s} {'lr':>7s} {'ep':>4s} "
          f"{'train_mse':>10s} {'val_mse':>9s} {'best_ep':>7s} {'val_dr2':>8s} "
          f"{'R2':>7s} {'relL2':>7s} {'closer':>7s} {'dR2med':>7s} {'axis_mm':>8s}")
    for row in rows:
        best_eval = row["evals"].get("checkpoint_best") or {}
        axis = best_eval.get("axis_error_m", {}).get("median")
        val_dr2 = row["best_val_dr2"]
        print(
            f"{row['run']:18s} {row['cell']:5s} {str(row['hidden']):>4s} "
            f"{str(row['lr']):>7s} {str(row['epochs']):>4s} "
            f"{(row['train_mse_last'] if row['train_mse_last'] is not None else float('nan')):10.3e} "
            f"{(row['val_mse_last'] if row['val_mse_last'] is not None else float('nan')):9.4f} "
            f"{str(row['best_epoch']):>7s} "
            f"{(val_dr2 if val_dr2 is not None else float('nan')):8.4f} "
            f"{(best_eval.get('mean_R2') or float('nan')):7.4f} "
            f"{(best_eval.get('mean_relL2') or float('nan')):7.4f} "
            f"{(best_eval.get('mean_closer') or float('nan')):7.3f} "
            f"{(best_eval.get('mean_dR2_median') or float('nan')):7.4f} "
            f"{(1e3 * axis) if axis is not None else float('nan'):8.3f}"
        )

    if args.fields:
        for row in rows:
            for label, payload in row["evals"].items():
                print(f"\n{row['run']} / {label}: "
                      f"R2={payload['mean_R2']:.4f} "
                      f"dR2(med)={payload.get('mean_dR2_median', float('nan')):.4f} "
                      f"derr/u={payload.get('mean_d_err_rel', float('nan')):.4f}")
                print(f"  {'field':9s} {'R2':>8s} {'MSE':>11s} {'relL2':>8s} "
                      f"{'derr/u':>8s} {'dR2med':>8s} {'flag':>5s}")
                for field, values in payload["fields"].items():
                    print(f"  {field:9s} {values['R2']:8.4f} {values['MSE']:11.3e} "
                          f"{values['relL2']:8.4f} {values.get('d_err_rel', float('nan')):8.4f} "
                          f"{values.get('dR2_median', float('nan')):8.4f} "
                          f"{'*' if values.get('below_floor') else '':>5s}")


if __name__ == "__main__":
    main()
