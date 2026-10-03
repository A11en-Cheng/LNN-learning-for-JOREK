#!/usr/bin/env bash
# Regenerate the three diagnostics for one run:
#   1. training curves (loss / val MSE / inner-val closer + skill_pers)
#   2. metrics vs dt on the held-out run (closer / relL2 / skill_pers) + CSV
#   3. field slices for one window (recorded / predicted / difference, 9 fields)
#
# Usage: tests/make_diagnostics.sh <run-dir> [window] [held-out-run]
#
# Environment overrides (needed for the 2026-10-03 manifest_h5 runs):
#   MANIFEST=.scratch/n/manifest_h5.json INNER=all MAXW=600 DEVICE=cpu
#   INNER=all maps to every window of the held-out run ("all" is translated to
#   inner=None inside the plotting tools); the historic default stays val.
set -uo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mpl}
PY=./.torch-cuda-env/bin/python
RUN=${1:?usage: make_diagnostics.sh <run-dir> [window] [held-out-run]}
WINDOW=${2:-0}
HELD=${3:-scan_1d22_ang15_5_massloss_neon_v400}
MANIFEST=${MANIFEST:-.scratch/n/manifest.json}
INNER=${INNER:-val}
MAXW=${MAXW:-45}
DEVICE=${DEVICE:-cuda}
NAME=$(basename "$RUN")
FIG=.scratch/n/fig
mkdir -p "$FIG"

$PY tests/plot_training_curves.py --run "$RUN" --out "$FIG/${NAME}_curves.png"
$PY tests/plot_metrics_vs_dt.py --manifest "$MANIFEST" \
    --checkpoint "$RUN/checkpoint_best.pt" \
    --held-out "$HELD" --inner "$INNER" --max-windows "$MAXW" --device "$DEVICE" \
    --out "$FIG/${NAME}_metrics_vs_dt.png" --csv "$FIG/${NAME}_metrics_vs_dt.csv"
$PY tests/plot_field_slices.py --manifest "$MANIFEST" \
    --checkpoint "$RUN/checkpoint_best.pt" \
    --held-out "$HELD" --inner "$INNER" --window "$WINDOW" --device "$DEVICE" \
    --out "$FIG/${NAME}_slices_w${WINDOW}.png"
echo "diagnostics written under $FIG"
