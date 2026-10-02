"""Build the Phase-1 interpolation window manifest (ADR-0012).

Example:

    .torch-cuda-env/bin/python tests/build_interp_manifest.py \
        --out .scratch/model502-interp/manifest.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.window_manifest import build_interp_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--out", default=".scratch/model502-interp/manifest.json")
    parser.add_argument("--burn-in", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--rel-threshold", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--block-windows", type=int, default=16)
    parser.add_argument("--runs", nargs="*", default=None, help="optional run-name filter")
    parser.add_argument(
        "--run-glob",
        default="scan_*",
        help="run-name glob (default: the ten production scan_* runs; pass '' for all 2D runs)",
    )
    args = parser.parse_args()

    started = time.time()
    manifest = build_interp_manifest(
        args.data_root,
        burn_in=args.burn_in,
        horizon=args.horizon,
        stride=args.stride,
        rel_threshold=args.rel_threshold,
        val_fraction=args.val_fraction,
        block_windows=args.block_windows,
        runs=args.runs,
        run_glob=args.run_glob or None,
    )
    out = write_manifest(manifest, args.out)

    print(f"{'run':40s} {'frames':>6} {'post_tn':>7} {'segments':>8} {'windows':>7}")
    for record in manifest["runs"]:
        post = record["post_tn_frames"] if record["post_tn_frames"] is not None else "-"
        print(
            f"{record['name']:40s} {record['n_frames']:6d} {str(post):>7} "
            f"{len(record['segments']):8d} {record['n_windows']:7d}"
        )
    summary = manifest["summary"]
    print(
        f"total: runs={summary['n_runs']} segments={summary['n_segments']} "
        f"windows={summary['n_windows']} inner={summary['inner']}"
    )
    print(f"manifest: {out}  (built in {time.time() - started:.1f}s)")


if __name__ == "__main__":
    main()
