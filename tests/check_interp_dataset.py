"""Smoke-check the Phase-1 interpolation dataset on real EHL-2 data.

Example:

    .torch-cuda-env/bin/python tests/check_interp_dataset.py \
        --runs scan_1d21_ang12_massloss_neon --cache-dir .scratch/model502-interp/frames
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import (
    InterpolationWindowDataset,
    RunFrameStore,
    compute_transform_stats,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/model502-interp/manifest.json")
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--inner", default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--stats", action="store_true", help="compute train-only stats for the selected runs")
    args = parser.parse_args()

    dataset = InterpolationWindowDataset(
        args.manifest,
        args.data_root,
        runs=args.runs,
        inner=args.inner,
        fold=args.fold,
        normalize=False,
        cache_dir=args.cache_dir,
    )
    print(f"windows={len(dataset)} held_out={dataset.held_out_run}")
    if len(dataset) == 0:
        raise SystemExit("no windows selected")
    item = dataset[args.index]
    u = item["u"]
    dt = item["dt"].numpy()
    print(f"window run={item['run']} segment={item['segment']} frames={item['frame_indices'].tolist()}")
    print(f"u {tuple(u.shape)}  condition {tuple(item['condition'].shape)}  source {tuple(item['source'].shape)}")
    print(
        f"dt[JU] min/med/max = {dt.min():.3g}/{np.median(dt):.3g}/{dt.max():.3g}"
        f"  source_available={int(item['source_available'].sum())}/{len(item['source_available'])}"
    )
    shard_counts = item["shard_mask"].sum(dim=1).tolist()
    print(f"alive shards per frame = {shard_counts}")
    print(
        f"topology nodes={item['topology'].pos.shape[0]} edges={item['topology'].edge_index.shape[1]} "
        f"edge_attr={tuple(item['topology'].edge_attr.shape)}"
    )
    if args.stats:
        runs = args.runs or [item["run"]]
        stores = {
            run: RunFrameStore(Path(args.data_root) / run, cache_dir=args.cache_dir).load()
            for run in runs
        }
        stats = compute_transform_stats(stores, runs=runs)
        print(
            f"stats: frames={stats['n_frames']} values={stats['n_state_values']} "
            f"source_values={stats['n_source_values']}"
        )
        print("field_std:", [round(value, 4) for value in stats["field_std"]])


if __name__ == "__main__":
    main()
