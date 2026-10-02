"""Marker -> projection reconstruction check (charge moments).

For a set of saved frames this compares the global marker charge moments

    N_imp = sum_p w_p * tstep
    Z_imp = sum_p w_p * <Z>_p * tstep
    Z_eff = sum_p w_p * <Z^2>_p * tstep

with the corresponding totals of the recorded ``projections*.h5`` channels
``[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]``.  A constant projection/marker ratio
across frames means the mapping is linear and computable; the per-frame spread
of the ratio is the reconstruction residual of this first (spatially
unresolved) check.

Example:

    .torch-cuda-env/bin/python tests/check_marker_projection_reconstruction.py \
        --run scan_1d22_ang15_5_massloss_neon_v400 --frames 12 --stride 5
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


CHANNELS = ["dE_ion", "rad_sink", "Z_eff", "Z_imp", "N_imp"]


def _index_map(run_dir: Path, pattern: str) -> dict[int, Path]:
    regex = re.compile(pattern)
    mapping: dict[int, Path] = {}
    for path in run_dir.iterdir():
        match = regex.fullmatch(path.name)
        if match:
            mapping[int(match.group(1))] = path
    return mapping


def _marker_moments(part_path: Path, *, stride: int) -> tuple[float, float, float]:
    with h5py.File(part_path, "r") as handle:
        group = handle["groups/001"]
        life = np.asarray(group["i_life"][::stride])
        mask = life > 0
        if not mask.any():
            return 0.0, 0.0, 0.0
        weight = np.asarray(group["weight"][::stride], dtype=np.float64)[mask]
        p_imp = np.asarray(group["P_imp"][::stride], dtype=np.float64)[mask]
    charge = np.arange(p_imp.shape[1], dtype=np.float64)
    mean_z = p_imp @ charge
    mean_z2 = p_imp @ (charge**2)
    scale = float(stride)
    return (
        float((weight).sum() * scale),
        float((weight * mean_z).sum() * scale),
        float((weight * mean_z2).sum() * scale),
    )


def _projection_totals(projection_path: Path) -> np.ndarray:
    with h5py.File(projection_path, "r") as handle:
        values = np.asarray(handle["values"][:5], dtype=np.float64)
    return values.reshape(values.shape[0], -1).sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.data_root) / args.run
    state_map = _index_map(run_dir, r"jorek(\d+)\.h5")
    part_map = _index_map(run_dir, r"part(\d+)\.h5")
    projection_map = _index_map(run_dir, r"projections(\d+)\.h5")
    indices = sorted(set(state_map) & set(part_map) & set(projection_map))
    if not indices:
        raise SystemExit(f"no frame files in {run_dir}")
    picks = [
        indices[int(round(fraction * (len(indices) - 1)))]
        for fraction in np.linspace(0, 1, args.frames)
    ]
    picks = sorted(set(int(index) for index in picks))

    rows = []
    for index in indices:
        state_path = state_map[index]
        part_path = part_map[index]
        projection_path = projection_map[index]
        with h5py.File(state_path, "r") as handle:
            tstep = float(np.ravel(handle["tstep"][()])[0]) if "tstep" in handle else 1.0
            t_now = float(np.ravel(handle["t_now"][()])[0]) if "t_now" in handle else float("nan")
        n_imp, z_imp, z_eff = _marker_moments(part_path, stride=args.stride)
        totals = _projection_totals(projection_path)
        rows.append(
            {
                "index": index,
                "t_now": t_now,
                "tstep": tstep,
                "marker": np.array([n_imp, z_imp, z_eff]),
                "projection": np.array([totals[4], totals[3], totals[2]]),
            }
        )

    if not rows:
        raise SystemExit("no usable frames")
    marker = np.stack([row["marker"] for row in rows])
    projection = np.stack([row["projection"] for row in rows])
    tstep = np.array([row["tstep"] for row in rows])
    print(f"run={args.run} frames={len(rows)} stride={args.stride}")
    print("alignments: same = P_k vs M_k*tstep_k | stagger = P_k vs M_{k-1}*tstep_k | trapezoid = P_k vs 0.5(M_k+M_{k-1})*tstep_k")

    def _report(label: str, predicted: np.ndarray, target: np.ndarray) -> None:
        for column, name in enumerate(["N_imp", "Z_imp", "Z_eff"]):
            m = predicted[:, column]
            p = target[:, column]
            valid = np.isfinite(m) & np.isfinite(p) & (m > 0) & (p != 0)
            if valid.sum() < 3:
                continue
            corr = float(np.corrcoef(m[valid], p[valid])[0, 1])
            ratio = p[valid] / m[valid]
            print(
                f"{label:10s} {name:8s} n={int(valid.sum()):3d} corr={corr:8.5f} "
                f"ratio_mean={float(np.mean(ratio)):11.4e} rel_std={float(np.std(ratio) / abs(np.mean(ratio))):8.4f}"
            )

    _report("same", marker * tstep[:, None], projection)
    shifted_marker = np.vstack([np.full((1, 3), np.nan), marker[:-1]])
    shifted_marker *= tstep[:, None]
    _report("stagger", shifted_marker, projection)
    trapezoid = 0.5 * (marker + np.vstack([np.full((1, 3), np.nan), marker[:-1]]))
    trapezoid *= tstep[:, None]
    _report("trapezoid", trapezoid, projection)



if __name__ == "__main__":
    main()
