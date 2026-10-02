"""FEM-projection reconstruction of the marker charge moments.

Unlike the naive "sum the projection values" check, this reproduces the JOREK
projection assembly:

    v_5 = HH(l,m) * element_size(l,m) * N_imp / n_norm      (N_imp)
    v_4 = HH(l,m) * element_size(l,m) * Z_imp / n_norm      (Z_imp)
    v_3 = HH(l,m) * element_size(l,m) * Z_eff / n_norm      (Z_eff)

with ``N_imp = weight * timesteps`` and the cubic 4x4 basis functions ported
from ``elements/mod_basisfunctions.f90``.  The comparison uses the global sum
of the assembled coefficients (element-vertex-DoF coefficients map one-to-one
onto node DoFs), so no node-assembly table is required.

Example:

    .torch-cuda-env/bin/python tests/check_marker_projection_fem.py \
        --run scan_1d22_ang15_5_massloss_neon_v400 --stride 5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def basis_2d_cubic(s: float, t: float) -> np.ndarray:
    """Cubic (n_order=3) 4x4 basis functions, indexed [vertex, dof]."""
    u = s - 1.0
    w = t - 1.0
    h = np.empty((4, 4), dtype=np.float64)
    h[0, 0] = u**2 * (1 + 2 * s) * w**2 * (1 + 2 * t)
    h[0, 1] = 3 * u**2 * s * w**2 * (1 + 2 * t)
    h[0, 2] = 3 * u**2 * (1 + 2 * s) * w**2 * t
    h[0, 3] = 9 * u**2 * s * w**2 * t
    h[1, 0] = -(s**2 * (2 * s - 3) * w**2 * (1 + 2 * t))
    h[1, 1] = -3 * u * s**2 * w**2 * (1 + 2 * t)
    h[1, 2] = -3 * s**2 * (2 * s - 3) * w**2 * t
    h[1, 3] = -9 * u * s**2 * w**2 * t
    h[2, 0] = s**2 * (2 * s - 3) * t**2 * (2 * t - 3)
    h[2, 1] = 3 * u * s**2 * t**2 * (2 * t - 3)
    h[2, 2] = 3 * s**2 * (2 * s - 3) * w * t**2
    h[2, 3] = 9 * u * s**2 * w * t**2
    h[3, 0] = -(u**2 * (1 + 2 * s) * t**2 * (2 * t - 3))
    h[3, 1] = -3 * u**2 * s * t**2 * (2 * t - 3)
    h[3, 2] = -3 * u**2 * (1 + 2 * s) * w * t**2
    h[3, 3] = -9 * u**2 * s * w * t**2
    return h


def _index_map(run_dir: Path, pattern: str) -> dict[int, Path]:
    regex = re.compile(pattern)
    return {
        int(match.group(1)): path
        for path in run_dir.iterdir()
        if (match := regex.fullmatch(path.name))
    }


def reconstruct_totals(
    part_path: Path,
    size: np.ndarray,
    *,
    stride: int,
) -> tuple[float, float, float]:
    """Return the reconstructed globals for (N_imp, Z_imp, Z_eff)."""
    n_imp = z_imp = z_eff = 0.0
    with h5py.File(part_path, "r") as handle:
        group = handle["groups/001"]
        life = np.asarray(group["i_life"][::stride])
        mask = life > 0
        if not mask.any():
            return 0.0, 0.0, 0.0
        weight = np.asarray(group["weight"][::stride], dtype=np.float64)[mask]
        p_imp = np.asarray(group["P_imp"][::stride], dtype=np.float64)[mask]
        i_elm = np.asarray(group["i_elm"][::stride], dtype=np.int64)[mask]
        st = np.asarray(group["st"][::stride], dtype=np.float64)[mask]
    charge = np.arange(p_imp.shape[1], dtype=np.float64)
    mean_z = p_imp @ charge
    mean_z2 = p_imp @ (charge**2)
    scale = float(stride)
    elements = np.clip(i_elm - 1, 0, size.shape[-1] - 1)
    s = st[:, 0]
    t = st[:, 1]
    u = s - 1.0
    w = t - 1.0
    h = np.empty((s.shape[0], 4, 4), dtype=np.float64)
    h[:, 0, 0] = u**2 * (1 + 2 * s) * w**2 * (1 + 2 * t)
    h[:, 0, 1] = 3 * u**2 * s * w**2 * (1 + 2 * t)
    h[:, 0, 2] = 3 * u**2 * (1 + 2 * s) * w**2 * t
    h[:, 0, 3] = 9 * u**2 * s * w**2 * t
    h[:, 1, 0] = -(s**2 * (2 * s - 3) * w**2 * (1 + 2 * t))
    h[:, 1, 1] = -3 * u * s**2 * w**2 * (1 + 2 * t)
    h[:, 1, 2] = -3 * s**2 * (2 * s - 3) * w**2 * t
    h[:, 1, 3] = -9 * u * s**2 * w**2 * t
    h[:, 2, 0] = s**2 * (2 * s - 3) * t**2 * (2 * t - 3)
    h[:, 2, 1] = 3 * u * s**2 * t**2 * (2 * t - 3)
    h[:, 2, 2] = 3 * s**2 * (2 * s - 3) * w * t**2
    h[:, 2, 3] = 9 * u * s**2 * w * t**2
    h[:, 3, 0] = -(u**2 * (1 + 2 * s) * t**2 * (2 * t - 3))
    h[:, 3, 1] = -3 * u**2 * s * t**2 * (2 * t - 3)
    h[:, 3, 2] = -3 * u**2 * (1 + 2 * s) * w * t**2
    h[:, 3, 3] = -9 * u**2 * s * w * t**2
    size_sel = np.moveaxis(size[:, :, elements], 2, 0)
    shape_sum = (h * size_sel).sum(axis=(1, 2))
    n_imp = float((shape_sum * weight).sum() * scale)
    z_imp = float((shape_sum * weight * mean_z).sum() * scale)
    z_eff = float((shape_sum * weight * mean_z2).sum() * scale)
    return n_imp, z_imp, z_eff


def projection_total(projection_path: Path, channel: int) -> float:
    with h5py.File(projection_path, "r") as handle:
        values = np.asarray(handle["values"][channel], dtype=np.float64)
    return float(values.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()

    run_dir = Path(args.data_root) / args.run
    states = _index_map(run_dir, r"jorek(\d+)\.h5")
    parts = _index_map(run_dir, r"part(\d+)\.h5")
    projections = _index_map(run_dir, r"projections(\d+)\.h5")
    indices = sorted(set(states) & set(parts) & set(projections))
    if not indices:
        raise SystemExit("no aligned frames")

    reconstructed: list[np.ndarray] = []
    recorded: list[np.ndarray] = []
    tsteps: list[float] = []
    for index in indices:
        with h5py.File(states[index], "r") as handle:
            size = np.asarray(handle["size"][()], dtype=np.float64)
            tstep = float(np.ravel(handle["tstep"][()])[0])
        n_imp, z_imp, z_eff = reconstruct_totals(parts[index], size, stride=args.stride)
        reconstructed.append(np.array([n_imp, z_imp, z_eff]) * tstep)
        recorded.append(
            np.array(
                [
                    projection_total(projections[index], 4),
                    projection_total(projections[index], 3),
                    projection_total(projections[index], 2),
                ]
            )
        )
        tsteps.append(tstep)

    reconstructed = np.stack(reconstructed)
    recorded = np.stack(recorded)

    def _report(label: str, predicted: np.ndarray) -> None:
        for column, name in enumerate(["N_imp", "Z_imp", "Z_eff"]):
            m = predicted[:, column]
            p = recorded[:, column]
            valid = np.isfinite(m) & np.isfinite(p) & (np.abs(m) > 0) & (np.abs(p) > 0)
            if valid.sum() < 3:
                continue
            corr = float(np.corrcoef(m[valid], p[valid])[0, 1])
            ratio = p[valid] / m[valid]
            print(
                f"{label:9s} {name:6s} n={int(valid.sum()):3d} corr={corr:8.5f} "
                f"ratio_med={float(np.median(ratio)):11.4e} "
                f"rel_iqr={float(np.percentile(ratio, 75) - np.percentile(ratio, 25)) / abs(float(np.median(ratio))):8.4f}"
            )

    print(f"run={args.run} frames={len(indices)} stride={args.stride}")
    _report("same", reconstructed)
    shifted = np.vstack([np.full((1, 3), np.nan), reconstructed[:-1]])
    _report("stagger", shifted)
    trapezoid = 0.5 * (reconstructed + shifted)
    _report("trapez", trapezoid)


if __name__ == "__main__":
    main()
