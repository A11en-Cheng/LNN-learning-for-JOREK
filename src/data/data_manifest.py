"""EHL-2 data discovery, frame alignment and the train/val/test manifest.

The naming contract pinned here is the one observed in ``data/EHL-2``:

* state frames are ``jorek<index>.h5``,
* marker frames are ``part<index>.h5``,
* a frame is a valid training frame only when the same integer index exists in
  both families,
* special names (``jorek_restart.h5``, ``jorek_grid.h5``, ``part_restart.h5``,
  ...) are never frames.

Only ``ntor=1, n_dim=2`` runs are training data for Phase 1 (ADR-0004); the
manifest excludes anything else.  Statistics are population statistics computed
on the training split only, after the schema transform (log1p for the
positive-only channels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from src.data.model502_schema import (
    FIELD_NAMES,
    H5_VALUES_INDICES,
    LOG1P_FIELD_INDICES,
    MARKER_PROJECTION_FIELD_NAMES,
    SCHEMA_VERSION,
)
from src.data.parameter_channel import build_condition_vector


STATE_FRAME_RE = re.compile(r"^jorek(?P<index>\d+)\.h5$")
MARKER_FRAME_RE = re.compile(r"^part(?P<index>\d+)\.h5$")
PROJECTION_FRAME_RE = re.compile(r"^projections(?P<index>\d+)\.h5$")


@dataclass(frozen=True)
class RunDir:
    path: Path

    @property
    def name(self) -> str:
        return self.path.name


@dataclass(frozen=True)
class FrameRef:
    index: int
    state_path: Path
    marker_path: Path


def discover_runs(data_root: str | Path) -> list[RunDir]:
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"data root is not a directory: {root}")
    return [RunDir(path) for path in sorted(root.iterdir()) if path.is_dir()]


def _numeric_frames(run_dir: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    frames: dict[int, Path] = {}
    for path in run_dir.iterdir():
        if not path.is_file() or path.suffix != ".h5":
            continue
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        frames[int(match.group("index"))] = path
    return frames


def aligned_frames(run_dir: str | Path) -> list[FrameRef]:
    run_dir = Path(run_dir)
    states = _numeric_frames(run_dir, STATE_FRAME_RE)
    markers = _numeric_frames(run_dir, MARKER_FRAME_RE)
    common = sorted(set(states) & set(markers))
    return [FrameRef(index, states[index], markers[index]) for index in common]


def _frame_metadata(state_path: Path) -> dict[str, int]:
    with h5py.File(state_path, "r") as handle:
        def scalar(key: str) -> int:
            if key not in handle:
                raise KeyError(f"{state_path.name} is missing '{key}' metadata")
            return int(handle[key][()].item())

        return {
            "n_tor": scalar("n_tor"),
            "n_dim": scalar("n_dim"),
            "n_var": scalar("n_var"),
            "model_id": scalar("jorek_model"),
            "nodes": int(handle["boundary"].shape[0]),
        }


def is_2d_run(run_dir: str | Path) -> bool:
    frames = aligned_frames(run_dir)
    if not frames:
        return False
    metadata = _frame_metadata(frames[0].state_path)
    return metadata["n_tor"] == 1 and metadata["n_dim"] == 2


def grid_fingerprint(state_path: str | Path) -> str:
    """Hash the coordinate array so two runs can be compared for mesh identity.

    The contract pinned here is the same as the parser seam: ``x`` must have the
    JOREK Bezier layout ``[2, 4, harmonics, nodes]`` with ``harmonics == 1`` for
    Phase 1.  A digest covers both the shape and every coordinate byte, so two
    meshes are identical only when the raw geometry is bit-for-bit equal.
    """
    state_path = Path(state_path)
    with h5py.File(state_path, "r") as handle:
        if "x" not in handle:
            raise KeyError(f"{state_path.name} is missing coordinate array 'x'")
        coords = handle["x"][()]
        if coords.ndim != 4 or coords.shape[0] != 2 or coords.shape[2] != 1:
            raise ValueError(
                f"{state_path.name}: expected 'x' shape [2, 4, 1, nodes], got {coords.shape}"
            )
        digest = hashlib.sha256()
        digest.update(str(coords.shape).encode())
        digest.update(np.ascontiguousarray(coords).tobytes())
        return digest.hexdigest()


def _eligible_2d_runs(data_root: Path) -> list[tuple[RunDir, list[FrameRef], dict[str, int]]]:
    """Select ADR-0004 eligible runs with their aligned frames and metadata."""
    eligible = []
    for run in discover_runs(data_root):
        frames = aligned_frames(run.path)
        if not frames:
            continue
        metadata = _frame_metadata(frames[0].state_path)
        if metadata["n_tor"] == 1 and metadata["n_dim"] == 2:
            eligible.append((run, frames, metadata))
    return eligible


def mesh_summary(data_root: str | Path) -> dict:
    """Report how many distinct grids the 2D-eligible runs sit on, cheaply.

    Only the first valid frame of each run is read, so this stays fast enough to
    call in tests.  Runs excluded by ADR-0004 (multi-harmonic, 3D) never appear.
    """
    grids: dict[str, int] = {}
    runs: list[dict] = []
    for run, frames, _ in _eligible_2d_runs(Path(data_root)):
        fingerprint = grid_fingerprint(frames[0].state_path)
        grid_id = grids.setdefault(fingerprint, len(grids))
        runs.append({"name": run.name, "grid_id": grid_id})
    return {"consistent": len(grids) == 1, "n_grids": len(grids), "runs": runs}


def _canonical_transformed_values(state_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(canonical [nodes, 9] float64, negative_count [9] int64)``."""
    with h5py.File(state_path, "r") as handle:
        raw = handle["values"]
        if raw.ndim != 4:
            raise ValueError(
                f"{state_path.name}: expected values shape [n_var, 4, harmonics, nodes], "
                f"got {raw.shape}"
            )
        if raw.shape[0] < 9:
            raise ValueError(f"{state_path.name}: expected at least 9 channels, got {raw.shape[0]}")
        values = raw[:, 0, 0, :].astype(np.float64)  # node DoF 0, harmonic 0
    canonical = values[list(H5_VALUES_INDICES), :].T
    negative_count = np.zeros(9, dtype=np.int64)
    for channel in LOG1P_FIELD_INDICES:
        negative_count[channel] = int((canonical[:, channel] < 0).sum())
        canonical[:, channel] = np.log1p(np.clip(canonical[:, channel], 0.0, None))
    return canonical, negative_count


def _dataset_id(data_root: Path) -> str:
    digest = hashlib.sha256()
    layout_files = sorted(
        list(data_root.rglob("jorek*.h5")) + list(data_root.rglob("part*.h5"))
    )
    for path in layout_files:
        digest.update(str(path.relative_to(data_root)).encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()


def _projection_path(state_path: Path) -> Path | None:
    match = STATE_FRAME_RE.fullmatch(state_path.name)
    if match is None:
        return None
    index = int(match.group("index"))
    for name in sorted(os.listdir(state_path.parent)):
        projection_match = PROJECTION_FRAME_RE.fullmatch(name)
        if projection_match is not None and int(projection_match.group("index")) == index:
            return state_path.parent / name
    return None


def _split_names(names: list[str], seed: int, val_frac: float, test_frac: float) -> dict[str, str]:
    if not 0 <= val_frac <= 1 or not 0 <= test_frac <= 1 or val_frac + test_frac >= 1:
        raise ValueError("val_frac and test_frac must satisfy 0 <= val+test < 1")
    total = len(names)
    n_val = int(round(total * val_frac))
    n_test = int(round(total * test_frac))
    if n_val + n_test >= total and total > 0:
        n_test = max(0, total - n_val - 1)
    if total - n_val - n_test < 1:
        raise ValueError("split leaves no training run")
    shuffled = list(names)
    random.Random(seed).shuffle(shuffled)
    assignment = {}
    cursor = 0
    for split, count in (("val", n_val), ("test", n_test)):
        for name in shuffled[cursor : cursor + count]:
            assignment[name] = split
        cursor += count
    for name in shuffled[cursor:]:
        assignment[name] = "train"
    return assignment


def build_manifest(
    data_root: str | Path,
    *,
    seed: int = 0,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> dict:
    data_root = Path(data_root)
    runs: list[dict] = []
    eligible_names: list[str] = []
    mesh = mesh_summary(data_root)
    grid_id_by_name = {entry["name"]: entry["grid_id"] for entry in mesh["runs"]}
    for run, frames, metadata in _eligible_2d_runs(data_root):
        eligible_names.append(run.name)
        runs.append(
            {
                "name": run.name,
                "split": None,
                "n_tor": metadata["n_tor"],
                "n_dim": metadata["n_dim"],
                "n_var": metadata["n_var"],
                "model_id": metadata["model_id"],
                "nodes": metadata["nodes"],
                "grid_id": grid_id_by_name[run.name],
                "first_index": frames[0].index,
                "last_index": frames[-1].index,
                "n_frames": len(frames),
            }
        )

    assignment = _split_names(eligible_names, seed, val_frac, test_frac)
    for run in runs:
        run["split"] = assignment[run["name"]]

    stats = _train_split_stats(data_root, runs)
    return {
        "schema": SCHEMA_VERSION,
        "dataset_id": _dataset_id(data_root),
        "seed": seed,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "mesh": {
            "consistent": mesh["consistent"],
            "n_grids": mesh["n_grids"],
        },
        "runs": sorted(runs, key=lambda run: run["name"]),
        "stats": stats,
    }


def _train_split_stats(data_root: Path, runs: list[dict]) -> dict:
    means = np.zeros(9, dtype=np.float64)
    sumsq = np.zeros(9, dtype=np.float64)
    counts = 0
    negative = np.zeros(9, dtype=np.int64)
    condition_means = None
    condition_sumsq = None
    condition_names: list[str] | None = None
    frame_count = 0
    source_means = np.zeros(len(MARKER_PROJECTION_FIELD_NAMES), dtype=np.float64)
    source_sumsq = np.zeros(len(MARKER_PROJECTION_FIELD_NAMES), dtype=np.float64)
    source_nodes = 0
    source_frames = 0
    namelist_cache: dict[str, str | None] = {}
    for run in runs:
        if run["split"] != "train":
            continue
        run_dir = data_root / run["name"]
        if run["name"] not in namelist_cache:
            namelist_text = None
            for candidate in ("jorek_namelist_spitest", "jorek_namelist"):
                candidate_path = run_dir / candidate
                if candidate_path.exists():
                    namelist_text = candidate_path.read_text(errors="replace")
                    break
            namelist_cache[run["name"]] = namelist_text
        namelist_text = namelist_cache[run["name"]]
        for frame in aligned_frames(data_root / run["name"]):
            values, negative_count = _canonical_transformed_values(frame.state_path)
            counts += values.shape[0]
            means += values.sum(axis=0)
            sumsq += np.square(values).sum(axis=0)
            negative += negative_count
            with h5py.File(frame.state_path, "r") as handle:
                condition = build_condition_vector(handle, namelist_text=namelist_text)
            vector = condition["vector"]
            if condition_means is None:
                condition_names = list(condition["names"])
                condition_means = np.zeros(len(vector), dtype=np.float64)
                condition_sumsq = np.zeros(len(vector), dtype=np.float64)
            condition_means += vector
            condition_sumsq += np.square(vector)
            frame_count += 1
            projection_path = _projection_path(frame.state_path)
            if projection_path is not None:
                with h5py.File(projection_path, "r") as projection_handle:
                    projection_values = projection_handle["values"][
                        : len(MARKER_PROJECTION_FIELD_NAMES), 0, 0, :
                    ].astype(np.float64)
                source_means += projection_values.sum(axis=1)
                source_sumsq += np.square(projection_values).sum(axis=1)
                source_nodes += projection_values.shape[1]
                source_frames += 1
    if counts == 0:
        raise ValueError("no training frames found")
    means /= counts
    variance = sumsq / counts - np.square(means)
    variance = np.clip(variance, 0.0, None)
    fields = {}
    for index, name in enumerate(FIELD_NAMES):
        fields[name] = {
            "mean": float(means[index]),
            "std": float(np.sqrt(variance[index])),
            "n_negative": int(negative[index]),
        }
    condition_mean = condition_means / frame_count if condition_means is not None else np.zeros(0)
    condition_variance = (
        condition_sumsq / frame_count - np.square(condition_mean)
        if condition_sumsq is not None
        else np.zeros(0)
    )
    return {
        "schema": SCHEMA_VERSION,
        "n_samples": counts,
        "fields": fields,
        "condition": {
            "names": condition_names or [],
            "mean": condition_mean.tolist(),
            "std": np.sqrt(np.clip(condition_variance, 0.0, None)).tolist(),
        },
        "source_stats": {
            "names": list(MARKER_PROJECTION_FIELD_NAMES),
            "mean": (source_means / source_nodes if source_nodes else source_means).tolist(),
            "std": np.sqrt(
                np.clip(
                    source_sumsq / source_nodes - np.square(source_means / source_nodes),
                    0.0,
                    None,
                )
                if source_nodes
                else np.zeros_like(source_means)
            ).tolist(),
            "n_frames": source_frames,
            "n_samples": source_nodes,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the EHL-2 data manifest.")
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--output", default=".scratch/model502-baseline/manifest.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    args = parser.parse_args()
    manifest = build_manifest(
        args.data_root, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote manifest for {len(manifest['runs'])} runs to {output}")


if __name__ == "__main__":
    main()
