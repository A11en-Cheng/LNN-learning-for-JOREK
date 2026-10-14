#!/usr/bin/env python
"""Rebuild EHL-2 pairs under ADR-0019 and register restart boundaries.

Phase A demonstration tool (roadmap_v2 A2/A3): reads one raw run directory,
builds the ``(u_k, p_{k-1}=part{k}, S_k=projections{k}) -> u_{k+1}`` pair index
with fail-fast timing assertions, and writes the index to a *new* output
directory (default ``.scratch/n/pairs_v3_demo/``).  Existing files are never
overwritten unless ``--force`` is given explicitly.

Artifacts written:

* ``pairs_index.jsonl``      one JSON record per valid pair;
* ``restart_boundaries.json`` registry + detected offsets + drop decisions;
* ``coverage.json``          contract report, time span, dt statistics.

Usage::

    .torch-cuda-env/bin/python tools/rebuild_ehl2_pairs_v3.py
    .torch-cuda-env/bin/python tools/rebuild_ehl2_pairs_v3.py \
        --run-dir data/EHL-2/scan_1d22_ang12_massloss_neon \
        --out-dir .scratch/n/pairs_v3_demo
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.output_contract import (  # noqa: E402
    ContractError,
    RESTART_BOUNDARY_INDICES,
    check_run_contract,
    collect_run_timing,
)

DEFAULT_RUN_DIR = Path("data/EHL-2/scan_1d22_ang12_massloss_neon")
DEFAULT_OUT_DIR = Path(".scratch/n/pairs_v3_demo")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--policy",
        choices=("drop", "error"),
        default="drop",
        help="registered restart boundaries: drop (default) or hard error",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite artifacts in an existing output directory",
    )
    return parser.parse_args(argv)


def _relative(path: Path, run_dir: Path) -> str:
    return str(path.relative_to(run_dir))


def _n_nodes(state_path: Path) -> int:
    with h5py.File(state_path, "r") as handle:
        return int(handle["boundary"].shape[0])


def _t_norm(state_path: Path) -> float:
    with h5py.File(state_path, "r") as handle:
        return float(np.ravel(handle["t_norm"][()])[0])


def build_artifacts(run_dir: Path, policy: str) -> dict[str, object]:
    timings = collect_run_timing(run_dir)
    report = check_run_contract(run_dir, policy=policy, timings=timings)
    timing_by_index = {timing.index: timing for timing in timings}

    pairs = []
    for record in report.pair_records:
        timing = timing_by_index[record.index]
        next_timing = timing_by_index[record.index_next]
        pairs.append(
            {
                "index": record.index,
                "index_next": record.index_next,
                "t_state_jorek": record.t_state,
                "t_next_jorek": record.t_next,
                "dt_step_jorek": record.dt_step,
                "dt_source_jorek": record.dt_source,
                "u_before": _relative(timing.state_path, run_dir),
                "u_after": _relative(next_timing.state_path, run_dir),
                "source": _relative(timing.projection_path, run_dir),
                "p_start": _relative(timing.particle_path, run_dir),
                "pairing": "u_k,[part{k}=p_{k-1}],projections{k}->u_k+1",
            }
        )

    dt_steps = [record.dt_step for record in report.pair_records]
    dt_sources = [record.dt_source for record in report.pair_records]
    first_state = timings[0].state_path
    coverage = {
        **report.to_dict(),
        "n_nodes": _n_nodes(first_state),
        "t_norm_seconds_per_jorek_unit": _t_norm(first_state),
        "t_first_jorek": timings[0].t_state,
        "t_last_jorek": timings[-1].t_state,
        "dt_step": {
            "min": min(dt_steps) if dt_steps else None,
            "median": statistics.median(dt_steps) if dt_steps else None,
            "max": max(dt_steps) if dt_steps else None,
        },
        "dt_source": {
            "min": min(dt_sources) if dt_sources else None,
            "median": statistics.median(dt_sources) if dt_sources else None,
            "max": max(dt_sources) if dt_sources else None,
        },
    }

    registry = list(RESTART_BOUNDARY_INDICES.get(run_dir.name, ()))
    detected = []
    for issue in report.issues:
        if issue.kind == "restart_boundary_projection_offset":
            timing = timing_by_index[issue.index]
            detected.append(
                {
                    "index": issue.index,
                    "t_state_jorek": timing.t_state,
                    "t_projection_jorek": timing.t_projection,
                    "deviation_jorek": None
                    if timing.t_projection is None
                    else timing.t_projection - timing.t_state,
                    "detail": issue.detail,
                }
            )
    restart = {
        "run": run_dir.name,
        "policy": policy,
        "registry": registry,
        "detected": detected,
        "dropped": list(report.dropped_restart),
        "rule": (
            "ADR-0019: projections{k} is written one fluid step early at restart "
            "checkpoints; part{k} still satisfies part{k}.time = T_{k-1}. "
            "Registered frames are dropped (or re-aligned only with explicit "
            "interval evidence)."
        ),
    }
    return {"pairs": pairs, "coverage": coverage, "restart": restart}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir: Path = args.run_dir
    out_dir: Path = args.out_dir
    if not run_dir.is_dir():
        print(f"error: run directory {run_dir} does not exist", file=sys.stderr)
        return 2
    targets = [
        out_dir / "pairs_index.jsonl",
        out_dir / "restart_boundaries.json",
        out_dir / "coverage.json",
    ]
    existing = [path for path in targets if path.exists()]
    if existing and not args.force:
        print(
            "error: refusing to overwrite existing artifacts "
            f"({', '.join(str(path) for path in existing)}); pass --force to override",
            file=sys.stderr,
        )
        return 2

    try:
        artifacts = build_artifacts(run_dir, args.policy)
    except ContractError as error:
        print(f"contract violation: {error}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    with targets[0].open("w", encoding="utf-8") as handle:
        for record in artifacts["pairs"]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    targets[1].write_text(
        json.dumps(artifacts["restart"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    targets[2].write_text(
        json.dumps(artifacts["coverage"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    coverage = artifacts["coverage"]
    print(f"run            : {coverage['run']}")
    print(f"state frames   : {coverage['n_state_frames']}")
    print(f"marker frames  : {coverage['n_marker_frames']}")
    print(f"valid pairs    : {coverage['n_pairs']}")
    print(f"dropped restart: {coverage['dropped_restart']}")
    print(f"issues         : {[issue['kind'] for issue in coverage['issues']]}")
    print(f"artifacts      : {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
