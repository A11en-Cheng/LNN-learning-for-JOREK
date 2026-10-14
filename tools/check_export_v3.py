#!/usr/bin/env python
"""Phase A' acceptance check for one fine-grained run (export_v3 or native).

Two layouts are auto-detected from ``--run-dir``:

* ``export_v3``: ``<run>/export_v3/{manifest.json, steps/step_*.h5}`` (frozen
  per-fluid-step export spec);
* ``native``: a raw stepwise run directory with ``jorek*/projections*/part*``
  files written one file per fluid step (``src/data/native_run.py``), where the
  interval metadata is derived from the recorded ``t_now`` values.

The same checks run for both layouts; the human summary and the JSON report
carry the layout-specific provenance (interval source, single-step method,
restart register, zero-dt policy).

Acceptance: the fail-fast contract checks (timing monotonicity, ADR-0019
pairing, single-step gate, shard presence/shape, boundary-code registry,
condition groups) must all pass for the export to be usable by Phase B/C.
Registered warnings (missing boundary diagnostics, missing condition, gaps)
do not fail the run; they are reported and counted.  Budget values are
registered only -- no performance gate.

Usage::

    .torch-cuda-env/bin/python tools/check_export_v3.py \
        --run-dir data/EHL-2/scan_1d22_ang12_massloss_neon \
        --raw-run-dir data/EHL-2/scan_1d22_ang12_massloss_neon \
        --out .scratch/n/export_v3_check.json

    # Native stepwise run (raw files, no export_v3):
    .torch-cuda-env/bin/python tools/check_export_v3.py \
        --run-dir /mnt/d/EHL-2/phaseA_dataset_..._injInner \
        --zero-dt-policy list --out .scratch/n/native_check.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import budget as budget_mod  # noqa: E402
from src.data.native_run import parse_index_sequence  # noqa: E402
from src.data.output_contract import ContractError  # noqa: E402
from src.data.step_dataset import (  # noqa: E402
    REGISTERED_BOUNDARY_CODES,
    StepRunDataset,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="run directory containing export_v3/ (or the export_v3 directory itself)",
    )
    parser.add_argument(
        "--raw-run-dir",
        type=Path,
        default=None,
        help="recorded run (jorek*/projections*/part*) for the ADR-0019 cross-check",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs=2,
        metavar=("FIRST", "LAST"),
        default=None,
        help="inclusive step-index window (default: all steps)",
    )
    parser.add_argument(
        "--allow-multi-step",
        action="store_true",
        help="disable require_single_step (diagnostics only; Phase B/C must NOT use it)",
    )
    parser.add_argument(
        "--single-step-tol",
        type=float,
        default=None,
        help="override the dt_step==dt_source tolerance (JOREK units)",
    )
    parser.add_argument("--policy", choices=("drop", "error"), default="drop")
    parser.add_argument(
        "--max-linkage-checks",
        type=int,
        default=200,
        help="max number of u_after==next u_before linkage checks (spread over the run)",
    )
    parser.add_argument("--no-budget", action="store_true", help="skip the budget summary")
    parser.add_argument(
        "--budget-step",
        type=int,
        default=None,
        help="raw jorek step index used for the budget series (default: last)",
    )
    parser.add_argument(
        "--zero-dt-policy",
        choices=("registered", "error", "list"),
        default="registered",
        help=(
            "code-restart zero-step frames: 'registered' (default; every dt=0 "
            "frame must be listed in the restart-event registry -- the "
            "cross-boundary pair and lagging projection are skipped), 'error' "
            "(fail-fast for newly generated data) or 'list' (legacy diagnostics "
            "only -- enumerate dt=0 frames and drop the zero-length records)"
        ),
    )
    parser.add_argument(
        "--restart-file",
        type=Path,
        default=None,
        help=(
            "restart checkpoint registry (default: <run>/available_restart_files.txt "
            "for native runs)"
        ),
    )
    parser.add_argument(
        "--restart-indices",
        type=str,
        default=None,
        help="explicit restart registry override, e.g. '2704,5681' (native runs)",
    )
    parser.add_argument(
        "--restart-events",
        type=str,
        default=None,
        help=(
            "registered zero-dt restart events, e.g. '2702,4111' (native runs; "
            "default registry: <run>/restart_events.json, then "
            "data/restart_events.json)"
        ),
    )
    parser.add_argument(
        "--gap-indices",
        type=str,
        default=None,
        help=(
            "registered index gaps for a deduped run, e.g. '2704' (native runs); "
            "only registered frames may be skipped"
        ),
    )
    parser.add_argument(
        "--namelist",
        type=Path,
        default=None,
        help="production namelist for the native condition channel (default: "
        "jorek_namelist_spitest, then jorek_namelist)",
    )
    parser.add_argument(
        "--mesh-verify-stride",
        type=int,
        default=128,
        help="native static-mesh hash verification stride in frames (>=1; 1 = every frame)",
    )
    parser.add_argument("--out", type=Path, default=None, help="write the JSON report here")
    parser.add_argument("--json", action="store_true", help="print the JSON report to stdout")
    parser.add_argument("--quiet", action="store_true", help="suppress the human summary")
    return parser.parse_args(argv)


def _quantiles(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "max": float(np.max(array)),
    }


def _sample_positions(length: int, maximum: int) -> list[int]:
    if length <= 0:
        return []
    if length <= maximum:
        return list(range(length))
    return sorted({int(round(i * (length - 1) / (maximum - 1))) for i in range(maximum)})


def _file_stats(paths: Iterable[Path]) -> dict[str, Any]:
    sizes = [path.stat().st_size for path in paths]
    if not sizes:
        return {"n_files": 0, "total_bytes": 0}
    return {
        "n_files": len(sizes),
        "total_bytes": int(sum(sizes)),
        "median_bytes": float(statistics.median(sizes)),
        "max_bytes": int(max(sizes)),
        "max_file": str(
            max(zip(paths, sizes), key=lambda pair: pair[1])[0]
        ),
    }


def _shard_stats(metas) -> dict[str, Any]:
    starts = [meta.n_spi_start for meta in metas]
    ends = [meta.n_spi_end for meta in metas]
    histogram: dict[str, int] = {}
    for count in starts + ends:
        key = str(count)
        histogram[key] = histogram.get(key, 0) + 1
    return {
        "n_steps": len(metas),
        "n_spi_start": _quantiles(starts),
        "n_spi_end": _quantiles(ends),
        "n_zero_start": int(sum(1 for count in starts if count == 0)),
        "n_zero_end": int(sum(1 for count in ends if count == 0)),
        "histogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
    }


def _boundary_stats(dataset: StepRunDataset, max_reads: int = 200) -> dict[str, Any]:
    positions = _sample_positions(len(dataset), max_reads)
    counts: dict[str, int] = {}
    missing = 0
    unregistered: set[int] = set()
    n_nodes: set[int] = set()
    for position in positions:
        codes = dataset.read_boundary(position)
        if codes is None:
            missing += 1
            continue
        n_nodes.add(int(codes.size))
        for code, count in zip(*np.unique(codes, return_counts=True)):
            key = str(int(code))
            counts[key] = counts.get(key, 0) + int(count)
            if int(code) not in REGISTERED_BOUNDARY_CODES:
                unregistered.add(int(code))
    return {
        "available": missing < len(positions),
        "steps_sampled": len(positions),
        "missing_steps": missing,
        "code_counts": dict(sorted(counts.items(), key=lambda item: int(item[0]))),
        "unregistered_codes": sorted(unregistered),
        "node_counts": sorted(n_nodes),
        "registry": sorted(REGISTERED_BOUNDARY_CODES),
    }


def _condition_stats(dataset: StepRunDataset, max_reads: int = 50) -> dict[str, Any]:
    positions = _sample_positions(len(dataset), max_reads)
    with_condition = 0
    groups: dict[str, int] = {}
    n_entries: set[int] = set()
    film_eligible: set[int] = set()
    for position in positions:
        condition = dataset.read_condition(position)
        if condition is None:
            continue
        with_condition += 1
        n_entries.add(len(condition.names))
        film_eligible.add(int(np.sum(condition.film_eligible)))
        for group, count in condition.group_counts.items():
            groups[group] = groups.get(group, 0) + count
    return {
        "available": with_condition > 0,
        "steps_sampled": len(positions),
        "steps_with_condition": with_condition,
        "n_entries": sorted(n_entries),
        "film_eligible_counts": sorted(film_eligible),
        "group_counts": dict(sorted(groups.items())),
    }


def _issue_table(dataset: StepRunDataset) -> dict[str, Any]:
    kinds: dict[str, list[int]] = {}
    for issue in dataset.issues:
        kinds.setdefault(issue.kind, []).append(issue.index)
    return {
        kind: {
            "n": len(indices),
            "indices": sorted(set(indices))[:50],
            "detail": next(
                issue.detail for issue in dataset.issues if issue.kind == kind
            ),
        }
        for kind, indices in kinds.items()
    }


def _restart_section(dataset: StepRunDataset) -> dict[str, Any]:
    """Restart registration + per-checkpoint checks (both layouts)."""
    report = getattr(dataset, "restart_report", None)
    if report:
        return report
    rep = dataset.report
    return {
        "registry": {
            "n": rep.restart_registry_size,
            "indices": list(rep.restart_events),
            "source": "export manifest restart_indices / explicit",
        },
        "events": [],
        "checks": [],
        "policy": {
            "zero_dt": rep.zero_dt_policy,
            "restart_boundary": dataset.policy,
            "require_single_step": rep.require_single_step,
        },
        "spec": (
            "export_v3 registers restart indices in the manifest; registered offsets are "
            "handled by policy=drop/error, new exports must carry explicit intervals"
        ),
    }


def _linkage(dataset: StepRunDataset, maximum: int) -> dict[str, Any]:
    if maximum <= 0:
        return {"checked": 0, "mismatches": [], "skipped": "max-linkage-checks=0"}
    consecutive = [
        position
        for position in range(len(dataset) - 1)
        if dataset.metas[position + 1].index == dataset.metas[position].index + 1
    ]
    positions = _sample_positions(len(consecutive), maximum)
    positions = [consecutive[position] for position in positions]
    mismatches: list[dict[str, Any]] = []
    for position in positions:
        try:
            dataset.check_linkage(position)
        except ContractError as error:
            mismatches.append({"position": position, "error": str(error)})
    return {
        "checked": len(positions),
        "mismatches": mismatches,
        "skipped_gaps": len(dataset) - 1 - len(consecutive),
        "criterion": "u_after[k] == u_before[k+1] (rtol=1e-9, atol=1e-12)",
    }


def _alignment_stats(args: argparse.Namespace, dataset: StepRunDataset) -> dict[str, Any]:
    if getattr(dataset.report, "layout_kind", "export_v3") == "native":
        restart = getattr(dataset, "restart_report", {}) or {}
        part_checked = [
            issue for issue in dataset.issues if issue.kind == "part_time_checked"
        ]
        return {
            "checked": True,
            "mode": "native_inline",
            "note": (
                "native runs are read from the recorded files directly: every record's "
                "jorek/projection alignment and every part frame time are verified "
                "during the index pass (no separate --raw-run-dir)"
            ),
            "n_state_frames": dataset.manifest.get("n_state_frames"),
            "n_projection_frames": dataset.manifest.get("n_projection_frames"),
            "n_part_frames": dataset.manifest.get("n_part_frames"),
            "part_check": part_checked[0].detail if part_checked else None,
            "restart_events": list(dataset.report.restart_events),
            "restart_registry": restart.get("registry"),
            "dropped_records": list(dataset.dropped_restart)
            + list(getattr(dataset, "dropped_zero_dt", ())),
        }
    out: dict[str, Any] = {
        "raw_run_dir": str(args.raw_run_dir) if args.raw_run_dir else None,
        "checked": False,
    }
    if args.raw_run_dir is None:
        return out
    raw = Path(args.raw_run_dir)
    n_jorek = n_projection = n_part = 0
    for meta in dataset.metas:
        index = meta.index
        n_jorek += int((raw / f"jorek{index:06d}.h5").exists())
        n_projection += int((raw / f"projections{index:06d}.h5").exists())
        n_part += int((raw / f"part{index:05d}.h5").exists())
    out.update(
        {
            "checked": True,
            "n_steps": len(dataset),
            "jorek_frames_found": n_jorek,
            "projection_frames_found": n_projection,
            "part_frames_found": n_part,
            "restart_registered": sorted(
                {issue.index for issue in dataset.issues if "restart" in issue.kind}
            ),
            "restart_dropped": list(dataset.dropped_restart),
            "mismatches": [],
            "note": (
                "time mismatches raise ContractError (exit 1) instead of being "
                "collected; projection/part counts are the frames cross-checked"
            ),
        }
    )
    return out


def _budget_summary(args: argparse.Namespace, dataset: StepRunDataset) -> dict[str, Any]:
    if args.no_budget:
        return {"available": False, "reason": "disabled by --no-budget"}
    if getattr(dataset.report, "layout_kind", "export_v3") == "native":
        summary = budget_mod.summarize_budget(
            dataset.export_root, step_index=args.budget_step
        )
        summary["available"] = bool(
            (summary.get("ablation") or {}).get("source")
            or (summary.get("ionization_radiation") or {}).get("source")
            or (summary.get("global_closure") or {}).get("available")
        )
        summary["source"] = "native_run_dir"
        if args.budget_step is not None:
            summary["step_budget"] = budget_mod.native_step_budget(
                dataset.export_root, args.budget_step
            )
        return summary
    if args.raw_run_dir is not None:
        summary = budget_mod.summarize_budget(args.raw_run_dir, step_index=args.budget_step)
        summary["available"] = True
        summary["source"] = "raw_run_dir"
        return summary
    # Fall back to series embedded in the export step files, if any.
    series = budget_mod.read_series(dataset.metas[0].path) if dataset.metas else {}
    if not series:
        return {
            "available": False,
            "reason": (
                "no --raw-run-dir and the export step files carry no xtime_* series; "
                "budget needs the raw jorek*.h5 history or an export budget block"
            ),
        }
    summaries = budget_mod.series_summaries(series)
    return {
        "available": True,
        "source": "export_step_file",
        "file": str(dataset.metas[0].path),
        "layout": budget_mod.detect_series_layout(series),
        "series": {name: summary.to_dict() for name, summary in summaries.items()},
        "global_closure": {"available": False, "reason": "no macroscopic_vars.dat in export"},
        "impurity_particle_budget": {
            "status": "needs rho_imp FEM volume integral + boundary particle flux",
            "owner": "Phase B1",
        },
        "gate": "no performance gate",
    }


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    require_single_step = not args.allow_multi_step
    restart_indices = (
        parse_index_sequence(args.restart_indices) if args.restart_indices else None
    )
    restart_events = (
        parse_index_sequence(args.restart_events) if args.restart_events else None
    )
    gap_indices = parse_index_sequence(args.gap_indices) if args.gap_indices else None
    try:
        dataset = StepRunDataset(
            args.run_dir,
            raw_run_dir=args.raw_run_dir,
            restart_indices=restart_indices,
            restart_events=restart_events,
            restart_file=args.restart_file,
            gap_indices=gap_indices,
            policy=args.policy,
            require_single_step=require_single_step,
            single_step_tol=args.single_step_tol,
            index_window=tuple(args.indices) if args.indices else None,
            zero_dt_policy=args.zero_dt_policy,
            namelist_file=args.namelist,
            mesh_verify_stride=args.mesh_verify_stride,
        )
    except ContractError as error:
        return {
            "ok": False,
            "run_dir": str(args.run_dir),
            "error": str(error),
            "require_single_step": require_single_step,
            "zero_dt_policy": args.zero_dt_policy,
        }, 1
    report = dataset.report
    step_paths = sorted(
        {
            path
            for meta in dataset.metas
            for path in (meta.path, meta.path_source, meta.path_after)
            if path is not None
        }
    )
    report_dict: dict[str, Any] = {
        "ok": True,
        "run_dir": str(args.run_dir),
        "export_root": str(dataset.export_root),
        "run": report.run,
        "layout_kind": report.layout_kind,
        "manifest": {
            key: dataset.manifest.get(key)
            for key in (
                "layout",
                "layout_kind",
                "cadence",
                "source_object",
                "budget_layout",
                "n_nodes",
                "jorek_version",
                "model_id",
                "interval_source",
                "mesh_verify_stride",
            )
        },
        "provenance": dataset.manifest.get("provenance"),
        "contract": {
            "n_steps": report.n_steps,
            "index_first": report.indices[0] if report.indices else None,
            "index_last": report.indices[-1] if report.indices else None,
            "n_windows": report.n_windows
            if report.n_windows is not None
            else len(report.pair_indices),
            "dt_step": {
                "min": report.dt_step_min,
                "median": report.dt_step_median,
                "max": report.dt_step_max,
            },
            "dt_source": {
                "min": report.dt_source_min,
                "median": report.dt_source_median,
                "max": report.dt_source_max,
            },
            "require_single_step": report.require_single_step,
            "single_step_tol": report.single_step_tol,
            "single_step_method": report.single_step_method,
            "step_table_source": report.step_table_source,
            "interval_source": report.interval_source,
            "dropped_restart": list(report.dropped_restart),
            "gaps": list(report.gaps),
            "issues": _issue_table(dataset),
            "zero_dt": {
                "policy": report.zero_dt_policy,
                "indices": list(report.zero_dt_indices),
                "affected_pairs": list(report.zero_dt_pairs),
                "note": (
                    "zero_dt policy='list' is legacy diagnostics only; Phase B/C "
                    "requires regenerated data with dedup at every restart"
                )
                if report.zero_dt_policy == "list"
                else (
                    "registered restart events accepted: cross-boundary pair and "
                    "lagging projection skipped; unregistered dt=0 remains fail-fast"
                    if report.zero_dt_policy == "registered"
                    else "no zero-dt frames (hard rule enforced)",
                ),
            },
            "restart": {
                "registered_events": list(report.restart_events_registered),
                "registered_events_source": report.restart_events_source,
                "cross_boundary_skipped": list(report.restart_cross_boundary_skipped),
                "lagging_projection_skipped": list(
                    report.restart_lagging_projection_skipped
                ),
                "degenerate_source_windows": list(
                    report.restart_degenerate_source_windows
                ),
            },
            "shard_time_semantics": report.shard_time_semantics,
            "boundary_frames": {
                "leading": list(report.boundary_frames_leading),
                "trailing": list(report.boundary_frames_trailing),
            },
            "mesh_verifications": report.mesh_verifications,
            "n_frame_files": report.n_frame_files,
        },
        "alignment": _alignment_stats(args, dataset),
        "restart": _restart_section(dataset),
        "shards": _shard_stats(dataset.metas),
        "boundary": _boundary_stats(dataset),
        "condition": _condition_stats(dataset),
        "files": _file_stats(step_paths),
        "linkage": _linkage(dataset, args.max_linkage_checks),
        "budget": _budget_summary(args, dataset),
    }
    if report_dict["linkage"]["mismatches"]:
        report_dict["ok"] = False
        report_dict["error"] = (
            f"{len(report_dict['linkage']['mismatches'])} linkage mismatches "
            "(u_after != next u_before)"
        )
        return report_dict, 1
    return report_dict, 0


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def print_summary(report: dict[str, Any]) -> None:
    if not report.get("ok", False):
        print(f"FAIL: {report.get('error', 'unknown error')}")
        return
    contract = report["contract"]
    shards = report["shards"]
    boundary = report["boundary"]
    condition = report["condition"]
    print(
        f"run            : {report['run']} ({report['export_root']}) "
        f"[layout={report.get('layout_kind', 'export_v3')}]"
    )
    print(
        f"steps          : {contract['n_steps']} "
        f"[{contract['index_first']}..{contract['index_last']}] "
        f"windows={contract['n_windows']}"
    )
    print(
        f"dt_step        : {_fmt(contract['dt_step']['min'])} / "
        f"{_fmt(contract['dt_step']['median'])} / {_fmt(contract['dt_step']['max'])} "
        f"| dt_source: {_fmt(contract['dt_source']['min'])} / "
        f"{_fmt(contract['dt_source']['median'])} / {_fmt(contract['dt_source']['max'])}"
    )
    print(
        f"single-step    : require={contract['require_single_step']} "
        f"tol={_fmt(contract['single_step_tol'])} "
        f"method={contract.get('single_step_method')}"
    )
    zero_dt = contract.get("zero_dt", {})
    print(
        f"zero-dt        : policy={zero_dt.get('policy')} "
        f"indices={zero_dt.get('indices')} pairs={zero_dt.get('affected_pairs')}"
    )
    restart = report.get("restart", {})
    registry = restart.get("registry", {})
    print(
        f"restart        : registry={registry.get('n')} "
        f"events={[event.get('index') for event in restart.get('events', [])]} "
        f"checked={restart.get('n_registry_checked', 0)}"
    )
    registered = contract.get("restart", {}).get("registered_events")
    if registered:
        print(
            f"restart events : registered={registered} "
            f"cross-boundary skipped="
            f"{contract['restart'].get('cross_boundary_skipped', [])} "
            f"lagging projection skipped="
            f"{contract['restart'].get('lagging_projection_skipped', [])}"
        )
    print(f"restart dropped: {contract['dropped_restart']} | gaps {contract['gaps']}")
    print(f"issues         : {sorted(contract['issues'])}")
    print(
        f"shards         : start {_fmt(shards['n_spi_start']['min'])}/"
        f"{_fmt(shards['n_spi_start']['median'])}/{_fmt(shards['n_spi_start']['max'])} "
        f"| end {_fmt(shards['n_spi_end']['min'])}/"
        f"{_fmt(shards['n_spi_end']['median'])}/{_fmt(shards['n_spi_end']['max'])}"
    )
    print(
        f"boundary       : available={boundary['available']} "
        f"codes={sorted(int(code) for code in boundary['code_counts'])} "
        f"missing_steps={boundary['missing_steps']}"
    )
    print(
        f"condition      : available={condition['available']} "
        f"n_entries={condition['n_entries']} groups={sorted(condition['group_counts'])}"
    )
    files = report["files"]
    print(
        f"files          : n={files['n_files']} total={files['total_bytes'] / 1e6:.2f} MB "
        f"median={files.get('median_bytes', 0) / 1e6:.3f} MB"
    )
    print(f"linkage        : checked={report['linkage']['checked']}")
    budget = report["budget"]
    if budget.get("available"):
        ablation = budget.get("ablation", {})
        ionization = budget.get("ionization_radiation", {})
        closure = budget.get("global_closure", {})
        print(
            f"budget         : atoms={_fmt(ablation.get('total_atoms'))} "
            f"| layout={ionization.get('layout')} "
            f"| energy residual={_fmt(closure.get('energy_conservation', {}).get('residual', {}).get('last'))}"
        )
    else:
        print(f"budget         : not available ({budget.get('reason')})")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.run_dir.exists():
        print(f"error: run directory {args.run_dir} does not exist", file=sys.stderr)
        return 2
    report, code = build_report(args)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    if not args.quiet:
        print_summary(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
