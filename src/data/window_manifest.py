"""Phase-1 window corpus for the interpolation-centred Model 502 route.

Implements the contract of ADR-0012:

* there is **one** window corpus; ``train``/``val`` are assignments on it;
* a run is cut into *segments* at (a) the SPI injection time ``t_ns``,
  (b) material changes of the zero-dimensional coefficient channel, and
  (c) gaps in the frame numbering;
* a *window* is ``burn_in`` input frames followed by ``horizon`` supervised
  intervals (``burn_in + horizon + 1`` frames in total); burn-in frames are
  inputs only and never receive a data loss;
* the inner split (model selection) is a blocked split inside each segment
  with a purge/embargo of at least one window span; the outer protocol is
  Leave-One-Run-Out (handled by :func:`loo_folds`).

Segments that are shorter than one window are dropped.  The manifest is the
single source for window generation; it stores run/segment/frame-index records
plus the blocked split assignment and the LOO fold table.
"""

from __future__ import annotations

import json
import math
import re
from fnmatch import fnmatch
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from src.data.data_manifest import aligned_frames, discover_runs, is_2d_run
from src.data.parameter_channel import build_condition_vector

MANIFEST_SCHEMA = "model502-interp-manifest-v1"
DEFAULT_BURN_IN = 30
DEFAULT_HORIZON = 1
DEFAULT_REL_THRESHOLD = 0.05
DEFAULT_VAL_FRACTION = 0.2

# Zero-dimensional coefficients that enter the field equations.  Names are
# compared by their stem so indexed entries ("D_perp[1]", ...) all match.
MATERIAL_PREFIXES: tuple[str, ...] = (
    "eta",
    "visco",
    "ZK_i_par",
    "ZK_e_par",
    "D_perp",
    "ZK_i_perp",
    "ZK_e_perp",
    "time_evol_theta",
    "time_evol_zeta",
    "F0",
    "R_domm",
)

_T_NS_RE = re.compile(r"^\s*t_ns\s*=\s*([0-9eEdD.+-]+)", re.MULTILINE)


@dataclass(frozen=True)
class FrameInfo:
    """Per-frame metadata needed for segmentation and window generation."""

    index: int
    position: int
    state_path: Path
    marker_path: Path
    projection_path: Path | None
    t_now: float
    n_spi_tot: int
    condition: np.ndarray
    condition_names: tuple[str, ...]


@dataclass(frozen=True)
class Segment:
    """A physically consistent, contiguous run of frames."""

    run: str
    segment: int
    start: int  # position in the run's frame list, inclusive
    end: int  # position, exclusive
    start_index: int  # file numbering
    end_index: int
    has_fragments: bool
    boundaries: tuple[str, ...] = ()

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class Window:
    """One burn-in + supervised window over a segment."""

    run: str
    segment: int
    start: int  # position in the run's frame list
    burn_in: int
    horizon: int
    frame_indices: tuple[int, ...]
    inner: str = "unassigned"
    block: int = -1


def material_mask(names: tuple[str, ...] | list[str]) -> np.ndarray:
    """Boolean mask selecting the material coefficient entries of a vector."""
    mask = []
    for name in names:
        stem = name.split("[", 1)[0]
        mask.append(any(stem == prefix or stem.startswith(prefix + "_") for prefix in MATERIAL_PREFIXES))
    return np.asarray(mask, dtype=bool)


def _run_namelist(run_dir: Path) -> str | None:
    for candidate in ("jorek_namelist_spitest", "jorek_namelist"):
        path = run_dir / candidate
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
    return None


def injection_time(namelist_text: str | None) -> float | None:
    """Parse ``t_ns`` (the SPI injection time in JOREK units) from a namelist."""
    if not namelist_text:
        return None
    match = _T_NS_RE.search(namelist_text)
    if match is None:
        return None
    return float(match.group(1).replace("d", "e").replace("D", "e"))


def _projection_path(state_path: Path) -> Path | None:
    match = re.fullmatch(r"jorek(?P<index>\d+)\.h5", state_path.name)
    if match is None:
        return None
    index = int(match.group("index"))
    candidate = state_path.parent / f"projections{index:06d}.h5"
    return candidate if candidate.exists() else None


def collect_frames(run_dir: str | Path) -> list[FrameInfo]:
    """Read per-frame metadata (time, shard count, condition vector)."""
    run_dir = Path(run_dir)
    namelist_text = _run_namelist(run_dir)
    frames: list[FrameInfo] = []
    for position, frame in enumerate(aligned_frames(run_dir)):
        with h5py.File(frame.state_path, "r") as handle:
            t_now = float(np.ravel(handle["t_now"][()])[0]) if "t_now" in handle else float(position)
            n_spi = int(np.ravel(handle["n_spi_tot"][()])[0]) if "n_spi_tot" in handle else 0
            condition = build_condition_vector(handle, namelist_text=namelist_text)
        frames.append(
            FrameInfo(
                index=frame.index,
                position=position,
                state_path=frame.state_path,
                marker_path=frame.marker_path,
                projection_path=_projection_path(frame.state_path),
                t_now=t_now,
                n_spi_tot=n_spi,
                condition=np.asarray(condition["vector"], dtype=np.float64),
                condition_names=tuple(condition["names"]),
            )
        )
    return frames


def _relative_change(old: float, new: float) -> float:
    if not np.isfinite(old) or not np.isfinite(new):
        return 0.0
    scale = max(abs(old), abs(new), 1e-30)
    if abs(old) <= 1e-30 and abs(new) <= 1e-30:
        return 0.0
    return abs(new - old) / scale


def _material_boundary(frames: list[FrameInfo], i: int, rel_threshold: float) -> bool:
    old = frames[i - 1]
    new = frames[i]
    mask = material_mask(old.condition_names)
    if mask.shape[0] != old.condition.shape[0] or mask.shape[0] != new.condition.shape[0]:
        mask = np.ones(min(old.condition.shape[0], new.condition.shape[0]), dtype=bool)
    old_values = old.condition[: mask.shape[0]][mask]
    new_values = new.condition[: mask.shape[0]][mask]
    return any(
        _relative_change(float(a), float(b)) > rel_threshold
        for a, b in zip(old_values, new_values)
    )


def _numbering_gap(frames: list[FrameInfo], i: int) -> bool:
    steps = [frames[k + 1].index - frames[k].index for k in range(len(frames) - 1)]
    if not steps:
        return False
    step = float(np.median(steps))
    return (frames[i].index - frames[i - 1].index) > 1.5 * step


def _time_reversal(frames: list[FrameInfo], i: int) -> bool:
    """True when the recorded time does not strictly increase at ``i``.

    A run whose H5 chain contains a restart/backup append can jump backwards
    (observed: ``scan_1d21_ang12_massloss_neon`` t=19710.5 -> 10621.1).  A
    window spanning that jump has a negative ``dt``; the propagator then
    overflows to inf/NaN and poisons the whole training run (pitfall #61).
    """
    old, new = frames[i - 1].t_now, frames[i].t_now
    if not (np.isfinite(old) and np.isfinite(new)):
        return True
    return new <= old


def segment_frames(
    run_name: str,
    frames: list[FrameInfo],
    *,
    t_ns: float | None = None,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
) -> list[Segment]:
    """Cut a run into physically consistent segments.

    Boundaries: frame-numbering gaps, material coefficient changes, the SPI
    injection time, a change of the fragment-availability flag, and any
    non-increasing recorded time (restart/backup append).
    """
    if not frames:
        return []
    boundaries: list[int] = [0]
    reasons: dict[int, list[str]] = {}
    for i in range(1, len(frames)):
        cuts: list[str] = []
        if _numbering_gap(frames, i):
            cuts.append("frame_gap")
        if _time_reversal(frames, i):
            cuts.append("time_reversal")
        if _material_boundary(frames, i, rel_threshold):
            cuts.append("material")
        if (frames[i - 1].n_spi_tot > 0) != (frames[i].n_spi_tot > 0):
            cuts.append("fragments")
        if t_ns is not None and (frames[i - 1].t_now < t_ns <= frames[i].t_now):
            cuts.append("injection")
        if cuts:
            boundaries.append(i)
            reasons[i] = cuts
    boundaries.append(len(frames))

    segments: list[Segment] = []
    for segment_id, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segments.append(
            Segment(
                run=run_name,
                segment=segment_id,
                start=start,
                end=end,
                start_index=frames[start].index,
                end_index=frames[end - 1].index,
                has_fragments=any(frames[k].n_spi_tot > 0 for k in range(start, end)),
                boundaries=tuple(reasons.get(start, ())),
            )
        )
    return segments


def build_windows(
    segments: list[Segment],
    *,
    burn_in: int = DEFAULT_BURN_IN,
    horizon: int = DEFAULT_HORIZON,
    stride: int = 1,
) -> list[Window]:
    """Generate every valid window of a segment (dense by default)."""
    if burn_in < 1 or horizon < 1:
        raise ValueError("burn_in and horizon must be >= 1")
    if stride < 1:
        raise ValueError("stride must be >= 1")
    span = burn_in + horizon + 1
    windows: list[Window] = []
    for segment in segments:
        for start in range(segment.start, segment.end - span + 1, stride):
            index_tuple = tuple(range(start, start + span))
            windows.append(
                Window(
                    run=segment.run,
                    segment=segment.segment,
                    start=start,
                    burn_in=burn_in,
                    horizon=horizon,
                    frame_indices=index_tuple,
                )
            )
    return windows


def assign_blocked_split(
    windows: list[Window],
    segments: list[Segment] | None = None,
    *, 
    val_fraction: float = DEFAULT_VAL_FRACTION,
    block_windows: int = 16,
) -> list[Window]:
    """Blocked inner split with purge/embargo (ADR-0012, interpolation protocol).

    Each run's windows (ordered by segment and start) are cut into contiguous
    blocks of ``block_windows`` windows.  The last ``val_fraction`` of a run's
    blocks becomes validation.  The hard requirement is a *purge*: training
    windows that share any frame with a validation window are dropped.  When a
    run has at least three blocks, an additional one-span embargo is applied
    (training windows within one window span of a validation window are also
    dropped); shorter runs stay purge-only so they still contribute training
    data.  Runs with a single block contribute training windows only.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must satisfy 0 <= val_fraction < 1")
    if block_windows < 1:
        raise ValueError("block_windows must be >= 1")
    grouped: dict[str, list[Window]] = {}
    for window in windows:
        grouped.setdefault(window.run, []).append(window)

    for run, group in grouped.items():
        group.sort(key=lambda window: (window.segment, window.start))
        span = group[0].burn_in + group[0].horizon + 1
        blocks = [group[i : i + block_windows] for i in range(0, len(group), block_windows)]
        for block_id, block in enumerate(blocks):
            for window in block:
                window.block = block_id
        if len(blocks) < 2 or val_fraction == 0.0:
            for window in group:
                window.inner = "train"
            continue
        n_val = max(1, int(math.ceil(len(blocks) * val_fraction)))
        val_blocks = set(range(len(blocks) - n_val, len(blocks)))
        for window in group:
            window.inner = "val" if window.block in val_blocks else "train"

        use_embargo = len(blocks) >= 3
        val_ranges = [
            (window.start, window.start + span)
            for window in group
            if window.inner == "val"
        ]
        for window in group:
            if window.inner != "train":
                continue
            start, end = window.start, window.start + span
            for val_start, val_end in val_ranges:
                overlap = not (end <= val_start or val_end <= start)
                if overlap:
                    window.inner = "dropped"
                    break
                if use_embargo and not (end + span <= val_start or val_end + span <= start):
                    window.inner = "dropped"
                    break
    return windows


def loo_folds(run_names: list[str]) -> list[dict[str, object]]:
    """Leave-One-Run-Out fold table (ADR-0012, scenario protocol)."""
    names = sorted(run_names)
    return [
        {"fold": fold, "held_out_run": name, "train_runs": [other for other in names if other != name]}
        for fold, name in enumerate(names)
    ]


def build_interp_manifest(
    data_root: str | Path,
    *,
    burn_in: int = DEFAULT_BURN_IN,
    horizon: int = DEFAULT_HORIZON,
    stride: int = 1,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    block_windows: int = 16,
    runs: list[str] | None = None,
    run_glob: str | None = "scan_*",
) -> dict:
    """Build the Phase-1 window corpus manifest for the eligible 2D runs.

    ``run_glob`` defaults to the ten production ``scan_*`` runs; pass ``None``
    to include every 2D-eligible run (``example``, ``spi_rkt5``, ...).
    """
    data_root = Path(data_root)
    wanted = set(runs) if runs else None
    run_records: list[dict] = []
    all_segments: list[Segment] = []
    all_windows: list[Window] = []

    for run in discover_runs(data_root):
        if wanted is not None and run.name not in wanted:
            continue
        if wanted is None and run_glob is not None and not fnmatch(run.name, run_glob):
            continue
        if not is_2d_run(run.path):
            continue
        frames = collect_frames(run.path)
        namelist_text = _run_namelist(run.path)
        t_ns = injection_time(namelist_text)
        segments = segment_frames(run.name, frames, t_ns=t_ns, rel_threshold=rel_threshold)
        windows = build_windows(segments, burn_in=burn_in, horizon=horizon, stride=stride)
        # Defensive second gate: even if segmentation ever misses a boundary,
        # never emit a window whose recorded time is not strictly increasing.
        # A single negative dt makes the LTC timespan negative, ``exp``
        # overflows and the NaN propagates through the whole model (pitfall #61).
        monotonic_windows: list[Window] = []
        for window in windows:
            times = [frames[k].t_now for k in window.frame_indices]
            if all(
                np.isfinite(a) and np.isfinite(b) and b > a
                for a, b in zip(times, times[1:])
            ):
                monotonic_windows.append(window)
        n_nonmonotonic = len(windows) - len(monotonic_windows)
        windows = monotonic_windows
        all_segments.extend(segments)
        all_windows.extend(windows)
        run_records.append(
            {
                "name": run.name,
                "n_frames": len(frames),
                "first_index": frames[0].index if frames else None,
                "last_index": frames[-1].index if frames else None,
                "t_ns": t_ns,
                "n_spi": int(max((frame.n_spi_tot for frame in frames), default=0)),
                "post_tn_frames": int(sum(frame.t_now >= t_ns for frame in frames)) if t_ns else None,
                "segments": [
                    {
                        "segment": segment.segment,
                        "start_index": segment.start_index,
                        "end_index": segment.end_index,
                        "length": segment.length,
                        "has_fragments": segment.has_fragments,
                        "boundaries": list(segment.boundaries),
                    }
                    for segment in segments
                ],
                "n_windows": len(windows),
                "n_nonmonotonic_windows": n_nonmonotonic,
            }
        )

    all_windows = assign_blocked_split(
        all_windows, all_segments, val_fraction=val_fraction, block_windows=block_windows
    )
    window_records = [
        {
            "run": window.run,
            "segment": window.segment,
            "start": window.start,
            "burn_in": window.burn_in,
            "horizon": window.horizon,
            "inner": window.inner,
            "block": window.block,
            "frame_indices": list(window.frame_indices),
        }
        for window in all_windows
    ]
    folds = loo_folds([record["name"] for record in run_records])
    inner_counts = {"train": 0, "val": 0, "dropped": 0, "unassigned": 0}
    for window in all_windows:
        inner_counts[window.inner] = inner_counts.get(window.inner, 0) + 1

    return {
        "schema": MANIFEST_SCHEMA,
        "data_root": str(data_root),
        "burn_in": burn_in,
        "horizon": horizon,
        "stride": stride,
        "rel_threshold": rel_threshold,
        "val_fraction": val_fraction,
        "block_windows": block_windows,
        "runs": sorted(run_records, key=lambda record: record["name"]),
        "windows": window_records,
        "folds": folds,
        "summary": {
            "n_runs": len(run_records),
            "n_segments": len(all_segments),
            "n_windows": len(all_windows),
            "n_nonmonotonic_windows": int(
                sum(record.get("n_nonmonotonic_windows", 0) for record in run_records)
            ),
            "inner": inner_counts,
        },
    }


def write_manifest(manifest: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return path


__all__ = [
    "DEFAULT_BURN_IN",
    "DEFAULT_HORIZON",
    "DEFAULT_REL_THRESHOLD",
    "DEFAULT_VAL_FRACTION",
    "MANIFEST_SCHEMA",
    "FrameInfo",
    "Segment",
    "Window",
    "assign_blocked_split",
    "build_interp_manifest",
    "build_windows",
    "collect_frames",
    "injection_time",
    "loo_folds",
    "material_mask",
    "segment_frames",
    "write_manifest",
]
