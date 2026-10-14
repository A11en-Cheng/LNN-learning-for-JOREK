"""Recorded-output timing contract v3 for Model 502 runs (ADR-0019).

This module implements the *file-level* timing contract that new architecture
work (roadmap_v2 Phase A) must enforce before any recorded frame may enter a
dataset.  The contract is defined in
``report/data-contract-v3.md`` and ``docs/adr/0019-recorded-output-timing-contract.md``:

* ``jorek{n}.h5`` is the nine-field state ``u_n`` at time ``T_n`` (JOREK time
  units in ``t_now``).
* ``projections{n}.h5`` is the source ``S_n`` sampled over the fluid-step
  window ``[T_{n-1}, T_n]``; it is consumed by the solve ``u_n -> u_{n+1}``.
  Its ``t_now`` is stored in **physical seconds** and equals ``T_n * t_norm``.
* ``part{n}.h5`` is the marker snapshot at the window start, ``p_{n-1}`` at
  ``T_{n-1}``; its ``/time`` is in **physical seconds** and equals
  ``T_{n-1} * t_norm``.
* Correct pairing: ``(u_k, p_{k-1} = part{k}, S_k = projections{k}) -> u_{k+1}``.

Restart-checkpoint frames (``scan_1d22_ang12_massloss_neon``: 3550/4150/4950/
5330/6600/7350; ``scan_1d22_ang15_5_massloss_neon_v400``: 5050) are the one
known exception: there ``projections{k}`` is written one fluid step early
(``t_projection = T_{k-1}``) while ``part{k}`` still satisfies the normal rule.
Such frames are registered here and dropped (or, with explicit evidence,
re-aligned); an unregistered offset is a hard error.

Everything in this module is fail-fast: non-positive ``dt``, non-monotonic
``jorek`` times, partially missing markers, or unregistered time offsets raise
:class:`ContractError`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import h5py
import numpy as np

STATE_FRAME_RE = re.compile(r"jorek(?P<index>\d+)\.h5\Z")
PROJECTION_FRAME_RE = re.compile(r"projections(?P<index>\d+)\.h5\Z")
PART_FRAME_RE = re.compile(r"part(?P<index>\d+)\.h5\Z")

N_STATE_FIELDS = 9
N_SOURCE_FIELDS = 5

#: Absolute tolerance when comparing times converted to JOREK units.  A fluid
#: step in the EHL-2 runs is 0.05--5.0 JOREK units, so 1e-4 can never hide a
#: real step-level offset.
TIME_TOL_JOREK = 1.0e-4

#: Restart-checkpoint frames whose ``projections{n}`` is offset by one fluid
#: step (ADR-0019 §5.4).  Keyed by run-directory name.
RESTART_BOUNDARY_INDICES: dict[str, tuple[int, ...]] = {
    "scan_1d22_ang12_massloss_neon": (3550, 4150, 4950, 5330, 6600, 7350),
    "scan_1d22_ang15_5_massloss_neon_v400": (5050,),
}

_MACROSCOPIC_TIMES_RE = re.compile(r"@times:\s+(\d+)\s+([0-9EeDd.+-]+)")


class ContractError(RuntimeError):
    """Raised when recorded output violates the ADR-0019 timing contract."""


@dataclass(frozen=True)
class TimingIssue:
    """A contract observation that is registered rather than fatal."""

    kind: str
    index: int
    detail: str


@dataclass(frozen=True)
class FrameTiming:
    """Per-index timing of the three recorded outputs, in JOREK time units."""

    index: int
    t_state: float
    t_norm: float
    t_projection: float | None
    t_particle: float | None
    state_path: Path
    projection_path: Path | None
    particle_path: Path | None

    @property
    def has_marker_pair(self) -> bool:
        return self.projection_path is not None and self.particle_path is not None


@dataclass(frozen=True)
class PairRecord:
    """One ADR-0019 pair ``(u_k, part{k}, projections{k}) -> u_{k+1}``.

    ``dt_source`` is the length of the source sampling window
    ``[T_{k-1}, T_k]``; ``dt_step`` is the target interval
    ``[T_k, T_{k+1}]`` used for the state rate.
    """

    index: int
    index_next: int
    t_state: float
    t_next: float
    dt_step: float
    dt_source: float


@dataclass(frozen=True)
class ContractReport:
    """Result of :func:`check_run_contract`."""

    run: str
    run_dir: Path
    n_state_frames: int
    n_marker_frames: int
    pair_records: tuple[PairRecord, ...]
    dropped_restart: tuple[int, ...]
    issues: tuple[TimingIssue, ...]
    step_table_source: str
    policy: str

    @property
    def pair_indices(self) -> tuple[int, ...]:
        return tuple(record.index for record in self.pair_records)

    def to_dict(self) -> dict:
        return {
            "run": self.run,
            "run_dir": str(self.run_dir),
            "n_state_frames": self.n_state_frames,
            "n_marker_frames": self.n_marker_frames,
            "n_pairs": len(self.pair_records),
            "pair_indices": list(self.pair_indices),
            "dropped_restart": list(self.dropped_restart),
            "issues": [issue.__dict__ for issue in self.issues],
            "step_table_source": self.step_table_source,
            "policy": self.policy,
        }


def _scalar(path: Path, key: str) -> float:
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise ContractError(f"{path.name} is missing dataset '{key}'")
        return float(np.ravel(handle[key][()])[0])


def _indexed_files(run_dir: Path, pattern: re.Pattern[str]) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in run_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match is not None:
            found[int(match.group("index"))] = path
    return found


def parse_macroscopic_times(path: str | Path) -> dict[int, float]:
    """Parse ``@times: <step> <time>`` records from ``macroscopic_vars.dat``.

    The file is JOREK's plottable block format; every fluid step contributes one
    ``@times`` record carrying the step index and the JOREK-unit time ``T_n``.
    """
    path = Path(path)
    times: dict[int, float] = {}
    with path.open("r", errors="replace") as handle:
        for line in handle:
            match = _MACROSCOPIC_TIMES_RE.match(line)
            if match is not None:
                index = int(match.group(1))
                value = float(match.group(2).replace("D", "E").replace("d", "e"))
                times[index] = value
    if not times:
        raise ContractError(f"{path} contains no '@times' records")
    return times


def resolve_step_times(
    run_dir: str | Path,
    step_times: Mapping[int, float] | None = None,
) -> tuple[dict[int, float] | None, str]:
    """Return ``(step_times, source)`` for the run.

    An explicit table wins; otherwise ``macroscopic_vars.dat`` is used when
    present.  ``(None, "absent")`` means only ordering checks are possible and
    callers must record that the per-fluid-step ``part`` assertion was not run.
    """
    if step_times is not None:
        return dict(step_times), "explicit"
    macro = Path(run_dir) / "macroscopic_vars.dat"
    if macro.exists():
        return parse_macroscopic_times(macro), "macroscopic_vars.dat"
    return None, "absent"


def read_frame_timing(run_dir: str | Path, index: int) -> FrameTiming:
    """Read the timing triple for one frame index."""
    run_dir = Path(run_dir)
    state_path = run_dir / f"jorek{index:06d}.h5"
    projection_path = run_dir / f"projections{index:06d}.h5"
    particle_path = run_dir / f"part{index:05d}.h5"
    if not state_path.exists():
        raise ContractError(f"{state_path} does not exist")
    t_state = _scalar(state_path, "t_now")
    t_norm = _scalar(state_path, "t_norm")
    if t_norm <= 0.0:
        raise ContractError(f"{state_path.name}: t_norm must be positive, got {t_norm}")
    has_projection = projection_path.exists()
    has_particle = particle_path.exists()
    t_projection = None
    t_particle = None
    if has_projection:
        t_projection = _scalar(projection_path, "t_now") / t_norm
    if has_particle:
        t_particle = _scalar(particle_path, "time") / t_norm
    return FrameTiming(
        index=index,
        t_state=t_state,
        t_norm=t_norm,
        t_projection=t_projection,
        t_particle=t_particle,
        state_path=state_path,
        projection_path=projection_path if has_projection else None,
        particle_path=particle_path if has_particle else None,
    )


def collect_run_timing(
    run_dir: str | Path,
    *,
    indices: Iterable[int] | None = None,
) -> tuple[FrameTiming, ...]:
    """Collect per-frame timings for every ``jorek`` frame in ``run_dir``.

    Frames without any marker files are legal unforced frames and are returned
    with ``projection_path=None``.  Frames carrying only one of the two marker
    files are returned as-is; :func:`check_run_contract` decides whether they
    are a legitimate leading/trailing edge or a gap.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ContractError(f"{run_dir} is not a directory")
    states = _indexed_files(run_dir, STATE_FRAME_RE)
    if not states:
        raise ContractError(f"{run_dir} contains no jorek*.h5 frames")
    if indices is None:
        selected: Iterable[int] = states.keys()
    else:
        requested = sorted(set(int(index) for index in indices))
        missing = [index for index in requested if index not in states]
        if missing:
            raise ContractError(f"{run_dir}: requested indices have no state frame: {missing}")
        selected = requested
    timings = [read_frame_timing(run_dir, index) for index in sorted(selected)]
    return tuple(timings)


def state_frame_indices(run_dir: str | Path) -> list[int]:
    """Return every ``jorek`` frame index in ``run_dir``, ascending."""
    run_dir = Path(run_dir)
    return sorted(_indexed_files(run_dir, STATE_FRAME_RE))


def assert_strictly_increasing(values: Sequence[float], *, what: str) -> None:
    """Raise :class:`ContractError` unless ``values`` is strictly increasing."""
    for previous, current in zip(values, values[1:]):
        if not current > previous:
            raise ContractError(
                f"{what} is not strictly increasing: {previous!r} -> {current!r}"
            )


def _resolve_restart_indices(run_dir: Path, restart_indices: Sequence[int] | None) -> set[int]:
    if restart_indices is not None:
        return set(int(index) for index in restart_indices)
    return set(RESTART_BOUNDARY_INDICES.get(run_dir.name, ()))


def check_run_contract(
    run_dir: str | Path,
    *,
    step_times: Mapping[int, float] | None = None,
    restart_indices: Sequence[int] | None = None,
    policy: str = "drop",
    tol: float = TIME_TOL_JOREK,
    indices: Iterable[int] | None = None,
    timings: Sequence[FrameTiming] | None = None,
) -> ContractReport:
    """Validate one run against the ADR-0019 timing contract.

    ``policy`` controls registered restart-boundary frames:

    * ``"drop"`` (default): drop them from the pair list and register the issue;
    * ``"error"``: treat them as fatal (for runs where they must be re-exported).

    Fatal (always raised): non-monotonic ``jorek`` times, ``dt <= 0``, partial
    marker pairs, unregistered projection/part time offsets, and a step table
    missing entries for a marker frame.
    """
    run_dir = Path(run_dir)
    if policy not in {"drop", "error"}:
        raise ValueError(f"unknown restart policy {policy!r}; use 'drop' or 'error'")
    if timings is None:
        timings = collect_run_timing(run_dir, indices=indices)
    elif indices is not None:
        raise ValueError("pass either 'indices' or 'timings', not both")
    assert_strictly_increasing(
        [timing.t_state for timing in timings], what=f"{run_dir.name}: jorek t_now"
    )
    table, table_source = resolve_step_times(run_dir, step_times)
    restart = _resolve_restart_indices(run_dir, restart_indices)

    issues: list[TimingIssue] = []
    dropped: list[int] = []
    complete = [timing.index for timing in timings if timing.has_marker_pair]
    if not complete:
        raise ContractError(f"{run_dir}: no complete part/projections pairs found")
    covered = (complete[0], complete[-1])
    for timing in timings:
        has_projection = timing.projection_path is not None
        has_particle = timing.particle_path is not None
        if has_projection == has_particle:
            continue
        if covered[0] < timing.index < covered[1]:
            raise ContractError(
                f"index {timing.index}: partial marker pairing inside the covered "
                f"window {covered} (projection={has_projection}, particle={has_particle})"
            )
        issues.append(
            TimingIssue(
                kind="unpaired_boundary_frame",
                index=timing.index,
                detail=(
                    f"marker coverage edge: projection={has_projection}, "
                    f"particle={has_particle}; skipped"
                ),
            )
        )
    for i, timing in enumerate(timings):
        if i > 0:
            dt_frames = timing.t_state - timings[i - 1].t_state
            if dt_frames <= 0.0:
                raise ContractError(
                    f"index {timing.index}: non-positive frame dt {dt_frames!r}"
                )
        if not timing.has_marker_pair:
            continue
        assert timing.t_projection is not None and timing.t_particle is not None
        is_restart = timing.index in restart

        dt_source: float | None = None
        if table is not None:
            if timing.index not in table or (timing.index - 1) not in table:
                raise ContractError(
                    f"index {timing.index}: step table ({table_source}) is missing "
                    f"entry {timing.index} or {timing.index - 1}"
                )
            dt_source = table[timing.index] - table[timing.index - 1]
            if dt_source <= 0.0:
                raise ContractError(
                    f"index {timing.index}: non-positive fluid-step dt {dt_source!r}"
                )

        projection_offset = timing.t_projection - timing.t_state
        if abs(projection_offset) > tol:
            if not is_restart:
                raise ContractError(
                    f"index {timing.index}: projections t_now is offset "
                    f"{projection_offset!r} JOREK units from the same-index jorek frame; "
                    "only registered restart boundaries may be offset"
                )
            if dt_source is not None and abs(projection_offset + dt_source) > tol:
                raise ContractError(
                    f"index {timing.index}: registered restart boundary offset "
                    f"{projection_offset!r} does not match the expected -dt_source "
                    f"{-dt_source!r}"
                )
            expected = f" (expected -dt_source={-dt_source!r})" if dt_source else ""
            if policy == "error":
                raise ContractError(
                    f"index {timing.index}: registered restart boundary is offset by "
                    f"{projection_offset!r}{expected} and policy='error'"
                )
            issues.append(
                TimingIssue(
                    kind="restart_boundary_projection_offset",
                    index=timing.index,
                    detail=(
                        f"registered restart boundary: projections t_now is "
                        f"{projection_offset!r} JOREK units early{expected}; dropped"
                    ),
                )
            )
            dropped.append(timing.index)
            continue
        if is_restart:
            issues.append(
                TimingIssue(
                    kind="restart_registry_stale",
                    index=timing.index,
                    detail=(
                        "index is registered as a restart boundary but projections "
                        "matches the same-index jorek frame; registry may be stale"
                    ),
                )
            )

        if dt_source is not None:
            expected_part = timing.t_state - dt_source
            if abs(timing.t_particle - expected_part) > tol:
                raise ContractError(
                    f"index {timing.index}: part time {timing.t_particle!r} != "
                    f"T_k - dt_source = {expected_part!r}"
                )
        else:
            if not timing.t_particle < timing.t_projection:
                raise ContractError(
                    f"index {timing.index}: part time {timing.t_particle!r} is not "
                    f"before its projection window end {timing.t_projection!r}"
                )
            issues.append(
                TimingIssue(
                    kind="part_window_unverified",
                    index=timing.index,
                    detail=(
                        "no fluid-step table available: only ordering of "
                        "part < projections was checked"
                    ),
                )
            )

    pair_records: list[PairRecord] = []
    dropped_set = set(dropped)
    for i in range(len(timings) - 1):
        timing = timings[i]
        if not timing.has_marker_pair or timing.index in dropped_set:
            continue
        assert timing.t_projection is not None and timing.t_particle is not None
        dt_step = timings[i + 1].t_state - timing.t_state
        if dt_step <= 0.0:
            raise ContractError(f"index {timing.index}: non-positive pair dt {dt_step!r}")
        if table is not None:
            dt_source = table[timing.index] - table[timing.index - 1]
        else:
            dt_source = timing.t_projection - timing.t_particle
        pair_records.append(
            PairRecord(
                index=timing.index,
                index_next=timings[i + 1].index,
                t_state=timing.t_state,
                t_next=timings[i + 1].t_state,
                dt_step=dt_step,
                dt_source=dt_source,
            )
        )

    return ContractReport(
        run=run_dir.name,
        run_dir=run_dir,
        n_state_frames=len(timings),
        n_marker_frames=sum(1 for timing in timings if timing.has_marker_pair),
        pair_records=tuple(pair_records),
        dropped_restart=tuple(dropped),
        issues=tuple(issues),
        step_table_source=table_source,
        policy=policy,
    )
