"""ADR-0019 paired loader: ``(u_k, p_{k-1}=part{k}, S_k=projections{k}) -> u_{k+1}``.

This is the Phase A replacement for the retired v1 interpolation pairing
(``source[:, k+1]``, one step of future leakage; ADR-0019).  It reads the raw
recorded files, enforces the timing contract from
``src/data/output_contract.py``, and exposes one sample per valid pair:

* ``u_before``  = ``jorek{k}`` nine-field state at ``T_k``;
* ``source``    = ``projections{k}`` five marker-projection channels, sampled
  over ``[T_{k-1}, T_k]`` and consumed by the solve ``u_k -> u_{k+1}``;
* ``p_start``   = ``part{k}`` marker snapshot at the window start ``T_{k-1}``;
* ``u_after``   = ``jorek{k+1}`` state at ``T_{k+1}``;
* ``dt_step``   = ``T_{k+1} - T_k`` (target interval for the state rate);
* ``dt_source`` = ``T_k - T_{k-1}`` (length of the source sampling window).

Fail-fast (all raise :class:`~src.data.output_contract.ContractError`):
``dt <= 0``, non-monotonic ``jorek`` times, partially missing marker pairs,
unregistered projection/part offsets, and shape mismatches between the state
and projection files.  Registered restart-boundary frames are dropped (or
raised on, with ``policy="error"``) rather than silently re-paired.

Marker snapshots stay lazy: :class:`ParticleSnapshot` gives access to the
``groups/<id>/<dataset>`` layout without loading the ~1e6-particle arrays until
they are actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import numpy as np

from src.data.output_contract import (
    ContractError,
    ContractReport,
    N_SOURCE_FIELDS,
    N_STATE_FIELDS,
    PairRecord,
    check_run_contract,
    state_frame_indices,
)


@dataclass(frozen=True)
class ParticleSnapshot:
    """Lazy view of one ``part{n}.h5`` snapshot (window-start marker state).

    ``time_seconds`` is the recorded physical time (seconds), matching
    ``part{n}./time``; divide by ``t_norm`` to get JOREK time units.
    """

    index: int
    path: Path
    time_seconds: float

    @property
    def groups(self) -> tuple[str, ...]:
        with h5py.File(self.path, "r") as handle:
            groups = handle.get("groups")
            if groups is None:
                return ()
            return tuple(sorted(groups.keys()))

    def read(self, group: str, name: str) -> np.ndarray:
        """Read one particle array, e.g. ``read("001", "P_imp")``."""
        with h5py.File(self.path, "r") as handle:
            try:
                return np.asarray(handle[f"groups/{group}/{name}"][()])
            except KeyError as exc:  # pragma: no cover - defensive
                raise ContractError(
                    f"{self.path.name}: groups/{group}/{name} not found"
                ) from exc

    def total_weight(self, group: str) -> float:
        """Sum of the marker weights of one group (cheap cross-check)."""
        return float(self.read(group, "weight").sum())


@dataclass(frozen=True)
class PairSample:
    """One ADR-0019 training pair."""

    index: int
    u_before: np.ndarray
    source: np.ndarray
    u_after: np.ndarray
    p_start: ParticleSnapshot
    t_before: float
    t_after: float
    t_norm: float
    dt_step: float
    dt_source: float

    @property
    def source_window(self) -> tuple[float, float]:
        """The sampling window ``[T_{k-1}, T_k]`` in JOREK time units."""
        return (self.t_before - self.dt_source, self.t_before)

    @property
    def field_interval(self) -> tuple[float, float]:
        """The field step ``[T_k, T_{k+1}]`` driven by ``source``."""
        return (self.t_before, self.t_after)


def _read_state(path: Path, n_nodes: int | None = None) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "values" not in handle:
            raise ContractError(f"{path.name}: missing 'values'")
        raw = handle["values"]
        if raw.ndim != 4 or raw.shape[0] < N_STATE_FIELDS or raw.shape[2] != 1:
            raise ContractError(
                f"{path.name}: expected values [>={N_STATE_FIELDS}, dof, 1, nodes], "
                f"got {raw.shape}"
            )
        values = np.asarray(raw[:N_STATE_FIELDS, 0, 0, :], dtype=np.float64).T
    if n_nodes is not None and values.shape[0] != n_nodes:
        raise ContractError(
            f"{path.name}: node count {values.shape[0]} does not match {n_nodes}"
        )
    return values


def _read_projection(path: Path, n_nodes: int) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "values" not in handle:
            raise ContractError(f"{path.name}: missing 'values'")
        raw = handle["values"]
        if raw.ndim != 4 or raw.shape[0] < N_SOURCE_FIELDS or raw.shape[2] != 1:
            raise ContractError(
                f"{path.name}: expected values [>={N_SOURCE_FIELDS}, dof, 1, nodes], "
                f"got {raw.shape}"
            )
        if raw.shape[-1] != n_nodes:
            raise ContractError(
                f"{path.name}: node count {raw.shape[-1]} does not match {n_nodes}"
            )
        return np.asarray(raw[:N_SOURCE_FIELDS, 0, 0, :], dtype=np.float64).T


class PairedRunDataset:
    """Index + reader for the pairs of one run, validated by ADR-0019.

    Parameters
    ----------
    run_dir:
        Run directory containing ``jorek*/projections*/part*`` files.
    step_times:
        Optional ``{step_index: T_n}`` table (JOREK units).  When omitted,
        ``macroscopic_vars.dat`` is used if present; without either, the
        per-fluid-step ``part`` equality cannot be verified and the contract
        report records that as ``part_window_unverified``.
    index_window:
        Optional ``(first, last)`` inclusive frame-index window, for tests and
        partial reads.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        step_times: Mapping[int, float] | None = None,
        restart_indices: Sequence[int] | None = None,
        policy: str = "drop",
        index_window: tuple[int, int] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        states = state_frame_indices(self.run_dir)
        if index_window is not None:
            first, last = index_window
            if last < first:
                raise ValueError(f"index_window must be ordered, got {index_window!r}")
            states = [index for index in states if first <= index <= last]
            if not states:
                raise ContractError(
                    f"{self.run_dir}: no state frames inside window {index_window!r}"
                )
        self.report: ContractReport = check_run_contract(
            self.run_dir,
            step_times=step_times,
            restart_indices=restart_indices,
            policy=policy,
            indices=states,
        )
        self._records: tuple[PairRecord, ...] = self.report.pair_records
        if not self._records:
            raise ContractError(
                f"{self.run_dir}: the contract report contains no valid pairs"
            )
        self._n_nodes = self._node_count()

    def _node_count(self) -> int:
        first_state = self._state_path(self._records[0].index)
        with h5py.File(first_state, "r") as handle:
            if "boundary" not in handle:
                raise ContractError(f"{self.run_dir}: state frames are missing 'boundary'")
            return int(handle["boundary"].shape[0])

    def _state_path(self, index: int) -> Path:
        return self.run_dir / f"jorek{index:06d}.h5"

    def _projection_path(self, index: int) -> Path:
        return self.run_dir / f"projections{index:06d}.h5"

    def _particle_path(self, index: int) -> Path:
        return self.run_dir / f"part{index:05d}.h5"

    @property
    def n_nodes(self) -> int:
        return self._n_nodes

    @property
    def pair_records(self) -> tuple[PairRecord, ...]:
        return self._records

    @property
    def dropped_restart(self) -> tuple[int, ...]:
        return self.report.dropped_restart

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, position: int) -> PairSample:
        record = self._records[position]
        state_path = self._state_path(record.index)
        u_before = _read_state(state_path, self._n_nodes)
        u_after = _read_state(self._state_path(record.index_next), self._n_nodes)
        source = _read_projection(self._projection_path(record.index), self._n_nodes)
        particle_path = self._particle_path(record.index)
        if not particle_path.exists():
            raise ContractError(
                f"index {record.index}: particle snapshot {particle_path.name} "
                "disappeared after the index was built"
            )
        t_norm = self._t_norm(record.index)
        p_start = ParticleSnapshot(
            index=record.index,
            path=particle_path,
            time_seconds=(record.t_state - record.dt_source) * t_norm,
        )
        return PairSample(
            index=record.index,
            u_before=u_before,
            source=source,
            u_after=u_after,
            p_start=p_start,
            t_before=record.t_state,
            t_after=record.t_next,
            t_norm=t_norm,
            dt_step=record.dt_step,
            dt_source=record.dt_source,
        )

    def _t_norm(self, index: int) -> float:
        with h5py.File(self._state_path(index), "r") as handle:
            if "t_norm" not in handle:
                raise ContractError(f"{self._state_path(index).name}: missing 't_norm'")
            return float(np.ravel(handle["t_norm"][()])[0])

    def pair_indices(self) -> tuple[int, ...]:
        return tuple(record.index for record in self._records)
