"""A' per-fluid-step export loader (data-export-spec-v3, ADR-0019).

This module reads the *new* fine-grained export layout
``<run>/export_v3/{manifest.json, steps/step_{n:06d}.h5}`` defined by
``report/data-export-spec-v3.md`` and exposes one self-described step per
record:

    ``(u_k = u_before, p_{k-1} = shard_start, S_k = S_used) -> u_{k+1} = u_after``

It does **not** touch the legacy frame-level loader
(:class:`src.data.paired_dataset.PairedRunDataset`): that one still handles the
recorded ``jorek*/projections*/part*`` triple by filename pairing.  Here every
record carries its own interval metadata, shard start/end states and (when the
generator provides them) mesh/condition/budget fields.  The checks are the
Phase A acceptance checks from ``report/data-contract-v3.md``:

* strictly increasing step times, positive ``dt``, no gaps unless registered;
* the source window ``[T_{n-1}, T_n]`` ends exactly at the field interval start
  ``T_n`` (the ADR-0019 pairing rule; unregistered offsets are fatal);
* ``require_single_step=True`` asserts ``dt_step == dt_source`` -- mandatory for
  data entering Phase B/C (frame-level legacy data has ``dt_step >> dt_source``);
* shard ``spi_*`` arrays are present on both window ends with consistent shape;
* boundary codes are registered (B0 report registry + manifest extensions);
* condition group labels are legal and ``excluded_*`` groups never enter FiLM.

Registered restart-boundary frames come from the manifest (``restart_indices``)
or the explicit ``restart_indices`` argument; they are dropped (or raised on
with ``policy="error"``) only when the optional raw-run cross-check actually
detects the known one-step projection offset.  New exports must be
offset-free, so a stale registry entry is registered as an issue, not hidden.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

from src.data.output_contract import (
    ContractError,
    TIME_TOL_JOREK,
    TimingIssue,
    assert_strictly_increasing,
    resolve_step_times,
)

MANIFEST_NAME = "manifest.json"
STEPS_DIRNAME = "steps"
STEP_FILE_RE = re.compile(r"step_(?P<index>\d+)\.h5\Z")

#: Fixed channel order (``report/model502-schema-and-source-protocol.md`` §1/§3.1).
STATE_FIELD_NAMES: tuple[str, ...] = (
    "psi",
    "Phi",
    "Z_j",
    "omega",
    "rho",
    "T_i",
    "v_par",
    "rho_imp",
    "T_e",
)
SOURCE_FIELD_NAMES: tuple[str, ...] = ("dE_ion", "rad_sink", "Z_eff", "Z_imp", "N_imp")
N_STATE_FIELDS = len(STATE_FIELD_NAMES)
N_SOURCE_FIELDS = len(SOURCE_FIELD_NAMES)

#: Shard (``type_SPI``) arrays required on ``/shards/start`` and ``/shards/end``
#: (``report/data-export-spec-v3.md`` §2.3; raw ``jorek*.h5`` spelling).
SHARD_FIELD_NAMES: tuple[str, ...] = (
    "spi_R_arr",
    "spi_Z_arr",
    "spi_phi_arr",
    "spi_Vel_R_arr",
    "spi_Vel_Z_arr",
    "spi_Vel_RxZ_arr",
    "spi_radius_arr",
    "spi_abl_arr",
    "spi_species_arr",
    "spi_vol_arr",
    "spi_psi_arr",
    "spi_grad_psi_arr",
    "spi_phi_init_arr",
    "spi_psi_arr_drift",
    "spi_grad_psi_arr_drift",
    "spi_vol_arr_drift",
)

LAYOUTS = ("node_dof0", "full_dof")
CADENCES = ("fluid_step", "substep_group")
#: Registered source objects (ADR-0020 decision 1 / data-contract-v3 §1):
#: ``inj_source`` and ``analytic_shard`` are the analytic shard density source in
#: JOREK semantics; ``projection`` / ``marker_feedback`` are the recorded marker
#: five-channel feedback, which **is** the true field source for the
#: energy/radiation/charge channels in a marker run.
SOURCE_OBJECTS = ("inj_source", "projection", "marker_feedback", "analytic_shard")
BUDGET_LAYOUTS = ("history", "slice", "absent")

#: Boundary codes registered from JOREK source + data (B0 report §3/§9,
#: ``grids/grid_utils/update_boundary_types.f90:630-702``).  0 = interior.
REGISTERED_BOUNDARY_CODES: frozenset[int] = frozenset(
    {0, 1, 2, 3, 4, 5, 9, 11, 12, 15, 19, 20, 21}
)

#: Condition group vocabulary (``report/data-export-spec-v3.md`` §4.2).
ALLOWED_CONDITION_GROUPS: frozenset[str] = frozenset(
    {
        "pellet_mass",
        "pellet_velocity",
        "pellet_angle",
        "pellet_species",
        "background_plasma",
        "boundary_flags",
        "excluded_transport",
        "excluded_numeric",
    }
)
FILM_FORBIDDEN_GROUPS: frozenset[str] = frozenset({"excluded_transport", "excluded_numeric"})

#: Tolerance for ``dt_step == dt_source``.  A fluid step is 0.01--5 JOREK
#: units; 1e-6 cannot hide a wrong step, but tolerates float64 round-trips.
SINGLE_STEP_TOL_JOREK = 1.0e-6


def resolve_export_root(root: str | Path) -> Path:
    """Return the ``export_v3`` directory for a run dir or export dir."""
    root = Path(root)
    if (root / MANIFEST_NAME).is_file():
        return root
    nested = root / "export_v3"
    if (nested / MANIFEST_NAME).is_file():
        return nested
    raise ContractError(
        f"{root}: no export manifest found (expected {root / MANIFEST_NAME} "
        f"or {nested / MANIFEST_NAME})"
    )


def _decode_strings(value: np.ndarray) -> tuple[str, ...]:
    flat = np.ravel(value)
    out: list[str] = []
    for item in flat:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return tuple(out)


def _read_scalar(handle: h5py.File, key: str, *, what: str) -> float:
    if key not in handle:
        raise ContractError(f"{what}: missing '{key}'")
    try:
        return float(np.ravel(handle[key][()])[0])
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ContractError(f"{what}: '{key}' is not a numeric scalar") from exc


def _read_int_scalar(handle: h5py.File, key: str, *, what: str) -> int:
    if key not in handle:
        raise ContractError(f"{what}: missing '{key}'")
    return int(np.ravel(handle[key][()])[0])


def _read_state_values(
    group: h5py.Group, layout: str, n_nodes: int | None, what: str, key: str = "u_before"
) -> np.ndarray:
    if key not in group:
        raise ContractError(f"{what}: missing '{key}'")
    raw = group[key]
    shape = raw.shape
    if layout == "full_dof":
        if len(shape) != 4 or shape[0] < N_STATE_FIELDS or shape[2] != 1:
            raise ContractError(
                f"{what}: full_dof 'values' must be [>={N_STATE_FIELDS}, dof, 1, nodes], "
                f"got {shape}"
            )
        values = np.asarray(raw[:N_STATE_FIELDS, 0, 0, :], dtype=np.float64).T
    elif layout == "node_dof0":
        if len(shape) == 2 and shape[0] >= N_STATE_FIELDS:
            values = np.asarray(raw[:N_STATE_FIELDS, :], dtype=np.float64).T
        elif len(shape) == 3 and shape[0] >= N_STATE_FIELDS and shape[1] == 1:
            values = np.asarray(raw[:N_STATE_FIELDS, 0, :], dtype=np.float64).T
        elif (
            len(shape) == 4
            and shape[0] >= N_STATE_FIELDS
            and shape[1] == 1
            and shape[2] == 1
        ):
            values = np.asarray(raw[:N_STATE_FIELDS, 0, 0, :], dtype=np.float64).T
        else:
            raise ContractError(
                f"{what}: node_dof0 'values' must be [{N_STATE_FIELDS}, nodes] or "
                f"[{N_STATE_FIELDS}, 1, nodes], got {shape}"
            )
    else:  # pragma: no cover - callers validate the layout first
        raise ContractError(f"{what}: unknown layout {layout!r}")
    if n_nodes is not None and values.shape[0] != n_nodes:
        raise ContractError(
            f"{what}: node count {values.shape[0]} does not match manifest n_nodes {n_nodes}"
        )
    return values


def _state_node_count(
    group: h5py.Group, layout: str, what: str, key: str = "u_before"
) -> tuple[int, int]:
    """Return ``(n_nodes, n_channels)`` for a state dataset without reading data."""
    if key not in group:
        raise ContractError(f"{what}: missing '{key}'")
    shape = group[key].shape
    if layout == "full_dof":
        if len(shape) != 4 or shape[0] < N_STATE_FIELDS or shape[2] != 1:
            raise ContractError(
                f"{what}: full_dof 'values' must be [>={N_STATE_FIELDS}, dof, 1, nodes], "
                f"got {shape}"
            )
        return int(shape[-1]), int(shape[0])
    if (len(shape) == 2 and shape[0] >= N_STATE_FIELDS) or (
        len(shape) == 3 and shape[0] >= N_STATE_FIELDS and shape[1] == 1
    ) or (
        len(shape) == 4
        and shape[0] >= N_STATE_FIELDS
        and shape[1] == 1
        and shape[2] == 1
    ):
        return int(shape[-1]), int(shape[0])
    raise ContractError(
        f"{what}: node_dof0 'values' must be [{N_STATE_FIELDS}, nodes] or "
        f"[{N_STATE_FIELDS}, 1, nodes], got {shape}"
    )


def _source_node_count(handle: h5py.File, layout: str, what: str) -> tuple[int, int]:
    if "S_used" in handle:
        dataset = "S_used"
    elif "source" in handle and "S_used" in handle["source"]:
        dataset = "source/S_used"
    elif "source" in handle and "values" in handle["source"]:
        dataset = "source/values"
    else:
        raise ContractError(f"{what}: missing 'S_used'")
    shape = handle[dataset].shape
    if layout == "full_dof":
        if len(shape) != 4 or shape[0] < N_SOURCE_FIELDS or shape[2] != 1:
            raise ContractError(
                f"{what}: full_dof '{dataset}' must be [>={N_SOURCE_FIELDS}, dof, 1, nodes], "
                f"got {shape}"
            )
        return int(shape[-1]), int(shape[0])
    if (len(shape) == 2 and shape[0] >= N_SOURCE_FIELDS) or (
        len(shape) == 3 and shape[0] >= N_SOURCE_FIELDS and shape[1] == 1
    ) or (
        len(shape) == 4
        and shape[0] >= N_SOURCE_FIELDS
        and shape[1] == 1
        and shape[2] == 1
    ):
        return int(shape[-1]), int(shape[0])
    raise ContractError(
        f"{what}: node_dof0 '{dataset}' must be [{N_SOURCE_FIELDS}, nodes] or "
        f"[{N_SOURCE_FIELDS}, 1, nodes], got {shape}"
    )


def _read_source_values(
    handle: h5py.File, layout: str, n_nodes: int | None, what: str
) -> np.ndarray:
    if "S_used" in handle:
        dataset = "S_used"
    elif "source" in handle and "S_used" in handle["source"]:
        dataset = "source/S_used"
    elif "source" in handle and "values" in handle["source"]:
        dataset = "source/values"
    else:
        raise ContractError(f"{what}: missing 'S_used'")
    raw = handle[dataset]
    shape = raw.shape
    if layout == "full_dof":
        if len(shape) != 4 or shape[0] < N_SOURCE_FIELDS or shape[2] != 1:
            raise ContractError(
                f"{what}: full_dof '{dataset}' must be [>={N_SOURCE_FIELDS}, dof, 1, nodes], "
                f"got {shape}"
            )
        values = np.asarray(raw[:N_SOURCE_FIELDS, 0, 0, :], dtype=np.float64).T
    else:
        if len(shape) == 2 and shape[0] >= N_SOURCE_FIELDS:
            values = np.asarray(raw[:N_SOURCE_FIELDS, :], dtype=np.float64).T
        elif len(shape) == 3 and shape[0] >= N_SOURCE_FIELDS and shape[1] == 1:
            values = np.asarray(raw[:N_SOURCE_FIELDS, 0, :], dtype=np.float64).T
        elif (
            len(shape) == 4
            and shape[0] >= N_SOURCE_FIELDS
            and shape[1] == 1
            and shape[2] == 1
        ):
            values = np.asarray(raw[:N_SOURCE_FIELDS, 0, 0, :], dtype=np.float64).T
        else:
            raise ContractError(
                f"{what}: node_dof0 '{dataset}' must be [{N_SOURCE_FIELDS}, nodes] or "
                f"[{N_SOURCE_FIELDS}, 1, nodes], got {shape}"
            )
    if n_nodes is not None and values.shape[0] != n_nodes:
        raise ContractError(
            f"{what}: node count {values.shape[0]} does not match manifest n_nodes {n_nodes}"
        )
    return values


def _read_shard(
    handle: h5py.File, side: str, index: int, what: str
) -> tuple[dict[str, np.ndarray], int]:
    group_name = f"shards/{side}"
    if group_name not in handle:
        raise ContractError(f"{what}: missing shard group '/{group_name}' (A' main path)")
    group = handle[group_name]
    n_total_key = f"shards/n_spi_tot_{side}"
    n_total = _read_int_scalar(handle, n_total_key, what=what) if n_total_key in handle else None
    arrays: dict[str, np.ndarray] = {}
    for name in SHARD_FIELD_NAMES:
        if name not in group:
            raise ContractError(f"{what}: missing '/{group_name}/{name}' (A' main path)")
        array = np.asarray(group[name][()], dtype=np.float64)
        if array.ndim != 1:
            raise ContractError(
                f"{what}: '/{group_name}/{name}' must be 1-D [n_spi], got {array.shape}"
            )
        if n_total is not None and array.shape[0] != n_total:
            raise ContractError(
                f"{what}: '/{group_name}/{name}' length {array.shape[0]} != "
                f"n_spi_tot_{side}={n_total}"
            )
        arrays[name] = array
    if n_total is None:
        n_total = len(next(iter(arrays.values())))
    return arrays, int(n_total)


def _shard_shape(handle: h5py.File, side: str, what: str) -> int:
    """Validate a shard group from metadata only and return ``n_spi_tot``."""
    group_name = f"shards/{side}"
    if group_name not in handle:
        raise ContractError(f"{what}: missing shard group '/{group_name}' (A' main path)")
    group = handle[group_name]
    n_total_key = f"shards/n_spi_tot_{side}"
    n_total = _read_int_scalar(handle, n_total_key, what=what) if n_total_key in handle else None
    for name in SHARD_FIELD_NAMES:
        if name not in group:
            raise ContractError(f"{what}: missing '/{group_name}/{name}' (A' main path)")
        shape = group[name].shape
        if len(shape) != 1:
            raise ContractError(
                f"{what}: '/{group_name}/{name}' must be 1-D [n_spi], got {shape}"
            )
        if n_total is not None and shape[0] != n_total:
            raise ContractError(
                f"{what}: '/{group_name}/{name}' length {shape[0]} != "
                f"n_spi_tot_{side}={n_total}"
            )
    if n_total is None:
        n_total = int(group[SHARD_FIELD_NAMES[0]].shape[0])
    return int(n_total)


def _read_condition(
    handle: h5py.File, what: str, *, required: bool
) -> "ConditionVector | None":
    if "condition" not in handle:
        if required:
            raise ContractError(f"{what}: missing '/condition' group")
        return None
    group = handle["condition"]
    if "names" not in group or "values" not in group:
        raise ContractError(f"{what}: /condition requires 'names' and 'values'")
    names = _decode_strings(group["names"][()])
    values = np.asarray(group["values"][()], dtype=np.float64).ravel()
    if len(names) != values.size:
        raise ContractError(
            f"{what}: /condition names ({len(names)}) and values ({values.size}) disagree"
        )
    if "groups" not in group:
        raise ContractError(f"{what}: /condition is missing 'groups' (FiLM group labels)")
    groups = _decode_strings(group["groups"][()])
    if len(groups) != len(names):
        raise ContractError(
            f"{what}: /condition groups ({len(groups)}) and names ({len(names)}) disagree"
        )
    illegal = sorted({g for g in groups if g not in ALLOWED_CONDITION_GROUPS})
    if illegal:
        raise ContractError(
            f"{what}: /condition groups outside the frozen vocabulary: {illegal}"
        )
    units = _decode_strings(group["units"][()]) if "units" in group else ()
    provenance = _decode_strings(group["provenance"][()]) if "provenance" in group else ()
    if "film_eligible" in group:
        film = np.asarray(group["film_eligible"][()], dtype=np.int64).ravel()
    else:
        film = np.zeros(len(names), dtype=np.int64)
    if film.size != len(names):
        raise ContractError(
            f"{what}: /condition film_eligible ({film.size}) and names ({len(names)}) disagree"
        )
    bad = [
        name
        for name, group_name, flag in zip(names, groups, film)
        if group_name in FILM_FORBIDDEN_GROUPS and flag != 0
    ]
    if bad:
        raise ContractError(
            f"{what}: excluded condition entries must not be FiLM-eligible: {bad}"
        )
    if units and len(units) != len(names):
        raise ContractError(f"{what}: /condition units/names length mismatch")
    if provenance and len(provenance) != len(names):
        raise ContractError(f"{what}: /condition provenance/names length mismatch")
    return ConditionVector(
        names=names,
        values=values,
        units=units,
        groups=groups,
        provenance=provenance,
        film_eligible=film,
    )


@dataclass(frozen=True)
class ConditionVector:
    """One exported condition vector with its frozen group labels."""

    names: tuple[str, ...]
    values: np.ndarray
    units: tuple[str, ...]
    groups: tuple[str, ...]
    provenance: tuple[str, ...]
    film_eligible: np.ndarray

    @property
    def group_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for group in self.groups:
            counts[group] = counts.get(group, 0) + 1
        return counts


@dataclass(frozen=True)
class ShardState:
    """One shard (``type_SPI``) state at the time recorded in ``time_jorek``."""

    index: int
    time_jorek: float
    n_total: int
    arrays: Mapping[str, np.ndarray]

    def __getitem__(self, name: str) -> np.ndarray:
        return self.arrays[name]


@dataclass(frozen=True)
class StepMeta:
    """Cheap per-step metadata collected while building the index."""

    index: int
    path: Path
    cadence: str
    substeps: int
    source_object: str
    t_interval_start: float
    t_interval_end: float
    t_source_start: float
    t_source_end: float
    t_norm: float
    n_nodes: int | None
    n_spi_start: int
    n_spi_end: int
    has_mesh: bool
    has_condition: bool
    budget_layout: str
    #: ``export_v3`` (one self-described file per window) or ``native`` (raw
    #: ``jorek*/projections*/part*`` run read one fluid step at a time).
    layout_kind: str = "export_v3"
    #: Native layout only: the successor frame path (``u_after``) and the source
    #: frame path (``projections{k}``); ``None`` in the export layout.
    path_after: Path | None = None
    path_source: Path | None = None
    #: Native layout only: recorded successor index (``u_after`` frame).
    index_next: int | None = None

    @property
    def dt_step(self) -> float:
        return self.t_interval_end - self.t_interval_start

    @property
    def dt_source(self) -> float:
        return self.t_source_end - self.t_source_start


@dataclass(frozen=True)
class StepSample:
    """One A' training pair ``(u_k, p_{k-1}, S_k) -> u_{k+1}``.

    ``shard_start``/``shard_end`` are the source-window endpoints
    ``[T_{k-1}, T_k]`` in the self-described ``export_v3`` layout.  The native
    derived layout stores the *field-interval* endpoints ``T_k``/``T_{k+1}``
    (true frame times); there the window-start state is the previous record's
    ``shard_start``.  ``meta.shard_time_semantics`` / ``report.shard_time_semantics``
    register which convention applies.
    """

    index: int
    u_before: np.ndarray
    source: np.ndarray
    u_after: np.ndarray
    shard_start: ShardState
    shard_end: ShardState
    t_before: float
    t_after: float
    t_source_start: float
    t_source_end: float
    t_norm: float
    dt_step: float
    dt_source: float
    meta: StepMeta
    condition: ConditionVector | None = None

    @property
    def source_window(self) -> tuple[float, float]:
        return (self.t_source_start, self.t_source_end)

    @property
    def field_interval(self) -> tuple[float, float]:
        return (self.t_before, self.t_after)


@dataclass(frozen=True)
class StepContractReport:
    """Result of the A' contract checks for one export run."""

    run: str
    export_root: Path
    layout: str
    cadence: str
    source_object: str
    n_steps: int
    indices: tuple[int, ...]
    pair_indices: tuple[int, ...]
    dt_step_min: float | None
    dt_step_median: float | None
    dt_step_max: float | None
    dt_source_min: float | None
    dt_source_median: float | None
    dt_source_max: float | None
    require_single_step: bool
    single_step_tol: float
    dropped_restart: tuple[int, ...]
    gaps: tuple[int, ...]
    issues: tuple[TimingIssue, ...]
    step_table_source: str
    budget_layout: str
    #: ``export_v3`` or ``native`` (see ``StepMeta.layout_kind``).
    layout_kind: str = "export_v3"
    #: How the single-fluid-step gate is evaluated (registered in the report).
    single_step_method: str = "dt_step==dt_source"
    #: Where the interval metadata comes from (manifest times or recorded t_now).
    interval_source: str = "export_metadata"
    #: Zero-dt policy actually used (``error`` / ``list``).
    zero_dt_policy: str = "registered"
    zero_dt_indices: tuple[int, ...] = ()
    zero_dt_pairs: tuple[int, ...] = ()
    #: Registered restart events (native layout; ADR-0019 addendum/ADR-0020 §5).
    restart_events_registered: tuple[int, ...] = ()
    restart_events_source: str | None = None
    restart_cross_boundary_skipped: tuple[int, ...] = ()
    restart_lagging_projection_skipped: tuple[int, ...] = ()
    #: Records right after a registered restart event: their source window is
    #: degenerate by construction; S_k = projections{k} stays usable.
    restart_degenerate_source_windows: tuple[int, ...] = ()
    #: Records kept in ``list`` mode whose source window is zero-length (left by
    #: the restart duplicate); registered, never silently dropped.
    zero_dt_source_windows: tuple[int, ...] = ()
    restart_registry_size: int = 0
    restart_events: tuple[int, ...] = ()
    #: ``source_window_endpoints`` (export_v3) or ``field_interval_endpoints``
    #: (native derived layout; see ``src/data/native_run.py``).
    shard_time_semantics: str = "source_window_endpoints"
    boundary_frames_leading: tuple[int, ...] = ()
    boundary_frames_trailing: tuple[int, ...] = ()
    n_frame_files: int = 0
    mesh_verifications: int = 0
    #: Number of ``(u_before, S_used, u_after)`` windows (native records are
    #: windows themselves; ``None`` falls back to ``len(pair_indices)``).
    n_windows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "export_root": str(self.export_root),
            "layout_kind": self.layout_kind,
            "layout": self.layout,
            "cadence": self.cadence,
            "source_object": self.source_object,
            "n_steps": self.n_steps,
            "n_windows": self.n_windows
            if self.n_windows is not None
            else len(self.pair_indices),
            "indices": list(self.indices),
            "pair_indices": list(self.pair_indices),
            "dt_step": {
                "min": self.dt_step_min,
                "median": self.dt_step_median,
                "max": self.dt_step_max,
            },
            "dt_source": {
                "min": self.dt_source_min,
                "median": self.dt_source_median,
                "max": self.dt_source_max,
            },
            "require_single_step": self.require_single_step,
            "single_step_tol": self.single_step_tol,
            "single_step_method": self.single_step_method,
            "dropped_restart": list(self.dropped_restart),
            "gaps": list(self.gaps),
            "issues": [issue.__dict__ for issue in self.issues],
            "step_table_source": self.step_table_source,
            "interval_source": self.interval_source,
            "budget_layout": self.budget_layout,
            "zero_dt": {
                "policy": self.zero_dt_policy,
                "indices": list(self.zero_dt_indices),
                "affected_pairs": list(self.zero_dt_pairs),
                "zero_length_source_windows": list(self.zero_dt_source_windows),
            },
            "restart": {
                "registry_size": self.restart_registry_size,
                "events": list(self.restart_events),
                "registered_events": list(self.restart_events_registered),
                "registered_events_source": self.restart_events_source,
                "cross_boundary_skipped": list(self.restart_cross_boundary_skipped),
                "lagging_projection_skipped": list(
                    self.restart_lagging_projection_skipped
                ),
                "degenerate_source_windows": list(
                    self.restart_degenerate_source_windows
                ),
            },
            "shard_time_semantics": self.shard_time_semantics,
            "boundary_frames": {
                "leading": list(self.boundary_frames_leading),
                "trailing": list(self.boundary_frames_trailing),
            },
            "n_frame_files": self.n_frame_files,
            "mesh_verifications": self.mesh_verifications,
        }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


class _ExportV3Run:
    """Index + reader for one A' fine-grained ``export_v3`` run.

    Parameters
    ----------
    root:
        Either the ``export_v3`` directory itself or the run directory that
        contains it.
    raw_run_dir:
        Optional directory with the recorded ``jorek*/projections*/part*``
        files, used to cross-check the export against the ADR-0019 timing
        contract (projection/part time equality, restart registration).
    require_single_step:
        Enforce ``dt_step == dt_source`` (within ``single_step_tol``).  Phase
        B/C training must set this to ``True`` so frame-level (multi-step)
        records can never enter a one-fluid-step propagator by accident.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        raw_run_dir: str | Path | None = None,
        restart_indices: Sequence[int] | None = None,
        policy: str = "drop",
        require_single_step: bool = False,
        single_step_tol: float | None = None,
        index_window: tuple[int, int] | None = None,
        condition_required: bool = False,
        extra_boundary_codes: Iterable[int] = (),
    ) -> None:
        if policy not in {"drop", "error"}:
            raise ValueError(f"unknown restart policy {policy!r}; use 'drop' or 'error'")
        self.export_root = resolve_export_root(root)
        self.raw_run_dir = Path(raw_run_dir) if raw_run_dir is not None else None
        self.policy = policy
        self.require_single_step = bool(require_single_step)
        self._manifest = self._load_manifest()
        self._single_step_tol = float(
            single_step_tol
            if single_step_tol is not None
            else self._manifest.get("single_step_tolerance_jorek", SINGLE_STEP_TOL_JOREK)
        )
        if self._single_step_tol <= 0.0:
            raise ContractError("single_step_tolerance_jorek must be positive")
        self.layout: str = self._manifest["layout"]
        self.cadence: str = self._manifest["cadence"]
        self.source_object: str = self._manifest["source_object"]
        self.budget_layout: str = str(self._manifest.get("budget_layout", "absent"))
        if self.budget_layout not in BUDGET_LAYOUTS:
            raise ContractError(
                f"manifest budget_layout must be one of {BUDGET_LAYOUTS}, "
                f"got {self.budget_layout!r}"
            )
        self._restart = self._resolve_restart_indices(restart_indices)
        self._gaps = {int(index) for index in self._manifest.get("gap_indices", ())}
        self._n_nodes_hint = (
            int(self._manifest["n_nodes"]) if "n_nodes" in self._manifest else None
        )
        self._extra_boundary_codes = {int(code) for code in extra_boundary_codes}
        self._mesh_file = self._resolve_mesh_file()
        self._condition_required = bool(condition_required)
        self._issues: list[TimingIssue] = []
        self._dropped: list[int] = []
        self._step_paths = self._collect_step_files(index_window)
        self._metas: tuple[StepMeta, ...] = ()
        self._check_index()
        self.report: StepContractReport = self._build_report()

    # ------------------------------------------------------------------ setup

    def _load_manifest(self) -> dict[str, Any]:
        path = self.export_root / MANIFEST_NAME
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ContractError(f"{path}: cannot read export manifest ({exc})") from exc
        except json.JSONDecodeError as exc:
            raise ContractError(f"{path}: invalid JSON ({exc})") from exc
        if not isinstance(manifest, dict):
            raise ContractError(f"{path}: manifest must be a JSON object")
        for key in ("layout", "cadence", "source_object"):
            if key not in manifest:
                raise ContractError(f"{path}: manifest is missing required key {key!r}")
        if manifest["layout"] not in LAYOUTS:
            raise ContractError(
                f"{path}: layout must be one of {LAYOUTS}, got {manifest['layout']!r}"
            )
        if manifest["cadence"] not in CADENCES:
            raise ContractError(
                f"{path}: cadence must be one of {CADENCES}, got {manifest['cadence']!r}"
            )
        if manifest["source_object"] not in SOURCE_OBJECTS:
            raise ContractError(
                f"{path}: source_object must be one of {SOURCE_OBJECTS}, "
                f"got {manifest['source_object']!r}"
            )
        return manifest

    def _resolve_mesh_file(self) -> Path | None:
        name = self._manifest.get("mesh_file")
        if name is None:
            return None
        path = self.export_root / str(name)
        if not path.is_file():
            raise ContractError(f"manifest mesh_file {name!r} not found under {self.export_root}")
        return path

    def _resolve_restart_indices(self, explicit: Sequence[int] | None) -> set[int]:
        if explicit is not None:
            return {int(index) for index in explicit}
        manifest_restart = self._manifest.get("restart_indices", ())
        return {int(index) for index in manifest_restart}

    def _collect_step_files(
        self, index_window: tuple[int, int] | None
    ) -> dict[int, Path]:
        steps_dir = self.export_root / STEPS_DIRNAME
        if not steps_dir.is_dir():
            raise ContractError(f"{steps_dir} does not exist")
        found: dict[int, Path] = {}
        for path in steps_dir.iterdir():
            match = STEP_FILE_RE.fullmatch(path.name)
            if match is not None:
                found[int(match.group("index"))] = path
        if not found:
            raise ContractError(f"{steps_dir} contains no step_*.h5 files")
        if index_window is not None:
            first, last = index_window
            if last < first:
                raise ValueError(f"index_window must be ordered, got {index_window!r}")
            found = {index: path for index, path in found.items() if first <= index <= last}
            if not found:
                raise ContractError(f"{steps_dir}: no steps inside window {index_window!r}")
        return found

    # ------------------------------------------------------------- validation

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    @property
    def issues(self) -> tuple[TimingIssue, ...]:
        return tuple(self._issues)

    @property
    def dropped_restart(self) -> tuple[int, ...]:
        return tuple(self._dropped)

    def _check_index(self) -> None:
        order = sorted(self._step_paths)
        metas: list[StepMeta] = []
        for index in order:
            if index in self._gaps:
                self._note(
                    "registered_gap",
                    index,
                    "index carries a step file but is registered as a gap; skipped",
                )
                continue
            path = self._step_paths[index]
            meta = self._read_meta(index, path)
            metas.append(meta)
        for gap in sorted(self._gaps):
            if gap not in self._step_paths:
                self._note(
                    "registered_gap",
                    gap,
                    "index is registered as a gap in the manifest and has no step file",
                )
        if not metas:
            raise ContractError(f"{self.export_root}: no usable steps after gap registration")
        for meta in metas:
            self._check_single_file(meta)
        assert_strictly_increasing(
            [meta.t_interval_start for meta in metas],
            what=f"{self.export_root.name}: interval_start",
        )
        for previous, current in zip(metas, metas[1:]):
            dt_step = current.t_interval_start - previous.t_interval_start
            if dt_step <= 0.0:  # pragma: no cover - strictly increasing already
                raise ContractError(f"index {current.index}: non-positive dt_step {dt_step!r}")
            if abs(previous.t_interval_end - current.t_interval_start) <= TIME_TOL_JOREK:
                continue
            if current.index - previous.index == 1:
                raise ContractError(
                    f"index {current.index}: previous interval_end "
                    f"{previous.t_interval_end!r} != interval_start "
                    f"{current.t_interval_start!r} (chain break; register gaps)"
                )
            missing = [
                index
                for index in range(previous.index + 1, current.index)
                if index not in self._gaps
            ]
            if missing:
                raise ContractError(
                    f"index {current.index}: interval times do not chain and no gap is "
                    f"registered for {missing} (register gap_indices)"
                )
            self._note(
                "registered_gap",
                current.index,
                f"registered gap between {previous.index} and {current.index}; "
                "interval times deliberately do not chain",
            )
        self._metas = tuple(metas)
        dropped = self._cross_check_raw()
        if dropped:
            self._dropped = sorted(set(self._dropped) | set(dropped))
            self._metas = tuple(meta for meta in self._metas if meta.index not in self._dropped)

    def _note(self, kind: str, index: int, detail: str, *, aggregate: bool = False) -> None:
        """Append an issue; ``aggregate=True`` keeps only the first of a kind."""
        if aggregate:
            for issue in self._issues:
                if issue.kind == kind:
                    return
        self._issues.append(TimingIssue(kind=kind, index=index, detail=detail))

    def _read_meta(self, index: int, path: Path) -> StepMeta:
        what = path.name
        with h5py.File(path, "r") as handle:
            step_index = None
            for attr in ("@step_index", "step_index"):
                if attr in handle.attrs:
                    step_index = int(np.ravel(handle.attrs[attr])[0])
                    break
            if step_index is not None and step_index != index:
                raise ContractError(
                    f"{what}: @step_index {step_index} does not match filename index {index}"
                )
            time_group = "time"
            if time_group not in handle:
                raise ContractError(f"{what}: missing '/time' group")
            t_group = handle[time_group]

            def t(key: str) -> float:
                if key not in t_group:
                    raise ContractError(f"{what}: missing '/time/{key}'")
                return float(np.ravel(t_group[key][()])[0])

            t_start = t("interval_start_jorek")
            t_end = t("interval_end_jorek")
            s_start = t("source_interval_start_jorek")
            s_end = t("source_interval_end_jorek")
            t_norm = t("t_norm")
            if t_norm <= 0.0:
                raise ContractError(f"{what}: /time/t_norm must be positive, got {t_norm}")
            for key, jorek in (
                ("interval_start_seconds", t_start),
                ("interval_end_seconds", t_end),
                ("source_interval_start_seconds", s_start),
                ("source_interval_end_seconds", s_end),
            ):
                if key not in t_group:
                    self._note(
                        "missing_seconds_timestamp",
                        index,
                        f"/time/{key} is missing (ADR-0019 requires both units); "
                        f"first seen at step {index}",
                        aggregate=True,
                    )
                    continue
                recorded = float(np.ravel(t_group[key][()])[0])
                if abs(recorded - jorek * t_norm) > max(1e-12, abs(jorek * t_norm) * 1e-6):
                    raise ContractError(
                        f"{what}: /time/{key}={recorded!r} != jorek*t_norm="
                        f"{jorek * t_norm!r}"
                    )
            substeps = _read_int_scalar(t_group, "substeps", what=what) if "substeps" in t_group else 1
            cadence = self._decode_attr(handle, "@cadence") or self._decode_attr(handle, "cadence") or self.cadence
            if cadence not in CADENCES:
                raise ContractError(f"{what}: @cadence {cadence!r} is not registered")
            if cadence != self.cadence:
                raise ContractError(
                    f"{what}: @cadence {cadence!r} disagrees with manifest "
                    f"cadence {self.cadence!r}"
                )
            state_object = self._read_string_dataset(handle, "provenance/source_object")
            if state_object and state_object != self.source_object:
                raise ContractError(
                    f"{what}: provenance/source_object {state_object!r} disagrees with "
                    f"manifest {self.source_object!r}"
                )
            # Metadata-only shape checks; arrays are read lazily in __getitem__.
            if "state" not in handle or "u_before" not in handle["state"]:
                raise ContractError(f"{what}: missing '/state/u_before'")
            if "u_after" not in handle["state"]:
                raise ContractError(f"{what}: missing '/state/u_after'")
            state_nodes, _ = _state_node_count(
                handle["state"], self.layout, f"{what}:/state/u_before", "u_before"
            )
            after_nodes, _ = _state_node_count(
                handle["state"], self.layout, f"{what}:/state/u_after", "u_after"
            )
            if state_nodes != after_nodes:
                raise ContractError(
                    f"{what}: u_before nodes {state_nodes} != u_after nodes {after_nodes}"
                )
            source_nodes, _ = _source_node_count(handle, self.layout, f"{what}:/source")
            if source_nodes != state_nodes:
                raise ContractError(
                    f"{what}: source nodes {source_nodes} != state nodes {state_nodes}"
                )
            has_mesh = (
                ("mesh" in handle and "boundary" in handle["mesh"])
                or "boundary" in handle
                or self._mesh_file is not None
            )
            has_condition = "condition" in handle
            n_spi_start = _shard_shape(handle, "start", what)
            n_spi_end = _shard_shape(handle, "end", what)
            if self.budget_layout != "absent":
                length = self._budget_series_length(handle)
                if length is not None:
                    if self.budget_layout == "slice" and length != 1:
                        raise ContractError(
                            f"{what}: manifest budget_layout='slice' but the step file "
                            f"carries {length} records"
                        )
                    if (
                        self.budget_layout == "history"
                        and length == 1
                        and len(self._step_paths) > 1
                    ):
                        raise ContractError(
                            f"{what}: manifest budget_layout='history' but the step file "
                            "carries a single-record slice"
                        )
        if self._n_nodes_hint is None:
            self._n_nodes_hint = int(state_nodes)
        elif state_nodes != self._n_nodes_hint:
            raise ContractError(
                f"{what}: node count {state_nodes} != manifest n_nodes {self._n_nodes_hint}"
            )
        return StepMeta(
            index=index,
            path=path,
            cadence=cadence,
            substeps=int(substeps),
            source_object=state_object or self.source_object,
            t_interval_start=t_start,
            t_interval_end=t_end,
            t_source_start=s_start,
            t_source_end=s_end,
            t_norm=t_norm,
            n_nodes=int(state_nodes),
            n_spi_start=n_spi_start,
            n_spi_end=n_spi_end,
            has_mesh=has_mesh,
            has_condition=has_condition,
            budget_layout=self.budget_layout,
        )

    @staticmethod
    def _decode_attr(handle: h5py.File, name: str) -> str | None:
        if name not in handle.attrs:
            return None
        value = handle.attrs[name]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)

    @staticmethod
    def _read_string_dataset(handle: h5py.File, key: str) -> str | None:
        if key not in handle:
            return None
        return _decode_strings(handle[key][()])[0]

    @staticmethod
    def _budget_series_length(handle: h5py.File) -> int | None:
        for name in ("xtime", "xtime_spi_ablation", "xtime_Ne_imp"):
            for prefix in ("", "budget/", "xtime/"):
                key = f"{prefix}{name}"
                if key in handle:
                    return int(handle[key].shape[0])
        return None

    def _check_single_file(self, meta: StepMeta) -> None:
        what = meta.path.name
        if not (meta.t_interval_end > meta.t_interval_start):
            raise ContractError(
                f"{what}: interval_end {meta.t_interval_end!r} <= interval_start "
                f"{meta.t_interval_start!r}"
            )
        if not (meta.t_source_end > meta.t_source_start):
            raise ContractError(
                f"{what}: source_interval_end {meta.t_source_end!r} <= "
                f"source_interval_start {meta.t_source_start!r}"
            )
        if abs(meta.t_source_end - meta.t_interval_start) > TIME_TOL_JOREK:
            raise ContractError(
                f"{what}: source window end {meta.t_source_end!r} != field interval "
                f"start {meta.t_interval_start!r}; S_n must drive u_n->u_n+1 (ADR-0019)"
            )
        if meta.cadence == "fluid_step" and meta.substeps != 1:
            raise ContractError(
                f"{what}: cadence='fluid_step' requires /time/substeps == 1, "
                f"got {meta.substeps}"
            )
        if meta.substeps < 1:
            raise ContractError(f"{what}: /time/substeps must be >= 1, got {meta.substeps}")
        if self.require_single_step:
            delta = meta.dt_step - meta.dt_source
            if abs(delta) > self._single_step_tol:
                raise ContractError(
                    f"{what}: require_single_step=True but dt_step {meta.dt_step!r} != "
                    f"dt_source {meta.dt_source!r} (delta {delta!r} > "
                    f"tol {self._single_step_tol!r}); frame-level data cannot enter "
                    "Phase B/C"
                )
        with h5py.File(meta.path, "r") as handle:
            codes = self._boundary_codes(handle, what)
            if codes is None:
                self._note(
                    "boundary_missing",
                    meta.index,
                    "no /mesh/boundary found (per-step or static mesh); boundary "
                    f"diagnostics registered as a warning (user-side), first at step {meta.index}",
                    aggregate=True,
                )
            else:
                registered = REGISTERED_BOUNDARY_CODES | self._extra_boundary_codes
                unknown = sorted({int(code) for code in np.unique(codes)} - registered)
                if unknown:
                    raise ContractError(
                        f"{what}: unregistered boundary codes {unknown}; register them in "
                        "the loader and B0 report before use"
                    )
            condition = _read_condition(handle, what, required=self._condition_required)
            if condition is None:
                self._note(
                    "condition_missing",
                    meta.index,
                    "no /condition group exported; FiLM conditioning unavailable "
                    f"(first seen at step {meta.index})",
                    aggregate=True,
                )

    def _boundary_codes(self, handle: h5py.File, what: str) -> np.ndarray | None:
        if "mesh" in handle and "boundary" in handle["mesh"]:
            return np.asarray(handle["mesh/boundary"][()], dtype=np.int64)
        if "boundary" in handle:
            return np.asarray(handle["boundary"][()], dtype=np.int64)
        if self._mesh_file is not None:
            with h5py.File(self._mesh_file, "r") as mesh:
                if "mesh" in mesh and "boundary" in mesh["mesh"]:
                    return np.asarray(mesh["mesh/boundary"][()], dtype=np.int64)
                if "boundary" in mesh:
                    return np.asarray(mesh["boundary"][()], dtype=np.int64)
        return None

    def _cross_check_raw(self) -> list[int]:
        """Check the export against the raw recorded triple when available."""
        if self.raw_run_dir is None:
            return []
        raw = self.raw_run_dir
        table, table_source = resolve_step_times(raw, None)
        dropped: list[int] = []
        for meta in self._metas:
            index = meta.index
            state_path = raw / f"jorek{index:06d}.h5"
            projection_path = raw / f"projections{index:06d}.h5"
            particle_path = raw / f"part{index:05d}.h5"
            if table is not None:
                if index not in table:
                    raise ContractError(
                        f"index {index}: raw step table ({table_source}) has no entry"
                    )
                if abs(table[index] - meta.t_interval_start) > TIME_TOL_JOREK:
                    raise ContractError(
                        f"index {index}: export interval_start {meta.t_interval_start!r} != "
                        f"raw T_n {table[index]!r} ({table_source})"
                    )
            if state_path.exists():
                with h5py.File(state_path, "r") as handle:
                    raw_t = float(np.ravel(handle["t_now"][()])[0])
                if abs(raw_t - meta.t_interval_start) > TIME_TOL_JOREK:
                    raise ContractError(
                        f"index {index}: export interval_start != raw jorek t_now "
                        f"({meta.t_interval_start!r} vs {raw_t!r})"
                    )
            if not projection_path.exists() and not particle_path.exists():
                continue
            is_restart = index in self._restart
            if projection_path.exists():
                with h5py.File(projection_path, "r") as handle:
                    t_projection = float(np.ravel(handle["t_now"][()])[0]) / meta.t_norm
                offset = t_projection - meta.t_interval_start
                if abs(offset) > TIME_TOL_JOREK:
                    if not is_restart:
                        raise ContractError(
                            f"index {index}: raw projections t_now is offset {offset!r} "
                            "JOREK units from the export; only registered restart "
                            "boundaries may be offset"
                        )
                    expected = -meta.dt_source
                    if abs(offset - expected) > TIME_TOL_JOREK:
                        raise ContractError(
                            f"index {index}: registered restart offset {offset!r} does not "
                            f"match -dt_source {expected!r}"
                        )
                    if self.policy == "error":
                        raise ContractError(
                            f"index {index}: registered restart boundary is offset by "
                            f"{offset!r} and policy='error'"
                        )
                    self._issues.append(
                        TimingIssue(
                            kind="restart_boundary_projection_offset",
                            index=index,
                            detail=(
                                f"registered restart: raw projections is {offset!r} JOREK "
                                "units early; record dropped"
                            ),
                        )
                    )
                    dropped.append(index)
                    continue
                if is_restart:
                    self._issues.append(
                        TimingIssue(
                            kind="restart_registry_stale",
                            index=index,
                            detail=(
                                "index is registered as a restart boundary but the raw "
                                "projections matches the same-index frame"
                            ),
                        )
                    )
            if particle_path.exists():
                with h5py.File(particle_path, "r") as handle:
                    t_particle = float(np.ravel(handle["time"][()])[0]) / meta.t_norm
                if abs(t_particle - meta.t_source_start) > TIME_TOL_JOREK:
                    raise ContractError(
                        f"index {index}: raw part time {t_particle!r} != source_interval_start "
                        f"{meta.t_source_start!r} (part{{n}} = p_n-1, ADR-0019)"
                    )
        self._issues.append(
            TimingIssue(
                kind="raw_cross_check",
                index=-1,
                detail=f"raw run cross-check against {raw} ({table_source}); "
                f"{len(self._metas)} steps checked",
            )
        )
        return dropped

    def _build_report(self) -> StepContractReport:
        metas = self._metas
        dt_steps = [meta.dt_step for meta in metas]
        dt_sources = [meta.dt_source for meta in metas]
        pair_indices = tuple(
            meta.index
            for meta, following in zip(metas, metas[1:])
            if following.index == meta.index + 1
        )
        table_source = "not_checked"
        if self.raw_run_dir is not None:
            _, table_source = resolve_step_times(self.raw_run_dir, None)
        return StepContractReport(
            run=str(self._manifest.get("run", self.export_root.parent.name)),
            export_root=self.export_root,
            layout=self.layout,
            cadence=self.cadence,
            source_object=self.source_object,
            n_steps=len(metas),
            indices=tuple(meta.index for meta in metas),
            pair_indices=pair_indices,
            dt_step_min=min(dt_steps) if dt_steps else None,
            dt_step_median=_median(dt_steps),
            dt_step_max=max(dt_steps) if dt_steps else None,
            dt_source_min=min(dt_sources) if dt_sources else None,
            dt_source_median=_median(dt_sources),
            dt_source_max=max(dt_sources) if dt_sources else None,
            require_single_step=self.require_single_step,
            single_step_tol=self._single_step_tol,
            dropped_restart=tuple(self._dropped),
            gaps=tuple(sorted(self._gaps)),
            issues=tuple(self._issues),
            step_table_source=table_source,
            budget_layout=self.budget_layout,
            layout_kind="export_v3",
            single_step_method="dt_step==dt_source",
            interval_source="export_metadata",
            restart_registry_size=len(self._restart),
            restart_events=tuple(sorted(self._restart)),
            n_frame_files=len(metas),
        )

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self._metas)

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(meta.index for meta in self._metas)

    @property
    def n_nodes(self) -> int | None:
        return self._n_nodes_hint

    @property
    def metas(self) -> tuple[StepMeta, ...]:
        return self._metas

    def pair_indices(self) -> tuple[int, ...]:
        return self.report.pair_indices

    def step_path(self, index: int) -> Path:
        for meta in self._metas:
            if meta.index == index:
                return meta.path
        raise KeyError(index)

    def read_condition(self, position: int) -> ConditionVector | None:
        meta = self._metas[position]
        with h5py.File(meta.path, "r") as handle:
            return _read_condition(handle, meta.path.name, required=self._condition_required)

    def read_boundary(self, position: int) -> np.ndarray | None:
        meta = self._metas[position]
        with h5py.File(meta.path, "r") as handle:
            return self._boundary_codes(handle, meta.path.name)

    def __getitem__(self, position: int) -> StepSample:
        meta = self._metas[position]
        what = meta.path.name
        with h5py.File(meta.path, "r") as handle:
            u_before = _read_state_values(
                handle["state"], self.layout, meta.n_nodes, f"{what}:/state/u_before", "u_before"
            ) if "u_before" in handle["state"] else None
            u_after = _read_state_values(
                handle["state"], self.layout, meta.n_nodes, f"{what}:/state/u_after", "u_after"
            ) if "u_after" in handle["state"] else None
            if u_before is None or u_after is None:
                raise ContractError(f"{what}: /state requires both 'u_before' and 'u_after'")
            source = _read_source_values(handle, self.layout, meta.n_nodes, f"{what}:/source")
            start_arrays, n_start = _read_shard(handle, "start", meta.index, what)
            end_arrays, n_end = _read_shard(handle, "end", meta.index, what)
            condition = _read_condition(handle, what, required=self._condition_required)
        return StepSample(
            index=meta.index,
            u_before=u_before,
            source=source,
            u_after=u_after,
            shard_start=ShardState(
                index=meta.index, time_jorek=meta.t_source_start, n_total=n_start, arrays=start_arrays
            ),
            shard_end=ShardState(
                index=meta.index, time_jorek=meta.t_source_end, n_total=n_end, arrays=end_arrays
            ),
            t_before=meta.t_interval_start,
            t_after=meta.t_interval_end,
            t_source_start=meta.t_source_start,
            t_source_end=meta.t_source_end,
            t_norm=meta.t_norm,
            dt_step=meta.dt_step,
            dt_source=meta.dt_source,
            meta=meta,
            condition=condition,
        )

    def check_linkage(self, position: int, *, rtol: float = 1e-9, atol: float = 1e-12) -> None:
        """Assert ``u_after`` of a step equals ``u_before`` of the next step."""
        if position < 0 or position >= len(self._metas) - 1:
            raise IndexError(position)
        current = self[position]
        following = self[position + 1]
        if following.index != current.index + 1:
            raise ContractError(
                f"index {current.index}: cannot check linkage across a registered gap "
                f"to {following.index}"
            )
        if current.u_after.shape != following.u_before.shape:
            raise ContractError(
                f"index {current.index}: u_after {current.u_after.shape} != next "
                f"u_before {following.u_before.shape}"
            )
        if not np.allclose(current.u_after, following.u_before, rtol=rtol, atol=atol):
            max_delta = float(np.max(np.abs(current.u_after - following.u_before)))
            raise ContractError(
                f"index {current.index}: u_after != next u_before (max |delta| "
                f"{max_delta!r}); the same window is not self-consistent across files"
            )


def detect_run_layout(root: str | Path) -> tuple[str, Path]:
    """Detect the A' data layout for ``root``.

    Returns ``(kind, resolved_root)`` where ``kind`` is

    * ``"export_v3"``: ``<root>/export_v3/{manifest.json, steps/step_*.h5}``
      (or ``root`` being the export directory itself);
    * ``"native"``: a raw stepwise run directory holding ``jorek*/projections*/
      part*`` files written one file per fluid step (``NativeRunDataset``).

    Raises :class:`ContractError` when neither layout is present.
    """
    root = Path(root)
    export_root: Path | None = None
    try:
        export_root = resolve_export_root(root)
    except ContractError:
        export_root = None
    if export_root is not None:
        return "export_v3", export_root
    from src.data.native_run import is_native_run

    if is_native_run(root):
        return "native", root
    raise ContractError(
        f"{root}: no A' data layout found (expected either "
        f"{root}/export_v3/manifest.json or a native stepwise run with "
        "jorek*/projections*/part* files)"
    )


class StepRunDataset:
    """Auto-detecting A' step-level dataset (``export_v3`` or native run).

    The class is a thin dispatcher: it detects the layout with
    :func:`detect_run_layout` and delegates to the matching backend
    (:class:`_ExportV3Run` for the self-described export layout,
    :class:`src.data.native_run.NativeRunDataset` for a raw stepwise run).
    Both backends expose the same public surface (``indices``, ``metas``,
    ``report``, ``__getitem__``, ``read_boundary``, ``read_condition``,
    ``check_linkage``), so downstream code and the acceptance CLI never branch
    on the layout themselves.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        raw_run_dir: str | Path | None = None,
        restart_indices: Sequence[int] | None = None,
        restart_events: Sequence[int] | None = None,
        restart_file: str | Path | None = None,
        gap_indices: Sequence[int] | None = None,
        policy: str = "drop",
        require_single_step: bool = False,
        single_step_tol: float | None = None,
        index_window: tuple[int, int] | None = None,
        condition_required: bool = False,
        extra_boundary_codes: Iterable[int] = (),
        zero_dt_policy: str = "registered",
        step_times: Mapping[int, float] | None = None,
        namelist_file: str | Path | None = None,
        mesh_verify_stride: int = 128,
    ) -> None:
        kind, resolved = detect_run_layout(root)
        if kind == "native":
            from src.data.native_run import NativeRunDataset

            self._impl: Any = NativeRunDataset(
                resolved,
                raw_run_dir=raw_run_dir,
                restart_indices=restart_indices,
                restart_events=restart_events,
                restart_file=restart_file,
                gap_indices=gap_indices,
                policy=policy,
                require_single_step=require_single_step,
                single_step_tol=single_step_tol,
                index_window=index_window,
                condition_required=condition_required,
                extra_boundary_codes=extra_boundary_codes,
                zero_dt_policy=zero_dt_policy,
                step_times=step_times,
                namelist_file=namelist_file,
                mesh_verify_stride=mesh_verify_stride,
            )
        else:
            self._impl = _ExportV3Run(
                resolved,
                raw_run_dir=raw_run_dir,
                restart_indices=restart_indices,
                policy=policy,
                require_single_step=require_single_step,
                single_step_tol=single_step_tol,
                index_window=index_window,
                condition_required=condition_required,
                extra_boundary_codes=extra_boundary_codes,
            )

    @property
    def impl(self) -> Any:
        """The layout backend (``_ExportV3Run`` or ``NativeRunDataset``)."""
        return self._impl

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, position: int) -> StepSample:
        return self._impl[position]

    def __getattr__(self, name: str) -> Any:
        impl = self.__dict__.get("_impl")
        if impl is None:
            raise AttributeError(name)
        return getattr(impl, name)


__all__ = [
    "ALLOWED_CONDITION_GROUPS",
    "BUDGET_LAYOUTS",
    "CADENCES",
    "ConditionVector",
    "ContractError",
    "LAYOUTS",
    "MANIFEST_NAME",
    "N_SOURCE_FIELDS",
    "N_STATE_FIELDS",
    "REGISTERED_BOUNDARY_CODES",
    "SHARD_FIELD_NAMES",
    "SINGLE_STEP_TOL_JOREK",
    "SOURCE_FIELD_NAMES",
    "STATE_FIELD_NAMES",
    "ShardState",
    "StepContractReport",
    "StepMeta",
    "StepRunDataset",
    "StepSample",
    "detect_run_layout",
    "resolve_export_root",
]
