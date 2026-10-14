"""Native stepwise run reader for Phase A' v2 (ADR-0019 timing contract).

Layout (b): the raw recorded JOREK output directory itself, i.e.
``<run>/{jorek{n:06d}.h5, projections{n:06d}.h5, part{n:05d}.h5, ...}`` written
one file per fluid step.  No ``export_v3`` files are involved; the per-record
interval metadata is *derived* from the recorded ``t_now`` values
(cross-checked against ``macroscopic_vars.dat``) and the provenance is
registered explicitly.

Pairing (``report/data-contract-v3.md`` §2.3, ADR-0019)::

    u_before = jorek{k}              (T_k)
    S_used   = projections{k}        (source window [T_{k-1}, T_k])
    u_after  = jorek{k+1}            (T_{k+1})
    shard_start/end = spi_*_arr of jorek{k} / jorek{k+1}   (true times T_k / T_{k+1})
    mesh/boundary   = jorek{k} (static, sha256-cached)
    part{k}         = p_{k-1} at T_{k-1} (verified wherever a part frame exists)

Shard timing (verified against ``spi_abl_arr(jorek{k})`` ==
``xtime_spi_ablation_rate[k]``): the arrays in ``jorek{k}`` are the state at
``T_k``; the source-window start ``p_{k-1}`` required by the B1 pairing is the
**previous record's** ``shard_start`` (``spi(jorek{k-1})`` at ``T_{k-1}``).
The old one-frame-early labels are gone (loader semantics fix, ADR-0020 §5).

Hard acceptance rules (fail-fast, no silent drops): strictly increasing
``jorek t_now``, no ``dt <= 0``, ``projections{k}`` aligned with the same-index
``jorek{k}`` (registered restart boundaries follow the ADR-0019 rule),
``part{n}.time == T_{n-1}``, and one-fluid-step records for Phase B/C.

Zero-dt frames produced by code restarts are handled with ``zero_dt_policy``
(ADR-0019 addendum / ADR-0020 §5):

* ``"registered"`` (default): the v400 production runs are **not** regenerated.
  Each detected ``dt=0`` frame must appear in the explicit restart-event
  registry (argument / ``phase_a_native_run.json`` / ``restart_events.json``);
  registered events are kept as frames while the cross-boundary pair
  (``n-1 -> n``) and the lagging ``projections{n-1}`` are skipped, and the
  event frame ``n`` itself stays usable (``projections{n}`` is aligned; its
  positive particle window is inferred from the startup zero projection time).  An **unregistered**
  ``dt=0`` frame is still fail-fast;
* ``"error"``: strict mode for newly generated data -- any ``dt=0`` frame fails
  with the exact indices and the generation-side dedup instructions;
* ``"list"``: legacy-data diagnostics only -- enumerate the ``dt=0`` frames and
  the affected pairs, drop the zero-length records, register issues.

The restart registry (``available_restart_files.txt`` or an explicit list) is
parsed as the checkpoint registry; the *detected* restart events
(``T_n <= T_{n-1}``) are cross-referenced against it and reported per index.
"""

from __future__ import annotations

import hashlib
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
from src.data.parameter_channel import build_condition_vector
from src.data.step_dataset import (
    ALLOWED_CONDITION_GROUPS,
    FILM_FORBIDDEN_GROUPS,
    REGISTERED_BOUNDARY_CODES,
    SHARD_FIELD_NAMES,
    SINGLE_STEP_TOL_JOREK,
    ConditionVector,
    ShardState,
    StepContractReport,
    StepMeta,
    StepSample,
    _median,
)

JOREK_FRAME_RE = re.compile(r"jorek(?P<index>\d+)\.h5\Z")
PROJECTION_FRAME_RE = re.compile(r"projections(?P<index>\d+)\.h5\Z")
PART_FRAME_RE = re.compile(r"part(?P<index>\d+)\.h5\Z")
_RESTART_LINE_RE = re.compile(r"jorek(?P<index>\d+)\.h5")

#: The native layout always stores the full JOREK array
#: ``[channels, dof, harmonics, nodes]`` (2D runs: ``harmonics == 1``).
NATIVE_LAYOUT = "full_dof"
NATIVE_CADENCE = "fluid_step"
#: ``S_used = projections{k}`` is the recorded marker feedback.  Per ADR-0020
#: decision 1 (and the JOREK source lines cited there) it **is** the true field
#: source for the energy/radiation/charge channels in a marker run -- not a
#: proxy.  ``provenance/source_object`` carries this string.
NATIVE_SOURCE_OBJECT = "marker_feedback"
NATIVE_BUDGET_LAYOUT = "history"
#: Registered single-step rule for native runs (see ``single_step_method``).
NATIVE_SINGLE_STEP_METHOD = "native_adjacent_frame"

#: Static-mesh arrays hashed to verify "one mesh per run" without rebuilding.
MESH_HASH_KEYS: tuple[str, ...] = (
    "x",
    "size",
    "vertex",
    "contain_node",
    "neighbours",
    "boundary",
)

NAMELIST_CANDIDATES: tuple[str, ...] = ("jorek_namelist_spitest", "jorek_namelist")

#: Side-car overrides for native runs (optional; never required).
SIDECAR_NAME = "phase_a_native_run.json"

ZERO_DT_POLICIES = ("registered", "error", "list")
#: Default policy: production runs carry registered restart events and are not
#: regenerated (user decision 2026-10-07, ADR-0020 §5).
DEFAULT_ZERO_DT_POLICY = "registered"
#: Registry file names / keys for the *restart-event* list (the checkpoint
#: registry in ``available_restart_files.txt`` is a different object).
RESTART_EVENTS_KEY = "restart_events"
RESTART_EVENTS_FILE = "restart_events.json"
REPO_RESTART_EVENTS = Path(__file__).resolve().parents[2] / "data" / RESTART_EVENTS_FILE

#: A recorded step shorter than this many JOREK time units counts as "zero
#: dt".  Fluid steps are 0.01--5.0 JOREK units, so 1e-8 can only match the
#: code-restart duplicates (whose ``t_now`` differs by float round-off,
#: e.g. 5.3e-10 at index 5681) -- never a real integration step.
ZERO_DT_TOL_JOREK = 1.0e-8


# --------------------------------------------------------------------------- #
# Registry / index helpers
# --------------------------------------------------------------------------- #


def parse_available_restart_files(path: str | Path) -> tuple[int, ...]:
    """Parse the restart-checkpoint registry written by JOREK.

    ``available_restart_files.txt`` lists the checkpoint files available to the
    code (one ``.../jorek{n:06d}.h5`` path per line).  The returned indices are
    the *checkpoint registry*; the actual code-restart events are detected from
    the time table and cross-referenced against it.
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - defensive
        raise ContractError(f"{path}: cannot read restart registry ({exc})") from exc
    indices = sorted(
        {int(match.group("index")) for match in _RESTART_LINE_RE.finditer(text)}
    )
    if not indices:
        raise ContractError(f"{path}: no 'jorek<index>.h5' entries found")
    return tuple(indices)


def parse_index_sequence(text: str) -> tuple[int, ...]:
    """Parse ``"2704, 5681"`` / ``"2704 5681"`` / ``"2700-2710"`` index lists."""
    indices: set[int] = set()
    for token in re.split(r"[,\s;]+", text.strip()):
        if not token:
            continue
        match = re.fullmatch(r"(\d+)\s*[-:]\s*(\d+)", token)
        if match is not None:
            first, last = int(match.group(1)), int(match.group(2))
            if last < first:
                raise ContractError(f"invalid index range {token!r}")
            indices.update(range(first, last + 1))
            continue
        if not token.isdigit():
            raise ContractError(f"invalid index token {token!r}")
        indices.add(int(token))
    return tuple(sorted(indices))


def read_restart_events_file(path: str | Path) -> tuple[int, ...]:
    """Read one restart-event registry file.

    Accepted shapes: a JSON list of indices, ``{"restart_events": [...]}`` or
    the repo-level ``{"_meta": ..., "runs": {...}}`` map (handled by
    :func:`resolve_restart_events`, which resolves the run name first).
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: cannot read restart-event registry ({exc})") from exc
    if isinstance(payload, list):
        return tuple(sorted({int(index) for index in payload}))
    if isinstance(payload, dict):
        events = payload.get(RESTART_EVENTS_KEY)
        if events is not None:
            return tuple(sorted({int(index) for index in events}))
    raise ContractError(
        f"{path}: expected a JSON list or an object with '{RESTART_EVENTS_KEY}'"
    )


def resolve_restart_events(
    run_dir: str | Path,
    *,
    explicit: Sequence[int] | None = None,
    sidecar: Mapping[str, Any] | None = None,
    repo_registry: str | Path | None = None,
) -> tuple[tuple[int, ...], str | None]:
    """Resolve the registered restart events for one native run.

    Resolution order (first hit wins):

    1. explicit ``restart_events`` argument;
    2. ``phase_a_native_run.json`` side-car key ``restart_events``;
    3. ``<run>/restart_events.json``;
    4. repo-level ``data/restart_events.json`` (``runs[<run name>]
       .restart_events``; also accepts a bare list under the run name).
    """
    run_dir = Path(run_dir)
    if explicit is not None:
        return tuple(sorted({int(index) for index in explicit})), "explicit"
    sidecar = sidecar or {}
    if sidecar.get(RESTART_EVENTS_KEY) is not None:
        return (
            tuple(sorted({int(index) for index in sidecar[RESTART_EVENTS_KEY]})),
            "sidecar:" + SIDECAR_NAME,
        )
    local = run_dir / RESTART_EVENTS_FILE
    if local.is_file():
        return read_restart_events_file(local), str(local)
    registry = Path(repo_registry) if repo_registry is not None else REPO_RESTART_EVENTS
    if registry.is_file():
        try:
            payload = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            payload = None
        if isinstance(payload, dict):
            runs = payload.get("runs")
            if isinstance(runs, Mapping) and run_dir.name in runs:
                entry = runs[run_dir.name]
                if isinstance(entry, Mapping):
                    events = entry.get(RESTART_EVENTS_KEY, ())
                else:
                    events = entry
                return tuple(sorted({int(index) for index in events})), str(registry)
    return (), None


def find_native_frames(
    run_dir: str | Path,
) -> tuple[dict[int, Path], dict[int, Path], dict[int, Path]]:
    """Return ``(jorek, projections, part)`` index->path maps for a raw run."""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ContractError(f"{run_dir} is not a directory")
    state: dict[int, Path] = {}
    projections: dict[int, Path] = {}
    part: dict[int, Path] = {}
    for path in run_dir.iterdir():
        for pattern, target in (
            (JOREK_FRAME_RE, state),
            (PROJECTION_FRAME_RE, projections),
            (PART_FRAME_RE, part),
        ):
            match = pattern.fullmatch(path.name)
            if match is not None:
                target[int(match.group("index"))] = path
    if not state:
        raise ContractError(f"{run_dir}: contains no jorek*.h5 frames")
    return state, projections, part


def is_native_run(root: str | Path) -> bool:
    """True when ``root`` is a raw stepwise run (jorek frames, no export_v3)."""
    root = Path(root)
    if not root.is_dir():
        return False
    if (root / "manifest.json").is_file() or (
        root / "export_v3" / "manifest.json"
    ).is_file():
        return False
    try:
        state, _, _ = find_native_frames(root)
    except ContractError:
        return False
    return bool(state)


def load_sidecar(root: str | Path) -> dict[str, Any]:
    """Load the optional native-run side-car JSON (``phase_a_native_run.json``)."""
    path = Path(root) / SIDECAR_NAME
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{path}: invalid side-car JSON ({exc})") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"{path}: side-car must be a JSON object")
    return payload


# --------------------------------------------------------------------------- #
# Condition channel (Task C: production namelist + H5 value chain)
# --------------------------------------------------------------------------- #

#: Group labels for the recorded 61-dimension parameter surface
#: (``report/data-export-spec-v3.md`` §4.2 vocabulary; mapping from the
#: ``phase-a-prime-data-audit-2026-10-04.md`` §6 audit).
_GROUP_BY_PARAMETER: dict[str, str] = {
    "n_spi": "pellet_mass",
    "pellet_density": "pellet_mass",
    "T_0": "background_plasma",
    "Ti_0": "background_plasma",
    "Te_0": "background_plasma",
    "particlesource": "background_plasma",
    "ellip": "background_plasma",
    "gamma_i_stangeby": "boundary_flags",
    "gamma_e_stangeby": "boundary_flags",
    "bc_natural_open": "boundary_flags",
    "freeboundary": "boundary_flags",
    "resistive_wall": "boundary_flags",
    "n_open": "boundary_flags",
    "n_limiter": "boundary_flags",
    "eta": "excluded_transport",
    "visco": "excluded_transport",
    "visco_par": "excluded_transport",
    "visco_par_par": "excluded_transport",
    "ZK_i_par": "excluded_transport",
    "ZK_e_par": "excluded_transport",
    "D_perp": "excluded_transport",
    "ZK_i_perp": "excluded_transport",
    "ZK_e_perp": "excluded_transport",
    "central_density": "excluded_transport",
    "central_mass": "excluded_transport",
    "F0": "excluded_transport",
    "R_domm": "excluded_transport",
    "sqrt_mu0_rho0": "excluded_transport",
    "t_norm": "excluded_transport",
    "ZK_par_max": "excluded_transport",
    "ZKpar_T_dependent": "excluded_transport",
    "Ti_min_ZKpar": "excluded_transport",
    "Te_min_ZKpar": "excluded_transport",
    "T_min": "excluded_transport",
    "T_min_neg": "excluded_transport",
    "rho_min": "excluded_transport",
    "rho_min_neg": "excluded_transport",
    "ZK_e_par_neg": "excluded_transport",
    "ZK_i_par_neg": "excluded_transport",
    "D_par": "excluded_transport",
    "D_par_imp": "excluded_transport",
    "time_evol_theta": "excluded_numeric",
    "time_evol_zeta": "excluded_numeric",
    "tstep": "excluded_numeric",
    "t_now": "excluded_numeric",
}

_UNIT_BY_PARAMETER: dict[str, str] = {
    "n_spi": "1",
    "pellet_density": "m^-3",
    "T_0": "JOREK energy",
    "Ti_0": "JOREK energy",
    "Te_0": "JOREK energy",
    "eta": "m^2 / JOREK time",
    "visco": "m^2 / JOREK time",
    "visco_par": "m^2 / JOREK time",
    "tstep": "JOREK time",
    "t_now": "JOREK time",
    "t_norm": "s / JOREK time",
    "ZK_par_max": "W / (m K^2.5) (JOREK-scaled)",
    "Ti_min_ZKpar": "JOREK energy",
    "Te_min_ZKpar": "JOREK energy",
}

#: Transport coefficients the A' spec requires to be cross-checked between the
#: H5 frame and the production namelist (H5 wins in the value chain).
TRANSPORT_CROSS_CHECK_NAMES: tuple[str, ...] = ("eta", "visco", "visco_par")


def _base_parameter_name(name: str) -> str:
    return name.split("[", 1)[0]


def _group_for_parameter(name: str) -> str:
    base = _base_parameter_name(name)
    try:
        return _GROUP_BY_PARAMETER[base]
    except KeyError as exc:
        raise ContractError(
            f"condition parameter {name!r} has no frozen group label; extend "
            "_GROUP_BY_PARAMETER and the export-spec vocabulary before use"
        ) from exc


def _dedup_condition_names(names: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(unique_names, notes)`` for repeated parameter names."""
    seen: dict[str, int] = {}
    unique: list[str] = []
    notes: list[str] = []
    for name in names:
        count = seen.get(name, 0) + 1
        seen[name] = count
        if count == 1:
            unique.append(name)
            continue
        new_name = f"{name}#{count}"
        unique.append(new_name)
        notes.append(
            f"{name!r} appears {count} times in the recorded parameter surface; "
            f"renamed to {new_name!r} (dedup before the A' export block is frozen)"
        )
    return tuple(unique), tuple(notes)


def resolve_namelist_path(
    run_dir: str | Path, namelist_file: str | Path | None = None
) -> Path | None:
    """Locate the production namelist (``jorek_namelist_spitest`` first)."""
    run_dir = Path(run_dir)
    if namelist_file is not None:
        path = Path(namelist_file)
        if not path.is_file():
            raise ContractError(f"namelist file {path} does not exist")
        return path
    for name in NAMELIST_CANDIDATES:
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def _vector_lookup(vector: ConditionVector, name: str) -> float | None:
    for entry, value in zip(vector.names, vector.values):
        if entry == name:
            return float(value)
    return None


def _namelist_scalar(namelist: Mapping[str, Any], key: str) -> float | None:
    if key not in namelist:
        return None
    value = namelist[key]
    if isinstance(value, list):
        return float(value[0]) if value else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class NativeConditionSummary:
    """Run-level condition provenance (vector + registration fields)."""

    vector: ConditionVector
    namelist_path: Path | None
    namelist_sha256: str | None
    cross_check: Mapping[str, Mapping[str, float | None]]
    notes: tuple[str, ...]
    parameter_surface: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_entries": len(self.vector.names),
            "namelist": str(self.namelist_path) if self.namelist_path else None,
            "namelist_sha256": self.namelist_sha256,
            "parameter_surface": self.parameter_surface,
            "group_counts": dict(self.vector.group_counts),
            "film_eligible": int(np.sum(self.vector.film_eligible)),
            "transport_cross_check": {
                key: dict(value) for key, value in self.cross_check.items()
            },
            "notes": list(self.notes),
            "zk_temperature_dependence": {
                "ZKpar_T_dependent": _vector_lookup(self.vector, "ZKpar_T_dependent"),
                "ZK_par_max": _vector_lookup(self.vector, "ZK_par_max"),
                "Ti_min_ZKpar": _vector_lookup(self.vector, "Ti_min_ZKpar"),
                "Te_min_ZKpar": _vector_lookup(self.vector, "Te_min_ZKpar"),
                "note": (
                    "ZK_par is applied as ZK_par*(T_corr/T_0)**2.5 clipped to "
                    "[neg branch, ZK_par_max] (mod_elt_matrix.f90:717-740); the "
                    "temperature-dependence flags and the cap are registered here"
                ),
            },
        }


def build_native_condition(
    run_dir: str | Path,
    jorek_path: str | Path,
    *,
    namelist_file: str | Path | None = None,
) -> NativeConditionSummary:
    """Build the run-level condition vector from the production namelist + H5.

    Value chain per ``src/data/parameter_channel.py`` (H5 -> namelist -> JOREK
    default).  The H5-stored ``eta``/``visco``/``visco_par`` are cross-checked
    against the namelist and registered (the H5 value wins, as documented for
    the recorded parameter surface).
    """
    from src.data.namelist import parse_namelist

    run_dir = Path(run_dir)
    namelist_path = resolve_namelist_path(run_dir, namelist_file)
    namelist_text = (
        namelist_path.read_text(encoding="utf-8", errors="replace")
        if namelist_path is not None
        else None
    )
    digest = (
        hashlib.sha256(namelist_text.encode("utf-8")).hexdigest()
        if namelist_text is not None
        else None
    )
    parsed = parse_namelist(namelist_text) if namelist_text else {}
    notes: list[str] = []
    if namelist_path is None:
        notes.append(
            "no production namelist found (jorek_namelist_spitest / jorek_namelist); "
            "condition entries fall back to H5 values and JOREK defaults"
        )
    cross_check: dict[str, dict[str, float | None]] = {}
    with h5py.File(jorek_path, "r") as handle:
        payload = build_condition_vector(handle, namelist_text)
        for name in TRANSPORT_CROSS_CHECK_NAMES:
            h5_value = float(np.ravel(handle[name][()])[0]) if name in handle else None
            namelist_value = _namelist_scalar(parsed, name)
            delta = None
            if h5_value is not None and namelist_value is not None:
                delta = h5_value - namelist_value
                if abs(delta) > 1e-12 * max(1.0, abs(h5_value)):
                    notes.append(
                        f"{name}: H5 value {h5_value!r} differs from the namelist value "
                        f"{namelist_value!r} (delta {delta!r}); the H5 value wins in the "
                        "recorded value chain"
                    )
            cross_check[name] = {
                "h5": h5_value,
                "namelist": namelist_value,
                "delta": delta,
            }
    raw_names = tuple(str(name) for name in payload["names"])
    names, duplicate_notes = _dedup_condition_names(raw_names)
    notes.extend(duplicate_notes)
    values = np.asarray(payload["vector"], dtype=np.float64)
    provenance_map = payload["provenance"]
    provenance: list[str] = []
    groups: list[str] = []
    units: list[str] = []
    film: list[int] = []
    for raw_name, _name in zip(raw_names, names):
        group = _group_for_parameter(raw_name)
        groups.append(group)
        units.append(_UNIT_BY_PARAMETER.get(_base_parameter_name(raw_name), "-"))
        provenance.append(str(provenance_map.get(raw_name, "unknown")))
        film.append(0 if group in FILM_FORBIDDEN_GROUPS else 1)
    illegal = sorted({group for group in groups if group not in ALLOWED_CONDITION_GROUPS})
    if illegal:  # pragma: no cover - guarded by the map
        raise ContractError(f"condition groups outside the frozen vocabulary: {illegal}")
    vector = ConditionVector(
        names=names,
        values=values,
        units=tuple(units),
        groups=tuple(groups),
        provenance=tuple(provenance),
        film_eligible=np.asarray(film, dtype=np.int64),
    )
    return NativeConditionSummary(
        vector=vector,
        namelist_path=namelist_path,
        namelist_sha256=digest,
        cross_check=cross_check,
        notes=tuple(notes),
        parameter_surface=(
            "recorded 61-dim parameter surface (src/data/parameter_channel.py); the "
            "frozen A' /condition export block is not present in the native layout"
        ),
    )


# --------------------------------------------------------------------------- #
# Frame-level helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NativeFrameMeta:
    """Cheap per-frame metadata collected during the native index pass."""

    index: int
    path: Path
    t_jorek: float
    t_norm: float
    n_nodes: int
    n_spi: int
    has_spi_arrays: bool


def _h5_scalar(handle: h5py.File, key: str, *, what: str) -> float:
    if key not in handle:
        raise ContractError(f"{what}: missing '{key}'")
    try:
        return float(np.ravel(handle[key][()])[0])
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ContractError(f"{what}: '{key}' is not a numeric scalar") from exc


def _read_frame_meta(path: Path, index: int) -> NativeFrameMeta:
    what = path.name
    with h5py.File(path, "r") as handle:
        if "values" not in handle:
            raise ContractError(f"{what}: missing 'values' state array")
        shape = handle["values"].shape
        if len(shape) != 4 or shape[0] < 9 or shape[2] != 1:
            raise ContractError(
                f"{what}: 'values' must be [>=9, dof, 1, nodes], got {shape}"
            )
        t_norm = _h5_scalar(handle, "t_norm", what=what)
        if t_norm <= 0.0:
            raise ContractError(f"{what}: 't_norm' must be positive, got {t_norm}")
        t_jorek = _h5_scalar(handle, "t_now", what=what)
        n_spi = (
            int(np.ravel(handle["n_spi_tot"][()])[0]) if "n_spi_tot" in handle else 0
        )
        has_spi = all(name in handle for name in SHARD_FIELD_NAMES)
        return NativeFrameMeta(
            index=index,
            path=path,
            t_jorek=t_jorek,
            t_norm=t_norm,
            n_nodes=int(shape[-1]),
            n_spi=n_spi,
            has_spi_arrays=has_spi,
        )


def _read_native_shard(
    path: Path, index: int, *, side: str
) -> tuple[dict[str, np.ndarray], int]:
    what = f"{path.name} (shard {side})"
    with h5py.File(path, "r") as handle:
        n_total = (
            int(np.ravel(handle["n_spi_tot"][()])[0]) if "n_spi_tot" in handle else 0
        )
        arrays: dict[str, np.ndarray] = {}
        for name in SHARD_FIELD_NAMES:
            if name not in handle:
                if n_total == 0:
                    arrays[name] = np.zeros(0, dtype=np.float64)
                    continue
                raise ContractError(
                    f"{what}: missing '/{name}' but n_spi_tot={n_total} (native layout "
                    "requires the 16 spi_* arrays; pre-injection frames must still "
                    "carry length-0 arrays)"
                )
            array = np.asarray(handle[name][()], dtype=np.float64)
            if array.ndim != 1:
                raise ContractError(f"{what}: '/{name}' must be 1-D, got {array.shape}")
            if array.shape[0] != n_total:
                raise ContractError(
                    f"{what}: '/{name}' length {array.shape[0]} != n_spi_tot={n_total}"
                )
            arrays[name] = array
    return arrays, int(n_total)


def _mesh_digest(handle: h5py.File, what: str) -> str:
    digest = hashlib.sha256()
    for key in MESH_HASH_KEYS:
        if key not in handle:
            if key == "neighbours":  # optional in the export spec
                continue
            raise ContractError(f"{what}: missing mesh array '{key}'")
        array = np.ascontiguousarray(handle[key][()])
        digest.update(key.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _read_state_values(path: Path, what: str, *, n_channels: int = 9) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "values" not in handle:
            raise ContractError(f"{what}: missing 'values' state array")
        values = np.asarray(handle["values"][:n_channels, 0, 0, :], dtype=np.float64).T
    if values.shape[1] < n_channels:
        raise ContractError(f"{what}: state has {values.shape[1]} channels, need {n_channels}")
    return values


def _read_projection_values(path: Path, what: str, *, n_channels: int = 5) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        if "values" not in handle:
            raise ContractError(f"{what}: missing 'values' source array")
        values = np.asarray(handle["values"][:n_channels, 0, 0, :], dtype=np.float64).T
    if values.shape[1] < n_channels:
        raise ContractError(f"{what}: source has {values.shape[1]} channels, need {n_channels}")
    return values


# --------------------------------------------------------------------------- #
# Native reader
# --------------------------------------------------------------------------- #


class NativeRunDataset:
    """Reader for one native stepwise run directory (A' layout b).

    The public surface mirrors :class:`src.data.step_dataset.StepRunDataset`
    (``indices``/``metas``/``report``/``__getitem__``/``read_boundary``/...),
    so the acceptance CLI and downstream code do not branch on the layout.
    """

    layout_kind = "native"

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
        zero_dt_policy: str = DEFAULT_ZERO_DT_POLICY,
        step_times: Mapping[int, float] | None = None,
        namelist_file: str | Path | None = None,
        mesh_verify_stride: int = 128,
    ) -> None:
        if policy not in {"drop", "error"}:
            raise ValueError(f"unknown restart policy {policy!r}; use 'drop' or 'error'")
        if zero_dt_policy not in ZERO_DT_POLICIES:
            raise ValueError(
                f"unknown zero-dt policy {zero_dt_policy!r}; use one of {ZERO_DT_POLICIES}"
            )
        if mesh_verify_stride < 1:
            raise ValueError("mesh_verify_stride must be >= 1")
        self.run_dir = Path(root)
        self.export_root = self.run_dir
        if raw_run_dir is not None and Path(raw_run_dir).resolve() != self.run_dir.resolve():
            raise ContractError(
                "native runs are read from the raw files directly: --raw-run-dir must "
                f"be omitted or equal the run dir ({self.run_dir})"
            )
        self.raw_run_dir = None
        self.policy = policy
        self.zero_dt_policy = zero_dt_policy
        self.require_single_step = bool(require_single_step)
        self._conditions_required = bool(condition_required)
        self._extra_boundary_codes = {int(code) for code in extra_boundary_codes}
        self._mesh_verify_stride = int(mesh_verify_stride)
        self._index_window = index_window
        self._issues: list[TimingIssue] = []
        self._dropped: list[int] = []
        self._dropped_zero_dt: list[int] = []
        self._dropped_cross_boundary: list[int] = []
        self._dropped_lagging: list[int] = []
        self._zero_dt_indices: tuple[int, ...] = ()
        self._zero_dt_pairs: tuple[int, ...] = ()
        self._mesh: dict[str, np.ndarray] | None = None
        self._mesh_digest: str | None = None
        self._mesh_verifications = 0
        self._boundary_cache: np.ndarray | None = None
        self._condition_summary: NativeConditionSummary | None = None
        self._boundary_frames_leading: tuple[int, ...] = ()
        self._boundary_frames_trailing: tuple[int, ...] = ()
        self._restart_report: dict[str, Any] = {}
        self._frame_metas: dict[int, NativeFrameMeta] = {}
        self._selected_window: tuple[int, int] | None = None

        sidecar = load_sidecar(self.run_dir)
        if restart_indices is None:
            restart_indices = sidecar.get("restart_indices")
        if gap_indices is None:
            gap_indices = sidecar.get("gap_indices")
        if mesh_verify_stride == 128 and "mesh_verify_stride" in sidecar:
            self._mesh_verify_stride = int(sidecar["mesh_verify_stride"])
        if namelist_file is None:
            namelist_file = sidecar.get("namelist_file")
        self._namelist_file = namelist_file
        self._state_frames, self._projection_frames, self._part_frames = find_native_frames(
            self.run_dir
        )
        self._restart_registry = self._resolve_restart_registry(restart_indices, restart_file)
        (
            self._restart_events,
            self._restart_events_source,
        ) = resolve_restart_events(self.run_dir, explicit=restart_events, sidecar=sidecar)
        self._restart_events = tuple(sorted(set(self._restart_events)))
        self._gaps = {int(index) for index in (gap_indices or ())}
        self._single_step_tol = float(
            single_step_tol if single_step_tol is not None else SINGLE_STEP_TOL_JOREK
        )
        if self._single_step_tol <= 0.0:
            raise ContractError("single_step_tol must be positive")
        self.layout = NATIVE_LAYOUT
        self.cadence = NATIVE_CADENCE
        self.source_object = NATIVE_SOURCE_OBJECT
        self.budget_layout = NATIVE_BUDGET_LAYOUT
        self._table, self.step_table_source = resolve_step_times(self.run_dir, step_times)
        if self._table is None:
            raise ContractError(
                f"{self.run_dir}: no fluid-step time table (macroscopic_vars.dat) and no "
                "explicit step_times; native interval metadata cannot be derived (only "
                "ordering checks would be possible, which is not sufficient for A')"
            )
        self._manifest = self._build_manifest(sidecar)
        self._n_nodes_hint: int | None = (
            int(self._manifest["n_nodes"]) if self._manifest.get("n_nodes") else None
        )
        self._metas: tuple[StepMeta, ...] = ()
        self._check_index()
        self.report = self._build_report()

    # ------------------------------------------------------------------ setup

    @property
    def manifest(self) -> dict[str, Any]:
        return self._manifest

    @property
    def issues(self) -> tuple[TimingIssue, ...]:
        return tuple(self._issues)

    @property
    def dropped_restart(self) -> tuple[int, ...]:
        return tuple(self._dropped)

    @property
    def dropped_zero_dt(self) -> tuple[int, ...]:
        return tuple(self._dropped_zero_dt)

    @property
    def metas(self) -> tuple[StepMeta, ...]:
        return self._metas

    @property
    def indices(self) -> tuple[int, ...]:
        return tuple(meta.index for meta in self._metas)

    @property
    def n_nodes(self) -> int | None:
        return self._n_nodes_hint

    @property
    def restart_report(self) -> dict[str, Any]:
        return self._restart_report

    @property
    def condition_summary(self) -> NativeConditionSummary | None:
        return self._condition_summary

    @property
    def mesh(self) -> dict[str, np.ndarray] | None:
        """The cached static mesh (read once, hash-verified on registered frames)."""
        return self._mesh

    def _resolve_restart_registry(
        self,
        restart_indices: Sequence[int] | None,
        restart_file: str | Path | None,
    ) -> tuple[int, ...]:
        if restart_file is not None:
            return parse_available_restart_files(restart_file)
        if restart_indices is not None:
            return tuple(sorted({int(index) for index in restart_indices}))
        default_file = self.run_dir / "available_restart_files.txt"
        if default_file.is_file():
            return parse_available_restart_files(default_file)
        return ()

    def _build_manifest(self, sidecar: Mapping[str, Any]) -> dict[str, Any]:
        registry = self._restart_registry
        return {
            "run": self.run_dir.name,
            "layout": self.layout,
            "layout_kind": "native",
            "cadence": self.cadence,
            "source_object": self.source_object,
            "budget_layout": self.budget_layout,
            "n_nodes": None,
            "jorek_version": sidecar.get("jorek_version"),
            "model_id": sidecar.get("model_id", 502),
            "interval_source": self.step_table_source,
            "provenance": {
                "layout": "native_stepwise_run",
                "interval_metadata": (
                    "derived from jorek t_now, cross-checked against "
                    f"{self.step_table_source}"
                ),
                "source_object": (
                    "marker_feedback (recorded projections; JOREK use_marker=.t. run: "
                    "the true field source for dE_ion/rad_sink/Z_eff/Z_imp/N_imp per "
                    "mod_elt_matrix.f90:989-1003,1129,1728 -- ADR-0020 decision 1)"
                ),
                "shard_start": (
                    "spi_*_arr of jorek{k}, true time T_k (field-interval start); the "
                    "source-window start p_{k-1} is the previous record's shard_start "
                    "(= spi(jorek{k-1}), T_{k-1})"
                ),
                "shard_end": (
                    "spi_*_arr of jorek{k+1}, true time T_{k+1} (field-interval end)"
                ),
                "shard_time_semantics": "field_interval_endpoints",
                "restart_events": {
                    "indices": list(self._restart_events),
                    "source": self._restart_events_source,
                    "semantics": (
                        "registered dt=0 frames: cross-boundary pair (n-1 -> n) and "
                        "projections{n-1} lagging source skipped; event frame n kept"
                    ),
                },
                "mesh": "jorek{k}, static, sha256-cached",
                "condition": "jorek_namelist_spitest (production) + H5 value chain",
            },
            "restart_registry": {
                "n": len(registry),
                "first": registry[0] if registry else None,
                "last": registry[-1] if registry else None,
                "source": (
                    str(self.run_dir / "available_restart_files.txt")
                    if (self.run_dir / "available_restart_files.txt").is_file()
                    else "explicit"
                ),
            },
            "gap_indices": sorted(self._gaps),
            "n_state_frames": len(self._state_frames),
            "n_projection_frames": len(self._projection_frames),
            "n_part_frames": len(self._part_frames),
            "mesh_verify_stride": self._mesh_verify_stride,
        }

    # ------------------------------------------------------------- validation

    def _note(self, kind: str, index: int, detail: str, *, aggregate: bool = False) -> None:
        if aggregate:
            for issue in self._issues:
                if issue.kind == kind:
                    return
        self._issues.append(TimingIssue(kind=kind, index=index, detail=detail))

    def _select_indices(self) -> list[int]:
        indices = sorted(self._state_frames)
        if self._index_window is not None:
            first, last = self._index_window
            if last < first:
                raise ValueError(f"index_window must be ordered, got {self._index_window!r}")
            indices = [index for index in indices if first <= index <= last]
            if not indices:
                raise ContractError(
                    f"{self.run_dir}: no jorek frames inside window {self._index_window!r}"
                )
        return indices

    def _zero_dt_frames(self, indices: Sequence[int]) -> list[int]:
        table = self._table or {}
        return [
            index
            for index in indices
            if index - 1 in table
            and abs(table[index] - table[index - 1]) <= ZERO_DT_TOL_JOREK
        ]

    def _check_index(self) -> None:
        indices = self._select_indices()
        self._selected_window = (indices[0], indices[-1])
        table = self._table or {}
        missing_table = [index for index in indices if index not in table]
        if missing_table:
            raise ContractError(
                f"{self.run_dir}: step table ({self.step_table_source}) has no entry for "
                f"jorek frames {missing_table[:10]}"
                + (" ..." if len(missing_table) > 10 else "")
            )
        negative_dt = [
            index
            for index in indices
            if index - 1 in table
            and table[index] - table[index - 1] < -ZERO_DT_TOL_JOREK
        ]
        if negative_dt:
            raise ContractError(
                f"{self.run_dir}: negative dt at frame indices {negative_dt}; the recorded "
                "time table is not monotonic and dedup cannot repair it"
            )
        zero_dt = self._zero_dt_frames(indices)
        self._zero_dt_indices = tuple(zero_dt)
        self._zero_dt_pairs = tuple(index - 1 for index in zero_dt)
        if zero_dt and self.zero_dt_policy == "error":
            detail = "; ".join(
                f"index {index}: T_{index}={table[index]!r} ~ T_{index - 1}="
                f"{table[index - 1]!r} (dt={table[index] - table[index - 1]!r})"
                for index in zero_dt
            )
            raise ContractError(
                f"{self.run_dir}: {len(zero_dt)} zero-dt frame(s) from code restarts: "
                f"{list(zero_dt)}. {detail}. Affected pairs (dt_step=0): "
                f"{list(self._zero_dt_pairs)}. Generation-side fix (plan 1: dedup at every "
                "code restart): keep exactly one frame with strictly increasing t_now, re-align "
                "projections{k} with jorek{k}, keep the part cadence, and register the "
                "removed duplicate indices as gaps. For legacy-data diagnostics only, "
                "pass --zero-dt-policy list (never for Phase B/C)."
            )
        if zero_dt and self.zero_dt_policy == "registered":
            registered = set(self._restart_events)
            unregistered = [index for index in zero_dt if index not in registered]
            if unregistered:
                raise ContractError(
                    f"{self.run_dir}: {len(unregistered)} unregistered zero-dt frame(s) "
                    f"from code restarts: {unregistered}. Registered restart events are "
                    "accepted (cross-boundary pair and lagging projection skipped), but "
                    "every dt=0 frame must be registered explicitly -- add the indices to "
                    f"{RESTART_EVENTS_FILE} (or pass --restart-events). Detected events: "
                    f"{list(zero_dt)}; registered: {list(self._restart_events)}."
                )
            stale = [index for index in self._restart_events if index not in set(zero_dt)]
            self._note(
                "restart_events_registered",
                -1,
                f"{len(zero_dt)} registered restart event(s) {list(zero_dt)} "
                f"(source: {self._restart_events_source or 'none'}); the cross-boundary "
                "pair (n-1 -> n) and projections{n-1} are skipped, the event frame n "
                "stays with S_n = projections{n} (positive particle window from startup event)"
                + (f"; {len(stale)} registered event(s) not detected in this window "
                   f"{stale[:10]}" if stale else ""),
            )
        if zero_dt and self.zero_dt_policy == "list":
            self._note(
                "zero_dt_frames_listed",
                0,
                f"zero_dt_policy='list' (legacy diagnostics only): frames {list(zero_dt)}; "
                f"affected pairs {list(self._zero_dt_pairs)} are dropped; regenerate with "
                "dedup at the restarts before Phase B/C",
            )

        zero_dt_set = set(zero_dt)
        if not zero_dt:
            assert_strictly_increasing(
                [table[index] for index in indices],
                what=f"{self.run_dir.name}: jorek t_now (step table)",
            )
        for previous, current in zip(indices, indices[1:]):
            if current == previous + 1:
                continue
            missing = [
                index
                for index in range(previous + 1, current)
                if index not in self._gaps and index not in zero_dt_set
            ]
            if missing:
                raise ContractError(
                    f"{self.run_dir}: jorek frame indices jump from {previous} to {current} "
                    f"and {missing} are not registered as gaps; register them explicitly "
                    "(gap_indices / phase_a_native_run.json) -- do not skip silently"
                )
        for gap in sorted(self._gaps):
            if gap in self._state_frames:
                self._note(
                    "registered_gap_stale",
                    gap,
                    "index is registered as a gap but a jorek frame exists; the frame is "
                    "kept (registry entry may be stale)",
                )

        # Edge frames (registered, never silent).  Only meaningful when the whole
        # frame range is checked; a windowed run registers its own edges instead.
        full_first = min(self._state_frames)
        full_last = max(self._state_frames)
        self._boundary_frames_leading = tuple(
            index
            for index in sorted(self._projection_frames)
            if index < indices[0] and indices[0] == full_first
        )
        self._boundary_frames_trailing = tuple(
            sorted(
                [
                    index
                    for index in self._projection_frames
                    if index > indices[-1] and indices[-1] == full_last
                ]
                + [indices[-1]]
            )
        )
        if indices[0] != full_first or indices[-1] != full_last:
            self._note(
                "checked_window",
                indices[-1],
                f"contract checked over the window [{indices[0]}..{indices[-1]}] of the "
                f"full frame range [{full_first}..{full_last}]; edge frames are registered "
                "for the full range only",
            )
        for index in self._boundary_frames_leading:
            self._note(
                "boundary_frame_leading",
                index,
                f"projections{index:06d}.h5 precedes the first jorek frame {indices[0]} "
                "(coverage edge; no pair can use it)",
            )
        self._note(
            "boundary_frame_trailing",
            indices[-1],
            f"jorek{indices[-1]:06d}.h5 is the last state frame inside the checked range "
            "(no successor); kept as a terminal state frame and cannot start a pair",
        )

        self._build_records(indices)

    def _build_records(self, indices: Sequence[int]) -> None:
        table = self._table or {}
        zero_dt_set = set(self._zero_dt_indices)
        metas: list[StepMeta] = []
        pairs = list(zip(indices[:-1], indices[1:]))
        for index, following in pairs:
            if (
                following in zero_dt_set
                and abs(table[following] - table[index]) <= ZERO_DT_TOL_JOREK
            ):
                if self.zero_dt_policy == "registered":
                    # Verify the lagging projection before skipping it: the
                    # record n-1 must carry projections{n-1} offset by
                    # -dt_source (ADR-0019 rule); any other offset is a hard
                    # error, never a silent skip.
                    projection_path = self._projection_frames.get(index)
                    if projection_path is None:
                        raise ContractError(
                            f"{self.run_dir}: registered restart event {following} needs "
                            f"projections{index:06d}.h5 to verify the lagging source "
                            "(missing)"
                        )
                    t_norm = self._read_state_frame(index).t_norm
                    with h5py.File(projection_path, "r") as handle:
                        t_projection = (
                            _h5_scalar(handle, "t_now", what=projection_path.name) / t_norm
                        )
                    dt_source = table[index] - table[index - 1]
                    offset = t_projection - table[index]
                    if abs(offset + dt_source) > TIME_TOL_JOREK:
                        raise ContractError(
                            f"{projection_path.name}: lagging projection of restart event "
                            f"{following} is offset {offset!r} JOREK units, expected "
                            f"{-dt_source!r} (= -dt_source); unregistered restart offsets "
                            "are fail-fast"
                        )
                    self._note(
                        "restart_lagging_projection_skipped",
                        index,
                        f"projections{index:06d}.h5 lags by {offset!r} JOREK units "
                        f"(= -dt_source {-dt_source!r}) and belongs to the registered "
                        f"restart event {following}; the record and its lagging source "
                        "are skipped (ADR-0019 addendum)",
                    )
                    self._dropped_lagging.append(index)
                    self._note(
                        "restart_cross_boundary_skipped",
                        index,
                        f"record {index} -> {following} crosses the registered restart "
                        f"event {following} (T={table[index]!r}); the zero-length pair and "
                        "the lagging projections{n-1} are skipped (ADR-0019 addendum)",
                    )
                    self._dropped_cross_boundary.append(index)
                else:
                    self._note(
                        "zero_dt_record_dropped",
                        index,
                        f"record {index} -> {following} is the zero-length restart "
                        f"duplicate (T={table[index]!r}); dropped under "
                        "zero_dt_policy='list'",
                    )
                    self._dropped_zero_dt.append(index)
                continue
            meta = self._read_record_meta(index, following)
            if self._n_nodes_hint is None:
                self._n_nodes_hint = meta.n_nodes
                self._manifest["n_nodes"] = meta.n_nodes
            elif meta.n_nodes != self._n_nodes_hint:
                raise ContractError(
                    f"jorek{index:06d}.h5: node count {meta.n_nodes} != run node count "
                    f"{self._n_nodes_hint}"
                )
            metas.append(meta)
        if not metas:
            raise ContractError(f"{self.run_dir}: no usable step records after registration")
        dropped_restart = self._cross_check_markers(metas)
        if self._dropped_cross_boundary:
            dropped_restart = sorted(
                set(dropped_restart) | set(self._dropped_cross_boundary)
            )
        if dropped_restart:
            self._dropped = sorted(set(self._dropped) | set(dropped_restart))
            metas = [meta for meta in metas if meta.index not in self._dropped]
        if not metas:
            raise ContractError(
                f"{self.run_dir}: every record was dropped by the restart registration; "
                "regenerate with dedup at the restarts (plan 1)"
            )
        self._metas = tuple(metas)
        self._restart_report = self._build_restart_report()

    def _read_state_frame(self, index: int) -> NativeFrameMeta:
        if index not in self._frame_metas:
            self._frame_metas[index] = _read_frame_meta(self._state_frames[index], index)
        return self._frame_metas[index]

    def _verify_frame(self, meta: NativeFrameMeta) -> None:
        """Hash-verify the static mesh on the registered stride."""
        first = min(self._state_frames)
        last = max(self._state_frames)
        if (
            meta.index == first
            or meta.index == last
            or (meta.index - first) % self._mesh_verify_stride == 0
        ):
            what = meta.path.name
            with h5py.File(meta.path, "r") as handle:
                digest = _mesh_digest(handle, what)
                codes = np.asarray(handle["boundary"][()], dtype=np.int64)
                if "freeboundary" in handle:
                    raw = handle["freeboundary"][()]
                    if isinstance(raw, (bytes, str, np.bytes_)) or (
                        hasattr(raw, "dtype") and raw.dtype.kind == "S"
                    ):
                        self._note(
                            "freeboundary_global_flag",
                            meta.index,
                            "freeboundary is the global 'F'/'T' flag; the per-node int8 "
                            "array required by the A' export spec is not available in the "
                            "native layout",
                            aggregate=True,
                        )
            registered = REGISTERED_BOUNDARY_CODES | self._extra_boundary_codes
            unknown = sorted({int(code) for code in np.unique(codes)} - registered)
            if unknown:
                raise ContractError(
                    f"{what}: unregistered boundary codes {unknown}; register them in the "
                    "loader and the B0 report before use"
                )
            if self._mesh_digest is None:
                self._mesh_digest = digest
            elif digest != self._mesh_digest:
                raise ContractError(
                    f"{what}: static mesh hash {digest[:16]} != first frame "
                    f"{self._mesh_digest[:16]}; the run does not share one mesh"
                )
            self._mesh_verifications += 1

    def _read_record_meta(self, index: int, following: int) -> StepMeta:
        table = self._table or {}
        meta = self._read_state_frame(index)
        next_meta = self._read_state_frame(following)
        self._verify_frame(meta)
        t_before = table[index]
        t_after = table[following]
        if abs(meta.t_jorek - t_before) > TIME_TOL_JOREK:
            raise ContractError(
                f"{meta.path.name}: t_now {meta.t_jorek!r} != step table T_{index} "
                f"{t_before!r} ({self.step_table_source})"
            )
        if abs(next_meta.t_jorek - t_after) > TIME_TOL_JOREK:
            raise ContractError(
                f"{next_meta.path.name}: t_now {next_meta.t_jorek!r} != step table "
                f"T_{following} {t_after!r} ({self.step_table_source})"
            )
        if abs(next_meta.t_norm - meta.t_norm) > 1e-12 * max(1.0, abs(meta.t_norm)):
            raise ContractError(
                f"{next_meta.path.name}: t_norm {next_meta.t_norm!r} differs from "
                f"{meta.path.name}: t_norm {meta.t_norm!r}"
            )
        if index - 1 not in table:
            raise ContractError(
                f"{self.run_dir}: step table has no T_{index - 1}; the source window "
                f"[T_{index - 1}, T_{index}] of record {index} cannot be derived"
            )
        t_source_start = table[index - 1]
        if index in self._zero_dt_indices and self.zero_dt_policy == "registered":
            # The prior zero projection is written at the particle restart
            # clock, BEFORE the duplicate-time fluid solve. It timestamps the
            # positive particle window; repeated fluid outputs do not.
            startup_path = self._projection_frames.get(index - 1)
            if startup_path is None:
                raise ContractError(f"restart {index}: missing startup zero projection time")
            with h5py.File(startup_path, "r") as handle:
                t_source_start = _h5_scalar(handle, "t_now", what=startup_path.name) / meta.t_norm
                if np.any(handle["values"][()] != 0):
                    raise ContractError(f"restart {index}: startup projection must be zero")
            if t_source_start >= t_before - ZERO_DT_TOL_JOREK:
                raise ContractError(f"restart {index}: particle window must have positive duration")
            self._note("restart_particle_window", index,
                       f"particle interval [{t_source_start!r}, {t_before!r}] inferred from "
                       "startup zero projection event; fluid duplicate is not a zero particle window")
        projection_path = self._projection_frames.get(index)
        if projection_path is None:
            raise ContractError(
                f"{self.run_dir}: record {index} needs S_used=projections{index:06d}.h5 "
                "(missing); a pair cannot be built without its source frame"
            )
        dt_step = t_after - t_before
        dt_source = t_before - t_source_start
        if dt_source <= ZERO_DT_TOL_JOREK:
            if self.zero_dt_policy == "error":
                raise ContractError(
                    f"{self.run_dir}: record {index}: the source window "
                    f"[T_{index - 1}, T_{index}] = [{t_source_start!r}, {t_before!r}] is "
                    f"zero-length (dt_source={dt_source!r}); a restart duplicate cannot "
                    "start a pair. Generation-side fix (plan 1): dedup so the kept frame "
                    "has a positive source window (drop the duplicate, keep the earlier "
                    "frame and continue with the next real step)"
                )
            self._note(
                "zero_dt_source_window", index,
                f"legacy diagnostic fluid-derived source interval [{t_source_start}, {t_before}]; "
                "not a physical particle sampling interval; use registered policy",
            )
        # Window-chain identity: the field interval [T_k, T_k'] must be exactly the
        # next record's source window whenever a single fluid step is required.
        next_source = table[following] - table[following - 1]
        single_step_violation: str | None = None
        if following != index + 1:
            skipped = list(range(index + 1, following))
            unregistered = [skip for skip in skipped if skip not in self._gaps]
            if unregistered:
                single_step_violation = (
                    f"record {index}->{following} skips frames {unregistered} which are "
                    "not registered as gaps"
                )
            else:
                not_zero_dt = [
                    skip
                    for skip in skipped
                    if abs(table[skip] - table[skip - 1]) > ZERO_DT_TOL_JOREK
                ]
                if not_zero_dt:
                    single_step_violation = (
                        f"record {index}->{following} skips frames {not_zero_dt} whose "
                        "dt != 0; only zero-length restart duplicates may be removed by "
                        "dedup"
                    )
        if single_step_violation is None and (
            abs(dt_step - next_source) > max(self._single_step_tol, TIME_TOL_JOREK)
        ):
            single_step_violation = (
                f"record {index}->{following}: dt_step {dt_step!r} != the next source "
                f"window {next_source!r}; the record covers more than one fluid step"
            )
        if single_step_violation is not None:
            if self.require_single_step:
                raise ContractError(
                    f"{self.run_dir}: {single_step_violation}; require_single_step=True "
                    "rejects non-stepwise records (frame-level dumps cannot enter "
                    "Phase B/C)"
                )
            self._note(
                "multi_step_record",
                index,
                f"{single_step_violation}; registered (require_single_step=False is a "
                "diagnostics-only mode)",
            )
        return StepMeta(
            index=index,
            path=meta.path,
            cadence=self.cadence,
            substeps=1,
            source_object=self.source_object,
            t_interval_start=t_before,
            t_interval_end=t_after,
            t_source_start=t_source_start,
            t_source_end=t_before,
            t_norm=meta.t_norm,
            n_nodes=meta.n_nodes,
            n_spi_start=meta.n_spi,
            n_spi_end=next_meta.n_spi,
            has_mesh=True,
            has_condition=True,
            budget_layout=self.budget_layout,
            path_after=next_meta.path,
            path_source=projection_path,
            layout_kind="native",
            index_next=following,
        )

    # ------------------------------------------------------- restart / markers

    def _cross_check_markers(self, metas: Sequence[StepMeta]) -> list[int]:
        """Check projections/part times and the restart-boundary registration."""
        dropped: list[int] = []
        restart_events = set(self._zero_dt_indices)
        aligned_registry: list[int] = []
        for meta in metas:
            index = meta.index
            projection_path = self._projection_frames[index]
            with h5py.File(projection_path, "r") as handle:
                t_projection = (
                    _h5_scalar(handle, "t_now", what=projection_path.name) / meta.t_norm
                )
            offset = t_projection - meta.t_interval_start
            in_registry = index in self._restart_registry
            is_restart = in_registry or (index + 1) in restart_events
            if abs(offset) > TIME_TOL_JOREK:
                if not is_restart:
                    raise ContractError(
                        f"projections{index:06d}.h5: t_now/t_norm {t_projection!r} is offset "
                        f"{offset!r} JOREK units from jorek{index:06d}.t_now "
                        f"{meta.t_interval_start!r}; only registered restart boundaries may "
                        "be offset (generation-side fix: re-align the restart projection)"
                    )
                if abs(offset + meta.dt_source) > TIME_TOL_JOREK:
                    raise ContractError(
                        f"projections{index:06d}.h5: registered restart offset {offset!r} "
                        f"does not match the ADR-0019 rule -dt_source {-meta.dt_source!r}"
                    )
                if self.policy == "error":
                    raise ContractError(
                        f"projections{index:06d}.h5: registered restart boundary offset "
                        f"{offset!r} while policy='error'; regenerate with dedup at the "
                        "restart (plan 1)"
                    )
                if (index + 1) in restart_events and self.zero_dt_policy == "registered":
                    self._note(
                        "restart_lagging_projection_skipped",
                        index,
                        f"projections{index:06d}.h5 lags by {offset!r} JOREK units "
                        f"(= -dt_source {-meta.dt_source!r}) and belongs to the registered "
                        f"restart event {index + 1}; the record and its lagging source are "
                        "skipped (ADR-0019 addendum)",
                    )
                    self._dropped_lagging.append(index)
                else:
                    self._note(
                        "restart_boundary_projection_offset",
                        index,
                        f"registered restart boundary: projections t_now is {offset!r} "
                        "JOREK units early (ADR-0019 rule); record dropped",
                    )
                dropped.append(index)
                continue
            if (index + 1) in restart_events:
                self._note(
                    "restart_frame_aligned",
                    index,
                    "index follows a detected restart event and projections is aligned "
                    "with jorek (dedup-compatible)",
                )
            elif in_registry:
                aligned_registry.append(index)
        if aligned_registry:
            self._note(
                "restart_registry_aligned",
                -1,
                f"{len(aligned_registry)} registered restart checkpoint(s) inside the run "
                f"window have projections aligned with the same-index jorek frame "
                f"(first: {aligned_registry[:10]}) -- no ADR-0019 offset; the checkpoint "
                "registry lists available restart files, not restart events",
            )
        # part{n}.time == T_{n-1} for every recorded part frame.
        table = self._table or {}
        if metas:
            window = (metas[0].index, metas[-1].index_next or metas[-1].index)
            checked = 0
            for index in sorted(self._part_frames):
                if index < window[0] or index > window[1] + 1:
                    continue
                if index - 1 not in table:
                    raise ContractError(f"part{index:05d}.h5: step table has no T_{index - 1}")
                path = self._part_frames[index]
                with h5py.File(path, "r") as handle:
                    t_particle = (
                        _h5_scalar(handle, "time", what=path.name) / metas[0].t_norm
                    )
                expected = table[index - 1]
                if abs(t_particle - expected) > TIME_TOL_JOREK:
                    raise ContractError(
                        f"{path.name}: time/t_norm {t_particle!r} != T_{index - 1}="
                        f"{expected!r} (ADR-0019: part{{n}} is the window-start particle "
                        "state p_n-1)"
                    )
                checked += 1
            if checked:
                self._note(
                    "part_time_checked",
                    -1,
                    f"{checked} part frames verified time == T_n-1",
                )
        return dropped

    def _build_restart_report(self) -> dict[str, Any]:
        table = self._table or {}
        registry = list(self._restart_registry)
        events = list(self._zero_dt_indices)
        event_rows: list[dict[str, Any]] = []
        for index in events:
            previous = index - 1
            row: dict[str, Any] = {
                "index": index,
                "previous_index": previous,
                "t_jorek": table.get(index),
                "t_previous_jorek": table.get(previous),
                "dt_jorek": (table[index] - table[previous])
                if previous in table
                else None,
                "affected_pair": previous,
                "in_checkpoint_registry": index in set(registry),
                "nearest_checkpoint_le_event": max(
                    [entry for entry in registry if entry <= index], default=None
                ),
            }
            t_norm_value = None
            if previous in self._state_frames:
                t_norm_value = self._read_state_frame(previous).t_norm
            for label, candidate in (("projection_prev", previous), ("projection_event", index)):
                path = self._projection_frames.get(candidate)
                if path is None or t_norm_value is None:
                    row[f"{label}_t_jorek"] = None
                    row[f"{label}_offset_jorek"] = None
                    continue
                with h5py.File(path, "r") as handle:
                    t_projection = _h5_scalar(handle, "t_now", what=path.name) / t_norm_value
                row[f"{label}_t_jorek"] = t_projection
                row[f"{label}_offset_jorek"] = t_projection - table.get(index, t_projection)
            row["particle_window_start_jorek"] = row.get("projection_prev_t_jorek")
            row["particle_window_end_jorek"] = row.get("projection_event_t_jorek")
            part_path = self._part_frames.get(index)
            if part_path is not None and t_norm_value is not None:
                with h5py.File(part_path, "r") as handle:
                    row["part_time_over_t_norm"] = (
                        _h5_scalar(handle, "time", what=part_path.name) / t_norm_value
                    )
            row["status"] = (
                "zero-dt duplicate frame: generation side must dedup (plan 1)"
                if self.zero_dt_policy == "error"
                else (
                    "registered restart event: cross-boundary pair and lagging "
                    "projection skipped in dataset pairing; event frame kept with positive particle window"
                    if self.zero_dt_policy == "registered"
                    else "listed (legacy diagnostics): zero-length record dropped"
                )
            )
            event_rows.append(row)
        window = getattr(
            self, "_selected_window", (min(self._state_frames), max(self._state_frames))
        )
        in_window = [entry for entry in registry if window[0] <= entry <= window[1]]
        checks: list[dict[str, Any]] = []
        for entry in in_window:
            checks.append(self._check_registry_entry(entry, table))
        if registry:
            self._note(
                "restart_registry",
                -1,
                f"{len(registry)} restart checkpoints registered "
                f"({registry[0]}..{registry[-1]}); {len(in_window)} inside the run window "
                f"({len(checks)} checked); {len(events)} detected restart event(s) {events}",
            )
        return {
            "registry": {
                "n": len(registry),
                "first": registry[0] if registry else None,
                "last": registry[-1] if registry else None,
                "indices_in_run_window": [
                    entry for entry in registry if window[0] <= entry <= window[1]
                ][:200],
                "source": (
                    "available_restart_files.txt"
                    if (self.run_dir / "available_restart_files.txt").is_file()
                    else "explicit/empty"
                ),
            },
            "events": event_rows,
            "checks": checks,
            "n_registry_checked": len(checks),
            "policy": {
                "zero_dt": self.zero_dt_policy,
                "restart_boundary": self.policy,
                "require_single_step": self.require_single_step,
                "restart_events_source": self._restart_events_source,
            },
            "spec": (
                "registered policy (ADR-0019 addendum/ADR-0020 §5): dt=0 frames must be "
                "listed in the restart-event registry; the cross-boundary pair n-1 -> n "
                "and the lagging projections{n-1} are skipped; event frame n keeps "
                "S_n = projections{n} (aligned) with a positive particle window timed by "
                "the startup zero projection. New exports should still dedup (strict 'error' policy) with "
                "exactly one frame per restart and projections aligned"
            ),
        }

    # ------------------------------------------------------------ static mesh

    def _check_registry_entry(self, index: int, table: Mapping[int, float]) -> dict[str, Any]:
        """Check one registered restart checkpoint (jorek / projections / part)."""
        row: dict[str, Any] = {
            "index": index,
            "t_jorek": table.get(index),
            "state_frame": index in self._state_frames,
            "projection_frame": index in self._projection_frames,
            "part_frame": index in self._part_frames,
        }
        t_norm = None
        if index in self._state_frames:
            meta = self._read_state_frame(index)
            t_norm = meta.t_norm
            row["t_jorek_h5"] = meta.t_jorek
        projection_path = self._projection_frames.get(index)
        if projection_path is not None and t_norm is not None:
            with h5py.File(projection_path, "r") as handle:
                t_projection = _h5_scalar(handle, "t_now", what=projection_path.name) / t_norm
            row["projection_t_jorek"] = t_projection
            row["projection_offset_jorek"] = t_projection - table.get(index, t_projection)
        part_path = self._part_frames.get(index)
        if part_path is not None and t_norm is not None:
            with h5py.File(part_path, "r") as handle:
                t_particle = _h5_scalar(handle, "time", what=part_path.name) / t_norm
            row["part_t_jorek"] = t_particle
            expected = table.get(index - 1)
            row["part_offset_jorek"] = (
                None if expected is None else t_particle - expected
            )
        offset = row.get("projection_offset_jorek")
        if offset is None:
            row["status"] = "frames absent (checkpoint outside the recorded file set)"
        elif abs(offset) <= TIME_TOL_JOREK:
            row["status"] = "projections aligned (no ADR-0019 restart offset)"
        elif abs(offset + (table.get(index, 0.0) - table.get(index - 1, 0.0))) <= TIME_TOL_JOREK:
            row["status"] = "projections offset by -dt_source (ADR-0019 restart boundary)"
        else:
            row["status"] = f"unregistered projection offset {offset!r}"
        return row

    def load_mesh(self, index: int | None = None) -> dict[str, np.ndarray] | None:
        """Read and cache the static mesh (first frame by default)."""
        if self._mesh is not None:
            return self._mesh
        indices = sorted(self._state_frames)
        if not indices:
            return None
        target = index if index is not None else indices[0]
        path = self._state_frames[target]
        mesh: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            for key in MESH_HASH_KEYS:
                if key in handle:
                    mesh[key] = np.asarray(handle[key][()])
            if "freeboundary" in handle:
                raw = handle["freeboundary"][()]
                if hasattr(raw, "dtype") and raw.dtype.kind == "S":
                    mesh["freeboundary_flag"] = np.asarray(
                        raw.item() if raw.size == 1 else raw
                    )
                else:
                    mesh["freeboundary_flag"] = np.asarray(raw)
            self._mesh_digest = _mesh_digest(handle, path.name)
        self._mesh = mesh
        return self._mesh

    # ------------------------------------------------------------------ access

    def __len__(self) -> int:
        return len(self._metas)

    def pair_indices(self) -> tuple[int, ...]:
        return tuple(
            meta.index
            for meta, following in zip(self._metas, self._metas[1:])
            if following.index == meta.index + 1
        )

    def step_path(self, index: int) -> Path:
        for meta in self._metas:
            if meta.index == index:
                return meta.path
        raise KeyError(index)

    def read_boundary(self, position: int) -> np.ndarray | None:
        meta = self._metas[position]
        if self._boundary_cache is None:
            with h5py.File(meta.path, "r") as handle:
                if "boundary" not in handle:
                    return None
                self._boundary_cache = np.asarray(handle["boundary"][()], dtype=np.int64)
        return self._boundary_cache

    def read_condition(self, position: int) -> ConditionVector | None:
        if self._condition_summary is None:
            summary = build_native_condition(
                self.run_dir,
                self._metas[0].path,
                namelist_file=self._namelist_file,
            )
            self._condition_summary = summary
            for note in summary.notes:
                self._note("condition_provenance", -1, note)
        return self._condition_summary.vector

    def __getitem__(self, position: int) -> StepSample:
        meta = self._metas[position]
        index = meta.index
        following = meta.index_next
        if following is None:  # pragma: no cover - always set by the reader
            raise ContractError(f"record {index} has no successor frame")
        projection_path = meta.path_source or self._projection_frames[index]
        path_after = meta.path_after or self._state_frames[following]
        u_before = _read_state_values(meta.path, f"jorek{index:06d}.h5")
        u_after = _read_state_values(path_after, f"jorek{following:06d}.h5")
        source = _read_projection_values(projection_path, projection_path.name)
        start_arrays, n_start = _read_native_shard(meta.path, index, side="start")
        end_arrays, n_end = _read_native_shard(path_after, following, side="end")
        condition = self.read_condition(position)
        return StepSample(
            index=index,
            u_before=u_before,
            source=source,
            u_after=u_after,
            shard_start=ShardState(
                index=index,
                # True frame times: jorek{k} stores the shard state at T_k and
                # jorek{k+1} at T_{k+1} (B1 evidence: spi_abl sum ==
                # xtime_spi_ablation_rate[k]).  The source-window start state
                # p_{k-1} is the *previous record's* shard_start
                # (= spi(jorek{k-1}), at T_{k-1}); registered in the manifest.
                time_jorek=meta.t_interval_start,
                n_total=n_start,
                arrays=start_arrays,
            ),
            shard_end=ShardState(
                index=following,
                time_jorek=meta.t_interval_end,
                n_total=n_end,
                arrays=end_arrays,
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
        """Assert ``u_after`` of a record equals ``u_before`` of the next one.

        In a native run both sides are literally the same file, so this checks
        index bookkeeping (and catches a mis-indexed successor or a gap that was
        not chained).
        """
        if position < 0 or position >= len(self._metas) - 1:
            raise IndexError(position)
        current = self._metas[position]
        following = self._metas[position + 1]
        if current.index_next != following.index:
            raise ContractError(
                f"record {current.index}: successor {current.index_next} != next record "
                f"{following.index}; index bookkeeping is inconsistent"
            )
        if Path(current.path_after or "") != following.path:
            raise ContractError(
                f"record {current.index}: u_after file {current.path_after} != next record "
                f"u_before file {following.path}"
            )

    # ------------------------------------------------------------------ report

    def _build_report(self) -> StepContractReport:
        metas = self._metas
        dt_steps = [meta.dt_step for meta in metas]
        dt_sources = [meta.dt_source for meta in metas]
        return StepContractReport(
            run=self.run_dir.name,
            export_root=self.run_dir,
            layout=self.layout,
            cadence=self.cadence,
            source_object=self.source_object,
            n_steps=len(metas),
            indices=tuple(meta.index for meta in metas),
            pair_indices=self.pair_indices(),
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
            step_table_source=self.step_table_source,
            budget_layout=self.budget_layout,
            layout_kind="native",
            single_step_method=NATIVE_SINGLE_STEP_METHOD,
            interval_source=self.step_table_source,
            zero_dt_policy=self.zero_dt_policy,
            zero_dt_indices=self._zero_dt_indices,
            zero_dt_pairs=self._zero_dt_pairs,
            zero_dt_source_windows=tuple(
                issue.index
                for issue in self._issues
                if issue.kind == "zero_dt_source_window"
            ),
            restart_events_registered=self._restart_events,
            restart_events_source=self._restart_events_source,
            restart_cross_boundary_skipped=tuple(self._dropped_cross_boundary),
            restart_lagging_projection_skipped=tuple(self._dropped_lagging),
            restart_degenerate_source_windows=tuple(
                issue.index
                for issue in self._issues
                if issue.kind == "restart_degenerate_source_window"
            ),
            restart_registry_size=len(self._restart_registry),
            restart_events=self._zero_dt_indices,
            shard_time_semantics="field_interval_endpoints",
            boundary_frames_leading=self._boundary_frames_leading,
            boundary_frames_trailing=self._boundary_frames_trailing,
            n_frame_files=(
                len(self._state_frames)
                + len(self._projection_frames)
                + len(self._part_frames)
            ),
            mesh_verifications=self._mesh_verifications,
            n_windows=len(metas),
        )


__all__ = [
    "MESH_HASH_KEYS",
    "NAMELIST_CANDIDATES",
    "NATIVE_BUDGET_LAYOUT",
    "NATIVE_CADENCE",
    "NATIVE_LAYOUT",
    "NATIVE_SINGLE_STEP_METHOD",
    "NATIVE_SOURCE_OBJECT",
    "NativeConditionSummary",
    "NativeFrameMeta",
    "NativeRunDataset",
    "SIDECAR_NAME",
    "TRANSPORT_CROSS_CHECK_NAMES",
    "DEFAULT_ZERO_DT_POLICY",
    "RESTART_EVENTS_FILE",
    "RESTART_EVENTS_KEY",
    "ZERO_DT_TOL_JOREK",
    "ZERO_DT_POLICIES",
    "build_native_condition",
    "find_native_frames",
    "is_native_run",
    "load_sidecar",
    "parse_available_restart_files",
    "parse_index_sequence",
    "read_restart_events_file",
    "resolve_restart_events",
    "resolve_namelist_path",
]
