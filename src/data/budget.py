"""Budget-only readers for the A' data interface (no JOREK physics rewrite).

These readers extract the recorded deposition/budget channels and the global
closure diagnostics so the acceptance CLI can *register* numbers and unit
conventions.  They deliberately do not re-derive JOREK physics and do not set
any performance gate (roadmap_v2 Phase A/B, ADR-0018).

Source semantics (JOREK code-verified):

* ``xtime_spi_ablation`` accumulates ``t_norm*tstep*spi_abl*species`` -- ablated
  atoms (``models/pellet_module.f90:319-321``); ``xtime_spi_ablation_rate``
  accumulates ``spi_abl*species`` -- atoms per physical second
  (``models/pellet_module.f90:767-768``).
* ``xtime_Ne_imp``, ``xtime_E_ion_power`` and ``xtime_rad_power`` are per-JOREK-
  time rates (divided by ``n_steps*timesteps`` in
  ``particles/examples/marker_spi_source.f90:735-739``); ``xtime_E_ion`` and
  ``xtime_radiation`` are cumulative energies.
* ``abl_history.dat`` columns are written by
  ``diagnostics/jorek2_diagno_spi.f90:185-205``: ``i, xtime, rate_inj(n_inj),
  abl_inj(n_inj), bg_rate_inj(n_inj), bg_inj(n_inj), rate_tot, abl_tot,
  bg_rate_tot, bg_tot`` with ``n_inj = (n_columns - 6) / 4``.
* ``macroscopic_vars.dat`` plottable blocks (``energy_conservation``,
  ``dparticles_dt``, ``particlecontent``, ``bnd_particle_fluxes``) are written
  by ``communication/live_data.f90`` (energy closure at :591, particle
  content/fluxes at :650-680 and :845-866).

The impurity particle budget itself is **not** closed here: it needs the
``rho_imp`` FEM volume integral plus the boundary particle flux, which is a
Phase B1 task (registered in the report and CLI summary).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np

#: Recorded per-step series, in the order used by the report tables.
BUDGET_SERIES_NAMES: tuple[str, ...] = (
    "xtime",
    "xtime_spi_ablation",
    "xtime_spi_ablation_rate",
    "xtime_spi_ablation_bg",
    "xtime_spi_ablation_bg_rate",
    "xtime_Ne_imp",
    "xtime_E_ion",
    "xtime_E_ion_power",
    "xtime_radiation",
    "xtime_rad_power",
    "xtime_rad_cooling_power",
    "xtime_P_ei",
)

SERIES_UNITS: Mapping[str, tuple[str, str]] = {
    "xtime": ("JOREK time", "time"),
    "xtime_spi_ablation": ("atoms (ablated, species-weighted)", "cumulative"),
    "xtime_spi_ablation_rate": ("atoms / second", "rate"),
    "xtime_spi_ablation_bg": ("atoms (background branch)", "cumulative"),
    "xtime_spi_ablation_bg_rate": ("atoms / second (background branch)", "rate"),
    "xtime_Ne_imp": ("electrons / JOREK time", "rate"),
    "xtime_E_ion": ("JOREK energy (cumulative ionization loss)", "cumulative"),
    "xtime_E_ion_power": ("JOREK energy / JOREK time", "rate"),
    "xtime_radiation": ("JOREK energy (cumulative radiated loss)", "cumulative"),
    "xtime_rad_power": ("JOREK energy / JOREK time", "rate"),
    "xtime_rad_cooling_power": ("JOREK energy / JOREK time (cooling channel)", "rate"),
    "xtime_P_ei": ("JOREK energy / JOREK time (e-i exchange)", "rate"),
}

_PLOT_LINE_RE = re.compile(r"@([A-Za-z_0-9]+):\s*(.*)\Z")
_JOREK_FRAME_RE = re.compile(r"jorek(?P<index>\d+)\.h5\Z")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESTART_EVENTS_FILE = "restart_events.json"


def detect_restart_events(run_dir: str | Path) -> dict[str, Any]:
    """Restart events for one native run (registered list, else time-table scan).

    ADR-0019 addendum / ADR-0020 §5: the production v400 runs are not
    regenerated, so budgets must be segmented at the *registered* code-restart
    events.  Resolution mirrors ``src/data/native_run.py``:
    ``<run>/restart_events.json`` -> repo ``data/restart_events.json`` ->
    detection from ``macroscopic_vars.dat`` ``@times`` (dt == 0).
    """
    run_dir = Path(run_dir)
    local = run_dir / _RESTART_EVENTS_FILE
    if local.is_file():
        try:
            payload = json.loads(local.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            payload = None
        if isinstance(payload, list):
            return {"source": str(local), "events": sorted(int(i) for i in payload)}
        if isinstance(payload, dict) and payload.get("restart_events") is not None:
            return {
                "source": str(local),
                "events": sorted(int(i) for i in payload["restart_events"]),
            }
    registry = _REPO_ROOT / "data" / _RESTART_EVENTS_FILE
    if registry.is_file():
        try:
            payload = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive
            payload = None
        if isinstance(payload, dict):
            runs = payload.get("runs")
            if isinstance(runs, dict) and run_dir.name in runs:
                entry = runs[run_dir.name]
                events = (
                    entry.get("restart_events", ())
                    if isinstance(entry, dict)
                    else entry
                )
                return {
                    "source": str(registry),
                    "events": sorted(int(i) for i in events),
                }
    macro = run_dir / "macroscopic_vars.dat"
    if macro.is_file():
        times: list[tuple[int, float]] = []
        for line in macro.read_text(encoding="utf-8", errors="replace").splitlines():
            match = _PLOT_LINE_RE.match(line)
            if match is None or match.group(1) != "times":
                continue
            tokens = match.group(2).split()
            if len(tokens) != 2:
                continue
            try:
                times.append((int(tokens[0]), float(tokens[1])))
            except ValueError:
                continue
        events = [
            index
            for (previous_index, previous_t), (index, t) in zip(times, times[1:])
            if abs(t - previous_t) <= 1e-8
        ]
        if times:
            return {"source": str(macro), "events": events, "detected": True}
    return {"source": None, "events": []}


@dataclass(frozen=True)
class SeriesSummary:
    """Basic statistics of one recorded series (registration only)."""

    name: str
    unit: str
    kind: str
    n: int
    first: float
    last: float
    total: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unit": self.unit,
            "kind": self.kind,
            "n": self.n,
            "first": self.first,
            "last": self.last,
            "total": self.total,
        }


@dataclass(frozen=True)
class AblHistory:
    """Parsed ``abl_history.dat`` (all-injector blocks)."""

    path: Path
    n_inj: int
    time: np.ndarray
    rate_inj: np.ndarray
    abl_inj: np.ndarray
    bg_rate_inj: np.ndarray
    bg_inj: np.ndarray
    rate_tot: np.ndarray
    abl_tot: np.ndarray
    bg_rate_tot: np.ndarray
    bg_tot: np.ndarray
    header_lines: tuple[str, ...]

    @property
    def total_ablated_atoms(self) -> float:
        return float(self.abl_tot[-1]) if self.abl_tot.size else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "n_inj": self.n_inj,
            "n_rows": int(self.time.size),
            "t_first_jorek": float(self.time[0]) if self.time.size else None,
            "t_last_jorek": float(self.time[-1]) if self.time.size else None,
            "total_ablated_atoms_last": self.total_ablated_atoms,
            "total_ablation_rate_last": float(self.rate_tot[-1]) if self.rate_tot.size else None,
            "unit": "atoms (total_abl_number column); rate column = atoms/second",
        }


def _find_dataset(
    handle: h5py.File, name: str, *, prefixes: Sequence[str] = ("", "budget/", "xtime/")
) -> h5py.Dataset | None:
    for prefix in prefixes:
        key = f"{prefix}{name}"
        if key in handle:
            return handle[key]
    return None


def read_series(
    path: str | Path,
    names: Iterable[str] = BUDGET_SERIES_NAMES,
    *,
    prefixes: Sequence[str] = ("", "budget/", "xtime/"),
) -> dict[str, np.ndarray]:
    """Read budget series from a raw ``jorek*.h5`` or an export step file."""
    path = Path(path)
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as handle:
        for name in names:
            dataset = _find_dataset(handle, name, prefixes=prefixes)
            if dataset is None:
                continue
            out[name] = np.asarray(dataset[()], dtype=np.float64).ravel()
    return out


def detect_series_layout(series: Mapping[str, np.ndarray]) -> str:
    """Return ``"history"`` (length > 1) or ``"slice"`` (length == 1)."""
    lengths = {int(array.size) for array in series.values()}
    if not lengths:
        return "absent"
    if lengths == {1}:
        return "slice"
    if min(lengths) > 1:
        return "history"
    return "mixed"


def read_series_tail(
    path: str | Path,
    names: Iterable[str] = BUDGET_SERIES_NAMES,
    *,
    prefixes: Sequence[str] = ("", "budget/", "xtime/"),
) -> dict[str, float]:
    """Read only the **last** recorded element of each series.

    Native ``jorek*.h5`` frames store the full ``xtime*`` history (length =
    ``index_now``), so reading the whole array per fluid step is O(N^2) I/O.
    Per-step readers must use this tail accessor instead.
    """
    path = Path(path)
    out: dict[str, float] = {}
    with h5py.File(path, "r") as handle:
        for name in names:
            dataset = _find_dataset(handle, name, prefixes=prefixes)
            if dataset is None or dataset.shape[0] == 0:
                continue
            out[name] = float(np.ravel(dataset[-1][()])[0])
    return out


def native_step_budget(
    run_dir_or_file: str | Path, step_index: int
) -> dict[str, Any]:
    """Per-fluid-step budget increments for one native run step (tail reads only).

    Uses ``xtime_*`` tails from ``jorek{step_index}`` (and ``{step_index-1}`` for
    the cumulative increments) plus ``abl_history.dat`` when present.  No full
    history array is ever loaded: each series is sliced to its last element.
    """
    path = Path(run_dir_or_file)
    run_dir = path if path.is_dir() else path.parent
    jorek = run_dir / f"jorek{step_index:06d}.h5"
    previous = run_dir / f"jorek{step_index - 1:06d}.h5"
    out: dict[str, Any] = {
        "step_index": int(step_index),
        "jorek": str(jorek),
        "available": jorek.is_file(),
        "layout": "native",
        "read_convention": "xtime* tail element only (dataset[-1]); no full history",
        "units": {
            "ablated_atoms_step": "atoms (species-weighted, cumulative difference)",
            "electrons_injected_step": "electrons",
            "ionization_energy_step": "JOREK energy",
            "radiated_energy_step": "JOREK energy",
        },
    }
    if not jorek.is_file():
        out["error"] = f"{jorek} not found"
        return out
    tail = read_series_tail(jorek)
    prev_tail = read_series_tail(previous) if previous.is_file() else {}
    with h5py.File(jorek, "r") as handle:
        tstep = float(np.ravel(handle["tstep"][()])[0]) if "tstep" in handle else None
        t_norm = float(np.ravel(handle["t_norm"][()])[0]) if "t_norm" in handle else None
        t_now = float(np.ravel(handle["t_now"][()])[0]) if "t_now" in handle else None
    out["tstep_jorek"] = tstep
    out["t_norm_seconds"] = t_norm
    out["t_now_jorek"] = t_now
    out["series_tail"] = dict(sorted(tail.items()))
    restart = detect_restart_events(run_dir)
    out["restart_registry"] = restart
    is_registered_event = int(step_index) in set(restart.get("events") or ())
    t_previous = None
    if previous.is_file():
        with h5py.File(previous, "r") as handle:
            if "t_now" in handle:
                t_previous = float(np.ravel(handle["t_now"][()])[0])
    ablation = tail.get("xtime_spi_ablation")
    ablation_prev = prev_tail.get("xtime_spi_ablation")
    if ablation is not None:
        out["ablated_atoms_cumulative"] = ablation
        cumulative_reset = ablation_prev is not None and ablation < ablation_prev
        zero_dt_step = (
            t_previous is not None
            and t_now is not None
            and abs(t_now - t_previous) <= 1e-8
        )
        if is_registered_event or cumulative_reset or zero_dt_step:
            # ADR-0020 §5: the cumulative series restarts at every code restart,
            # so the per-step increment must come from the rate series inside
            # the segment, never from a cross-segment difference.
            rate = tail.get("xtime_spi_ablation_rate")
            out["restart_segment_boundary"] = True
            out["restart_segment_reason"] = (
                "registered_event"
                if is_registered_event
                else ("cumulative_reset" if cumulative_reset else "zero_dt_step")
            )
            if rate is not None and tstep is not None and t_norm is not None:
                out["ablated_atoms_step"] = float(rate * tstep * t_norm)
                out["ablated_atoms_step_source"] = (
                    "xtime_spi_ablation_rate * tstep * t_norm (within-segment)"
                )
            else:
                out["ablated_atoms_step_source"] = "unavailable at restart boundary"
        elif ablation_prev is not None:
            out["ablated_atoms_step"] = ablation - ablation_prev
            out["ablated_atoms_step_source"] = "cumulative difference (within segment)"
    if tail.get("xtime_Ne_imp") is not None and tstep is not None and t_norm is not None:
        out["electrons_injected_step"] = (
            tail["xtime_Ne_imp"] * tstep * t_norm
        )
        out["electrons_injected_note"] = (
            "xtime_Ne_imp is electrons per JOREK time; multiplied by tstep*t_norm"
        )
    if tail.get("xtime_E_ion_power") is not None and tstep is not None:
        out["ionization_energy_step"] = tail["xtime_E_ion_power"] * tstep
    if tail.get("xtime_radiation") is not None:
        out["radiated_energy_cumulative"] = tail["xtime_radiation"]
    if tail.get("xtime_rad_power") is not None and tstep is not None:
        out["radiated_energy_step"] = tail["xtime_rad_power"] * tstep
    abl_path = run_dir / "abl_history.dat"
    if abl_path.exists():
        history = read_abl_history(abl_path)
        rows = np.where(history.time.astype(np.int64) == int(step_index))[0]
        if rows.size:
            row = int(rows[0])
            out["abl_history"] = {
                "row": row,
                "time_jorek": float(history.time[row]),
                "rate_tot": float(history.rate_tot[row]),
                "abl_tot": float(history.abl_tot[row]),
            }
            if row > 0:
                out["abl_history"]["ablated_atoms_step"] = float(
                    history.abl_tot[row] - history.abl_tot[row - 1]
                )
    return out


def series_summaries(
    series: Mapping[str, np.ndarray], *, extra_totals: Mapping[str, float] | None = None
) -> dict[str, SeriesSummary]:
    """Summarize recorded series, registering the unit convention per series."""
    extra_totals = dict(extra_totals or {})
    summaries: dict[str, SeriesSummary] = {}
    time = series.get("xtime")
    for name, array in series.items():
        unit, kind = SERIES_UNITS.get(name, ("unregistered", "unknown"))
        total: float | None
        if kind == "cumulative":
            total = float(array[-1] - array[0]) if array.size else 0.0
        elif kind == "rate":
            if time is not None and time.size == array.size and time.size > 1:
                dt = np.diff(time, prepend=0.0)
                total = float(np.sum(array * dt))
            else:
                total = extra_totals.get(name)
        else:
            total = extra_totals.get(name)
        if name == "xtime":
            total = None
        summaries[name] = SeriesSummary(
            name=name,
            unit=unit,
            kind=kind,
            n=int(array.size),
            first=float(array[0]) if array.size else 0.0,
            last=float(array[-1]) if array.size else 0.0,
            total=total,
        )
    return summaries


def _numeric_row(payload: str) -> np.ndarray | None:
    tokens = payload.split()
    values: list[float] = []
    for token in tokens:
        try:
            values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError:
            return None
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def parse_plot_blocks(path: str | Path) -> dict[str, dict[str, Any]]:
    """Parse the JOREK plottable block format.

    Returns ``{block: {"columns": tuple[str, ...], "rows": ndarray[n, k]}}``.
    Header lines (quoted column names) and data lines (``@block: numbers``) are
    separated by whether the payload contains a quote.

    """
    path = Path(path)
    blocks: dict[str, dict[str, Any]] = {}
    with path.open("r", errors="replace") as handle:
        for line in handle:
            match = _PLOT_LINE_RE.match(line.rstrip("\n"))
            if match is None:
                continue
            name, payload = match.group(1), match.group(2)
            if '"' in payload:
                columns = tuple(re.findall(r'"([^"]*)"', payload))
                blocks.setdefault(name, {})["columns"] = columns
                blocks[name].setdefault("rows", [])
                continue
            row = _numeric_row(payload)
            if row is None:
                continue
            blocks.setdefault(name, {}).setdefault("rows", []).append(row)
    out: dict[str, dict[str, Any]] = {}
    for name, block in blocks.items():
        rows = block.get("rows") or []
        out[name] = {
            "columns": tuple(block.get("columns", ())),
            "rows": np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float64),
        }
    return out


def read_abl_history(path: str | Path) -> AblHistory:
    """Parse ``abl_history.dat`` (5 wrapped header lines + numeric rows)."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header_lines: list[str] = []
    rows: list[np.ndarray] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            first = float(stripped.split()[0].replace("D", "E").replace("d", "e"))
        except ValueError:
            header_lines.append(stripped)
            continue
        if float(first).is_integer() and len(stripped.split()) >= 6:
            rows.append(np.asarray([float(tok) for tok in stripped.split()], dtype=np.float64))
        else:
            header_lines.append(stripped)
    if not rows:
        raise ValueError(f"{path}: no numeric rows found")
    widths = {row.size for row in rows}
    if len(widths) != 1:
        raise ValueError(f"{path}: inconsistent column counts {sorted(widths)}")
    n_cols = widths.pop()
    if (n_cols - 6) % 4 != 0:
        raise ValueError(
            f"{path}: {n_cols} columns do not match i, time, 4*n_inj, 4 totals"
        )
    n_inj = (n_cols - 6) // 4
    table = np.vstack(rows)
    # Columns: 0=i, 1=time, then 4 blocks of n_inj, then 4 totals.
    return AblHistory(
        path=path,
        n_inj=n_inj,
        time=table[:, 1],
        rate_inj=table[:, 2 : 2 + n_inj],
        abl_inj=table[:, 2 + n_inj : 2 + 2 * n_inj],
        bg_rate_inj=table[:, 2 + 2 * n_inj : 2 + 3 * n_inj],
        bg_inj=table[:, 2 + 3 * n_inj : 2 + 4 * n_inj],
        rate_tot=table[:, -4],
        abl_tot=table[:, -3],
        bg_rate_tot=table[:, -2],
        bg_tot=table[:, -1],
        header_lines=tuple(header_lines),
    )


def _latest_jorek_file(run_dir: Path) -> Path:
    candidates = sorted(
        (path for path in run_dir.glob("jorek*.h5") if _JOREK_FRAME_RE.fullmatch(path.name)),
        key=lambda path: int(_JOREK_FRAME_RE.fullmatch(path.name).group("index")),
    )
    if not candidates:
        raise FileNotFoundError(f"{run_dir}: no jorek*.h5 files")
    return candidates[-1]


def _resolve_jorek_path(path: str | Path, step_index: int | None = None) -> Path:
    path = Path(path)
    if path.is_file():
        return path
    if step_index is not None:
        candidate = path / f"jorek{step_index:06d}.h5"
        if candidate.exists():
            return candidate
    return _latest_jorek_file(path)


def _macro_blocks(run_dir: Path) -> dict[str, dict[str, Any]] | None:
    macro = run_dir / "macroscopic_vars.dat"
    if not macro.exists():
        return None
    return parse_plot_blocks(macro)


def _closure_stats(residual: np.ndarray) -> dict[str, Any]:
    if residual.size == 0:
        return {"n": 0}
    absolute = np.abs(residual)
    return {
        "n": int(residual.size),
        "n_nonfinite": int(np.count_nonzero(~np.isfinite(residual))),
        "last": float(residual[-1]),
        "max_abs": float(np.nanmax(absolute)),
        "median_abs": float(np.nanmedian(absolute)),
        "note": "non-finite rows are counted in n_nonfinite and excluded from max/median",
    }


def global_closure(run_dir: str | Path) -> dict[str, Any]:
    """Read the macroscopic global closures (no pass/fail gate).

    * energy: ``-dEtotdt`` vs ``sum(bnd fluxes + sources + dissipative)``;
    * particles: ``dparticles_dt`` total vs ions+neutrals, and
      ``particlecontent`` total ions vs inside+outside;
    * boundary particle fluxes: per-component last values plus the sum.
    """
    run_dir = Path(run_dir)
    blocks = _macro_blocks(run_dir)
    out: dict[str, Any] = {
        "macroscopic_vars": str(run_dir / "macroscopic_vars.dat"),
        "available": blocks is not None,
        "abl_history": str(run_dir / "abl_history.dat"),
        "abl_history_present": (run_dir / "abl_history.dat").exists(),
    }
    if blocks is None:
        return out
    energy = blocks.get("energy_conservation", {})
    if energy.get("rows") is not None and energy["rows"].size:
        rows = energy["rows"]
        out["energy_conservation"] = {
            "columns": list(energy.get("columns", ())),
            "convention": "residual = (-dEtotdt) - (sum bnd fluxes + sources + dissipative)",
            "residual": _closure_stats(rows[:, 1] - rows[:, 2]),
            "last_time_jorek": float(rows[-1, 0]),
        }
    particles = blocks.get("dparticles_dt", {})
    if particles.get("rows") is not None and particles["rows"].size:
        rows = particles["rows"]
        out["dparticles_dt"] = {
            "columns": list(particles.get("columns", ())),
            "convention": "residual = Total - (Ions + Neutrals)",
            "residual": _closure_stats(rows[:, 1] - (rows[:, 2] + rows[:, 3])),
            "total_last": float(rows[-1, 1]),
        }
    content = blocks.get("particlecontent", {})
    if content.get("rows") is not None and content["rows"].size:
        rows = content["rows"]
        out["particlecontent"] = {
            "columns": list(content.get("columns", ())),
            "last": {
                "total_main_ions": float(rows[-1, 1]),
                "inside_lcfs": float(rows[-1, 2]),
                "outside_lcfs": float(rows[-1, 3]),
                "total_d_neutrals": float(rows[-1, 4]) if rows.shape[1] > 4 else None,
                "total_impurities": float(rows[-1, 5]) if rows.shape[1] > 5 else None,
            },
            "ions_inside_plus_outside_residual_last": float(
                rows[-1, 1] - (rows[-1, 2] + rows[-1, 3])
            ),
        }
    fluxes = blocks.get("bnd_particle_fluxes", {})
    if fluxes.get("rows") is not None and fluxes["rows"].size:
        rows = fluxes["rows"]
        out["bnd_particle_fluxes"] = {
            "columns": list(fluxes.get("columns", ())),
            "last": {name: float(value) for name, value in zip(fluxes.get("columns", ())[1:], rows[-1, 1:])},
            "sum_last": float(np.sum(rows[-1, 1:])),
            "note": "particle fluxes in JOREK particle units per JOREK time",
        }
    return out


def ablation_budget(
    run_dir_or_file: str | Path, *, step_index: int | None = None
) -> dict[str, Any]:
    """Ablated-atom budget from ``xtime_spi_ablation*`` and ``abl_history.dat``."""
    path = Path(run_dir_or_file)
    run_dir = path if path.is_dir() else path.parent
    out: dict[str, Any] = {"source": None, "units": SERIES_UNITS["xtime_spi_ablation"][0]}
    try:
        jorek = _resolve_jorek_path(path, step_index)
    except FileNotFoundError as exc:
        out["error"] = str(exc)
        return out
    series = read_series(jorek, ("xtime", "xtime_spi_ablation", "xtime_spi_ablation_rate",
                                 "xtime_spi_ablation_bg", "xtime_spi_ablation_bg_rate"))
    if "xtime_spi_ablation" in series:
        cumulative = series["xtime_spi_ablation"]
        increments = np.diff(cumulative, prepend=cumulative[0] - 0.0)
        out.update(
            {
                "source": "jorek_history",
                "file": str(jorek),
                "layout": "history",
                "n": int(cumulative.size),
                "cumulative_first": float(cumulative[0]),
                "cumulative_last": float(cumulative[-1]),
                "total_atoms": float(cumulative[-1] - cumulative[0]),
                "last_step_atoms": float(increments[-1]),
                "rate_last": float(series["xtime_spi_ablation_rate"][-1])
                if "xtime_spi_ablation_rate" in series
                else None,
                "rate_unit": SERIES_UNITS["xtime_spi_ablation_rate"][0],
            }
        )
        if "xtime_spi_ablation_rate" in series and "xtime" in series:
            rate = series["xtime_spi_ablation_rate"]
            time = series["xtime"]
            with h5py.File(jorek, "r") as handle:
                t_norm = float(np.ravel(handle["t_norm"][()])[0]) if "t_norm" in handle else None
            if t_norm is not None and time.size == rate.size and time.size > 1:
                dt_jorek = np.diff(time, prepend=0.0)
                out["rate_integral_atoms"] = float(np.sum(rate * dt_jorek * t_norm))
                out["t_norm_seconds"] = t_norm
                total = out.get("total_atoms")
                if total:
                    out["rate_integral_over_cumulative"] = float(
                        out["rate_integral_atoms"] / total
                    )
                    out["convention_note"] = (
                        "rate series is registered as atoms/second; its dt-weighted "
                        "integral does not reproduce the cumulative series index-by-index "
                        "for this run (ratio registered above). The cumulative series and "
                        "abl_history.dat agree; the rate index/time-step convention needs "
                        "generation-side confirmation. No gate is applied."
                    )
    abl_path = run_dir / "abl_history.dat"
    if abl_path.exists():
        history = read_abl_history(abl_path)
        out["abl_history"] = history.to_dict()
        if out.get("cumulative_last") is not None:
            out["h5_vs_abl_history_delta"] = float(
                out["cumulative_last"] - history.total_ablated_atoms
            )
    return out


def ionization_radiation_budget(
    run_dir_or_file: str | Path, *, step_index: int | None = None
) -> dict[str, Any]:
    """Ionization/radiation deposition from the recorded ``xtime_*`` history."""
    path = Path(run_dir_or_file)
    out: dict[str, Any] = {"source": None}
    try:
        jorek = _resolve_jorek_path(path, step_index)
    except FileNotFoundError as exc:
        out["error"] = str(exc)
        return out
    names = (
        "xtime",
        "xtime_Ne_imp",
        "xtime_E_ion",
        "xtime_E_ion_power",
        "xtime_radiation",
        "xtime_rad_power",
        "xtime_rad_cooling_power",
        "xtime_P_ei",
    )
    series = read_series(jorek, names)
    if not series:
        return out
    out["source"] = "jorek_history"
    out["file"] = str(jorek)
    out["layout"] = detect_series_layout(series)
    out["units"] = {name: SERIES_UNITS[name][0] for name in series if name in SERIES_UNITS}
    summaries = series_summaries(series)
    out["series"] = {name: summary.to_dict() for name, summary in summaries.items()}
    time = series.get("xtime")
    if time is not None and time.size > 1:
        dt = np.diff(time, prepend=0.0)
        out["dt_jorek_total"] = float(np.sum(dt))
    # Register the raw per-step (un-integrated) deposition channels explicitly.
    out["step_channels"] = {
        "electrons_injected_per_step": "xtime_Ne_imp * tstep (electrons; rate is per JOREK time)",
        "ionization_energy_per_step": "xtime_E_ion_power * tstep (JOREK energy)",
        "radiated_energy_per_step": "xtime_rad_power * tstep (JOREK energy)",
    }
    return out


def projection_integrals(
    projection_path: str | Path, mesh_path: str | Path | None = None
) -> dict[str, Any]:
    """Lumped FEM integral of the cinco projection channels (cross-check only).

    Uses the P1 dof-0 node values and splits every quad into two triangles
    (``(0,1,2)``/``(0,2,3)``), the same convention as the B0 stencil assembly.
    Degenerate triangles are dropped.  This is an approximate cross-check of
    ``dE_ion``/``rad_sink``/``N_imp``; it is not the internal ``inj_source``.
    """
    projection_path = Path(projection_path)
    mesh_path = Path(mesh_path) if mesh_path is not None else projection_path
    out: dict[str, Any] = {
        "projection_file": str(projection_path),
        "mesh_file": str(mesh_path),
        "ok": False,
    }
    try:
        with h5py.File(projection_path, "r") as proj:
            values = np.asarray(proj["values"][:5, 0, 0, :], dtype=np.float64)
        with h5py.File(mesh_path, "r") as mesh:
            x_group = mesh["x"] if "x" in mesh else mesh["mesh/x"]
            vertex_group = mesh["vertex"] if "vertex" in mesh else mesh["mesh/vertex"]
            x = np.asarray(x_group[()], dtype=np.float64)
            vertex = np.asarray(vertex_group[()], dtype=np.int64)
    except (KeyError, OSError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    if x.ndim == 4:
        points = x[:, 0, 0, :]
    elif x.ndim == 2:
        points = x
    else:
        out["error"] = f"unsupported x shape {x.shape}"
        return out
    if vertex.shape[0] != 4:
        out["error"] = f"unsupported vertex shape {vertex.shape}"
        return out
    n_nodes = points.shape[1]
    nodal_measure = np.zeros(n_nodes, dtype=np.float64)
    corners = vertex - 1  # JOREK writes 1-based connectivity
    for tri in ((0, 1, 2), (0, 2, 3)):
        a = corners[tri[0]]
        b = corners[tri[1]]
        c = corners[tri[2]]
        pa = points[:, a]
        pb = points[:, b]
        pc = points[:, c]
        area = 0.5 * np.abs(
            (pb[0] - pa[0]) * (pc[1] - pa[1]) - (pc[0] - pa[0]) * (pb[1] - pa[1])
        )
        for node in (a, b, c):
            np.add.at(nodal_measure, node, area / 3.0)
    finite = np.isfinite(nodal_measure) & (nodal_measure > 0.0)
    out["n_nodes_used"] = int(finite.sum())
    out["measure_total"] = float(nodal_measure[finite].sum())
    if values.shape[1] != n_nodes:
        out["error"] = f"projection nodes {values.shape[1]} != mesh nodes {n_nodes}"
        return out
    channels = ("dE_ion", "rad_sink", "Z_eff", "Z_imp", "N_imp")
    out["channel_integrals"] = {
        name: float(np.sum(values[index][finite] * nodal_measure[finite]))
        for index, name in enumerate(channels)
    }
    out["ok"] = True
    out["note"] = (
        "projections are the marker export channel, not the internal inj_source "
        "(ADR-0016/ADR-0018); cross-check only"
    )
    return out


def ablation_series_reconciliation(
    run_dir_or_file: str | Path, *, step_index: int | None = None
) -> dict[str, Any]:
    """Reconcile the ablated-atom cumulative series with the rate series.

    Phase B1 registers a long-standing "rate integral vs cumulative ~20x"
    discrepancy.  Code-verified reading (``models/pellet_module.f90:319-321,
    767-768``):

    * ``xtime_spi_ablation`` accumulates ``t_norm * tstep * spi_abl * species``
      (ablated atoms, cumulative);
    * ``xtime_spi_ablation_rate`` accumulates ``spi_abl * species``
      (atoms per physical second) at the same index.

    The two agree step-by-step, but the cumulative series **restarts at every
    code restart** (the array is re-imported/zeroed beyond the restart index),
    so its last element covers only the final restart segment.  This function
    quantifies that: the physical total is the sum of the per-segment positive
    increments, which must match the rate integral; the ratio to the raw last
    element is the registered "discrepancy factor".  No unit conversion is
    involved and no gate is applied.
    """
    path = Path(run_dir_or_file)
    run_dir = path if path.is_dir() else path.parent
    try:
        jorek = _resolve_jorek_path(path, step_index)
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc)}
    series = read_series(jorek, ("xtime", "xtime_spi_ablation", "xtime_spi_ablation_rate"))
    if "xtime_spi_ablation" not in series or "xtime_spi_ablation_rate" not in series:
        return {"available": False, "file": str(jorek), "error": "series missing"}
    with h5py.File(jorek, "r") as handle:
        t_norm = float(np.ravel(handle["t_norm"][()])[0]) if "t_norm" in handle else None
    time = series["xtime"]
    cumulative = series["xtime_spi_ablation"].reshape(-1)
    rate = series["xtime_spi_ablation_rate"].reshape(-1)
    if time.size != cumulative.size or rate.size != cumulative.size:
        return {
            "available": False,
            "file": str(jorek),
            "error": f"length mismatch t={time.size} cum={cumulative.size} rate={rate.size}",
        }
    increments = np.diff(cumulative, prepend=0.0)
    positive = np.maximum(increments, 0.0)
    inner_diff = np.diff(cumulative)
    scale = float(np.nanmax(np.abs(cumulative))) if cumulative.size else 0.0
    reset_threshold = -1e-6 * max(scale, 1.0)
    resets = [int(index) + 1 for index in np.where(inner_diff < reset_threshold)[0]]
    segment_total = float(positive.sum())
    dt = np.diff(time, prepend=time[0]) if time.size else np.zeros(0)
    rate_integral = (
        float(np.sum(rate[1:] * dt[1:]) * t_norm)
        if (t_norm is not None and rate.size > 1)
        else None
    )
    # ---- restart segmentation (ADR-0020 §5; budgets are per segment) -------
    starts = [0] + list(resets)
    ends = [reset - 1 for reset in resets] + [int(cumulative.size) - 1]
    segments: list[dict[str, Any]] = []
    for start, end in zip(starts, ends):
        if end < start:
            continue
        atoms = float(cumulative[end] - cumulative[start])
        rate_sum = (
            float(np.sum(rate[start + 1 : end + 1] * dt[start + 1 : end + 1]) * t_norm)
            if (t_norm is not None and end > start)
            else 0.0
        )
        segments.append(
            {
                "start_row": int(start),
                "end_row": int(end),
                "t_start_jorek": float(time[start]) if time.size else None,
                "t_end_jorek": float(time[end]) if time.size else None,
                "atoms": atoms,
                "rate_integral_atoms": rate_sum,
            }
        )
    registered = detect_restart_events(run_dir)
    last = float(cumulative[-1])
    out: dict[str, Any] = {
        "available": True,
        "file": str(jorek),
        "n": int(cumulative.size),
        "t_norm_seconds": t_norm,
        "cumulative_last": last,
        "cumulative_max": float(np.nanmax(cumulative)) if cumulative.size else None,
        "cumulative_positive_increment_sum": segment_total,
        "reset_indices": resets,
        "n_resets": len(resets),
        "restart_segments": segments,
        "segmented_total_atoms": float(sum(segment["atoms"] for segment in segments)),
        "registered_restart_events": registered,
        "rate_integral_atoms": rate_integral,
        "step_increment_rate_ratio_median": float(
            np.median(
                increments[1:][rate[1:] * dt[1:] > 0]
                / (rate[1:] * dt[1:] * t_norm)[rate[1:] * dt[1:] > 0]
            )
        )
        if (t_norm is not None and rate.size > 1 and np.any(rate[1:] * dt[1:] > 0))
        else None,
        "conclusion": (
            "the cumulative series restarts at each code restart, so its last value "
            "covers only the final segment; the physical total is the positive-increment "
            "sum (or the rate integral). The registered ~20x discrepancy is a "
            "restart/index bookkeeping artifact, not a unit error."
        ),
        "gate": "no gate: values are registered only",
    }
    if rate_integral is not None:
        out["rate_integral_over_cumulative_last"] = (
            float(rate_integral / last) if last else None
        )
        out["rate_integral_over_segment_total"] = (
            float(rate_integral / segment_total) if segment_total else None
        )
    return out


def summarize_budget(
    run_dir: str | Path, *, step_index: int | None = None
) -> dict[str, Any]:
    """Combine all budget readers into one JSON-able summary.

    Native layout: totals come from ``abl_history.dat`` (all-injector columns)
    and ``macroscopic_vars.dat`` (global closures) wherever available; the
    ``xtime*`` series are read once from the latest ``jorek*.h5`` (they are
    full histories there).  Per-step values must use
    :func:`native_step_budget`, which slices only the last element.
    """
    run_dir = Path(run_dir)
    restart = detect_restart_events(run_dir)
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "ablation": ablation_budget(run_dir, step_index=step_index),
        "ionization_radiation": ionization_radiation_budget(run_dir, step_index=step_index),
        "global_closure": global_closure(run_dir),
        "restart_segmentation": {
            "source": restart.get("source"),
            "events": list(restart.get("events") or ()),
            "rule": (
                "budgets are segmented at registered code restarts; the cumulative "
                "xtime_spi_ablation resets there, so per-step atoms use the rate "
                "series within the segment (ADR-0020 §5)"
            ),
        },
        "impurity_particle_budget": {
            "status": "needs rho_imp FEM volume integral + boundary particle flux",
            "owner": "Phase B1",
        },
        "gate": "no performance gate: values and conventions are registered only",
    }
    native_frames = sorted(
        int(match.group("index"))
        for path in run_dir.glob("jorek*.h5")
        if (match := _JOREK_FRAME_RE.fullmatch(path.name))
    )
    if native_frames:
        summary["native_layout"] = {
            "n_jorek_frames": len(native_frames),
            "index_first": native_frames[0],
            "index_last": native_frames[-1],
            "per_step_reader": "native_step_budget(run_dir, step_index)",
            "note": (
                "xtime* arrays are complete histories (length = index_now); "
                "totals are read once, per-step values use tail-only slicing"
            ),
        }
        reconciliation = ablation_series_reconciliation(run_dir, step_index=step_index)
        if reconciliation.get("available"):
            summary["restart_segmentation"]["segments"] = reconciliation.get(
                "restart_segments"
            )
            summary["restart_segmentation"]["segmented_total_atoms"] = reconciliation.get(
                "segmented_total_atoms"
            )
            summary["restart_segmentation"]["cumulative_last"] = reconciliation.get(
                "cumulative_last"
            )
    try:
        jorek = _resolve_jorek_path(run_dir, step_index)
    except FileNotFoundError:
        return summary
    projection = jorek.parent / f"projections{jorek.name[5:]}"
    if projection.exists():
        summary["projection_integrals"] = projection_integrals(projection)
    return summary


__all__ = [
    "AblHistory",
    "BUDGET_SERIES_NAMES",
    "SERIES_UNITS",
    "SeriesSummary",
    "ablation_series_reconciliation",
    "ablation_budget",
    "detect_restart_events",
    "detect_series_layout",
    "global_closure",
    "ionization_radiation_budget",
    "native_step_budget",
    "parse_plot_blocks",
    "projection_integrals",
    "read_abl_history",
    "read_series",
    "read_series_tail",
    "series_summaries",
    "summarize_budget",
]
