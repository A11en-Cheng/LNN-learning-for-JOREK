"""Zero-dimensional parameter channel for Model 502 (#0005).

Reads the conditioning scalars from the HDF5 frame, the run's Fortran
namelist, and JOREK's documented defaults, then flattens everything into one
canonical float64 vector with per-entry provenance.  Quantities that JOREK
derives at assembly time from raw inputs are computed here with the same
logic (``time_evol_theta``/``time_evol_zeta`` from ``time_evol_scheme``,
per ADR-0007).  Missing namelist entries are filled with the JOREK default
and marked ``provenance="default"`` so consumers can always tell explicit
input from a filled value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.data.namelist import parse_namelist


@dataclass(frozen=True)
class _Spec:
    name: str
    source: str  # "h5" | "namelist" | "derived"
    key: str | None = None
    default: Any = None
    size: int = 1


_DEFAULT_DIFFUSIVITY = [1e-5, 0.0, 0.0, 99.0, 99.0, 0.0]
_DEFAULT_SCHEME = "Crank-Nicholson"

_SPEC: tuple[_Spec, ...] = (
    _Spec("eta", "h5", key="eta", default=1e-5),
    _Spec("visco", "h5", key="visco", default=1e-5),
    _Spec("visco_par", "h5", key="visco_par", default=1e-5),
    _Spec("visco_par_par", "namelist", key="visco_par_par", default=0.0),
    _Spec("ZK_i_par", "namelist", key="zk_i_par", default=1.0),
    _Spec("ZK_e_par", "namelist", key="zk_e_par", default=1.0),
    _Spec("D_perp", "namelist", key="d_perp", default=_DEFAULT_DIFFUSIVITY, size=6),
    _Spec("ZK_i_perp", "namelist", key="zk_i_perp", default=_DEFAULT_DIFFUSIVITY, size=6),
    _Spec("ZK_e_perp", "namelist", key="zk_e_perp", default=_DEFAULT_DIFFUSIVITY, size=6),
    _Spec("time_evol_theta", "derived", key="time_evol_scheme", default=_DEFAULT_SCHEME),
    _Spec("time_evol_zeta", "derived", key="time_evol_scheme", default=_DEFAULT_SCHEME),
    _Spec("T_0", "namelist", key="t_0", default=1e-6),
    _Spec("Ti_0", "namelist", key="ti_0", default=5e-7),
    _Spec("Te_0", "namelist", key="te_0", default=5e-7),
    _Spec("tstep", "h5", key="tstep", default=1.0),
    _Spec("t_now", "h5", key="t_now", default=0.0),
    _Spec("central_density", "h5", key="central_density", default=1.0),
    _Spec("central_mass", "h5", key="central_mass", default=1.0),
    _Spec("F0", "h5", key="F0", default=1.0),
    _Spec("R_domm", "h5", key="R_domm", default=1.0),
    _Spec("sqrt_mu0_rho0", "h5", key="sqrt_mu0_rho0", default=1.0),
    _Spec("t_norm", "h5", key="t_norm", default=1.0),
)


def _flattened_name(spec: _Spec, index: int | None = None) -> str:
    return f"{spec.name}[{index}]" if index is not None else spec.name


def derive_time_evol_params(scheme: str) -> tuple[float, float]:
    """Replicate ``update_time_evol_params.f90``."""
    normalized = scheme.strip().lower()
    if normalized == "implicit euler":
        return 1.0, 0.0
    if normalized == "crank-nicholson":
        return 0.5, 0.0
    if normalized == "gears":
        return 1.0, 0.5
    raise ValueError(f"unknown time_evol_scheme: {scheme!r}")


def rescale_zeta(zeta: float, tstep: float, tstep_prev: float) -> float:
    """Replicate the variable-dt zeta rescaling at ``mod_elt_matrix.f90:225``."""
    return zeta * 2.0 * tstep / (tstep + tstep_prev)


def build_condition_vector(h5_handle, namelist_text: str | None = None) -> dict:
    """Build the canonical condition vector with per-entry provenance.

    ``h5_handle`` is an open :class:`h5py.File` (or mapping-like object with a
    ``__contains__``/``__getitem__`` interface).  Values are resolved h5 first,
    then namelist, then the JOREK default; ``provenance`` records which source
    won for each flattened entry.
    """
    namelist = parse_namelist(namelist_text) if namelist_text else {}

    def h5_value(key: str) -> float:
        return float(h5_handle[key][()].item())

    names: list[str] = []
    vector: list[float] = []
    provenance: dict[str, str] = {}
    scheme = str(namelist.get("time_evol_scheme", _DEFAULT_SCHEME))
    theta, zeta = derive_time_evol_params(scheme)

    for spec in _SPEC:
        if spec.source == "derived":
            values = [theta if spec.name == "time_evol_theta" else zeta]
            source = "derived"
        elif spec.source == "h5":
            if spec.key in h5_handle:
                values, source = [h5_value(spec.key)], "h5"
            else:
                values, source = [spec.default], "default"
        else:
            namelist_key = spec.key
            if namelist_key in namelist:
                raw = namelist[namelist_key]
                if isinstance(raw, list):
                    values = list(raw[: spec.size])
                    while len(values) < spec.size:
                        values.append(spec.default[len(values)] if isinstance(spec.default, list) else 0.0)
                else:
                    values = [float(raw)]
                source = "namelist"
            else:
                values = (
                    list(spec.default[: spec.size])
                    if isinstance(spec.default, list)
                    else [spec.default]
                )
                source = "default"
        for index, value in enumerate(values):
            name = _flattened_name(spec, index + 1 if spec.size > 1 else None)
            names.append(name)
            vector.append(float(value))
            provenance[name] = source

    return {
        "names": names,
        "vector": np.asarray(vector, dtype=np.float64),
        "provenance": provenance,
    }
