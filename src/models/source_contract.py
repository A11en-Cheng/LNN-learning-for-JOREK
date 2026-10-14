"""Shared five-channel ``S`` interface (roadmap_v2 §3.1, ADR-0020).

Phase B1 (self-generated source), Phase B2 (stencil-conv nine-field propagator)
and Phase C (staggered closed loop) must consume the **same** source object.
This module freezes that object so no downstream component redefines it:

* **channel order (frozen)**: ``dE_ion, rad_sink, Z_eff, Z_imp, N_imp``;
* **node layout (frozen)**: node dof-0 values, shape ``[nodes, 5]`` in memory
  (the recorded files store ``[5, dof, harmonics, nodes]``; readers slice
  ``dof=0`` and transpose -- see ``src/data/native_run.py``);
* **gather-time semantics (frozen)**: each value is the time average of the
  marker deposits accumulated over the source window ``[T_{k-1}, T_k]``,
  i.e. ``sum_substeps(deposit) / (n_steps * timesteps)``
  (``particles/examples/marker_spi_source.f90:442-443,670-676``); the
  averaging divides by actual particle gather time. A registered duplicate
  fluid output is not a zero-length particle window (ADR-0020 decision 5);
* **units (frozen)**: the recorded ``projections*.h5`` normalisation.
  ``dE_ion``/``rad_sink`` carry the ``t_norm/E_norm`` scaling
  (``marker_spi_source.f90:257-261``); ``Z_eff``/``Z_imp`` are charge-square
  and charge density moments, NOT the normalized means ``<q^2>``/``<q>``;
  the consumer divides by its floored channel 5 at Gauss points. ``N_imp`` is the time-averaged
  impurity number density in units of ``n_norm = central_density*1e20``
  (``marker_spi_source.f90:535-537,626-640``).  The exporter registers the
  per-run conversion into the field-equation source units; the shared object
  itself is never silently renormalised downstream;
* **true-source declaration for marker runs**: with ``use_marker=.t.`` the
  recorded five-channel marker feedback *is* the actual field source for the
  energy/radiation/charge channels (``models/model502/mod_elt_matrix.f90``:
  ``:989-1003`` consumes ``eq_aux_g``, ``:1129`` builds ``Z_eff``,
  ``:1728`` feeds ``aux_dEion_dt + aux_rad`` into equation 9), while the
  analytic ``total_imp_source`` path only supplies the impurity density
  source (``:1219``, ``rhs_ij_8 :1717``, ``rhs_ij_5 :1542``).  The marker
  feedback is therefore the B1 target/source object -- not a proxy
  (ADR-0020 decision 1).

The deterministic consumption boundary is ``marker_feedback.GaussMarkerFeedback``:
full-basis values, reference derivatives and raw projection deltas already
evaluated at Gauss points, with explicit physical normalization. The shared
node-dof0 view above cannot directly satisfy that boundary. Channel 5 is neither
the native fluid impurity MASS density ``rho_imp`` nor its particle injection
source. The learned ``FieldPropagator.source_proj`` remains an unaligned
approximation; it does not implement the deterministic formula consumer or FEM
assembly. See ``report/marker-feedback-consumption-2026-10-07.md``.

The complete native cubic reference boundary is now provided by
``native_gauss.evaluate_native_element`` and
``native_gauss.marker_electron_energy_local_loads``: all four degrees and all
declared harmonics, signed element sizes, geometry and raw projection deltas
are required. It supplies local electron-energy source/history loads only;
it does not replace the shared dof0 view or align ``source_proj``. See
``report/native-gauss-local-marker-loads-2026-10-08.md`` for the input/output
contract and remaining assembly requirements.

The small native coupled Adapter is ``native_coupling.CoupledNativeField``:
its marker object is explicitly full ``[5,4,1,N]`` with independent raw history,
not the legacy dof0 observation view. ``NativeDensitySource`` is a separate
``[2,4,1,N]`` bg/imp mass-density rate object. Concrete small-cloud event
integration reuses ``ReferenceRollout``. Under ADR-0022 the default
``joint_learning`` mode encodes full source coefficients, native Gauss values/
gradients, deterministic charge closures, raw history and source clocks before
the stencil/CfC; it predicts the full joint nine-field increment. No frozen
capacity increment is computed or added in that mode. The mutually exclusive
``frozen99_reference`` mode retains the old approximate response. Pressure,
Zeff/ne and quotient-history features now enter learning, while pressure/
resistivity/exchange physical operators remain unimplemented. Source events are NumPy;
the conditional-on-source field Interface is differentiable Torch. Versions
are checkpoint-bound; field-only checkpoints are not silently upgraded. See
``report/native-joint-source-learning-2026-10-08.md``; the prior frozen-capacity
experiment remains in ``report/native-source-field-coupling-2026-10-08.md``.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

__all__ = [
    "N_SOURCE_FIELDS",
    "SOURCE_CHANNEL_ORDER",
    "SOURCE_CHANNEL_UNITS",
    "SOURCE_LAYOUT",
    "SOURCE_OBJECT_ANALYTIC_SHARD",
    "SOURCE_OBJECT_INJ_SOURCE",
    "SOURCE_OBJECT_MARKER_FEEDBACK",
    "SHARED_S_SPEC",
    "describe_source_object",
    "shared_s_spec",
    "validate_shared_source",
]

#: Number of source channels in the shared object.
N_SOURCE_FIELDS = 5

#: Frozen channel order (ADR-0020 decision 6; never reorder downstream).
SOURCE_CHANNEL_ORDER: tuple[str, ...] = (
    "dE_ion",
    "rad_sink",
    "Z_eff",
    "Z_imp",
    "N_imp",
)

#: Human-readable unit string per channel (registration only; the numeric
#: convention is the recorded projection normalisation, see module docstring).
SOURCE_CHANNEL_UNITS: Mapping[str, str] = {
    "dE_ion": "JOREK energy source, gather-time average (t_norm/E_norm scaling)",
    "rad_sink": "JOREK energy sink, gather-time average (t_norm/E_norm scaling)",
    "Z_eff": "sum(weight*q^2) / n_norm (recorded convention; the field-level "
    "<q^2>/<1> is recovered downstream by dividing by channel N_imp)",
    "Z_imp": "sum(weight*q) / n_norm (recorded convention; <q>/<1> recovered by "
    "dividing by channel N_imp)",
    "N_imp": "sum(weight) / n_norm (recorded convention; n_norm = central_density*1e20)",
}

#: Recorded array layout after reading.
SOURCE_LAYOUT = "node_dof0"

#: Registered source-object kinds (``provenance.source_object``).
SOURCE_OBJECT_MARKER_FEEDBACK = "marker_feedback"
SOURCE_OBJECT_INJ_SOURCE = "inj_source"
SOURCE_OBJECT_ANALYTIC_SHARD = "analytic_shard"

_SOURCE_OBJECT_DESCRIPTIONS: Mapping[str, str] = {
    SOURCE_OBJECT_MARKER_FEEDBACK: (
        "recorded marker feedback (projections*.h5 five channels); in a marker run "
        "this is the actual field source for dE_ion/rad_sink/Z_eff/Z_imp/N_imp "
        "(mod_elt_matrix.f90:989-1003,1129,1728) -- the true source object"
    ),
    SOURCE_OBJECT_INJ_SOURCE: (
        "internal model502 total_imp_source -> inj_source (Gauss-point shard "
        "deposition into the RHS; explicit in the fields); supplies the impurity "
        "density source (mod_elt_matrix.f90:1219,1542,1717)"
    ),
    SOURCE_OBJECT_ANALYTIC_SHARD: (
        "deterministic analytic shard deposition in inj_source semantics; B1's "
        "deterministic main term, never a substitute for the marker feedback when "
        "the run is a marker run"
    ),
}


def describe_source_object(kind: str) -> str:
    """Return the registered description of a source-object kind."""

    try:
        return _SOURCE_OBJECT_DESCRIPTIONS[kind]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown source object {kind!r}; registered kinds: "
            f"{sorted(_SOURCE_OBJECT_DESCRIPTIONS)}"
        ) from exc


def shared_s_spec() -> dict[str, Any]:
    """JSON-able freeze record shared by B1/B2/C and the docs."""

    return {
        "n_channels": N_SOURCE_FIELDS,
        "channel_order": list(SOURCE_CHANNEL_ORDER),
        "channel_units": dict(SOURCE_CHANNEL_UNITS),
        "layout": SOURCE_LAYOUT,
        "recorded_file_layout": "[5, dof, harmonics, nodes] (dof-0 slice)",
        "gather_time_average": (
            "sum_substeps(deposit) / (n_steps * timesteps); actual particle gather time; "
            "registered duplicate fluid times do not imply zero particle duration (ADR-0020 §5)"
        ),
        "marker_run_true_source": SOURCE_OBJECT_MARKER_FEEDBACK,
        "source_objects": {
            kind: description for kind, description in _SOURCE_OBJECT_DESCRIPTIONS.items()
        },
        "rhs_mapping": (
            "B2 consumes the same tensor object as an explicit RHS forcing; the "
            "exporter registers any per-run conversion into field-equation units "
            "(identity otherwise)"
        ),
        "authority": "roadmap_v2.md §3.1 + docs/adr/0020-marker-source-contract.md",
    }


def validate_shared_source(
    source: torch.Tensor,
    *,
    what: str = "S",
    n_nodes: int | None = None,
    allow_nan: bool = False,
) -> torch.Tensor:
    """Validate a tensor against the shared ``S`` interface and return it.

    Accepted shapes: ``[nodes, 5]`` (single window, B2/C) or ``[batch, nodes, 5]``
    (batched B1).  The check is structural only (order/shape/finiteness); the
    object/unit declaration lives in the caller's provenance.
    """

    if source.ndim not in (2, 3) or source.shape[-1] != N_SOURCE_FIELDS:
        raise ValueError(
            f"{what} must be [nodes, {N_SOURCE_FIELDS}] or [batch, nodes, "
            f"{N_SOURCE_FIELDS}] in the shared channel order "
            f"{SOURCE_CHANNEL_ORDER}, got {tuple(source.shape)}"
        )
    if n_nodes is not None and source.shape[-2] != int(n_nodes):
        raise ValueError(f"{what} has {source.shape[-2]} nodes, expected {n_nodes}")
    if not allow_nan and not bool(torch.isfinite(source).all()):
        raise ValueError(f"{what} contains non-finite values; the shared S contract is fail-fast")
    return source
