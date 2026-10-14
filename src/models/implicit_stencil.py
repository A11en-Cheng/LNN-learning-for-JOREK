"""Implicit / semi-implicit stencil update for the nine Model-502 fields.

Phase B2 built the spatial operator (``src.models.stencil_conv``) but its
*explicit* layer is not a propagator: the geometric bound is
``dt <= 2 / lambda_max ~ 1.7e-7`` JOREK time and the per-term physical bounds
``dt_term = 2 / (alpha_term * lambda_max)`` drop to 1e-13..1e-11 for the
parallel transport terms (``docs/jorek-normalization.md`` section 3,
ADR-0018 decision 13).  At the physical cadence ``dt = 0.05`` JOREK time the
update must therefore solve

    (I + dt * sum_t alpha_t * S_t) x_new = rhs,     S_t = A_t / m >= 0

B2.1 (2026-10-04) revisions, all code-verified against
``models/model502/mod_elt_matrix.f90``:

* **Per-term metric weights** (``TERM_METRIC_WEIGHT``).  JOREK does *not* use a
  single geometric weight: parallel terms carry ``BigR/BB2``
  (:1546-1547, 1595, 1642, 1754), perpendicular/viscous terms carry ``BigR``
  (:1492, 1565, 1598, 1641, 1649, 1720), and the numerical viscosity
  ``visco_num`` carries no weight at all (:1496, 2031).  Because
  ``Bgrad_* = B . grad`` with ``B`` in tesla (:1455-1460), the parallel weight
  applied to a *unit-direction* P1 kernel collapses exactly::

      BigR/BB2 * (B.grad N_i)(B.grad N_j) = BigR * (bhat.grad N_i)(bhat.grad N_j)

  so the assembled operator is the B0 field-aligned kernel scaled by ``BigR``
  (``R_REF = 1 m`` keeps the relative weight dimensionless).  A dedicated test
  re-derives the equality from a B-magnitude kernel and the literal
  ``BigR/BB2`` factor.
* **Exact channel-term map** (``DEFAULT_CHANNEL_TERMS``): ``ZK_i_par -> T_i``
  (:1595), ``ZK_e_par -> T_e`` (:1754), ``ZK_perp_num -> T_i/T_e``
  (:1598, :1757, previously missing), ``D_par/D_par_imp -> rho/rho_imp``
  (:1546-1547), ``D_perp* -> rho/rho_imp`` (:1565, :1720), ``visco* -> u/omega``
  (:1492, :2029), ``visco_par* -> v_par`` (:1641-1649).
* **True implicit solver** (ADR-0018 decision 13, "stability is not fidelity").
  The default solver is a sparse direct LU solve of the per-channel system (the
  same class of solver JOREK uses), differentiable through a custom autograd
  function (implicit function theorem).  Jacobi-preconditioned CG in the mass
  inner product is kept as an alternative (``solver="cg"``); on real frames the
  parallel operator's near-null space (functions constant along field lines)
  makes the Jacobi-preconditioned condition number ~1e9, so CG is registered as
  an un-converged option there.  The B2 damped spectral iteration is kept as
  ``solver="richardson"`` for stability tests / the "frozen but bounded"
  comparator only.

The per-term operators are assembled exactly: P1 stiffness kernels reused from
B0 for the isotropic terms and ``int (bhat.grad N_i)(bhat.grad N_j)``
(``fem_weak.triangle_parallel_stiffness``) for the field-aligned terms, scaled
by the per-term metric weight.  ``lambda_max(M^-1 A)`` is computed with a sparse
generalized eigensolve, so the explicit limits in the report are exact for this
discretisation.

Boundary handling reuses the Phase B0 layer (ADR-0018 decision 11), extended in
B2.1 with the corner-pair skip and the sheath linearisation terms
(:mod:`src.models.jorek_boundary`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigvalsh as dense_generalized_eigvalsh
from scipy.sparse.linalg import eigsh

import torch
import torch.nn as nn

from src.data.model502_schema import FIELD_NAMES, NUM_STATE_FIELDS
from src.models import jorek_boundary as jb
from src.models.stencil_conv import FemStencilGeometry, R_MIN
from src.physics import fem_weak

# ---------------------------------------------------------------------------
# Per-term coefficients and metric weights
# ---------------------------------------------------------------------------
#: Per-term transport coefficients in "m^2 per JOREK time" (they multiply the
#: weak-form operator directly, no extra ``t_norm`` factor; docs section 3).
#: B2.1 values come from the production namelist
#: ``data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek_namelist_spitest``
#: (which agrees with the recorded H5 for ``visco``/``visco_par``/``eta``) and
#: from the ``docs/jorek-normalization.md`` section 3.2 table; per-entry
#: provenance is in :data:`ALPHA_PROVENANCE`.
DOC_ALPHA_TABLE: Mapping[str, float] = {
    "eta": 1.443e-10,
    "visco": 2.4e-3,
    "visco_num": 9.0e-4,
    "D_perp": 5.0e-6,
    "D_perp_imp": 5.0e-6,
    "D_perp_num": 1.0e-12,
    "Dn_perp_num": 0.0,
    "ZK_i_perp": 2.0e-6,
    "ZK_e_perp": 2.0e-6,
    "ZK_perp_num": 1.0e-12,
    "visco_par": 5.0e-4,
    "visco_par_num": 1.0e-4,
    "visco_par_par": 0.0,
    "D_par": 5.0e-2,
    "D_par_imp": 5.0e-2,
    "ZK_i_par": 1.6666e5,
    "ZK_e_par": 4.1037e6,
}

#: Provenance of every alpha entry (source file/line + namelist key).
ALPHA_PROVENANCE: Mapping[str, str] = {
    "eta": "namelist jorek_namelist_spitest:391 eta=1.443d-10; weak form mod_elt_matrix.f90:1946-1947",
    "visco": "namelist jorek_namelist_spitest:381 visco=2.4d-3; H5 stored; mod_elt_matrix.f90:1492,2029",
    "visco_num": "namelist jorek_namelist_spitest:385 visco_num=9.d-4; mod_elt_matrix.f90:1496,2031",
    "D_perp": "namelist jorek_namelist_spitest:433 D_perp(1)=5.d-6 (profile D_prof); mod_elt_matrix.f90:1551",
    "D_perp_imp": "namelist jorek_namelist_spitest:430 D_perp_imp(1)=5.d-6; mod_elt_matrix.f90:1552",
    "D_perp_num": "namelist jorek_namelist_spitest:395 d_perp_num=1.d-12; mod_elt_matrix.f90:1565",
    "Dn_perp_num": "preset_parameters.f90:297 Dn_perp_num=0 (namelist unset); mod_elt_matrix.f90:1720",
    "ZK_i_perp": "namelist jorek_namelist_spitest:467 ZK_i_perp(1)=2d-6; mod_elt_matrix.f90:1596",
    "ZK_e_perp": "namelist jorek_namelist_spitest:460 ZK_e_perp(1)=2d-6; mod_elt_matrix.f90:1755",
    "ZK_perp_num": "namelist jorek_namelist_spitest:396 zk_perp_num=1.d-12; mod_elt_matrix.f90:1598,1757",
    "visco_par": "namelist jorek_namelist_spitest:383 visco_par=5.d-4; H5 stored; mod_elt_matrix.f90:1641",
    "visco_par_num": "namelist jorek_namelist_spitest:386 visco_par_num=1.d-4; mod_elt_matrix.f90:1649",
    "visco_par_par": "preset_parameters.f90:49 visco_par_par=0 (namelist unset); mod_elt_matrix.f90:1642",
    "D_par": "namelist jorek_namelist_spitest:422 D_par=5.d-2; mod_elt_matrix.f90:1546",
    "D_par_imp": "namelist jorek_namelist_spitest:429 D_par_imp=5.d-2; mod_elt_matrix.f90:1547,1702",
    "ZK_i_par": "namelist jorek_namelist_spitest:456 ZK_i_par=1.6666d5 (ZK_par_max=1d4 caps the effective value); mod_elt_matrix.f90:1595",
    "ZK_e_par": "namelist jorek_namelist_spitest:457 ZK_e_par=4.1037d6 (ZK_par_max=1d4 caps the effective value); mod_elt_matrix.f90:1754",
}

#: Relative radial reference of the metric weights (metres).  JOREK's toroidal
#: measure is ``dV = 2 pi BigR dR dZ``; dividing by 1 m keeps the per-term
#: relative weights dimensionless while preserving the O(R) term ratios.
R_REF = 1.0

#: ``ZK_par_max`` cap of the temperature-dependent parallel conduction
#: (``mod_elt_matrix.f90:717-740``; docs section 3.2 lists it as a row of the
#: table).  It caps the coefficient, it is not a separate operator.
ZK_PAR_MAX_REFERENCE = 1.0e4

#: Terms carrying the JOREK ``BigR`` metric weight (perpendicular/viscous):
#: ``mod_elt_matrix.f90:1492 (visco), 1565 (D_perp_num), 1598 (ZK_perp_num),
#: 1641 (visco_par), 1649 (visco_par_num), 1720 (Dn_perp_num)`` plus the
#: physical perpendicular profiles at :1551 (D_perp = D_prof), :1552
#: (D_perp_imp), :1596 (ZK_i_perp = ZK_i_prof), :1755 (ZK_e_perp).
RADIAL_TERMS: tuple[str, ...] = (
    "visco",
    "D_perp",
    "D_perp_imp",
    "D_perp_num",
    "Dn_perp_num",
    "ZK_i_perp",
    "ZK_e_perp",
    "ZK_perp_num",
    "visco_par",
    "visco_par_num",
)

#: Field-aligned terms carrying ``BigR/BB2`` in JOREK (equivalent to ``BigR`` on
#: the unit-direction kernel, see the module docstring):
#: ``mod_elt_matrix.f90:1546-1547 (D_par, D_par_imp), 1595 (ZK_i_par),
#: 1642 (visco_par_par), 1754 (ZK_e_par)``.
PARALLEL_TERMS: tuple[str, ...] = (
    "D_par",
    "D_par_imp",
    "ZK_i_par",
    "ZK_e_par",
    "visco_par_par",
)

#: Terms with no geometric weight: ``visco_num`` (:1496, 2031) and the
#: magnetic diffusion ``eta`` (B2.1 keeps the unweighted P1 Laplacian; the
#: cylindrical ``1/BigR`` factor of :1947 is registered as an open item).
UNWEIGHTED_TERMS: tuple[str, ...] = ("visco_num", "eta")

#: Backwards-compatible alias of the isotropic (radial + unweighted) terms.
ISOTROPIC_TERMS: tuple[str, ...] = RADIAL_TERMS + UNWEIGHTED_TERMS
TERM_NAMES: tuple[str, ...] = RADIAL_TERMS + PARALLEL_TERMS + UNWEIGHTED_TERMS

#: Per-term metric weight class: ``"radius"`` (BigR), ``"parallel"``
#: (BigR/BB2 with a B-magnitude kernel; BigR on the unit-direction kernel) or
#: ``"none"``.
TERM_METRIC_WEIGHT: Mapping[str, str] = {
    **{term: "radius" for term in RADIAL_TERMS},
    **{term: "parallel" for term in PARALLEL_TERMS},
    **{term: "none" for term in UNWEIGHTED_TERMS},
}

#: Terms whose explicit bound is at or below the physical cadence ``dt = 0.05``
#: for the reference spectrum and therefore must be integrated implicitly.
#: The physical perpendicular diffusion (``D_perp``/``D_perp_imp``/``ZK_*_perp``)
#: is included because its limit is ~5e-2..0.13 (critical at dt = 0.05); all
#: parallel terms are stiff by orders of magnitude.
DEFAULT_IMPLICIT_TERMS: tuple[str, ...] = tuple(
    term for term in TERM_NAMES if term != "eta"
)
#: The only term that is not stiff at the physical cadence (eta limit ~1e3
#: JOREK time) and can stay explicit in a semi-implicit split.
DEFAULT_EXPLICIT_TERMS: tuple[str, ...] = ("eta",)

#: Field -> transport terms of the same equation, code-verified in B2.1
#: (``models/model502/mod_elt_matrix.f90``; see :data:`TERM_PROVENANCE`).
#: Registered architectural choice for the nine-field contract (ADR-0018
#: decision 7); it does not split the state, it only assigns each channel's
#: Laplacian-type terms.
DEFAULT_CHANNEL_TERMS: Mapping[str, tuple[str, ...]] = {
    "psi": ("eta",),                                               # :1946-1947
    "Phi": ("visco", "visco_num"),                                 # :1492,1496
    "Z_j": ("eta",),                                               # :1946-1947
    "omega": ("visco", "visco_num"),                               # :2029,2031
    "rho": ("D_perp", "D_perp_num", "D_par"),                      # :1551,1565,1546
    "T_i": ("ZK_i_perp", "ZK_perp_num", "ZK_i_par"),               # :1596,1598,1595
    "v_par": ("visco_par", "visco_par_num", "visco_par_par"),      # :1641,1649,1642
    "rho_imp": ("D_perp_imp", "Dn_perp_num", "D_par_imp"),         # :1552,1720,1702
    "T_e": ("ZK_e_perp", "ZK_perp_num", "ZK_e_par"),               # :1755,1757,1754
}

#: Code-line provenance of the channel-term map (for the B2.1 report/tests).
CHANNEL_TERM_PROVENANCE: Mapping[str, Mapping[str, str]] = {
    "psi": {"eta": "mod_elt_matrix.f90:1946-1947"},
    "Phi": {"visco": ":1492", "visco_num": ":1496"},
    "Z_j": {"eta": ":1946-1947"},
    "omega": {"visco": ":2029", "visco_num": ":2031"},
    "rho": {"D_perp": ":1551 (D_prof)", "D_perp_num": ":1565", "D_par": ":1546"},
    "T_i": {"ZK_i_perp": ":1596 (ZK_i_prof)", "ZK_perp_num": ":1598", "ZK_i_par": ":1595"},
    "v_par": {"visco_par": ":1641", "visco_par_num": ":1649", "visco_par_par": ":1642"},
    "rho_imp": {"D_perp_imp": ":1552", "Dn_perp_num": ":1720", "D_par_imp": ":1702"},
    "T_e": {"ZK_e_perp": ":1755 (ZK_e_prof)", "ZK_perp_num": ":1757", "ZK_e_par": ":1754"},
}


def channel_term_mask(
    channel_terms: Mapping[str, Sequence[str]] = DEFAULT_CHANNEL_TERMS,
    *,
    term_names: Sequence[str] = TERM_NAMES,
    n_channels: int = NUM_STATE_FIELDS,
) -> np.ndarray:
    """``[n_terms, n_channels]`` 0/1 mask of the registered channel-term map."""

    mask = np.zeros((len(term_names), n_channels), dtype=np.float64)
    for channel, name in enumerate(FIELD_NAMES):
        for term in channel_terms.get(name, ()):
            if term not in term_names:
                raise ValueError(f"unknown term {term!r} for channel {name!r}")
            mask[term_names.index(term), channel] = 1.0
    return mask


class _EmptyH5:
    """Mapping-like empty handle for the ``h5_handle is None`` value chain."""

    def __contains__(self, key) -> bool:  # noqa: D105 - trivial
        return False

    def __getitem__(self, key):  # pragma: no cover - never reached
        raise KeyError(key)


@dataclass(frozen=True)
class TransportCoefficients:
    """Per-term transport coefficients with provenance (docs section 3.2)."""

    values: Mapping[str, float]
    provenance: Mapping[str, str]
    reference: str = "docs/jorek-normalization.md section 3.2"

    def alpha(self, term: str) -> float:
        if term not in self.values:
            raise KeyError(f"unknown transport term {term!r}")
        return float(self.values[term])

    def as_table(self) -> list[dict]:
        return [
            {
                "term": term,
                "alpha": self.alpha(term),
                "provenance": self.provenance.get(term, self.reference),
            }
            for term in TERM_NAMES
            if term in self.values
        ]

    @classmethod
    def from_docs(cls) -> "TransportCoefficients":
        return cls(
            values=dict(DOC_ALPHA_TABLE),
            provenance={term: cls.reference for term in DOC_ALPHA_TABLE},
        )

    @classmethod
    def from_h5_and_namelist(
        cls,
        h5_path: str | None = None,
        namelist_path: str | None = None,
    ) -> "TransportCoefficients":
        """Resolve every term alpha through the H5 -> namelist -> docs chain.

        Reuses :func:`src.data.parameter_channel.build_condition_vector`
        (H5 -> namelist -> JOREK default, ADR-0006 (历史) value chain still
        valid for *values*).  Entries absent from both sources fall back to the
        docs section 3.2 table and are registered as ``docs_fallback``.

        B2.1 adds the numerical/numerical-profile coefficients
        (``visco_num``, ``visco_par_num``, ``d_perp_num``, ``zk_perp_num``,
        ``dn_perp_num``) by parsing the production namelist directly (the
        parameter channel does not expose them).
        """

        import h5py

        from src.data.parameter_channel import build_condition_vector
        from src.data.namelist import parse_namelist

        h5_handle = None
        namelist_text = None
        parsed_namelist: dict = {}
        try:
            if namelist_path is not None:
                namelist_text = Path(namelist_path).read_text(encoding="utf-8", errors="replace")
                parsed_namelist = parse_namelist(namelist_text)
            if h5_path is not None:
                h5_handle = h5py.File(h5_path, "r")
            else:
                h5_handle = _EmptyH5()
            condition = build_condition_vector(h5_handle, namelist_text)
            names = list(condition["names"])
            vector = np.asarray(condition["vector"], dtype=np.float64)
            provenance = dict(condition["provenance"])

            def lookup(*candidates: str) -> tuple[float | None, str | None]:
                for candidate in candidates:
                    if candidate in names:
                        index = names.index(candidate)
                        return float(vector[index]), provenance.get(candidate)
                return None, None

            mapping = {
                "eta": ("eta",),
                "visco": ("visco",),
                "visco_par": ("visco_par",),
                "visco_par_par": ("visco_par_par",),
                "D_perp": ("d_perp[1]",),
                "D_perp_imp": ("d_perp_imp[1]",),
                "ZK_i_perp": ("zk_i_perp[1]",),
                "ZK_e_perp": ("zk_e_perp[1]",),
                "D_par": ("d_par",),
                "D_par_imp": ("d_par_imp",),
                "ZK_i_par": ("zk_i_par",),
                "ZK_e_par": ("zk_e_par",),
            }
            values: dict[str, float] = {}
            sources: dict[str, str] = {}
            for term, candidates in mapping.items():
                value, source = lookup(*candidates)
                if value is None or not np.isfinite(value):
                    # the parameter channel may raise for unknown candidates;
                    # try the raw namelist parse before the docs fallback
                    raw = _namelist_scalar(parsed_namelist, candidates)
                    if raw is not None and np.isfinite(raw):
                        value, source = raw, "namelist"
                    else:
                        value = float(DOC_ALPHA_TABLE[term])
                        source = "docs_fallback"
                values[term] = float(value)
                sources[term] = f"parameter_channel:{source}"
            # Terms not covered by the parameter channel: parse the namelist
            # directly (production run values), else keep the registered table.
            extra_keys = {
                "visco_num": ("visco_num",),
                "visco_par_num": ("visco_par_num",),
                "D_perp_num": ("d_perp_num",),
                "ZK_perp_num": ("zk_perp_num",),
                "Dn_perp_num": ("dn_perp_num",),
                "D_perp_imp": ("d_perp_imp",),
            }
            for term, keys in extra_keys.items():
                raw = _namelist_scalar(parsed_namelist, keys)
                if raw is not None and np.isfinite(raw):
                    values[term] = float(raw)
                    sources[term] = f"namelist:{keys[0]}"
                else:
                    values[term] = float(DOC_ALPHA_TABLE[term])
                    sources[term] = "docs_fallback"
            return cls(values=values, provenance=sources)
        finally:
            if h5_handle is not None:
                h5_handle.close()


def _namelist_scalar(parsed: Mapping[str, object], keys: Sequence[str]) -> float | None:
    """Best-effort float lookup in a parsed namelist (accepts array syntax)."""

    for key in keys:
        if key not in parsed:
            continue
        raw = parsed[key]
        try:
            as_array = np.asarray(raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if as_array.size and np.isfinite(as_array[0]):
            return float(as_array[0])
    return None


# ---------------------------------------------------------------------------
# Per-term weak-form operators
# ---------------------------------------------------------------------------
def parallel_element_kernels(
    geometry: FemStencilGeometry,
    b_hat_nodes: np.ndarray,
    *,
    metric_weight: str = "none",
    element_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Local 4x4 P1 kernels of ``int (b.grad N_i)(b.grad N_j)``.

    Mirrors the B0 triangularisation (``(0,1,2)`` / ``(0,2,3)``, incident
    quadrilaterals).  ``metric_weight="jorek"`` multiplies each element by
    ``R_e / R_REF``; this is the JOREK parallel weight ``BigR/BB2`` acting on
    the unit-direction kernel, because ``BigR/BB2 * (B.grad N_i)(B.grad N_j)
    = BigR * (bhat.grad N_i)(bhat.grad N_j)`` (``Bgrad_*`` uses the full B in
    tesla, ``mod_elt_matrix.f90:1455-1460``).  An explicit ``element_weights``
    array (one entry per element) overrides the mode.
    """

    if metric_weight not in {"none", "jorek"}:
        raise ValueError(f"metric_weight must be 'none' or 'jorek', got {metric_weight!r}")
    pos = getattr(geometry, "pos", None)
    if pos is None:
        raise ValueError("geometry has no node coordinates; rebuild it from a frame")
    b_hat_nodes = np.asarray(b_hat_nodes, dtype=np.float64)
    if b_hat_nodes.shape != (geometry.n_nodes, 2):
        raise ValueError(
            f"b_hat_nodes must be [n_nodes, 2] = [{geometry.n_nodes}, 2], got {b_hat_nodes.shape}"
        )
    vertex = geometry.vertex
    kernels = np.zeros((geometry.n_elements, 4, 4), dtype=np.float64)
    for element in range(geometry.n_elements):
        corners = vertex[:, element]
        coords = pos[corners]
        b_e = b_hat_nodes[corners].mean(axis=0)
        norm = float(np.linalg.norm(b_e))
        if norm < 1e-12:
            continue
        b_e = b_e / norm
        if element_weights is not None:
            weight = float(element_weights[element])
        elif metric_weight == "jorek":
            radius = float(np.clip(coords[:, 0].mean(), R_MIN, None))
            weight = radius / R_REF
        else:
            weight = 1.0
        for triangle in ((0, 1, 2), (0, 2, 3)):
            idx = np.asarray(triangle)
            a_e = fem_weak.triangle_parallel_stiffness(coords[idx], b_e)
            if not np.any(a_e):
                continue
            for i in range(3):
                for j in range(3):
                    kernels[element, idx[i], idx[j]] += weight * a_e[i, j]
    return kernels


def element_metric_weights(
    geometry: FemStencilGeometry,
    weight: str,
) -> np.ndarray:
    """Per-element metric weight of one term class.

    ``"none"`` -> 1; ``"radius"`` -> ``R_e / R_REF`` (JOREK ``BigR`` on
    perpendicular/viscous terms); ``"parallel"`` -> ``R_e / R_REF`` on the
    unit-direction kernel (= ``BigR/BB2`` on the B-magnitude kernel).
    """

    if weight == "none":
        return np.ones(int(geometry.n_elements), dtype=np.float64)
    if weight not in {"radius", "parallel"}:
        raise ValueError(f"unknown metric weight {weight!r}")
    pos = getattr(geometry, "pos", None)
    if pos is None:
        raise ValueError("geometry has no node coordinates; rebuild it from a frame")
    vertex = geometry.vertex
    radii = np.asarray(pos, dtype=np.float64)[vertex, 0].mean(axis=0)
    return np.clip(radii, R_MIN, None) / R_REF


@dataclass
class TermOperatorSet:
    """Assembly of one local 4x4 kernel per transport term on a shared mesh."""

    names: tuple[str, ...]
    element_kernels: dict[str, np.ndarray]
    mass: np.ndarray
    cluster: np.ndarray
    vertex_cluster: np.ndarray
    n_nodes: int
    metric: dict[str, str] = field(default_factory=dict)
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    _matrix_cache: dict = field(default_factory=dict, repr=False)
    _lambda_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_clusters(self) -> int:
        return int(self.mass.shape[0])

    @property
    def n_elements(self) -> int:
        return int(self.vertex_cluster.shape[1])

    def node_mass(self) -> np.ndarray:
        return self.mass[self.cluster]

    def matrix(self, term: str) -> sp.csr_matrix:
        """Assemble the cluster-space CSR stiffness of one term (cached)."""

        if term not in self.element_kernels:
            raise KeyError(f"term {term!r} is not assembled; have {self.names}")
        if term not in self._matrix_cache:
            kernels = self.element_kernels[term]
            n_elem = kernels.shape[0]
            i = np.repeat(np.arange(4), 4)
            j = np.tile(np.arange(4), 4)
            rows = self.vertex_cluster[i, :].T.reshape(-1)
            cols = self.vertex_cluster[j, :].T.reshape(-1)
            vals = kernels.reshape(n_elem, 16).reshape(-1)
            self._matrix_cache[term] = sp.coo_matrix(
                (vals, (rows, cols)), shape=(self.n_clusters, self.n_clusters)
            ).tocsr()
        return self._matrix_cache[term]

    def diagonal(self, term: str) -> np.ndarray:
        """Diagonal of the cluster-space operator (sum of local diagonal)."""

        kernels = self.element_kernels[term]
        diag = np.zeros(self.n_clusters, dtype=np.float64)
        for corner in range(4):
            np.add.at(diag, self.vertex_cluster[corner], kernels[:, corner, corner])
        return diag

    def lambda_max(
        self,
        term: str,
        *,
        interior_mask: np.ndarray | None = None,
    ) -> float:
        """Largest generalized eigenvalue of ``(A, diag(m))`` (exact, cached).

        ``interior_mask`` is a node-space boolean mask (BC nodes excluded, the
        JOREK Dirichlet rows do not take part in the interior operator).

        Terms that share a metric weight class (and, for parallel terms, the
        same field direction) share the kernel and therefore the spectrum; the
        cache is keyed on the weight label so the eigensolve runs once per
        class instead of once per term.
        """

        kernel = np.ascontiguousarray(self.element_kernels[term])
        key: tuple = (hash(kernel.tobytes()), None)
        if interior_mask is not None:
            mask = np.asarray(interior_mask, dtype=bool)
            if mask.shape[0] != self.n_nodes:
                raise ValueError("interior_mask must be a boolean mask over the mesh nodes")
            key = (hash(kernel.tobytes()), hash(mask.tobytes()))
        if key in self._lambda_cache:
            return self._lambda_cache[key]

        matrix = self.matrix(term)
        mass = np.maximum(self.mass, 1e-300)
        if interior_mask is not None:
            # collapse the node-space mask to clusters (a cluster is interior
            # only when every node of it is interior)
            counts = np.bincount(self.cluster, minlength=self.n_clusters)
            keep_nodes = np.asarray(interior_mask, dtype=np.float64)
            keep = np.bincount(
                self.cluster, weights=keep_nodes, minlength=self.n_clusters
            ) >= counts
            index = np.flatnonzero(keep)
            matrix = matrix[index][:, index]
            mass = mass[index]
        if matrix.shape[0] == 0:
            lambda_max = 0.0
        elif matrix.shape[0] <= 128:
            dense = np.asarray(matrix.todense(), dtype=np.float64)
            eigenvalues = dense_generalized_eigvalsh(dense, np.diag(mass))
            lambda_max = float(eigenvalues.max())
        else:
            eigenvalues = eigsh(
                matrix, k=1, M=sp.diags(mass), which="LM", tol=1e-9, maxiter=100000
            )[0]
            lambda_max = float(eigenvalues[0])
        self._lambda_cache[key] = lambda_max
        return lambda_max

    def gershgorin_bound(self, term: str) -> float:
        """Upper bound of ``max_i sum_j |A_ij| / m_i`` (B0's proxy, for reference)."""

        row_bound = self.row_bound(term)
        return float((row_bound / np.maximum(self.mass, 1e-300)).max()) if row_bound.size else 0.0

    def row_bound(self, term: str) -> np.ndarray:
        """Per-cluster ``sum_j |A_ij|`` (Jacobi majorant, >= diagonal)."""

        matrix = self.matrix(term)
        return np.asarray(abs(matrix).sum(axis=1)).ravel()


def build_term_operators(
    geometry: FemStencilGeometry,
    *,
    b_hat_nodes: np.ndarray | None = None,
    terms: Sequence[str] = TERM_NAMES,
    metric_weight: str | Mapping[str, str] | None = None,
    element_weights: Mapping[str, np.ndarray] | None = None,
) -> TermOperatorSet:
    """Assemble the per-term local kernels on top of the B0 geometry.

    ``metric_weight`` is either ``None`` (use the per-term registry
    :data:`TERM_METRIC_WEIGHT`), a string legacy mode (``"none"`` or
    ``"jorek"``) or a mapping ``term -> "none"|"radius"|"parallel"``.
    ``element_weights`` may override the per-element weight array of any term.
    """

    terms = tuple(terms)
    kernels: dict[str, np.ndarray] = {}
    metric: dict[str, str] = {}
    weights: dict[str, np.ndarray] = {}
    parallel_cache: np.ndarray | None = None
    parallel_cache_weight: np.ndarray | None = None
    for term in terms:
        if isinstance(metric_weight, Mapping):
            weight_mode = metric_weight.get(term, TERM_METRIC_WEIGHT.get(term))
        elif metric_weight is None:
            weight_mode = TERM_METRIC_WEIGHT.get(term)
        elif metric_weight == "jorek":
            weight_mode = "parallel" if term in PARALLEL_TERMS else "radius"
        elif metric_weight == "none":
            weight_mode = "none"
        else:
            raise ValueError(f"unknown metric_weight {metric_weight!r}")
        if weight_mode is None:
            raise ValueError(f"no metric weight registered for term {term!r}")
        if element_weights is not None and term in element_weights:
            weight_array = np.asarray(element_weights[term], dtype=np.float64)
            if weight_array.shape != (geometry.n_elements,):
                raise ValueError(
                    f"element_weights[{term!r}] must be [{geometry.n_elements}]"
                )
        else:
            weight_array = element_metric_weights(geometry, weight_mode)
        weights[term] = weight_array

        if term in PARALLEL_TERMS:
            if b_hat_nodes is None:
                raise ValueError(f"term {term!r} needs b_hat_nodes to assemble the parallel operator")
            if parallel_cache is None or parallel_cache_weight is None or not np.array_equal(
                parallel_cache_weight, weight_array
            ):
                parallel_cache = parallel_element_kernels(
                    geometry, b_hat_nodes, element_weights=weight_array
                )
                parallel_cache_weight = np.array(weight_array, copy=True)
            kernels[term] = np.array(parallel_cache, copy=True)
            metric[term] = (
                f"P1 field-aligned stiffness | weight=BigR/BB2 (unit-bhat: BigR/R_REF)"
            )
        elif term in ISOTROPIC_TERMS:
            kernels[term] = np.array(geometry.element_kernel, dtype=np.float64, copy=True)
            kernels[term] *= weight_array[:, None, None]
            metric[term] = (
                "P1 stiffness (B0 element_kernel) | weight=BigR/R_REF"
                if weight_mode == "radius"
                else "P1 stiffness (B0 element_kernel) | weight=none"
            )
        else:
            raise ValueError(f"unknown transport term {term!r}")
    return TermOperatorSet(
        names=terms,
        element_kernels=kernels,
        mass=np.array(geometry.mass, dtype=np.float64, copy=True),
        cluster=np.array(geometry.cluster, dtype=np.int64, copy=True),
        vertex_cluster=np.array(geometry.vertex_cluster, dtype=np.int64, copy=True),
        n_nodes=int(geometry.n_nodes),
        metric=metric,
        weights=weights,
    )


@dataclass(frozen=True)
class TermStability:
    """One row of the per-term explicit-stability table."""

    term: str
    alpha: float
    lambda_max: float
    dt_limit: float
    metric_weight: str = ""
    provenance: str = ""

    def stable_at(self, dt: float) -> bool:
        return bool(dt <= self.dt_limit)


def explicit_stability_table(
    operators: TermOperatorSet,
    *,
    alphas: Mapping[str, float] | TransportCoefficients | None = None,
    interior_mask: np.ndarray | None = None,
) -> list[TermStability]:
    """Exact per-term ``lambda_max`` and explicit limit ``dt = 2/(alpha*lambda)``."""

    if alphas is None:
        alphas = DOC_ALPHA_TABLE
    if isinstance(alphas, TransportCoefficients):
        lookup = alphas.alpha
    else:
        merged = {**DOC_ALPHA_TABLE, **{k: float(v) for k, v in alphas.items()}}
        lookup = merged.get
    rows: list[TermStability] = []
    for term in operators.names:
        alpha = lookup(term)
        if alpha is None:
            raise KeyError(f"no alpha registered for term {term!r}")
        lambda_max = operators.lambda_max(term, interior_mask=interior_mask)
        limit = 2.0 / (float(alpha) * lambda_max) if lambda_max > 0 and alpha > 0 else float("inf")
        rows.append(
            TermStability(
                term=term,
                alpha=float(alpha),
                lambda_max=lambda_max,
                dt_limit=limit,
                metric_weight=TERM_METRIC_WEIGHT.get(term, "?"),
                provenance=ALPHA_PROVENANCE.get(term, ""),
            )
        )
    return rows


def b_hat_from_gradients(
    psi_gradient: np.ndarray,
    radius: np.ndarray,
    f0: float,
) -> np.ndarray:
    """Unit poloidal magnetic field direction ``[N, 2]`` (R, Z).

    ``B = [psi_Z, -psi_R, F0]/R`` (docs section 2); ``b_hat`` is the (R, Z)
    direction of that vector, so ``b_hat . grad`` is the field-aligned
    derivative used by the parallel weak form.
    """

    gradient = np.asarray(psi_gradient, dtype=np.float64)
    if gradient.shape[1] != 2:
        raise ValueError(f"psi_gradient must be [N, 2], got {gradient.shape}")
    radius = np.clip(np.asarray(radius, dtype=np.float64), R_MIN, None)
    b_r = gradient[:, 1] / radius
    b_z = -gradient[:, 0] / radius
    b_phi = float(f0) / radius
    norm = np.sqrt(b_r**2 + b_z**2 + b_phi**2)
    norm = np.maximum(norm, 1e-300)
    return np.stack([b_r / norm, b_z / norm], axis=1)


# ---------------------------------------------------------------------------
# Implicit layer
# ---------------------------------------------------------------------------
@dataclass
class _DirectSolvePlan:
    """Static data of the per-channel sparse direct implicit solve.

    All arrays are numpy/scipy objects derived from the (frozen) mesh and the
    term coefficients; only ``alpha`` (the learnable per-term/per-channel
    multiplier) and the right-hand side / reference enter per call.
    """

    term_names: tuple[str, ...]
    matrices: tuple[sp.csr_matrix, ...]          # A_t (cluster space)
    mass: np.ndarray                             # lumped cluster mass
    dirichlet: np.ndarray                        # [K, C] bool
    alpha0: np.ndarray                           # [T]
    base: sp.csr_matrix | None = None            # cached diagonal of A_t/m
    factor_cache: dict = field(default_factory=dict)   # (dt, alpha) -> SuperLU list
    max_cache_entries: int = 4

    @property
    def n_terms(self) -> int:
        return len(self.term_names)

    @property
    def n_channels(self) -> int:
        return int(self.dirichlet.shape[1])

    @property
    def n_clusters(self) -> int:
        return int(self.dirichlet.shape[0])

    def scalar_matrices(self) -> tuple[sp.csr_matrix, ...]:
        """``A_t / m`` as CSR (cached)."""

        if self.base is None:
            inv_mass = sp.diags(1.0 / np.maximum(self.mass, 1e-300))
            self.base = tuple((inv_mass @ matrix).tocsr() for matrix in self.matrices)
        return self.base

    def build_matrix(self, channel: int, alpha: np.ndarray, dt: float) -> sp.csc_matrix:
        """``M = I + dt * sum_t alpha[t, c] * A_t / m`` with Dirichlet rows."""

        scaled = None
        for index, matrix in enumerate(self.scalar_matrices()):
            value = float(alpha[index, channel])
            if value == 0.0:
                continue
            contribution = matrix * (dt * value)
            scaled = contribution if scaled is None else scaled + contribution
        if scaled is None:
            scaled = sp.csr_matrix((self.n_clusters, self.n_clusters))
        scaled = scaled.tocsr()
        fixed = np.asarray(self.dirichlet[:, channel], dtype=bool)
        if fixed.any():
            keep = (~fixed).astype(np.float64)
            scaled = scaled + sp.identity(self.n_clusters, format="csr")
            scaled = sp.diags(keep) @ scaled
            scaled = scaled + sp.diags(fixed.astype(np.float64))
        else:
            scaled = scaled + sp.identity(self.n_clusters, format="csr")
        return scaled.tocsc()


class _DirectSparseSolve(torch.autograd.Function):
    """Differentiable exact solve of ``M x = b`` for the assembled stencil.

    The forward factorises the per-channel sparse matrix with SuperLU; the
    backward solves the adjoint system with the same factors and applies the
    implicit function theorem to the learnable per-term multipliers, so the
    solver is exact (no truncation of the stiff modes) *and* differentiable.
    """

    @staticmethod
    def forward(
        ctx,
        rhs: torch.Tensor,
        reference: torch.Tensor,
        alpha: torch.Tensor,
        plan: "_DirectSolvePlan",
        dt: float,
    ) -> torch.Tensor:
        import scipy.sparse.linalg as spla

        alpha_np = np.asarray(alpha.detach().cpu().numpy(), dtype=np.float64)
        rhs_np = np.asarray(rhs.detach().cpu().numpy(), dtype=np.float64)
        ref_np = np.asarray(reference.detach().cpu().numpy(), dtype=np.float64)
        x_np = np.array(rhs_np, copy=True)
        cache_key = (round(float(dt), 15), alpha_np.tobytes())
        factors = plan.factor_cache.get(cache_key)
        if factors is None:
            factors = []
            for channel in range(plan.n_channels):
                matrix = plan.build_matrix(channel, alpha_np, float(dt))
                factors.append(spla.splu(matrix))
            if len(plan.factor_cache) < plan.max_cache_entries:
                plan.factor_cache[cache_key] = factors
        for channel in range(plan.n_channels):
            fixed = np.asarray(plan.dirichlet[:, channel], dtype=bool)
            b = np.array(rhs_np[:, channel], copy=True)
            b[fixed] = ref_np[fixed, channel]
            x_np[:, channel] = factors[channel].solve(b)
        ctx.plan = plan
        ctx.dt = float(dt)
        ctx.factors = factors
        ctx.save_for_backward(rhs, reference, alpha)
        ctx.x = x_np
        return torch.as_tensor(x_np, dtype=rhs.dtype, device=rhs.device)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        plan = ctx.plan
        dt = ctx.dt
        _, reference, alpha = ctx.saved_tensors
        x_np = ctx.x
        g_np = np.asarray(grad_out.detach().cpu().numpy(), dtype=np.float64)
        ref_np = np.asarray(reference.detach().cpu().numpy(), dtype=np.float64)
        grad_rhs = np.zeros_like(x_np)
        grad_ref = np.zeros_like(x_np)
        alpha_np = np.asarray(alpha.detach().cpu().numpy(), dtype=np.float64)
        grad_alpha = np.zeros_like(alpha_np)
        for channel in range(plan.n_channels):
            fixed = np.asarray(plan.dirichlet[:, channel], dtype=bool)
            interior = ~fixed
            v = np.zeros(plan.n_clusters, dtype=np.float64)
            rhs_adj = np.zeros(plan.n_clusters, dtype=np.float64)
            rhs_adj[interior] = g_np[interior, channel]
            v = np.asarray(ctx.factors[channel].solve(rhs_adj, trans="T"), dtype=np.float64)
            grad_rhs[interior, channel] = v[interior]
            grad_ref[fixed, channel] = g_np[fixed, channel] + v[fixed]
            x_c = x_np[:, channel]
            for index, matrix in enumerate(plan.scalar_matrices()):
                weight = float(alpha_np[index, channel])
                if weight == 0.0 and not ctx.needs_input_grad[2]:
                    continue
                ax = matrix @ x_c
                grad_alpha[index, channel] = -dt * float(np.dot(v[interior], ax[interior]))
        return (
            torch.as_tensor(grad_rhs, dtype=grad_out.dtype, device=grad_out.device),
            torch.as_tensor(grad_ref, dtype=grad_out.dtype, device=grad_out.device),
            torch.as_tensor(grad_alpha, dtype=alpha.dtype, device=alpha.device),
            None,
            None,
        )


class ImplicitStencilLayer(nn.Module):
    """Implicit stencil layer: ``(I + dt * sum_t alpha_t * A_t/m) x = rhs``.

    ``solver`` selects the implicit solve (ADR-0018 decision 13: stability is
    not fidelity - the solver must reproduce the quasi-steady response):

    * ``"direct"`` (default): exact sparse LU solve per channel, differentiable
      through :class:`_DirectSparseSolve` (implicit function theorem).  The
      stiff parallel modes are solved, not frozen.
    * ``"cg"``: Jacobi-preconditioned CG in the mass inner product, built from
      the stencil matvec in torch (fully in-graph).  On real divertor frames the
      field-aligned operator's near-null space (functions constant along field
      lines) makes the Jacobi-preconditioned condition number ~1e9, so CG may
      fail to reach tolerance with a practical iteration cap; registered as an
      open item of B2.1.
    * ``"richardson"``: the B2 symmetric-majorant damped iteration.  It is
      unconditionally contractive for any ``dt``/``alpha`` but its contraction
      factor under ZK-parallel transport is ``1 - 1e-12``: it *freezes* the
      stiff modes.  Kept as the "stable but not faithful" comparator.

    Per-channel/per-term multipliers (``alpha_scale``, identity at init) are
    learnable in every mode; ``damping`` (sigmoid) drives the Richardson steps
    and exists only for ``solver="richardson"``; ``relax`` (zero at init,
    bounded by ``max_relax``) over/under-relaxes the implicit correction for
    every solver while leaving the initial output exactly the implicit solve.
    """

    def __init__(
        self,
        geometry: FemStencilGeometry,
        *,
        operators: TermOperatorSet | None = None,
        b_hat_nodes: np.ndarray | None = None,
        term_names: Sequence[str] = TERM_NAMES,
        alphas: TransportCoefficients | Mapping[str, float] | None = None,
        channel_terms: Mapping[str, Sequence[str]] = DEFAULT_CHANNEL_TERMS,
        n_channels: int = NUM_STATE_FIELDS,
        dt: float = 0.05,
        n_iters: int = 12,
        solver: str = "direct",
        cg_tol: float = 1e-10,
        max_cg_iters: int = 400,
        boundary: jb.BoundaryContext | None = None,
        explicit_terms: Sequence[str] = DEFAULT_EXPLICIT_TERMS,
        implicit_terms: Sequence[str] = DEFAULT_IMPLICIT_TERMS,
        learnable: bool = True,
        damping_init: float = 0.8,
        max_alpha_scale_delta: float = 0.5,
        max_relax: float = 0.5,
        preconditioner: str = "spectral",
        gamma: float = 5.0 / 3.0,
        c_angle: float = 0.0174524,
        metric_weight: str | Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if n_iters < 1:
            raise ValueError("n_iters must be >= 1")
        if not 0.0 < damping_init < 1.0:
            raise ValueError("damping_init must be in (0, 1)")
        if preconditioner not in {"spectral", "jacobi"}:
            raise ValueError(
                f"preconditioner must be 'spectral' or 'jacobi', got {preconditioner!r}"
            )
        if solver not in {"direct", "cg", "richardson"}:
            raise ValueError(
                f"solver must be 'direct', 'cg' or 'richardson', got {solver!r}"
            )
        if cg_tol <= 0.0 or max_cg_iters < 1:
            raise ValueError("cg_tol must be positive and max_cg_iters >= 1")
        self.preconditioner = preconditioner
        self.solver = solver
        self.cg_tol = float(cg_tol)
        self.max_cg_iters = int(max_cg_iters)
        self.dt = float(dt)
        self.last_solve_stats: dict = {"solver": solver, "n_solves": 0}
        coefficients = (
            alphas if isinstance(alphas, TransportCoefficients) else TransportCoefficients.from_docs()
        )
        if isinstance(alphas, Mapping) and not isinstance(alphas, TransportCoefficients):
            coefficients = TransportCoefficients(
                values={**DOC_ALPHA_TABLE, **{k: float(v) for k, v in alphas.items()}},
                provenance={k: "caller" for k in alphas},
            )
        if operators is None:
            operators = build_term_operators(
                geometry, b_hat_nodes=b_hat_nodes, terms=term_names, metric_weight=metric_weight
            )
        self.geometry = geometry
        self.operators = operators
        self.term_names = tuple(operators.names)
        self.n_channels = int(n_channels)
        self.n_iters = int(n_iters)
        self.boundary = boundary
        self.gamma = float(gamma)
        self.c_angle = float(c_angle)
        self.max_alpha_scale_delta = float(max_alpha_scale_delta)
        self.max_relax = float(max_relax)
        self.coefficients = coefficients

        explicit = tuple(term for term in explicit_terms if term in self.term_names)
        implicit = tuple(term for term in implicit_terms if term in self.term_names)
        unknown = set(explicit_terms) | set(implicit_terms)
        unknown -= set(self.term_names)
        if unknown:
            raise ValueError(f"term split references unassembled terms: {sorted(unknown)}")
        self.explicit_terms = explicit
        self.implicit_terms = implicit

        kernels = np.stack([operators.element_kernels[t] for t in self.term_names], axis=0)
        self.register_buffer(
            "element_kernels", torch.as_tensor(kernels, dtype=torch.float64)
        )
        self.register_buffer("mass", torch.as_tensor(operators.mass, dtype=torch.float64))
        self.register_buffer(
            "vertex_cluster",
            torch.as_tensor(operators.vertex_cluster, dtype=torch.long),
        )
        self.register_buffer("cluster", torch.as_tensor(operators.cluster, dtype=torch.long))
        node_mass = torch.as_tensor(operators.node_mass(), dtype=torch.float64)
        self.register_buffer("node_mass", node_mass)
        row_bound = np.stack([operators.row_bound(t) for t in self.term_names], axis=0)
        self.register_buffer(
            "operator_row_bound", torch.as_tensor(row_bound, dtype=torch.float64)
        )
        operator_diagonal = np.stack(
            [operators.diagonal(t) for t in self.term_names], axis=0
        )
        self.register_buffer(
            "operator_diagonal", torch.as_tensor(operator_diagonal, dtype=torch.float64)
        )
        # Exact per-term generalized spectra (the B2 replacement for B0's A/m
        # proxy); used by the symmetric majorant preconditioner.
        term_lambda = np.array(
            [operators.lambda_max(term) for term in self.term_names], dtype=np.float64
        )
        self.register_buffer("term_lambda", torch.as_tensor(term_lambda, dtype=torch.float64))
        alpha0 = np.array([coefficients.alpha(t) for t in self.term_names], dtype=np.float64)
        self.register_buffer("alpha0", torch.as_tensor(alpha0, dtype=torch.float64))
        mask = channel_term_mask(channel_terms, term_names=self.term_names, n_channels=n_channels)
        self.register_buffer("channel_mask", torch.as_tensor(mask, dtype=torch.float64))
        if boundary is None:
            boundary_rows = np.zeros(geometry.n_nodes, dtype=bool)
            dirichlet_mask = np.zeros((geometry.n_nodes, n_channels), dtype=bool)
        else:
            boundary_rows = boundary.bc_node_mask
            dirichlet_mask = boundary.dirichlet_mask
        self.register_buffer("boundary_rows", torch.as_tensor(boundary_rows, dtype=torch.bool))
        self.register_buffer("dirichlet_mask", torch.as_tensor(dirichlet_mask, dtype=torch.bool))
        # Cluster-space Dirichlet mask used by the direct solver (a cluster is
        # fixed only when every node of it is BC-coded, matching the operator
        # restriction used for the exact spectra).
        counts = np.bincount(geometry.cluster, minlength=geometry.n_clusters).astype(np.float64)
        if boundary is None:
            dirichlet_cluster = np.zeros((geometry.n_clusters, n_channels), dtype=bool)
        else:
            fixed_nodes = boundary.dirichlet_mask.astype(np.float64)
            fixed_counts = np.stack(
                [
                    np.bincount(geometry.cluster, weights=fixed_nodes[:, c], minlength=geometry.n_clusters)
                    for c in range(n_channels)
                ],
                axis=1,
            )
            dirichlet_cluster = fixed_counts >= counts[:, None]
        self._dirichlet_cluster = dirichlet_cluster
        self._direct_plan: _DirectSolvePlan | None = None

        self.alpha_scale_raw = nn.Parameter(
            torch.zeros(len(self.term_names), n_channels, dtype=torch.float64),
            requires_grad=learnable,
        )
        # ``relax`` is zero at init so the initial output is exactly the
        # implicit solve for every solver; it gives the network a bounded
        # over/under-relaxation knob that is differentiable by construction.
        self.relax_raw = nn.Parameter(
            torch.zeros(n_channels, dtype=torch.float64), requires_grad=learnable
        )
        if solver == "richardson":
            damping_raw = torch.full(
                (n_channels,),
                float(np.log(damping_init / (1.0 - damping_init))),
                dtype=torch.float64,
            )
            self.damping_raw = nn.Parameter(damping_raw, requires_grad=learnable)
        else:
            self.damping_raw = None
        self.damping_init = float(damping_init)

    # -- parameterization --------------------------------------------------
    def alpha_scale(self) -> torch.Tensor:
        """``[n_terms, n_channels]`` positive multiplier, exactly 1 at init."""

        return 1.0 + self.max_alpha_scale_delta * torch.tanh(self.alpha_scale_raw)

    def effective_alpha(self) -> torch.Tensor:
        """``[n_terms, n_channels]`` = table alpha * mask * learnable scale."""

        return self.alpha0[:, None] * self.channel_mask * self.alpha_scale()

    def damping(self) -> torch.Tensor:
        if self.damping_raw is None:
            raise ValueError(
                "damping is only a parameter of solver='richardson'; "
                f"the current solver is {self.solver!r}"
            )
        return torch.sigmoid(self.damping_raw)

    def relax(self) -> torch.Tensor:
        """``[C]`` bounded relaxation of the implicit correction (0 at init)."""

        return self.max_relax * torch.tanh(self.relax_raw)

    def term_matrix(self, term: str) -> sp.csr_matrix:
        """Diagnostic access to the assembled cluster-space operator."""

        return self.operators.matrix(term)

    # -- operators ---------------------------------------------------------
    def _local_messages(self, values: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """``sum_t alpha_t,c * (A_t x)/m`` in node space ``[N, C]``."""

        x_cluster = self.geometry.to_cluster(values)
        # element-major corner order (see ``StencilConv._element_flux``)
        flat = self.vertex_cluster.transpose(0, 1).reshape(-1)
        local = x_cluster.index_select(0, flat).reshape(self.operators.n_elements, 4, self.n_channels)
        message = torch.einsum("teij,ejc->teic", self.element_kernels, local)
        message = message.reshape(len(self.term_names), -1, self.n_channels)
        acc = torch.zeros(
            (len(self.term_names), self.operators.n_clusters, self.n_channels),
            dtype=values.dtype,
            device=values.device,
        ).index_add(1, flat, message)
        acc = acc / self.mass.clamp_min(1e-300).unsqueeze(0).unsqueeze(-1)
        acc = torch.einsum("tc,tkc->kc", alpha, acc)
        return self.geometry.from_cluster(acc)

    def operator(self, values: torch.Tensor, *, terms: Sequence[str] | None = None) -> torch.Tensor:
        """Node-space ``sum_t alpha_t * A_t x / m`` (no dt, no BC)."""

        alpha = self.effective_alpha().clone()
        if terms is not None:
            allowed = [self.term_names.index(term) for term in terms]
            mask = torch.zeros_like(alpha)
            if allowed:
                mask[allowed] = 1.0
            alpha = alpha * mask
        return self._local_messages(values, alpha)

    def _diagonal(self, alpha: torch.Tensor, dt: float | None = None) -> torch.Tensor:
        """Jacobi majorant ``1 + dt * sum_t alpha_t * sum_j|A_t,ij| / m``.

        The absolute row sums (not just the diagonal) are required for the
        parallel operator: ``A_par`` has *positive* off-diagonal entries.  This
        variant is kept for the isotropic-only case and tests; for the mixed
        nine-field operator the spectral majorant below is the default because a
        diagonal-scaled iteration with a non-normal matrix of norm ``> 1``
        suffers catastrophic transient growth.
        """

        dt = self.dt if dt is None else float(dt)
        scale = torch.einsum(
            "tc,tk->kc", alpha, self.operator_row_bound
        ) / self.mass.clamp_min(1e-300).unsqueeze(1)
        return self.geometry.from_cluster(1.0 + dt * scale)

    def spectral_majorant(self, alpha: torch.Tensor, dt: float | None = None) -> torch.Tensor:
        """``[1, C]`` symmetric preconditioner ``1 + dt * sum_t alpha_tc * lambda_t``."""

        dt = self.dt if dt is None else float(dt)
        return 1.0 + dt * torch.einsum("tc,t->c", alpha, self.term_lambda).unsqueeze(0)

    def _implicit_alpha(self) -> torch.Tensor:
        alpha = self.effective_alpha().clone()
        mask = torch.zeros_like(alpha)
        for term in self.implicit_terms:
            mask[self.term_names.index(term)] = 1.0
        return alpha * mask

    def preconditioner_scale(
        self, alpha: torch.Tensor, dt: float | None = None
    ) -> torch.Tensor:
        """Diagonal (node-space) or symmetric-majorant (per-channel) scale."""

        if self.preconditioner == "spectral":
            return self.spectral_majorant(alpha, dt)
        return self._diagonal(alpha, dt)

    def contraction_factor(self, dt: float | None = None) -> torch.Tensor:
        """Per-channel contraction bound of the *Richardson* iteration (``< 1``).

        Spectral majorant: the iteration matrix is symmetric in the
        mass-weighted inner product with spectrum in
        ``[1 - omega, 1 - omega / D_c]``, so its operator norm is
        ``1 - omega / D_c``.  Jacobi variant: the Gershgorin row bound
        ``dt * B_ii / (1 + dt * B_ii)``.

        The direct and CG solvers are not damped iterations; they report a zero
        contraction bound (exact solve / data-dependent CG convergence is
        recorded through :attr:`last_solve_stats` instead).
        """

        dt = self.dt if dt is None else float(dt)
        alpha = self._implicit_alpha()
        if self.solver != "richardson":
            return torch.zeros(self.n_channels, dtype=torch.float64)
        if self.preconditioner == "spectral":
            majorant = self.spectral_majorant(alpha, dt).reshape(-1)
            return 1.0 - self.damping() / majorant
        scale = torch.einsum(
            "tc,tk->kc", alpha, self.operator_row_bound
        ) / self.mass.clamp_min(1e-300).unsqueeze(1)
        return (dt * scale / (1.0 + dt * scale)).max(dim=0).values

    def gershgorin_contraction(self, dt: float | None = None) -> torch.Tensor:
        """Backward-compatible alias of :meth:`contraction_factor`."""

        return self.contraction_factor(dt)

    # -- solver ------------------------------------------------------------
    def residual(self, rhs: torch.Tensor, x: torch.Tensor, dt: float | None = None) -> torch.Tensor:
        dt = self.dt if dt is None else float(dt)
        return rhs - (x + dt * self.operator(x, terms=self.implicit_terms))

    def _direct_plan_obj(self) -> _DirectSolvePlan:
        """Assemble (once) the static data of the sparse direct solve."""

        if self._direct_plan is None:
            matrices = tuple(self.operators.matrix(term).tocsr() for term in self.term_names)
            self._direct_plan = _DirectSolvePlan(
                term_names=self.term_names,
                matrices=matrices,
                mass=np.array(self.operators.mass, dtype=np.float64),
                dirichlet=np.array(self._dirichlet_cluster, dtype=bool),
                alpha0=np.array(
                    [self.coefficients.alpha(t) for t in self.term_names], dtype=np.float64
                ),
            )
        return self._direct_plan

    def _solve_richardson(
        self,
        rhs: torch.Tensor,
        reference: torch.Tensor,
        *,
        dt: float,
        iterations: int,
        keep_history: bool,
    ) -> tuple[torch.Tensor, list[float]]:
        import time

        start = time.perf_counter()
        alpha_implicit = self._implicit_alpha()
        scale = self.preconditioner_scale(alpha_implicit, dt)
        damping = self.damping().unsqueeze(0)
        x = rhs.clone()
        dirichlet = self.dirichlet_mask
        if bool(dirichlet.any()):
            rhs = torch.where(dirichlet, reference, rhs)
            x = torch.where(dirichlet, reference, x)
        history: list[float] = []
        for _ in range(iterations):
            residual = rhs - (x + dt * self._local_messages(x, alpha_implicit))
            if keep_history:
                history.append(float(residual.detach().abs().max()))
            x = x + damping * residual / scale.clamp_min(1e-300)
            if bool(dirichlet.any()):
                x = torch.where(dirichlet, reference, x)
        x = torch.where(torch.isfinite(x), x, reference)
        self.last_solve_stats = {
            "solver": "richardson",
            "n_solves": int(self.last_solve_stats.get("n_solves", 0)) + 1,
            "dt": float(dt),
            "iterations": int(iterations),
            "residual_relative": (
                float(history[-1]) if history else None
            ),
            "converged": None,
            "time_s": float(time.perf_counter() - start),
        }
        return x, history

    def _solve_direct(
        self,
        rhs: torch.Tensor,
        reference: torch.Tensor,
        *,
        dt: float,
    ) -> torch.Tensor:
        """Exact sparse LU solve of the per-channel implicit system."""

        import time

        alpha = self._implicit_alpha()
        rhs_cluster = self.geometry.to_cluster(rhs)
        reference_cluster = self.geometry.to_cluster(reference)
        plan = self._direct_plan_obj()
        cache_key = (round(float(dt), 15), alpha.detach().cpu().numpy().tobytes())
        cache_hit = cache_key in plan.factor_cache
        start = time.perf_counter()
        x_cluster = _DirectSparseSolve.apply(
            rhs_cluster, reference_cluster, alpha, plan, float(dt)
        )
        elapsed = time.perf_counter() - start
        x = self.geometry.from_cluster(x_cluster)
        self.last_solve_stats = {
            "solver": "direct",
            "n_solves": int(self.last_solve_stats.get("n_solves", 0)) + 1,
            "n_channels": int(self.n_channels),
            "n_clusters": int(plan.n_clusters),
            "matrix_nnz": int(sum(m.nnz for m in plan.matrices)),
            "dt": float(dt),
            "iterations": None,
            "time_s": float(elapsed),
            "cache_hit": bool(cache_hit),
        }
        return torch.where(torch.isfinite(x), x, reference)

    def _solve_cg(
        self,
        rhs: torch.Tensor,
        reference: torch.Tensor,
        *,
        dt: float,
        iterations: int,
        keep_history: bool,
    ) -> tuple[torch.Tensor, list[float]]:
        """Jacobi-preconditioned CG in the mass inner product (stencil matvec).

        SPD system ``M = I + dt * sum_{t in implicit} alpha_t A_t / m`` under
        ``<a, b>_m = sum_i m_i a_i b_i``; the preconditioner is
        ``diag(M) = 1 + dt * sum_t alpha_t diag(A_t)/m`` (node space).
        """

        import time

        alpha = self._implicit_alpha()
        mass = self.node_mass.reshape(-1, 1)
        diagonal = 1.0 + dt * torch.einsum("tc,tk->kc", alpha, self.operator_diagonal)
        diagonal = self.geometry.from_cluster(diagonal)
        dirichlet = self.dirichlet_mask
        x = rhs.clone()
        if bool(dirichlet.any()):
            rhs = torch.where(dirichlet, reference, rhs)
            x = torch.where(dirichlet, reference, x)

        def apply_matrix(vector: torch.Tensor) -> torch.Tensor:
            return vector + dt * self._local_messages(vector, alpha)

        def mass_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return (mass * a * b).sum()

        residual = rhs - apply_matrix(x)
        if bool(dirichlet.any()):
            residual = torch.where(dirichlet, torch.zeros_like(residual), residual)
        z = residual / diagonal.clamp_min(1e-300)
        direction = z.clone()
        rho = mass_dot(residual, z)
        norm0 = float(torch.sqrt(torch.clamp(rho, min=0.0)).detach())
        history: list[float] = []
        start = time.perf_counter()
        iterations_used = 0
        for _ in range(iterations):
            iterations_used += 1
            if norm0 == 0.0:
                break
            product = apply_matrix(direction)
            denominator = mass_dot(direction, product)
            if float(denominator.detach()) <= 0.0:
                iterations_used -= 1
                break
            step = rho / denominator
            x = x + step * direction
            residual = residual - step * product
            if bool(dirichlet.any()):
                residual = torch.where(dirichlet, torch.zeros_like(residual), residual)
            z = residual / diagonal.clamp_min(1e-300)
            rho_new = mass_dot(residual, z)
            history.append(float(torch.sqrt(torch.clamp(rho_new, min=0.0)).detach()) / norm0)
            if history[-1] <= self.cg_tol:
                break
            direction = z + (rho_new / rho) * direction
            rho = rho_new
        x = torch.where(torch.isfinite(x), x, reference)
        self.last_solve_stats = {
            "solver": "cg",
            "n_solves": int(self.last_solve_stats.get("n_solves", 0)) + 1,
            "dt": float(dt),
            "iterations": int(iterations_used),
            "residual_relative": None if not history else float(history[-1]),
            "converged": bool(history and history[-1] <= self.cg_tol),
            "time_s": float(time.perf_counter() - start),
        }
        return x, history

    def implicit_solve(
        self,
        rhs: torch.Tensor,
        reference: torch.Tensor,
        *,
        dt: float | None = None,
        n_iters: int | None = None,
        keep_history: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
        """Solve ``(I + dt * sum_t alpha_t A_t/m) x = rhs`` at physical dt.

        Dispatch on ``self.solver`` (see the class docstring); the Dirichlet
        rows are held at ``reference`` in every mode, and the learnable
        ``relax`` factor is applied to the correction afterwards.
        """

        dt = self.dt if dt is None else float(dt)
        iterations = self.n_iters if n_iters is None else int(n_iters)
        if self.solver == "direct":
            x = self._solve_direct(rhs, reference, dt=dt)
            history: list[float] = []
        elif self.solver == "cg":
            x, history = self._solve_cg(
                rhs, reference, dt=dt, iterations=min(iterations, self.max_cg_iters),
                keep_history=keep_history,
            )
        else:
            x, history = self._solve_richardson(
                rhs, reference, dt=dt, iterations=iterations, keep_history=keep_history
            )
        relax = self.relax().reshape(1, -1)
        x = x + relax * (x - reference)
        if bool(self.dirichlet_mask.any()):
            x = torch.where(self.dirichlet_mask, reference, x)
        if keep_history:
            return x, history
        return x

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        values: torch.Tensor,
        *,
        dt: float | None = None,
        forcing: torch.Tensor | None = None,
        rhs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = torch.as_tensor(values)
        if values.ndim != 2 or values.shape[1] != self.n_channels:
            raise ValueError(
                f"implicit stencil expects [n_nodes, {self.n_channels}], got {tuple(values.shape)}"
            )
        if values.shape[0] != self.geometry.n_nodes:
            raise ValueError(
                f"geometry has {self.geometry.n_nodes} nodes, got {values.shape[0]}"
            )
        dt = self.dt if dt is None else float(dt)
        reference = values
        if self.boundary is not None:
            # A boundary value (or NaN) must not leak into the interior update.
            values = jb.sanitize_boundary_inputs(values, self.boundary, gamma=self.gamma)
        if rhs is None:
            rhs = values
            if self.explicit_terms:
                rhs = rhs - dt * self.operator(values, terms=self.explicit_terms)
            if forcing is not None:
                rhs = rhs + dt * forcing
        out = self.implicit_solve(rhs, reference, dt=dt)
        if bool(self.boundary_rows.any()):
            out = torch.where(self.boundary_rows.unsqueeze(1), values, out)
        if self.boundary is not None:
            out = jb.apply_boundary_conditions(
                out,
                self.boundary,
                reference=values,
                dt=dt,
                gamma=self.gamma,
                c_angle=self.c_angle,
                mass=self.node_mass,
            )
        return out

    # -- reporting ---------------------------------------------------------
    def parameter_report(self) -> dict:
        alpha_scale = self.alpha_scale().detach()
        contraction = self.contraction_factor().detach()
        return {
            "n_channels": self.n_channels,
            "terms": list(self.term_names),
            "metric_weight": {
                term: TERM_METRIC_WEIGHT.get(term, "?") for term in self.term_names
            },
            "explicit_terms": list(self.explicit_terms),
            "implicit_terms": list(self.implicit_terms),
            "n_iters": self.n_iters,
            "solver": self.solver,
            "cg_tol": self.cg_tol,
            "max_cg_iters": self.max_cg_iters,
            "dt": self.dt,
            "n_parameters": int(sum(p.numel() for p in self.parameters())),
            "alpha_table": {term: self.coefficients.alpha(term) for term in self.term_names},
            "alpha_provenance": {
                term: self.coefficients.provenance.get(term, self.coefficients.reference)
                for term in self.term_names
            },
            "alpha_scale": {
                term: {
                    name: float(alpha_scale[index, channel])
                    for channel, name in enumerate(FIELD_NAMES[: self.n_channels])
                }
                for index, term in enumerate(self.term_names)
            },
            "alpha_scale_init_identity": bool(
                torch.allclose(alpha_scale, torch.ones_like(alpha_scale))
            ),
            "preconditioner": self.preconditioner,
            "relax": [float(value) for value in self.relax().detach()],
            "damping": (
                [float(value) for value in self.damping().detach()]
                if self.damping_raw is not None
                else None
            ),
            "term_lambda": {
                term: float(self.term_lambda[index])
                for index, term in enumerate(self.term_names)
            },
            "contraction_factor_max": float(contraction.max()),
            "last_solve_stats": dict(self.last_solve_stats),
        }


class ExplicitStencilLayer(nn.Module):
    """Explicit Euler stencil step with B0's safety cap (component test only).

    ADR-0018 decision 13: the explicit cap is *not* a B2 propagation scheme.  It
    exists so tests can demonstrate that the physical ``dt`` cannot be realised
    explicitly and that the implicit layer is the stable counterpart.
    """

    def __init__(
        self,
        geometry: FemStencilGeometry,
        *,
        operators: TermOperatorSet | None = None,
        b_hat_nodes: np.ndarray | None = None,
        term_names: Sequence[str] = TERM_NAMES,
        alphas: TransportCoefficients | Mapping[str, float] | None = None,
        channel_terms: Mapping[str, Sequence[str]] = DEFAULT_CHANNEL_TERMS,
        n_channels: int = NUM_STATE_FIELDS,
        dt: float = 0.05,
        boundary: jb.BoundaryContext | None = None,
        cap: bool = True,
        metric_weight: str | Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if operators is None:
            operators = build_term_operators(
                geometry,
                b_hat_nodes=b_hat_nodes,
                terms=term_names,
                metric_weight=metric_weight,
            )
        self.geometry = geometry
        self.operators = operators
        self.term_names = tuple(operators.names)
        self.n_channels = int(n_channels)
        self.dt = float(dt)
        self.boundary = boundary
        coefficients = (
            alphas if isinstance(alphas, TransportCoefficients) else TransportCoefficients.from_docs()
        )
        self.coefficients = coefficients
        kernels = np.stack([operators.element_kernels[t] for t in self.term_names], axis=0)
        self.register_buffer("element_kernels", torch.as_tensor(kernels, dtype=torch.float64))
        self.register_buffer("mass", torch.as_tensor(operators.mass, dtype=torch.float64))
        self.register_buffer(
            "vertex_cluster", torch.as_tensor(operators.vertex_cluster, dtype=torch.long)
        )
        self.register_buffer(
            "node_mass", torch.as_tensor(operators.node_mass(), dtype=torch.float64)
        )
        alpha0 = np.array([coefficients.alpha(t) for t in self.term_names], dtype=np.float64)
        mask = channel_term_mask(channel_terms, term_names=self.term_names, n_channels=n_channels)
        alpha = alpha0[:, None] * mask
        self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float64))
        # Gershgorin row-sum bound *including the physical alpha*: the cap of
        # the alpha=1 layer (B0's geometric cap) is NOT the cap of the physical
        # operator (ADR-0018 decision 13).
        row_bounds = np.array(
            [operators.gershgorin_bound(term) for term in self.term_names], dtype=np.float64
        )
        per_channel = (alpha * row_bounds[:, None]).sum(axis=0)
        self.spectral_radius = float(per_channel.max()) if per_channel.size else 0.0
        self.dt_effective = (
            min(self.dt, 1.0 / max(self.spectral_radius, 1e-30)) if cap else self.dt
        )
        self.dt_capped = bool(self.dt_effective < self.dt)

    def flux(self, values: torch.Tensor) -> torch.Tensor:
        """Node-space ``sum_t alpha_t * A_t x / m`` (positive-semidefinite)."""

        values = torch.as_tensor(values)
        flat = self.vertex_cluster.transpose(0, 1).reshape(-1)
        x_cluster = self.geometry.to_cluster(values)
        local = x_cluster.index_select(0, flat).reshape(self.operators.n_elements, 4, self.n_channels)
        message = torch.einsum("teij,ejc->teic", self.element_kernels, local)
        message = message.reshape(len(self.term_names), -1, self.n_channels)
        acc = torch.zeros(
            (len(self.term_names), self.operators.n_clusters, self.n_channels),
            dtype=values.dtype,
            device=values.device,
        ).index_add(1, flat, message)
        acc = acc / self.mass.clamp_min(1e-300).unsqueeze(0).unsqueeze(-1)
        acc = (self.alpha.unsqueeze(1) * acc).sum(dim=0)
        return self.geometry.from_cluster(acc)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.as_tensor(values)
        out = values - self.dt_effective * self.flux(values)
        out = torch.where(torch.isfinite(out), out, values)
        if self.boundary is not None:
            out = jb.apply_boundary_conditions(
                out,
                self.boundary,
                reference=values,
                dt=self.dt_effective,
                mass=self.node_mass,
            )
        return out


__all__ = [
    "DOC_ALPHA_TABLE",
    "ALPHA_PROVENANCE",
    "CHANNEL_TERM_PROVENANCE",
    "ZK_PAR_MAX_REFERENCE",
    "R_REF",
    "RADIAL_TERMS",
    "ISOTROPIC_TERMS",
    "PARALLEL_TERMS",
    "UNWEIGHTED_TERMS",
    "TERM_NAMES",
    "TERM_METRIC_WEIGHT",
    "DEFAULT_IMPLICIT_TERMS",
    "DEFAULT_EXPLICIT_TERMS",
    "DEFAULT_CHANNEL_TERMS",
    "TransportCoefficients",
    "TermOperatorSet",
    "TermStability",
    "ImplicitStencilLayer",
    "ExplicitStencilLayer",
    "build_term_operators",
    "parallel_element_kernels",
    "element_metric_weights",
    "explicit_stability_table",
    "channel_term_mask",
    "b_hat_from_gradients",
]
