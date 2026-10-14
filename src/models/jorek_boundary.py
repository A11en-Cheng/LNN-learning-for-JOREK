"""JOREK boundary-condition layer for the Phase B0 stencil operator.

The contract is taken from the JOREK sources, not invented here:

* node ``boundary`` codes are assigned by
  ``grids/grid_utils/update_boundary_types.f90`` (RECAP block at the end of the
  file defines the Stan/Guidio code meaning);
* the matrix/BC application logic is ``models/model502/mod_boundary_conditions.f90``
  (``apply_psi_BC`` / ``apply_current_BC`` / ``apply_dirichlet_1234`` /
  ``apply_dirichlet_all`` / the Mach-1 target condition gated by ``apply_cs``);
* the natural/open-field-line sheath terms are
  ``models/model502/mod_boundary_matrix_open.f90`` (``rhs_ij_5..9`` and
  ``amat_*``), selected by ``bc_natural_open = .true.`` in the run namelist;
* the magnetic-axis dof sharing is ``communication/mod_import_gvec.f90:387-404``
  (all axis nodes get ``index(1:4) = 1,2,3,4``); the coincident-node clustering
  below mirrors it.

Semantics implemented per node/field (fixed boundary, ``freeboundary = F``):

* ``psi`` (0) and ``Z_j`` (2): Dirichlet when the node is BC-coded and the run
  is not free-boundary (``apply_psi_BC`` / ``apply_current_BC``);
* ``Phi`` (1) and ``omega`` (3): Dirichlet at every BC-coded node
  (``apply_dirichlet_1234`` is set for all boundary nodes);
* ``rho`` (4), ``T_i`` (5), ``v_par`` (6), ``rho_imp`` (7), ``T_e`` (8):
  Dirichlet for the ``apply_dirichlet_all`` codes (2, 3, 12, 20, 21), otherwise
  the Mach-1 target/sheath treatment for the ``apply_cs`` codes
  (1, 3, 4, 5, 9, 11, 15, 19): ``v_par = sign(B.n) * c_s`` and a one-sided
  particle/energy outflow sink for the density/temperature channels;
* nodes with ``boundary = 0`` are *not* BC-managed even when they sit on the
  geometric wall (this is how JOREK leaves the X-point and free nodes).

Dirichlet is implemented the way JOREK's matrix does it: the boundary row is
replaced by ``zbig * I`` with the previous value on the RHS, i.e. the dof is
*held*, not extrapolated.  This is what makes "interior perturbation does not
change the Dirichlet boundary value" hold by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np

try:  # torch is a hard dependency of the model stack but keep the import lazy
    import torch
except Exception:  # pragma: no cover - the test suite imports torch anyway
    torch = None

from src.data.model502_schema import FIELD_INDEX, FIELD_NAMES

# ---------------------------------------------------------------------------
# Boundary-kind contract
# ---------------------------------------------------------------------------
INTERIOR = 0
DIRICHLET = 1
NEUMANN = 2
SHEATH_SONIC = 3
SHEATH_FLUX = 4
FREE = 5

KIND_NAMES: Mapping[int, str] = {
    INTERIOR: "interior",
    DIRICHLET: "dirichlet",
    NEUMANN: "neumann_one_sided",
    SHEATH_SONIC: "sheath_sonic",
    SHEATH_FLUX: "sheath_flux_sink",
    FREE: "free_natural",
}

#: Fields that are Dirichlet at *every* BC-coded node (``apply_dirichlet_1234``
#: for u/w plus ``apply_psi_BC``/``apply_current_BC`` for psi/Z_j).
UNCONDITIONAL_DIRICHLET_FIELDS: tuple[int, ...] = (
    FIELD_INDEX["psi"],
    FIELD_INDEX["Phi"],
    FIELD_INDEX["Z_j"],
    FIELD_INDEX["omega"],
)

#: Fields that JOREK closes with ``apply_dirichlet_all`` (codes 2, 3, 12, 20, 21).
ALL_FIELD_DIRICHLET_CODES: frozenset[int] = frozenset({2, 3, 12, 20, 21})

#: Codes carrying the Mach-1 target condition (``apply_cs`` in
#: ``mod_boundary_conditions.f90``, gated by ``no_mach1_bc``).
MACH1_SHEATH_CODES: frozenset[int] = frozenset({1, 3, 4, 5, 9, 11, 15, 19})

#: Corner codes whose adjacent pairs skip the sheath integration
#: (``construct_matrix_mod.f90:124-130`` ``is_corner_pair -> cycle``).
CORNER_CODES: frozenset[int] = frozenset({3, 9, 19, 20, 21})

#: ``grid_to_wall`` wall-side map used by ``construct_matrix_mod.f90:113-121``.
WALL_SIDE_BY_CODE: Mapping[int, int] = {
    1: 2,
    11: 2,
    5: 3,
    15: 3,
    2: 3,
    12: 2,
    4: 2,
}

#: Accepted boundary-edge code combinations of the ``grid_to_wall = .false.``
#: branch (``construct_matrix_mod.f90:141-196``); the value is the wall side
#: (2 or 3) and ``None`` means JOREK prints a warning and skips the edge.
EDGE_DIRECTION_RULES: tuple[tuple[frozenset[int], frozenset[int], int], ...] = (
    (frozenset({1, 3}), frozenset({1, 3}), 2),
    (frozenset({1, 9}), frozenset({1, 9}), 2),
    (frozenset({4, 9}), frozenset({4, 9}), 2),
    (frozenset({1, 4}), frozenset({4, 1}), 2),
    (frozenset({5, 9}), frozenset({5, 9}), 3),
    (frozenset({2, 3}), frozenset({2, 3}), 3),
)


def edge_wall_side(
    bnd1: int,
    bnd2: int,
    *,
    grid_to_wall: bool = False,
) -> int | None:
    """Wall side of a boundary edge, or ``None`` when JOREK skips it.

    Mirrors ``matrix/construct_matrix_mod.f90:110-196``: with
    ``grid_to_wall``/wall blocks the side comes from the per-code map and a
    corner pair (both codes in :data:`CORNER_CODES`) is skipped; otherwise the
    accepted (code1, code2) combinations of the ``else`` branch are used.
    """

    if grid_to_wall:
        side1 = WALL_SIDE_BY_CODE.get(int(bnd1), 0)
        side2 = WALL_SIDE_BY_CODE.get(int(bnd2), 0)
        if side1 == 0 and side2 == 0:
            return None  # is_corner_pair (or undefined wall side) -> cycle
        if side1 and side2 and side1 != side2:
            return None  # incoherent element -> warning + cycle
        return side1 or side2
    for allowed1, allowed2, side in EDGE_DIRECTION_RULES:
        if int(bnd1) in allowed1 and int(bnd2) in allowed2:
            return side
    return None


@dataclass(frozen=True)
class SheathParameters:
    """Sheath/Mach-1 parameters of the open-boundary routine.

    Defaults are ``models/preset_parameters.f90`` (:80-94); a namelist may
    override them, and ``gamma_sheath_i/e`` are derived from
    ``gamma_i_stangeby``/``gamma_e_stangeby`` exactly like
    ``models/model502/initialise_parameters.f90:241-251``.
    """

    density_reflection: float = 0.0
    imp_reflection: float = 0.0
    gamma_sheath_i: float = -0.111
    gamma_sheath_e: float = 3.0
    visco_par_heating: float = 0.0
    vpar_smoothing: bool = False
    vpar_smoothing_coef: tuple[float, float, float] = (0.01, 0.0, 0.0)
    mach_one_bnd_integral: bool = False
    min_sheath_angle: float = 1.0
    penalty: float = 1.0e12  # Zbig of mod_boundary_matrix_open.f90:73
    provenance: Mapping[str, str] = field(default_factory=dict)

    @property
    def c_angle(self) -> float:
        """Angle factor of the minimum heat/particle flux (radians)."""

        return float(np.deg2rad(self.min_sheath_angle))


def sheath_parameters_from_namelist(parsed: Mapping[str, object] | None) -> SheathParameters:
    """Build :class:`SheathParameters` from a parsed namelist (defaults = preset)."""

    parsed = dict(parsed or {})
    provenance: dict[str, str] = {}

    def number(key: str, default: float) -> float:
        value = parsed.get(key)
        if value is None:
            provenance[key] = "preset_parameters.f90"
            return float(default)
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        provenance[key] = "namelist"
        return float(array[0]) if array.size else float(default)

    def logical(key: str, default: bool) -> bool:
        value = parsed.get(key)
        if value is None:
            provenance[key] = "preset_parameters.f90"
            return bool(default)
        provenance[key] = "namelist"
        return str(value).strip().strip(".").lower() in {"t", "true", "1"}

    gamma = 5.0 / 3.0
    gamma_sheath_e = number("gamma_sheath_e", 3.0)
    gamma_sheath_i = number("gamma_sheath_i", -0.111)
    stangeby_e = parsed.get("gamma_e_stangeby")
    stangeby_i = parsed.get("gamma_i_stangeby")
    if stangeby_e is not None:
        value = float(np.asarray(stangeby_e, dtype=np.float64).reshape(-1)[0])
        if value > -1e89:
            gamma_sheath_e = (gamma - 1.0) * (value - 1.0)
            provenance["gamma_sheath_e"] = "derived from gamma_e_stangeby (initialise_parameters.f90:243)"
    if stangeby_i is not None:
        value = float(np.asarray(stangeby_i, dtype=np.float64).reshape(-1)[0])
        if value > -1e89:
            gamma_sheath_i = (gamma - 1.0) * (value - 1.0 - gamma)
            provenance["gamma_sheath_i"] = "derived from gamma_i_stangeby (initialise_parameters.f90:248)"
    coef = parsed.get("vpar_smoothing_coef")
    if coef is None:
        coef_tuple = (0.01, 0.0, 0.0)
    else:
        coef_array = np.asarray(coef, dtype=np.float64).reshape(-1)
        coef_tuple = tuple(float(v) for v in coef_array[:3])
        if len(coef_tuple) < 3:
            coef_tuple = coef_tuple + (0.0,) * (3 - len(coef_tuple))
    return SheathParameters(
        density_reflection=number("density_reflection", 0.0),
        imp_reflection=number("imp_reflection", 0.0),
        gamma_sheath_i=gamma_sheath_i,
        gamma_sheath_e=gamma_sheath_e,
        visco_par_heating=number("visco_par_heating", 0.0),
        vpar_smoothing=logical("vpar_smoothing", False),
        vpar_smoothing_coef=coef_tuple,
        mach_one_bnd_integral=logical("mach_one_bnd_integral", False),
        min_sheath_angle=number("min_sheath_angle", 1.0),
        penalty=number("zbig", 1.0e12),
        provenance=provenance,
    )

#: Fields carried by the sheath outflow RHS (density/temperature channels).
SHEATH_FLUX_FIELDS: tuple[int, ...] = (
    FIELD_INDEX["rho"],
    FIELD_INDEX["T_i"],
    FIELD_INDEX["rho_imp"],
    FIELD_INDEX["T_e"],
)


@dataclass(frozen=True)
class BoundaryCodeSpec:
    """One row of the JOREK ``boundary`` code -> BC mapping."""

    code: int
    name: str
    side: str
    field_direction: str
    description: str
    apply_cs: bool
    apply_dirichlet_all: bool
    provenance: str
    in_new_grid: bool = True


BOUNDARY_CODE_TABLE: Mapping[int, BoundaryCodeSpec] = {
    0: BoundaryCodeSpec(
        0,
        "interior",
        "-",
        "-",
        "no boundary condition (also used for free/X-point nodes on the wall)",
        False,
        False,
        "update_boundary_types.f90:60-62 (initialise)",
    ),
    1: BoundaryCodeSpec(
        1,
        "target_side2_inward",
        "2",
        "inward",
        "limiter/divertor target; Mach-1 sheath for v_par, fluxes for n/T",
        True,
        False,
        "update_boundary_types.f90:636 (RECAP), 133-176; mod_boundary_conditions.f90:234-245",
    ),
    2: BoundaryCodeSpec(
        2,
        "tangent_side3",
        "3",
        "tangent",
        "flux surface tangent to the wall; all nine fields Dirichlet",
        False,
        True,
        "update_boundary_types.f90:642 (RECAP); mod_boundary_conditions.f90:226-233",
    ),
    3: BoundaryCodeSpec(
        3,
        "corner_1_2",
        "-",
        "inward",
        "old-grid corner between type 1 and 2; Dirichlet-all + Mach-1",
        True,
        True,
        "update_boundary_types.f90:647 (old grid); mod_boundary_conditions.f90:226-245",
        in_new_grid=False,
    ),
    4: BoundaryCodeSpec(
        4,
        "target_side2_old",
        "2",
        "inward",
        "old-grid duplicate of type 1; Mach-1 sheath",
        True,
        False,
        "update_boundary_types.f90:648 (old grid)",
        in_new_grid=False,
    ),
    5: BoundaryCodeSpec(
        5,
        "target_side3_inward",
        "3",
        "inward",
        "target on side 3; Mach-1 sheath for v_par, fluxes for n/T",
        True,
        False,
        "update_boundary_types.f90:638 (RECAP), 559; mod_boundary_conditions.f90:234-245",
    ),
    9: BoundaryCodeSpec(
        9,
        "corner_free_inward",
        "corner",
        "inward",
        "free corner (inward field); Mach-1 sheath constraints",
        True,
        False,
        "update_boundary_types.f90:640 (RECAP); mod_boundary_conditions.f90:234-245",
    ),
    11: BoundaryCodeSpec(
        11,
        "target_side2_outward",
        "2",
        "outward",
        "target on side 2 with outward field; Mach-1 sheath",
        True,
        False,
        "update_boundary_types.f90:637 (RECAP), 553; mod_boundary_conditions.f90:234-245",
    ),
    12: BoundaryCodeSpec(
        12,
        "tangent_side2",
        "2",
        "tangent",
        "side-2 tangent field; all nine fields Dirichlet (new grids keep code 2)",
        False,
        True,
        "update_boundary_types.f90:565-566, 643 (RECAP); mod_boundary_conditions.f90:226-233",
    ),
    15: BoundaryCodeSpec(
        15,
        "target_side3_outward",
        "3",
        "outward",
        "target on side 3 with outward field; Mach-1 sheath",
        True,
        False,
        "update_boundary_types.f90:639 (RECAP), 555; mod_boundary_conditions.f90:234-245",
    ),
    19: BoundaryCodeSpec(
        19,
        "corner_free_outward",
        "corner",
        "outward",
        "free corner (outward field); Mach-1 sheath constraints",
        True,
        False,
        "update_boundary_types.f90:641 (RECAP), 465; mod_boundary_conditions.f90:234-245",
    ),
    20: BoundaryCodeSpec(
        20,
        "corner_fixed_tangent",
        "corner",
        "tangent",
        "corner fixed because the two sides disagree (tangent field); Dirichlet-all",
        False,
        True,
        "update_boundary_types.f90:644 (RECAP), 596-623; mod_boundary_conditions.f90:226-233",
    ),
    21: BoundaryCodeSpec(
        21,
        "corner_inverted",
        "corner",
        "-",
        "inverted three-element corner (special case); Dirichlet-all",
        False,
        True,
        "update_boundary_types.f90:645 (RECAP), 473; mod_boundary_conditions.f90:226-233",
    ),
}


def boundary_code_table() -> list[dict]:
    """The BC mapping table as plain records (used by the tests and the report)."""

    return [
        {
            "code": spec.code,
            "name": spec.name,
            "side": spec.side,
            "field_direction": spec.field_direction,
            "apply_cs": spec.apply_cs,
            "apply_dirichlet_all": spec.apply_dirichlet_all,
            "in_new_grid": spec.in_new_grid,
            "description": spec.description,
            "provenance": spec.provenance,
        }
        for spec in sorted(BOUNDARY_CODE_TABLE.values(), key=lambda item: item.code)
    ]


def field_kind_table(
    *,
    freeboundary: bool = False,
    no_mach1_bc: bool = False,
    n_fields: int = len(FIELD_NAMES),
) -> list[list[str]]:
    """``[code, field] -> kind name`` matrix matching the module contract."""

    codes = sorted(BOUNDARY_CODE_TABLE)
    return [
        [
            KIND_NAMES[
                field_kind_for_code(code, f, freeboundary=freeboundary, no_mach1_bc=no_mach1_bc)
            ]
            for f in range(n_fields)
        ]
        for code in codes
    ]


def field_kind_for_code(
    code: int,
    field_index: int,
    *,
    freeboundary: bool = False,
    no_mach1_bc: bool = False,
) -> int:
    """Map one ``boundary`` code and one state channel to its :class:`kind`."""

    spec = BOUNDARY_CODE_TABLE.get(int(code))
    if spec is None or spec.code == 0:
        return INTERIOR
    # psi/Z_j first: ``apply_psi_BC``/``apply_current_BC`` are gated by
    # ``is_freebound`` independently of ``apply_dirichlet_all``.
    if field_index in (FIELD_INDEX["psi"], FIELD_INDEX["Z_j"]):
        return FREE if freeboundary else DIRICHLET
    if field_index in UNCONDITIONAL_DIRICHLET_FIELDS:
        return DIRICHLET
    # ``no_mach1_bc = .true.`` makes every boundary node Dirichlet-all
    # (mod_boundary_conditions.f90:247-248).
    dirichlet_all = spec.apply_dirichlet_all or no_mach1_bc
    if dirichlet_all:
        return DIRICHLET
    if spec.apply_cs and not no_mach1_bc:
        if field_index == FIELD_INDEX["v_par"]:
            return SHEATH_SONIC
        if field_index in SHEATH_FLUX_FIELDS:
            return SHEATH_FLUX
    return NEUMANN


# ---------------------------------------------------------------------------
# Boundary context
# ---------------------------------------------------------------------------
@dataclass
class BoundaryContext:
    """Per-node boundary metadata plus the derived per-field masks."""

    boundary_code: np.ndarray                 # [N] int
    field_kind: np.ndarray                    # [N, 9] int (KIND_* constants)
    freeboundary: bool
    axis_node: np.ndarray                     # [N] bool
    normal: np.ndarray                        # [N, 2] outward unit normal
    direction: np.ndarray                     # [N] B.n sign at sheath nodes
    boundary_length: np.ndarray               # [N] wall-edge length (dl)
    interior_neighbour: np.ndarray            # [N] one-sided inward node id
    radius: np.ndarray                        # [N] R coordinate (flux-sink metric)
    unknown_codes: tuple[int, ...] = ()
    notes: list[str] = field(default_factory=list)
    sheath: SheathParameters = field(default_factory=SheathParameters)
    edge_nodes: np.ndarray | None = None      # [Ne, 2] oriented boundary edges
    edge_dl: np.ndarray | None = None         # [Ne] edge length
    edge_side: np.ndarray | None = None       # [Ne] 2/3 wall side (0 = skipped)
    edge_skipped: np.ndarray | None = None    # [Ne] corner-pair / undefined edge
    grid_to_wall: bool = False
    f0: float = 1.0
    pos: np.ndarray | None = None             # [N, 2] node coordinates (R, Z)
    psi_gradient: np.ndarray | None = None    # [N, 2] d psi / d (R, Z)
    provenance: str = (
        "models/model502/mod_boundary_conditions.f90; "
        "models/model502/mod_boundary_matrix_open.f90; "
        "grids/grid_utils/update_boundary_types.f90"
    )

    @property
    def n_nodes(self) -> int:
        return int(self.boundary_code.shape[0])

    @property
    def dirichlet_mask(self) -> np.ndarray:
        return self.field_kind == DIRICHLET

    @property
    def neumann_mask(self) -> np.ndarray:
        return self.field_kind == NEUMANN

    @property
    def sonic_mask(self) -> np.ndarray:
        return self.field_kind == SHEATH_SONIC

    @property
    def flux_mask(self) -> np.ndarray:
        return self.field_kind == SHEATH_FLUX

    @property
    def free_mask(self) -> np.ndarray:
        return self.field_kind == FREE

    @property
    def bc_node_mask(self) -> np.ndarray:
        return self.boundary_code != 0

    def summary(self) -> dict:
        codes, counts = np.unique(self.boundary_code, return_counts=True)
        per_field = {}
        for index, name in enumerate(FIELD_NAMES):
            kinds, kind_counts = np.unique(self.field_kind[:, index], return_counts=True)
            per_field[name] = {
                KIND_NAMES[int(k)]: int(c) for k, c in zip(kinds, kind_counts)
            }
        return {
            "n_nodes": self.n_nodes,
            "freeboundary": self.freeboundary,
            "axis_nodes": int(self.axis_node.sum()),
            "boundary_code_counts": {int(k): int(c) for k, c in zip(codes, counts)},
            "unknown_codes": list(self.unknown_codes),
            "per_field_kinds": per_field,
            "notes": list(self.notes),
            "sheath": {
                "density_reflection": self.sheath.density_reflection,
                "imp_reflection": self.sheath.imp_reflection,
                "gamma_sheath_i": self.sheath.gamma_sheath_i,
                "gamma_sheath_e": self.sheath.gamma_sheath_e,
                "vpar_smoothing": self.sheath.vpar_smoothing,
                "mach_one_bnd_integral": self.sheath.mach_one_bnd_integral,
                "min_sheath_angle": self.sheath.min_sheath_angle,
                "c_angle": self.sheath.c_angle,
            },
            "n_boundary_edges": (
                None if self.edge_nodes is None else int(self.edge_nodes.shape[0])
            ),
            "n_skipped_edges": (
                None if self.edge_skipped is None else int(self.edge_skipped.sum())
            ),
        }


def _parse_logical(value) -> bool:
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value.strip().upper() in {"T", "TRUE", "1"}
    array = np.asarray(value).reshape(-1)
    if array.dtype.kind in {"S", "U"}:
        return str(array[0]).strip().upper() in {"T", "TRUE", "1"}
    return bool(int(array[0]))


def boundary_normals(
    pos: np.ndarray,
    vertex: np.ndarray,
    boundary_code: np.ndarray,
) -> np.ndarray:
    """Outward unit normals at BC-coded nodes from the element geometry.

    Mirrors the JOREK construction ``(R_mid - R_center, Z_mid - Z_center)``:
    for every element side whose two corners are BC-coded, the side midpoint
    minus the element centroid gives the outward direction; contributions are
    averaged over the incident elements.
    """

    pos = np.asarray(pos, dtype=np.float64)
    vertex = np.asarray(vertex, dtype=np.int64)
    code = np.asarray(boundary_code)
    normals = np.zeros_like(pos)
    counts = np.zeros(pos.shape[0], dtype=np.float64)
    boundary = code != 0
    for element in range(vertex.shape[1]):
        corners = vertex[:, element]
        centroid = pos[corners].mean(axis=0)
        found_side = False
        for side in range(4):
            a = corners[side]
            b = corners[(side + 1) % 4]
            if not (boundary[a] and boundary[b]):
                continue
            midpoint = 0.5 * (pos[a] + pos[b])
            direction = midpoint - centroid
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-30:
                continue
            found_side = True
            for node in (a, b):
                normals[node] += direction / norm
                counts[node] += 1.0
        if not found_side:
            # Isolated BC-coded corner (e.g. code 21): fall back to the
            # node - centroid direction so the Mach-1 sign is still defined.
            for node in corners:
                if not boundary[node]:
                    continue
                direction = pos[node] - centroid
                norm = float(np.linalg.norm(direction))
                if norm > 1e-30:
                    normals[node] += direction / norm
                    counts[node] += 1.0
    counts = np.where(counts > 0, counts, 1.0)
    normals = normals / counts[:, None]
    norm = np.linalg.norm(normals, axis=1)
    safe = np.where(norm > 1e-30, norm, 1.0)
    return normals / safe[:, None]


def boundary_edge_lengths(
    pos: np.ndarray,
    vertex: np.ndarray,
    boundary_code: np.ndarray,
) -> np.ndarray:
    """Half the sum of the wall-edge lengths incident to each BC-coded node."""

    pos = np.asarray(pos, dtype=np.float64)
    vertex = np.asarray(vertex, dtype=np.int64)
    code = np.asarray(boundary_code)
    length = np.zeros(pos.shape[0], dtype=np.float64)
    for element in range(vertex.shape[1]):
        corners = vertex[:, element]
        for side in range(4):
            a = corners[side]
            b = corners[(side + 1) % 4]
            if not (code[a] != 0 and code[b] != 0):
                continue
            dl = float(np.linalg.norm(pos[a] - pos[b]))
            length[a] += 0.5 * dl
            length[b] += 0.5 * dl
    return length


def one_sided_interior_neighbour(
    pos: np.ndarray,
    vertex: np.ndarray,
    boundary_code: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Nearest inward neighbour used by the one-sided Neumann/sheath operators."""

    pos = np.asarray(pos, dtype=np.float64)
    vertex = np.asarray(vertex, dtype=np.int64)
    code = np.asarray(boundary_code)
    neighbour = np.arange(pos.shape[0], dtype=np.int64)
    best_score = np.full(pos.shape[0], -np.inf, dtype=np.float64)
    for element in range(vertex.shape[1]):
        corners = vertex[:, element]
        for corner in range(4):
            node = corners[corner]
            if code[node] == 0:
                continue
            inward = -normal[node]
            for other in range(4):
                if other == corner:
                    continue
                candidate = corners[other]
                offset = pos[candidate] - pos[node]
                norm = float(np.linalg.norm(offset))
                if norm <= 1e-30:
                    continue
                score = float(np.dot(offset / norm, inward))
                interior_bonus = 1.0 if code[candidate] == 0 else 0.0
                score += 0.5 * interior_bonus
                if score > best_score[node]:
                    best_score[node] = score
                    neighbour[node] = candidate
    return neighbour


def boundary_edge_topology(
    pos: np.ndarray,
    vertex: np.ndarray,
    boundary_code: np.ndarray,
    *,
    grid_to_wall: bool = False,
    min_length: float = 1e-30,
) -> dict:
    """Wall edges of the boundary mesh with their JOREK sheath acceptance.

    Returns ``{"nodes": [Ne, 2], "dl": [Ne], "side": [Ne], "skipped": [Ne],
    "skip_reasons": {...}}``.  An edge is a quadrilateral side whose two nodes
    both carry a non-zero ``boundary`` code (``construct_matrix_mod.f90:99-103``
    requires ``bnd1 != 0 and bnd2 != 0``).  Edges that JOREK drops
    (``is_corner_pair -> cycle`` at :124-130, or the undefined combinations of
    the ``else`` branch at :141-196) are flagged in ``skipped`` so the sheath
    integration can be masked without losing the registration.
    """

    pos = np.asarray(pos, dtype=np.float64)
    vertex = np.asarray(vertex, dtype=np.int64)
    code = np.asarray(boundary_code).reshape(-1)
    nodes: list[tuple[int, int]] = []
    lengths: list[float] = []
    sides: list[int] = []
    skipped: list[bool] = []
    corner_pairs = 0
    undefined = 0
    for element in range(vertex.shape[1]):
        corners = vertex[:, element]
        for side in range(4):
            a = int(corners[side])
            b = int(corners[(side + 1) % 4])
            bnd1, bnd2 = int(code[a]), int(code[b])
            if bnd1 == 0 or bnd2 == 0:
                continue
            dl = float(np.linalg.norm(pos[a] - pos[b]))
            if dl <= min_length:
                continue
            wall_side = edge_wall_side(bnd1, bnd2, grid_to_wall=grid_to_wall)
            is_corner_pair = (
                bnd1 in CORNER_CODES and bnd2 in CORNER_CODES
            )
            skip = wall_side is None
            if skip:
                if is_corner_pair:
                    corner_pairs += 1
                else:
                    undefined += 1
            nodes.append((a, b))
            lengths.append(dl)
            sides.append(0 if wall_side is None else int(wall_side))
            skipped.append(bool(skip))
    return {
        "nodes": np.asarray(nodes, dtype=np.int64).reshape(-1, 2),
        "dl": np.asarray(lengths, dtype=np.float64),
        "side": np.asarray(sides, dtype=np.int64),
        "skipped": np.asarray(skipped, dtype=bool),
        "skip_reasons": {
            "corner_pair": int(corner_pairs),
            "undefined_combination": int(undefined),
            "provenance": (
                "matrix/construct_matrix_mod.f90:99-196 (bc_natural_open sheath loop); "
                "is_corner_pair cycle at :124-130"
            ),
        },
    }


def build_boundary_context(
    boundary_code: np.ndarray,
    pos: np.ndarray,
    vertex: np.ndarray,
    *,
    freeboundary: bool = False,
    axis_node: np.ndarray | None = None,
    psi_gradient: np.ndarray | None = None,
    no_mach1_bc: bool = False,
    sheath: SheathParameters | None = None,
    grid_to_wall: bool = False,
    f0: float = 1.0,
    n_fields: int = len(FIELD_NAMES),
) -> BoundaryContext:
    """Assemble the per-node context and the per-field BC masks."""

    code = np.asarray(boundary_code, dtype=np.int64).reshape(-1)
    pos = np.asarray(pos, dtype=np.float64)
    vertex = np.asarray(vertex, dtype=np.int64)
    if vertex.ndim != 2 or vertex.shape[0] != 4:
        raise ValueError(f"vertex must be [4, n_elements], got {vertex.shape}")
    if axis_node is None:
        axis = np.zeros(code.shape[0], dtype=bool)
    else:
        axis = np.asarray(axis_node).reshape(-1).astype(bool)

    unknown = tuple(
        sorted({int(c) for c in np.unique(code) if int(c) != 0 and int(c) not in BOUNDARY_CODE_TABLE})
    )
    kinds = np.full((code.shape[0], n_fields), INTERIOR, dtype=np.int64)
    for channel in range(n_fields):
        for node, node_code in enumerate(code):
            kinds[node, channel] = field_kind_for_code(
                int(node_code),
                channel,
                freeboundary=freeboundary,
                no_mach1_bc=no_mach1_bc,
            )

    normal = boundary_normals(pos, vertex, code)
    length = boundary_edge_lengths(pos, vertex, code)
    neighbour = one_sided_interior_neighbour(pos, vertex, code, normal)

    # Mach-1 flow direction: JOREK uses sign(dot((psi_y, -psi_x), normal)),
    # with (psi_x, psi_y) = physical (d/dR, d/dZ) flux gradient.
    direction = np.ones(code.shape[0], dtype=np.float64)
    notes: list[str] = []
    if psi_gradient is not None and np.asarray(psi_gradient).shape[0] == code.shape[0]:
        grad = np.asarray(psi_gradient, dtype=np.float64)
        projection = grad[:, 1] * normal[:, 0] - grad[:, 0] * normal[:, 1]
        sign = np.sign(projection)
        direction = np.where(sign == 0.0, 1.0, sign)
    else:
        notes.append(
            "psi_gradient not supplied: Mach-1 direction defaults to +1 "
            "(sign(B.n) unknown); open question for the JOREK-side review"
        )
    if freeboundary:
        notes.append(
            "freeboundary=T: psi/Z_j Dirichlet suppressed; the vacuum coupling "
            "of mod_boundary_conditions.f90 / mod_vacuum_fields.f90 is not modelled"
        )
    if no_mach1_bc:
        notes.append("no_mach1_bc=T: apply_cs disabled, all BC nodes Dirichlet-all")
    if unknown:
        notes.append(f"unknown boundary codes {unknown} treated as Dirichlet for psi/Phi/Z_j/omega")

    sheath = sheath if sheath is not None else SheathParameters()
    topology = boundary_edge_topology(
        pos, vertex, code, grid_to_wall=bool(grid_to_wall)
    )
    if topology["skipped"].any():
        notes.append(
            "corner-pair / undefined boundary edges skipped by the sheath integration "
            f"(construct_matrix_mod.f90:124-130): {int(topology['skipped'].sum())} edges "
            f"(corner_pair={topology['skip_reasons']['corner_pair']}, "
            f"undefined={topology['skip_reasons']['undefined_combination']})"
        )
    if sheath.mach_one_bnd_integral:
        notes.append(
            "mach_one_bnd_integral=T: the Mach-1 condition is imposed through the "
            "boundary integral (rhs_ij_7/amat_76-77); the node-level implementation "
            "applies the same converged relation"
        )
    if sheath.vpar_smoothing:
        notes.append(
            "vpar_smoothing=T: the tanh smoothing factor of "
            "mod_boundary_conditions.f90:466-486 is applied to the Mach-1 target"
        )

    return BoundaryContext(
        boundary_code=code,
        field_kind=kinds,
        freeboundary=bool(freeboundary),
        axis_node=axis,
        normal=normal,
        direction=direction,
        boundary_length=length,
        interior_neighbour=neighbour,
        radius=pos[:, 0].copy(),
        unknown_codes=unknown,
        notes=notes,
        sheath=sheath,
        edge_nodes=topology["nodes"],
        edge_dl=topology["dl"],
        edge_side=topology["side"],
        edge_skipped=topology["skipped"],
        grid_to_wall=bool(grid_to_wall),
        f0=float(f0),
        pos=pos.copy(),
        psi_gradient=(
            None if psi_gradient is None else np.asarray(psi_gradient, dtype=np.float64).copy()
        ),
    )


def load_boundary_context(
    h5_path: str | Path,
    *,
    vertex: np.ndarray | None = None,
    pos: np.ndarray | None = None,
    freeboundary: bool | None = None,
    psi_gradient: np.ndarray | None = None,
    no_mach1_bc: bool = False,
    sheath: SheathParameters | None = None,
    grid_to_wall: bool = False,
    f0: float = 1.0,
    namelist_path: str | Path | None = None,
) -> BoundaryContext:
    """Read ``boundary``/``freeboundary``/``axis_node`` and build the context.

    ``vertex``/``pos`` should be passed in 0-based form when the caller already
    loaded the frame (``src.models.stencil_conv.load_frame_geometry``); when they
    are ``None`` they are read (and converted) from the same HDF5 frame.
    """

    import h5py

    with h5py.File(h5_path, "r") as handle:
        code = np.asarray(handle["boundary"][()], dtype=np.int64).reshape(-1)
        if freeboundary is None:
            freeboundary = _parse_logical(handle["freeboundary"][()]) if "freeboundary" in handle else False
        axis_node = None
        if "axis_node" in handle:
            raw_axis = np.asarray(handle["axis_node"][()]).reshape(-1)
            if raw_axis.dtype.kind in {"S", "U"}:
                axis_node = np.array([str(item).strip().upper() in {"T", "TRUE", "1"} for item in raw_axis])
            else:
                axis_node = raw_axis.astype(bool)
        if vertex is None:
            vertex = np.asarray(handle["vertex"][()], dtype=np.int64) - 1
        if pos is None:
            x = np.asarray(handle["x"][()], dtype=np.float64)
            if x.ndim == 4:
                pos = x[:, 0, 0, :].T
            elif x.ndim == 3:
                pos = x[:, 0, :].T
            else:
                pos = x.T
    if sheath is None and namelist_path is not None:
        from src.data.namelist import parse_namelist

        text = Path(namelist_path).read_text(encoding="utf-8", errors="replace")
        sheath = sheath_parameters_from_namelist(parse_namelist(text))
    return build_boundary_context(
        code,
        pos,
        vertex,
        freeboundary=bool(freeboundary),
        axis_node=axis_node,
        psi_gradient=psi_gradient,
        no_mach1_bc=_parse_logical(no_mach1_bc) if not isinstance(no_mach1_bc, bool) else no_mach1_bc,
        sheath=sheath,
        grid_to_wall=grid_to_wall,
        f0=f0,
    )


# ---------------------------------------------------------------------------
# Applying the boundary conditions to a layer output
# ---------------------------------------------------------------------------
def sanitize_boundary_inputs(
    values,
    context: BoundaryContext,
    *,
    gamma: float = 5.0 / 3.0,
    cs_clip: tuple[float, float] = (1e-12, 1e6),
):
    """Make BC-coded rows finite before they feed the interior stencil.

    A non-finite boundary value (e.g. produced by a diverging layer) is replaced
    by the one-sided inward neighbour value (0 when that is also non-finite) and
    the Mach-1 rows are re-imposed, so a boundary NaN cannot leak into the
    interior update of the next layer.
    """

    values = torch.as_tensor(values)
    out = values.clone()
    dirichlet = torch.as_tensor(context.dirichlet_mask, device=values.device)
    neumann = torch.as_tensor(context.neumann_mask, device=values.device)
    sonic = torch.as_tensor(context.sonic_mask, device=values.device)
    flux = torch.as_tensor(context.flux_mask, device=values.device)
    free = torch.as_tensor(context.free_mask, device=values.device)
    bc_rows = dirichlet | neumann | sonic | flux | free
    if not bool(bc_rows.any()):
        return out
    index = torch.as_tensor(context.interior_neighbour, device=values.device)
    one_sided = out.index_select(0, index)
    fallback = torch.where(torch.isfinite(one_sided), one_sided, torch.zeros_like(one_sided))
    out = torch.where(bc_rows & ~torch.isfinite(out), fallback, out)
    temperature = out[:, FIELD_INDEX["T_i"]] + out[:, FIELD_INDEX["T_e"]]
    cs = torch.sqrt(torch.clamp(gamma * temperature, min=cs_clip[0] ** 2)).clamp(
        min=cs_clip[0], max=cs_clip[1]
    )
    if bool(sonic.any()):
        direction = torch.as_tensor(context.direction, device=values.device, dtype=values.dtype)
        out = torch.where(sonic, (direction * cs).unsqueeze(1).expand_as(out), out)
    return out


def apply_boundary_conditions(
    values,
    context: BoundaryContext,
    *,
    reference=None,
    dt: float = 1.0,
    gamma: float = 5.0 / 3.0,
    c_angle: float = 0.0174524,  # 1 degree, mod_boundary_matrix_open.f90:62
    mass=None,
    cs_clip: tuple[float, float] = (1e-12, 1e6),
):
    """Reset the BC-coded nodes of one stencil-conv output.

    ``values``/``reference`` are ``[n_nodes, n_fields]`` tensors (the layer
    output and the pre-layer/previous state).  Dirichlet rows are held at the
    reference value; Neumann rows take the one-sided inward neighbour value
    (zero normal flux); Mach-1 rows use ``v_par = direction * c_s``; the
    density/temperature channels lose the one-sided sheath outflow
    ``rate = c_angle * c_s * R * dl / m`` (clipped so a layer cannot flip the
    sign).  Rows that are not BC-coded are returned untouched.
    """

    if torch is None:  # pragma: no cover
        raise RuntimeError("torch is required for apply_boundary_conditions")
    values = torch.as_tensor(values)
    if values.ndim != 2:
        raise ValueError(f"values must be [n_nodes, n_fields], got {tuple(values.shape)}")
    if values.shape[0] != context.n_nodes:
        raise ValueError(
            f"values has {values.shape[0]} nodes but the boundary context has {context.n_nodes}"
        )
    reference = values if reference is None else torch.as_tensor(reference, device=values.device, dtype=values.dtype)
    finite_reference = torch.where(torch.isfinite(reference), reference, torch.zeros_like(reference))
    out = values.clone()

    dirichlet = torch.as_tensor(context.dirichlet_mask, device=values.device)
    neumann = torch.as_tensor(context.neumann_mask, device=values.device)
    sonic = torch.as_tensor(context.sonic_mask, device=values.device)
    flux = torch.as_tensor(context.flux_mask, device=values.device)
    free = torch.as_tensor(context.free_mask, device=values.device)
    any_bc = dirichlet | neumann | sonic | flux | free

    # Boundary rows are always sanitised against the (finite part of the)
    # reference, so a non-finite layer output cannot leak inward.
    out = torch.where(any_bc & ~torch.isfinite(out), finite_reference, out)

    out = torch.where(dirichlet, finite_reference, out)

    # Free-boundary psi/Z_j have no Dirichlet row in JOREK (they are coupled to
    # the vacuum solution); the component-level analogue is the same one-sided
    # natural (zero normal flux) treatment as Neumann.
    one_sided_rows = neumann | free
    if bool(one_sided_rows.any()):
        index = torch.as_tensor(context.interior_neighbour, device=values.device)
        one_sided = finite_reference.index_select(0, index)
        out = torch.where(one_sided_rows, one_sided, out)

    if (bool(flux.any()) or bool(sonic.any())) and mass is not None:
        out = _apply_open_boundary_terms(
            out,
            context,
            reference=finite_reference,
            dt=float(dt),
            gamma=gamma,
            # the namelist ``min_sheath_angle`` (via SheathParameters) wins over
            # the legacy explicit argument; the defaults are identical (1 deg)
            c_angle=context.sheath.c_angle if context.sheath is not None else c_angle,
            mass=mass,
            flux=flux,
            sonic=sonic,
            cs_clip=cs_clip,
        )

    return out


def _edge_nodal_averages(
    context: BoundaryContext,
    reference,
) -> dict:
    """Node-level edge averages used by the sheath RHS/linearisation.

    The tangential derivatives along the wall edge and the normal derivatives
    from the one-sided interior neighbour mirror the Gaussian-point quantities
    of ``mod_boundary_matrix_open.f90:206-266``.  Orientation-dependent
    combinations are stored as the invariant products ``ps0_s*normal_sign3``
    and ``u0_s*normal_sign3`` (both are unchanged when the arbitrary edge
    orientation is flipped, see ``normal_sign3 = sign(ps0_s)*normal_sign`` at
    :266).
    """

    n_nodes = context.n_nodes
    device, dtype = reference.device, reference.dtype
    edges = context.edge_nodes
    if edges is None or edges.size == 0:
        zeros = torch.zeros(n_nodes, device=device, dtype=dtype)
        return {
            "psi_s_abs": zeros,
            "u_s_sign": zeros,
            "dl": zeros,
            "weight": zeros,
        }
    dl = torch.as_tensor(context.edge_dl, device=device, dtype=dtype)
    active = ~torch.as_tensor(context.edge_skipped, device=device)
    index_i = torch.as_tensor(edges[:, 0], device=device, dtype=torch.long)
    index_j = torch.as_tensor(edges[:, 1], device=device, dtype=torch.long)
    psi = reference[:, FIELD_INDEX["psi"]]
    u = reference[:, FIELD_INDEX["Phi"]]
    dpsi = (psi.index_select(0, index_j) - psi.index_select(0, index_i)) / dl
    du = (u.index_select(0, index_j) - u.index_select(0, index_i)) / dl
    sign_psi = torch.sign(dpsi)
    sign_psi = torch.where(sign_psi == 0.0, torch.ones_like(sign_psi), sign_psi)
    half = 0.5 * dl
    acc_abs = torch.zeros(n_nodes, device=device, dtype=dtype)
    acc_u = torch.zeros(n_nodes, device=device, dtype=dtype)
    acc_w = torch.zeros(n_nodes, device=device, dtype=dtype)
    for index in (index_i, index_j):
        acc_abs = acc_abs.index_add(0, index, torch.where(active, half * dpsi.abs(), torch.zeros_like(half)))
        acc_u = acc_u.index_add(0, index, torch.where(active, half * du * sign_psi, torch.zeros_like(half)))
        acc_w = acc_w.index_add(0, index, torch.where(active, half, torch.zeros_like(half)))
    weight = acc_w.clamp_min(1e-300)
    return {
        "psi_s_abs": acc_abs / weight,
        "u_s_sign": acc_u / weight,
        "dl": acc_w,
        "weight": weight,
    }


def _apply_open_boundary_terms(
    values,
    context: BoundaryContext,
    *,
    reference,
    dt: float,
    gamma: float,
    c_angle: float,
    mass,
    flux,
    sonic,
    cs_clip: tuple[float, float],
):
    """JOREK open-boundary (``bc_natural_open``) sheath terms, node-level.

    Replicates the ``boundary_matrix_open`` RHS (``rhs_ij_5..9``,
    ``mod_boundary_matrix_open.f90:288-319``) together with the diagonal part of
    its linearisation (``amat_55/66/88/99``, :362-416) and the Mach-1 relation
    of ``mod_boundary_conditions.f90:511-520``::

        v_par = direction * factor * c_s / Btot + factor * R^2 * U_b / psi_b

    The off-diagonal linearisation couplings (``amat_51/52/56/57/61/62/65/67/
    76/77/81/82/86/87/91/92/95/96/97``, i.e. the cross-field/cross-channel
    Jacobian entries) are *not* assembled; the node-level diagonal solve keeps
    the sign structure and the (gamma_sheath - 1) energy weights.  Corners whose
    adjacent pair is skipped by ``construct_matrix_mod.f90:124-130`` do not
    receive the sheath flux.
    """

    if torch is None:  # pragma: no cover - torch is imported by the caller
        raise RuntimeError("torch is required for the sheath boundary terms")
    sheath = context.sheath
    device, dtype = values.device, values.dtype
    mass_t = torch.as_tensor(mass, device=device, dtype=dtype).reshape(-1)
    if mass_t.numel() != values.shape[0]:
        raise ValueError(
            "mass must be the per-node lumped mass with one entry per node "
            f"(got {mass_t.numel()} entries for {values.shape[0]} nodes)"
        )
    mass_t = mass_t.clamp_min(1e-30)
    edge = _edge_nodal_averages(context, reference)
    radius = torch.as_tensor(context.radius, device=device, dtype=dtype).clamp_min(0.0)
    direction = torch.as_tensor(context.direction, device=device, dtype=dtype)

    temperature = reference[:, FIELD_INDEX["T_i"]] + reference[:, FIELD_INDEX["T_e"]]
    cs = torch.sqrt(torch.clamp(gamma * temperature, min=cs_clip[0] ** 2)).clamp(
        min=cs_clip[0], max=cs_clip[1]
    )
    cs_T = gamma * reference[:, FIELD_INDEX["T_i"]] / (2.0 * cs)

    # ---- sheath particle/energy RHS + diagonal linearisation --------------
    out = values
    if bool(flux.any()):
        vpar = reference[:, FIELD_INDEX["v_par"]]
        rho = reference[:, FIELD_INDEX["rho"]]
        ti = reference[:, FIELD_INDEX["T_i"]]
        te = reference[:, FIELD_INDEX["T_e"]]
        rho_imp = reference[:, FIELD_INDEX["rho_imp"]]
        ps0_s_n3 = direction * edge["psi_s_abs"]
        u0_s_n3 = direction * edge["u_s_sign"]
        c_flux_rate = c_angle * cs * radius * edge["dl"] / mass_t
        # per-mass v_par advection and perpendicular-flow reflection terms of
        # amat_55/66/88/99 (mod_boundary_matrix_open.f90:362-416)
        vpar_term = vpar * ps0_s_n3 / mass_t
        u_term = radius ** 2 * u0_s_n3 / mass_t

        def update_rate(field_ref, gamma_factor, reflection, extra_cs_term=0.0):
            diagonal = gamma_factor * (c_flux_rate + extra_cs_term + vpar_term) + (
                -reflection * vpar_term + u_term
            )
            source = reflection * vpar_term
            diagonal = torch.clamp(diagonal, min=0.0)
            # Component-level boundedness guard: the recycling source can only
            # slow the sink, never invert it (the full JOREK matrix has extra
            # off-diagonal couplings which are registered as gaps).
            source = torch.clamp(source, max=diagonal)
            numerator = field_ref * (1.0 + dt * source)
            denominator = 1.0 + dt * diagonal
            return numerator / denominator

        rho_new = update_rate(rho, 1.0, sheath.density_reflection)
        ti_new = update_rate(
            ti,
            sheath.gamma_sheath_i - 1.0,
            0.0,
            extra_cs_term=cs_T * ti * c_angle * radius * edge["dl"] / mass_t,
        )
        imp_new = update_rate(rho_imp, 1.0, sheath.imp_reflection)
        te_new = update_rate(te, sheath.gamma_sheath_e - 1.0, 0.0)
        if sheath.visco_par_heating:
            neighbour = torch.as_tensor(
                context.interior_neighbour, device=device, dtype=torch.long
            )
            distance = (
                node_pos.index_select(0, neighbour) - node_pos
            ).norm(dim=1).clamp_min(1e-30)
            grad_vpar = (vpar.index_select(0, neighbour) - vpar) / distance
            ti_new = ti_new - dt * (gamma - 1.0) * sheath.visco_par_heating * grad_vpar * (
                vpar * radius * edge["dl"] / mass_t
            )
        rows = {
            FIELD_INDEX["rho"]: rho_new,
            FIELD_INDEX["T_i"]: ti_new,
            FIELD_INDEX["rho_imp"]: imp_new,
            FIELD_INDEX["T_e"]: te_new,
        }
        for field, updated in rows.items():
            mask = flux[:, field]
            column = torch.zeros(values.shape[1], device=device, dtype=torch.bool)
            column[field] = True
            replacement = out.clone()
            replacement[:, field] = torch.where(mask, updated, out[:, field])
            out = torch.where(
                mask.reshape(-1, 1) & column.reshape(1, -1), replacement, out
            )

    # ---- Mach-1 target on v_par (apply_cs branch, penalty limit) ----------
    # Evaluated on the flux-updated state so ``v_par`` satisfies the same
    # relation that ``bc_consistency_report`` checks; psi/Phi are Dirichlet-held
    # on the sheath nodes, so the normal-derivative correction is unchanged.
    if bool(sonic.any()):
        sonic_value = _mach1_target(context, out, gamma=gamma).clamp(
            -cs_clip[1], cs_clip[1]
        )
        replacement = out.clone()
        replacement[:, FIELD_INDEX["v_par"]] = torch.where(
            sonic[:, FIELD_INDEX["v_par"]],
            sonic_value,
            out[:, FIELD_INDEX["v_par"]],
        )
        column = torch.zeros(values.shape[1], device=device, dtype=torch.bool)
        column[FIELD_INDEX["v_par"]] = True
        out = torch.where(sonic & column.reshape(1, -1), replacement, out)
    return out


def bc_consistency_report(
    values,
    context: BoundaryContext,
    *,
    reference=None,
    gamma: float = 5.0 / 3.0,
    tolerance: float = 1e-6,
) -> dict:
    """Check a layer output against the BC contract (used by tests/report)."""

    values = torch.as_tensor(values)
    reference = values if reference is None else torch.as_tensor(reference, device=values.device, dtype=values.dtype)
    report: dict = {
        "finite": bool(torch.isfinite(values).all()),
        "n_bc_nodes": int(context.bc_node_mask.sum()),
        "n_dirichlet_rows": int(context.dirichlet_mask.sum()),
        "n_neumann_rows": int(context.neumann_mask.sum()),
        "n_sonic_rows": int(context.sonic_mask.sum()),
        "n_flux_rows": int(context.flux_mask.sum()),
    }
    dirichlet = torch.as_tensor(context.dirichlet_mask, device=values.device)
    if bool(dirichlet.any()):
        delta = (values - reference).abs()
        report["max_dirichlet_deviation"] = float(delta[dirichlet].max()) if bool(dirichlet.any()) else 0.0
        report["dirichlet_ok"] = report["max_dirichlet_deviation"] <= tolerance
    sonic = torch.as_tensor(context.sonic_mask, device=values.device)
    if bool(sonic.any()):
        # Same converged Mach-1 relation as ``_apply_open_boundary_terms``:
        # v_par = direction * factor * c_s / Btot + factor * R^2 * U_b / psi_b
        # (mod_boundary_conditions.f90:511-520, penalty limit).
        with torch.no_grad():
            # the constraint is a property of the *output* state (the temperatures
            # may have been updated by the sheath flux in the same application)
            target_full = _mach1_target(context, values, gamma=gamma)
        target = target_full.unsqueeze(1).expand_as(values)
        report["max_sonic_deviation"] = float((values - target).abs()[sonic].max())
        report["sonic_ok"] = report["max_sonic_deviation"] <= tolerance
    return report


def _mach1_target(context: BoundaryContext, reference, *, gamma: float = 5.0 / 3.0):
    """Node-level Mach-1 ``v_par`` target ``direction * factor * c_s / Btot (+corr)``."""

    device, dtype = reference.device, reference.dtype
    radius = torch.as_tensor(context.radius, device=device, dtype=dtype).clamp_min(0.0)
    temperature = reference[:, FIELD_INDEX["T_i"]] + reference[:, FIELD_INDEX["T_e"]]
    cs = torch.sqrt(torch.clamp(gamma * temperature, min=1e-24)).clamp(max=1e6)
    direction = torch.as_tensor(context.direction, device=device, dtype=dtype)
    if context.pos is None:
        node_pos = torch.stack([radius, torch.zeros_like(radius)], dim=1)
    else:
        node_pos = torch.as_tensor(context.pos, device=device, dtype=dtype)
    normal = torch.as_tensor(context.normal, device=device, dtype=dtype)
    factor = torch.ones_like(cs)
    local_direction = direction.clone()
    if context.psi_gradient is None:
        b_total = torch.ones(context.n_nodes, device=device, dtype=dtype)
        b_dot_n = torch.zeros(context.n_nodes, device=device, dtype=dtype)
    else:
        gradient = torch.as_tensor(context.psi_gradient, device=device, dtype=dtype)
        b_total = torch.sqrt(
            float(context.f0) ** 2 + gradient[:, 0] ** 2 + gradient[:, 1] ** 2
        ) / radius.clamp_min(1e-8)
        b_dot_n = (
            gradient[:, 1] * normal[:, 0] - gradient[:, 0] * normal[:, 1]
        ) / (radius.clamp_min(1e-8) * b_total.clamp_min(1e-12))
    if context.sheath.vpar_smoothing:
        c1, c2, c3 = context.sheath.vpar_smoothing_coef
        if c2 > 0.0:
            factor = 0.25 * (1.0 + torch.tanh((b_dot_n.abs() - c1) / c2)) ** 2 - c3
        else:
            factor = torch.tanh(b_dot_n / max(c1, 1e-12))
            local_direction = torch.ones_like(direction)
    neighbour = torch.as_tensor(context.interior_neighbour, device=device, dtype=torch.long)
    distance = (node_pos.index_select(0, neighbour) - node_pos).norm(dim=1).clamp_min(1e-30)
    psi = reference[:, FIELD_INDEX["psi"]]
    u = reference[:, FIELD_INDEX["Phi"]]
    psi_b = (psi.index_select(0, neighbour) - psi) / distance
    u_b = (u.index_select(0, neighbour) - u) / distance
    sonic_term = local_direction * factor * cs / b_total.clamp_min(1e-12)
    if context.psi_gradient is None:
        # without a physical flux gradient the poloidal-flow correction is not
        # defined; registering the approximation by dropping it is the only
        # non-arbitrary choice for synthetic states
        correction = torch.zeros_like(psi_b)
    else:
        # guard the 0/0 branch: ``torch.where`` would otherwise propagate the
        # NaN of the unselected branch into the backward pass
        valid = psi_b.abs() > 1e-12
        psi_b_safe = torch.where(valid, psi_b, torch.ones_like(psi_b))
        correction = torch.where(
            valid,
            radius ** 2 * u_b / psi_b_safe,
            torch.zeros_like(psi_b),
        )
        # boundedness guard: the poloidal-flow correction may not exceed the
        # sound-speed scale (JOREK's linearisation assumes it is a correction)
        correction = correction.clamp(-sonic_term.abs(), sonic_term.abs())
    return (sonic_term + factor * correction).clamp(-1e6, 1e6)
