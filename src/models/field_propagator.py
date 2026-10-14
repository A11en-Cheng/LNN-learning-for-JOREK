"""Phase B2 nine-field propagator: shallow FiLM encoder + implicit stencil stack.

Architecture (ADR-0018 decisions 5/6/7/11/12/13, ``roadmap_v2.md`` section 4
Phase B2):

``u -> shallow encoder -> FiLM (shallowest layer only) -> LTC closed-form time
integration -> n x implicit stencil layers -> decoder -> BC reset``

* **Nine fields, one contract**: all nine Model-502 channels flow through the
  same stack (no 7+2 split, ADR-0018 decision 7).
* **Condition (decision 12)**: FiLM/AdaLN modulates only the shallowest encoded
  representation; it is a *global scalar* gain/bias with identity
  initialisation (``gamma = 1``, ``beta = 0``).  Deep stencil layers receive no
  condition argument at all, so their local Jacobian with respect to the
  condition is structurally zero; a group permutation can only change the
  overall gain (response shape invariant - the behavioral acceptance).
  Condition groups follow the export-v3 vocabulary
  (``report/data-export-spec-v3.md`` section 4.2): ``pellet_*`` /
  ``background_plasma`` / ``boundary_flags`` are FiLM-eligible; the
  ``excluded_transport`` / ``excluded_numeric`` groups raise.
* **Time integration (decision 13)**: the stencil layers are implicit
  (``src.models.implicit_stencil``); the LTC component is the exact closed-form
  solution of a per-channel leaky integrator (``rate > 0`` by ``softplus``), so
  ``dt = 0`` is the identity and any ``dt`` is non-expansive.
* **Boundary (decision 11)**: every layer sanitises and resets the BC rows via
  the Phase B0 layer (``src.models.jorek_boundary``).  Boundary codes are
  checked fail-fast: unknown codes raise, codes pending JOREK-side
  confirmation (B0 report section 6) require an explicit acknowledgement and
  are registered in the returned report.  ``use_simple_bnd_types`` /
  ``no_mach1_bc`` are read from the namelist value chain.

The B2 acceptance is component-level and physical: boundedness / no NaN / BC
consistency at physical ``dt`` (0.05 JOREK time), exact discrete relations,
determinism, gradient flow, FiLM depth and response-shape invariants.  No
performance baseline or gate is defined here (ADR-0018 decision 2/10).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.model502_schema import NUM_STATE_FIELDS
from src.models import jorek_boundary as jb
from src.models import stencil_conv as sc
from src.models.implicit_stencil import (
    DEFAULT_CHANNEL_TERMS,
    DEFAULT_EXPLICIT_TERMS,
    DEFAULT_IMPLICIT_TERMS,
    TERM_NAMES,
    ImplicitStencilLayer,
    TermOperatorSet,
    TransportCoefficients,
    b_hat_from_gradients,
    build_term_operators,
)
from src.models.ltc_cell import inverse_softplus
from src.models.source_contract import N_SOURCE_FIELDS, validate_shared_source

# ---------------------------------------------------------------------------
# Condition-group vocabulary (report/data-export-spec-v3.md section 4.2)
# ---------------------------------------------------------------------------
FILM_GROUPS: tuple[str, ...] = (
    "pellet_mass",
    "pellet_velocity",
    "pellet_angle",
    "pellet_species",
    "background_plasma",
    "boundary_flags",
)
EXCLUDED_GROUPS: tuple[str, ...] = ("excluded_transport", "excluded_numeric")

# ---------------------------------------------------------------------------
# Boundary-code policy (B0 report section 6 / ADR-0018 decision 11)
# ---------------------------------------------------------------------------
#: Codes that are registered and used by the B0 layer without further review.
CONFIRMED_BOUNDARY_CODES: frozenset[int] = frozenset({0, 1, 2, 5, 11, 15})
#: Codes whose JOREK-side meaning still needs confirmation (B0 report section 6);
#: a frame carrying them requires ``acknowledge_pending_codes=True`` and the
#: decision is registered in the returned report.
PENDING_BOUNDARY_CODES: frozenset[int] = frozenset({3, 4, 9, 12, 19, 20, 21})
#: ``use_simple_bnd_types = .true.`` folds these codes
#: (``update_boundary_types.f90:665-676``, B0 report section 6.4).
SIMPLE_BOUNDARY_FOLD: Mapping[int, int] = {11: 1, 15: 5, 19: 9, 12: 4, 20: 9, 21: 9}


def fold_simple_boundary_codes(codes: np.ndarray) -> np.ndarray:
    """Apply ``use_simple_bnd_types`` folding to the per-node boundary codes."""

    folded = np.array(codes, dtype=np.int64, copy=True)
    for source, target in SIMPLE_BOUNDARY_FOLD.items():
        folded[folded == source] = target
    return folded


def read_namelist_bc_flags(namelist_path: str | Path | None) -> dict:
    """Read the boundary flags through the namelist value chain.

    Returns ``{"values": {...}, "provenance": {...}, "source": path|None}``;
    per ADR-0018 decision 11 / roadmap_v2 section 5.10 the values must be
    registered in the context rather than assumed from JOREK defaults.
    """

    flags: dict = {
        "use_simple_bnd_types": False,
        "no_mach1_bc": False,
        "bc_natural_open": None,
        "freeboundary": None,
        "mach_one_bnd_integral": None,
        "grid_to_wall": None,
    }
    provenance = {key: "default" for key in flags}
    if namelist_path is None:
        return {"values": flags, "provenance": provenance, "source": None}

    from src.data.namelist import parse_namelist

    text = Path(namelist_path).read_text(encoding="utf-8", errors="replace")
    parsed = parse_namelist(text)

    def truthy(value) -> bool:
        if isinstance(value, (bool, int, float)):
            return bool(value)
        return str(value).strip().strip(".").lower() in {"true", "t"}

    logical = {
        "use_simple_bnd_types",
        "no_mach1_bc",
        "bc_natural_open",
        "freeboundary",
        "grid_to_wall",
        "mach_one_bnd_integral",
    }
    for key in flags:
        if key in parsed:
            raw = parsed[key]
            flags[key] = truthy(raw) if key in logical else raw
            provenance[key] = "namelist"
    return {"values": flags, "provenance": provenance, "source": str(namelist_path)}


def read_namelist_sheath_parameters(
    namelist_path: str | Path | None,
) -> jb.SheathParameters:
    """Read the sheath/Mach-1 parameters through the namelist value chain.

    Defaults are ``models/preset_parameters.f90:80-94``; ``gamma_sheath_i/e``
    are derived from ``gamma_*_stangeby`` when the run sets them
    (``models/model502/initialise_parameters.f90:241-251``).
    """

    parsed: dict = {}
    if namelist_path is not None:
        from src.data.namelist import parse_namelist

        text = Path(namelist_path).read_text(encoding="utf-8", errors="replace")
        parsed = parse_namelist(text)
    return jb.sheath_parameters_from_namelist(parsed)


def check_boundary_codes(
    codes: np.ndarray,
    *,
    acknowledge_pending_codes: bool = False,
) -> dict:
    """Fail-fast boundary-code audit for the B2 propagator.

    Unknown codes (not in the B0 mapping table) always raise.  Codes pending
    JOREK-side confirmation raise unless the caller explicitly acknowledges
    them; either way the outcome is registered in the returned report.
    """

    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    present = {int(code) for code in np.unique(codes)}
    unknown = sorted(code for code in present if code not in jb.BOUNDARY_CODE_TABLE)
    if unknown:
        raise ValueError(
            "unregistered boundary code(s) "
            f"{unknown}: not in the B0 mapping table (report/phase-b0-stencil-boundary-"
            "2026-10-04.md section 3); fail-fast per ADR-0018 decision 11"
        )
    pending = sorted(present & PENDING_BOUNDARY_CODES)
    if pending and not acknowledge_pending_codes:
        raise ValueError(
            f"boundary code(s) {pending} are registered but pending JOREK-side confirmation "
            "(B0 report section 6); pass acknowledge_pending_codes=True to register and proceed"
        )
    counts = {int(code): int((codes == code).sum()) for code in sorted(present)}
    return {
        "unknown_codes": unknown,
        "pending_codes": pending,
        "acknowledged_pending_codes": bool(pending) and bool(acknowledge_pending_codes),
        "code_counts": counts,
        "provenance": "report/phase-b0-stencil-boundary-2026-10-04.md sections 3/6",
    }


def validate_condition_groups(
    groups: Sequence[str],
    film_eligible: Sequence[bool] | None = None,
) -> list[bool]:
    """Check the FiLM-eligible group vocabulary; unknown/excluded groups raise."""

    groups = [str(group) for group in groups]
    known = set(FILM_GROUPS) | set(EXCLUDED_GROUPS)
    unknown = sorted({group for group in groups if group not in known})
    if unknown:
        raise ValueError(
            f"unknown condition group(s) {unknown}; allowed FiLM groups {list(FILM_GROUPS)} "
            f"and excluded groups {list(EXCLUDED_GROUPS)} "
            "(report/data-export-spec-v3.md section 4.2)"
        )
    if film_eligible is None:
        return [group in FILM_GROUPS for group in groups]
    eligible = [bool(flag) for flag in film_eligible]
    if len(eligible) != len(groups):
        raise ValueError("film_eligible must align with groups")
    for group, flag in zip(groups, eligible):
        if flag and group in EXCLUDED_GROUPS:
            raise ValueError(
                f"condition group {group!r} is excluded from FiLM modulation "
                "(ADR-0018 decision 6; roadmap_v2 section 5.5)"
            )
    return eligible


def scramble_group(
    condition: np.ndarray,
    groups: Sequence[str],
    group: str,
    *,
    shift: int = 1,
) -> np.ndarray:
    """Roll the entries of one condition group (grouped-permutation probe)."""

    condition = np.array(condition, dtype=np.float64, copy=True)
    index = [i for i, name in enumerate(groups) if name == group]
    if len(index) < 2:
        raise ValueError(f"group {group!r} has fewer than two entries to permute")
    values = condition[index]
    condition[index] = np.roll(values, shift)
    return condition


def symlog_condition(condition: torch.Tensor) -> torch.Tensor:
    """Sign-preserving log compression of raw condition values.

    Raw namelist/H5 entries span many decades (``n_spi ~ 1e22`` up to
    ``eta ~ 1e-10``); the FiLM head needs a bounded input.  The transform is
    the same ``symlog`` used for the state data contract
    (``src/data/model502_schema.py``), so it is exact and invertible.
    """

    return torch.sign(condition) * torch.log1p(torch.abs(condition))


def normalize_condition(condition: torch.Tensor, input_range: float = 5.0) -> torch.Tensor:
    """Bounded FiLM input: ``symlog`` compressed into ``(-input_range, input_range)``.

    The raw namelist/H5 entries span ``1e-10 .. 1e22``; after ``symlog`` they
    still span decades, which saturates the FiLM head and makes different
    conditions indistinguishable.  The soft compression
    ``x / (1 + |x| / input_range)`` is monotone, sign preserving and keeps the
    head in its responsive range.
    """

    compressed = symlog_condition(condition)
    return compressed / (1.0 + compressed.abs() / float(input_range))


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------
class ShallowEncoder(nn.Module):
    """Per-node input embedding; identity-initialised on the nine field channels."""

    def __init__(
        self, in_channels: int = NUM_STATE_FIELDS, hidden: int = NUM_STATE_FIELDS
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden = int(hidden)
        self.proj = nn.Linear(self.in_channels, self.hidden, dtype=torch.float64)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            for index in range(min(self.in_channels, self.hidden)):
                self.proj.weight[index, index] = 1.0

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.proj(state)


class ShallowDecoder(nn.Module):
    """Per-node decode back to the nine field channels (identity at init)."""

    def __init__(
        self, hidden: int = NUM_STATE_FIELDS, out_channels: int = NUM_STATE_FIELDS
    ) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.out_channels = int(out_channels)
        self.proj = nn.Linear(self.hidden, self.out_channels, dtype=torch.float64)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            for index in range(min(self.hidden, self.out_channels)):
                self.proj.weight[index, index] = 1.0

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.proj(hidden)


class FiLMHead(nn.Module):
    """Shallow AdaLN/FiLM modulation from the physical-scenario condition subset.

    The modulation is a *global scalar* gain/bias applied to the encoded state
    (``h -> gamma(c) * h + beta(c)``), identity-initialised (``gamma = 1``,
    ``beta = 0``).  Because the deep path (LTC + implicit stencil + decoder) is
    affine in ``h``, a condition change can only alter the overall gain/offset:
    the normalized local response shape is invariant by construction
    (ADR-0018 decision 12).
    """

    def __init__(
        self,
        n_conditions: int,
        groups: Sequence[str],
        *,
        film_eligible: Sequence[bool] | None = None,
        hidden: int = 32,
        per_channel: bool = False,
        n_channels: int = NUM_STATE_FIELDS,
        gamma_span: float = 0.5,
        beta_span: float = 0.5,
        condition_input_range: float = 5.0,
    ) -> None:
        super().__init__()
        groups = [str(group) for group in groups]
        eligible = validate_condition_groups(groups, film_eligible)
        if len(groups) != int(n_conditions):
            raise ValueError(
                f"n_conditions={n_conditions} does not match {len(groups)} group labels"
            )
        self.groups = tuple(groups)
        self.per_channel = bool(per_channel)
        self.n_channels = int(n_channels)
        self.gamma_span = float(gamma_span)
        self.beta_span = float(beta_span)
        self.condition_input_range = float(condition_input_range)
        if self.gamma_span <= 0.0 or self.gamma_span >= 1.0:
            raise ValueError("gamma_span must be in (0, 1) so the gain stays positive and bounded")
        if self.condition_input_range <= 0.0:
            raise ValueError("condition_input_range must be positive")
        self.register_buffer(
            "eligible_index",
            torch.as_tensor([i for i, flag in enumerate(eligible) if flag], dtype=torch.long),
        )
        self.register_buffer(
            "group_index",
            torch.as_tensor(
                [FILM_GROUPS.index(group) if group in FILM_GROUPS else -1 for group in groups],
                dtype=torch.long,
            ),
        )
        n_inputs = int(self.eligible_index.numel())
        if n_inputs == 0:
            raise ValueError("no FiLM-eligible condition entries in the supplied vector")
        out_features = 2 * self.n_channels if self.per_channel else 2
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden, dtype=torch.float64),
            nn.SiLU(),
            nn.Linear(hidden, out_features, dtype=torch.float64),
        )
        # Identity initialisation: gamma = 1, beta = 0 for every condition.
        last = self.net[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()

    def forward(
        self,
        condition: torch.Tensor,
        groups: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = torch.as_tensor(condition, dtype=torch.float64).reshape(-1)
        if condition.numel() != len(self.groups):
            raise ValueError(
                f"condition vector has {condition.numel()} entries, expected {len(self.groups)}"
            )
        if groups is not None:
            validate_condition_groups(groups)
            if [str(group) for group in groups] != list(self.groups):
                raise ValueError("groups do not match the groups this FiLM head was built with")
        x = normalize_condition(condition, self.condition_input_range).index_select(
            0, self.eligible_index
        )
        raw = self.net(x)
        if self.per_channel:
            gamma = 1.0 + self.gamma_span * torch.tanh(raw[: self.n_channels])
            beta = self.beta_span * torch.tanh(raw[self.n_channels :])
        else:
            gamma = 1.0 + self.gamma_span * torch.tanh(raw[0])
            beta = self.beta_span * torch.tanh(raw[1])
        return gamma, beta


class LTCFieldStep(nn.Module):
    """Closed-form linear LTC time integration (exact, unconditionally stable).

    Per-channel rate ``r > 0`` (``softplus``) and a per-channel affine target
    ``target(f) = W f + b`` built from the *step features* ``f`` (the shallow
    encoder output, frozen inside the step) and integrated exactly over ``dt``::

        decay  = exp(-r * dt)
        h_new  = decay * h + (1 - decay) * target(f)

    ``target`` is identity-initialised, so the component starts as an exact
    identity (``dt = 0`` is the identity for any parameters) and its rates
    cannot amplify: ``decay in (0, 1]``.  Because the target is a function of the
    frozen features (not of ``h``), the split ``dt1 + dt2`` is exact at
    machine precision - the same property verified for
    ``src/models/ltc_cell.py`` with input-only coefficients.
    """

    def __init__(
        self,
        n_channels: int = NUM_STATE_FIELDS,
        *,
        dt_reference: float = 0.05,
        init_decay: float = 0.5,
        target_clamp: float | None = 50.0,
    ) -> None:
        super().__init__()
        if not 0.0 < init_decay < 1.0:
            raise ValueError("init_decay must be in (0, 1)")
        if dt_reference <= 0.0:
            raise ValueError("dt_reference must be positive")
        self.n_channels = int(n_channels)
        self.dt_reference = float(dt_reference)
        self.rate_floor = 1e-6
        rate_init = -np.log(init_decay) / dt_reference
        self.rate_raw = nn.Parameter(
            torch.tensor(inverse_softplus(rate_init), dtype=torch.float64)
        )
        self.target = nn.Linear(self.n_channels, self.n_channels, dtype=torch.float64)
        with torch.no_grad():
            self.target.weight.zero_()
            self.target.bias.zero_()
            for index in range(self.n_channels):
                self.target.weight[index, index] = 1.0
        self.target_clamp = target_clamp

    def rate(self) -> torch.Tensor:
        return F.softplus(self.rate_raw) + self.rate_floor

    def decay(self, dt: float) -> torch.Tensor:
        return torch.exp(-self.rate() * float(dt))

    def forward(
        self,
        hidden: torch.Tensor,
        dt: float,
        features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        decay = self.decay(dt)
        target = self.target(hidden if features is None else features)
        if self.target_clamp is not None:
            target = target.clamp(-float(self.target_clamp), float(self.target_clamp))
        return hidden * decay + target * (1.0 - decay)


@dataclass
class LoadedPropagator:
    """Frame + boundary context + propagator built from one ``jorek*.h5`` file."""

    frame: sc.FrameGeometry
    geometry: sc.FemStencilGeometry
    context: jb.BoundaryContext
    propagator: "FieldPropagator"
    namelist_flags: dict
    boundary_report: dict
    sheath: jb.SheathParameters | None = None


class FieldPropagator(nn.Module):
    """The B2 nine-field stack (shallow encoder + FiLM + LTC + implicit stencil).

    source_proj is a learned approximation, not model502 physical consumption.
    The separately verified Gauss consumer in marker_feedback is not yet wired
    into this nodal latent stack; full FEM and coupled nine-field assembly remain
    pending. A free 5->hidden projection cannot supply those dependencies.
    """

    def __init__(
        self,
        geometry: sc.FemStencilGeometry,
        *,
        boundary: jb.BoundaryContext | None = None,
        operators: TermOperatorSet | None = None,
        b_hat_nodes: np.ndarray | None = None,
        n_channels: int = NUM_STATE_FIELDS,
        source_channels: int = N_SOURCE_FIELDS,
        hidden: int | None = None,
        n_layers: int = 2,
        n_iters: int = 12,
        dt: float = 0.05,
        alphas: TransportCoefficients | Mapping[str, float] | None = None,
        channel_terms: Mapping[str, Sequence[str]] = DEFAULT_CHANNEL_TERMS,
        explicit_terms: Sequence[str] = DEFAULT_EXPLICIT_TERMS,
        implicit_terms: Sequence[str] = DEFAULT_IMPLICIT_TERMS,
        learnable: bool = True,
        condition_dim: int | None = None,
        condition_groups: Sequence[str] | None = None,
        condition_film_eligible: Sequence[bool] | None = None,
        film_hidden: int = 32,
        film_per_channel: bool = False,
        metric_weight: str | Mapping[str, str] | None = None,
        solver: str = "direct",
        cg_tol: float = 1e-10,
        max_cg_iters: int = 400,
        boundary_codes: np.ndarray | None = None,
        acknowledge_pending_codes: bool = False,
        gamma: float = 5.0 / 3.0,
        c_angle: float = 0.0174524,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        hidden = int(hidden if hidden is not None else n_channels)
        self.geometry = geometry
        self.boundary = boundary
        self.n_channels = int(n_channels)
        self.source_channels = int(source_channels)
        self.hidden = hidden
        self.n_layers = int(n_layers)
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.c_angle = float(c_angle)
        self.metric_weight = metric_weight
        self.solver = solver
        codes = (
            boundary_codes
            if boundary_codes is not None
            else (boundary.boundary_code if boundary is not None else np.zeros(geometry.n_nodes))
        )
        self.boundary_report = check_boundary_codes(
            codes, acknowledge_pending_codes=acknowledge_pending_codes
        )
        if operators is None:
            operators = build_term_operators(
                geometry, b_hat_nodes=b_hat_nodes, terms=TERM_NAMES, metric_weight=metric_weight
            )
        self.operators = operators
        self.encoder = ShallowEncoder(n_channels, hidden)
        self.decoder = ShallowDecoder(hidden, n_channels)
        self.ltc = LTCFieldStep(hidden, dt_reference=self.dt)
        self.film: FiLMHead | None = None
        if condition_dim is not None or condition_groups is not None:
            if condition_dim is None or condition_groups is None:
                raise ValueError("condition_dim and condition_groups must be supplied together")
            self.film = FiLMHead(
                int(condition_dim),
                condition_groups,
                film_eligible=condition_film_eligible,
                hidden=film_hidden,
                per_channel=film_per_channel,
                n_channels=hidden,
            )
        self.stencil_layers = nn.ModuleList(
            [
                ImplicitStencilLayer(
                    geometry,
                    operators=operators,
                    term_names=self.operators.names,
                    alphas=alphas,
                    channel_terms=channel_terms,
                    n_channels=hidden,
                    dt=self.dt,
                    n_iters=n_iters,
                    boundary=boundary,
                    explicit_terms=explicit_terms,
                    implicit_terms=implicit_terms,
                    learnable=learnable,
                    solver=solver,
                    cg_tol=cg_tol,
                    max_cg_iters=max_cg_iters,
                    gamma=gamma,
                    c_angle=c_angle,
                )
                for _ in range(self.n_layers)
            ]
        )
        # Approximate learned forcing, NOT the physical five-channel consumer
        # (ADR-0020 decision 4). Zero initialization does not align its equations.
        self.source_proj = nn.Linear(self.source_channels, hidden, dtype=torch.float64)
        with torch.no_grad():
            self.source_proj.weight.zero_()
            self.source_proj.bias.zero_()

    # -- helpers -----------------------------------------------------------
    def set_dt(self, dt: float) -> None:
        """Set the propagation interval in JOREK time units on every layer."""

        self.dt = float(dt)
        for layer in self.stencil_layers:
            layer.dt = float(dt)

    def film_parameters(
        self,
        condition: torch.Tensor,
        condition_groups: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.film is None:
            raise ValueError("this propagator was built without a condition head")
        return self.film(condition, condition_groups)

    def encode(
        self,
        state: torch.Tensor,
        *,
        condition: torch.Tensor | None = None,
        condition_groups: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """Shallow encoding path; the *only* place the condition is consumed."""

        hidden = self.encoder(state)
        if self.film is not None and condition is not None:
            gamma, beta = self.film(condition, condition_groups)
            hidden = gamma * hidden + beta
        return hidden

    def run_stencil_stack(
        self,
        hidden: torch.Tensor,
        dt: float | None = None,
        *,
        forcing: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deep local path: implicit stencil layers only (no condition input)."""

        dt = self.dt if dt is None else float(dt)
        for index, layer in enumerate(self.stencil_layers):
            hidden = layer(hidden, dt=dt, forcing=forcing if index == 0 else None)
        return hidden

    def forward(
        self,
        state: torch.Tensor,
        *,
        condition: torch.Tensor | None = None,
        condition_groups: Sequence[str] | None = None,
        source: torch.Tensor | None = None,
        dt: float | None = None,
        n_layers: int | None = None,
    ) -> torch.Tensor:
        state = torch.as_tensor(state)
        if state.ndim != 2 or state.shape[1] != self.n_channels:
            raise ValueError(
                f"state must be [n_nodes, {self.n_channels}], got {tuple(state.shape)}"
            )
        if state.shape[0] != self.geometry.n_nodes:
            raise ValueError(
                f"geometry has {self.geometry.n_nodes} nodes, got {state.shape[0]}"
            )
        dt = self.dt if dt is None else float(dt)
        hidden = self.encode(state, condition=condition, condition_groups=condition_groups)
        hidden = self.ltc(hidden, dt, features=hidden)
        forcing = None
        if source is not None:
            validate_shared_source(
                source, what="S (shared five-channel interface)", n_nodes=self.geometry.n_nodes
            )
            forcing = self.source_proj(source.to(dtype=hidden.dtype))
        layers = self.stencil_layers if n_layers is None else self.stencil_layers[: int(n_layers)]
        propagated = hidden
        for index, layer in enumerate(layers):
            propagated = layer(propagated, dt=dt, forcing=forcing if index == 0 else None)
        # Residual (flux) form: the deep stack contributes the local increment
        # ``S(h) - h``.  This keeps the map stable for a bounded FiLM gain
        # (``u -> (1 - gamma) u + gamma S(u)``, eigenvalues in ``(1 - gamma, 1]``)
        # and makes the condition act exactly as an overall dissipation /
        # driving strength (ADR-0018 decision 12).
        out = state + self.decoder(propagated - hidden)
        return self.reset_boundary(out, state, dt=dt)

    def reset_boundary(
        self, output: torch.Tensor, reference: torch.Tensor, *, dt: float | None = None
    ) -> torch.Tensor:
        """Final BC reset (each layer already resets; the decoder may mix rows)."""

        if self.boundary is None:
            return output
        dt = self.dt if dt is None else float(dt)
        output = torch.where(
            torch.as_tensor(self.boundary.bc_node_mask, device=output.device).unsqueeze(1),
            reference,
            output,
        )
        return jb.apply_boundary_conditions(
            output,
            self.boundary,
            reference=reference,
            dt=dt,
            gamma=self.gamma,
            c_angle=self.c_angle,
            mass=self.stencil_layers[0].node_mass,
        )

    # -- reporting ---------------------------------------------------------
    def parameter_report(self) -> dict:
        """Parameter count / condition-depth report (B2 deliverable)."""

        by_module = {}
        for name, module in self.named_children():
            if name == "stencil_layers":
                total = sum(p.numel() for p in module.parameters())
                by_module[name] = {
                    "n_parameters": int(total),
                    "n_layers": len(module),
                    "per_layer": int(sum(p.numel() for p in module[0].parameters())),
                }
            else:
                by_module[name] = int(sum(p.numel() for p in module.parameters()))
        total = int(sum(p.numel() for p in self.parameters()))
        return {
            "total_parameters": total,
            "by_module": by_module,
            "n_channels": self.n_channels,
            "hidden": self.hidden,
            "n_layers": self.n_layers,
            "dt": self.dt,
            "solver": self.solver,
            "condition_dim": None if self.film is None else len(self.film.groups),
            "film_eligible_dim": (
                None if self.film is None else int(self.film.eligible_index.numel())
            ),
            "film_scope": "shallow_encoder_only",
            "film_per_channel": None if self.film is None else bool(self.film.per_channel),
            "film_identity_init": bool(
                self.film is not None
                and torch.allclose(
                    self.film.net[-1].weight.detach(), torch.zeros_like(self.film.net[-1].weight)
                )
            ),
            "boundary_report": dict(self.boundary_report),
        }

    def condition_depth_report(self) -> dict:
        """Structural check: the deep stencil layers consume no condition.

        The condition is consumed exclusively by :class:`FiLMHead` inside
        :meth:`encode`; the stencil layers' forward signatures take only the
        hidden state, a timestep and an (optional) explicit source forcing.
        """

        film_ids = {id(p) for p in (self.film.parameters() if self.film is not None else [])}
        deep_ids = {
            id(parameter)
            for layer in self.stencil_layers
            for parameter in layer.parameters()
        }
        return {
            "film_parameters": len(film_ids),
            "deep_stencil_parameters": len(deep_ids),
            "shared_parameters": sorted(film_ids & deep_ids),
            "shared_with_condition_head": False,
            "stencil_layer_inputs": ["hidden", "dt", "forcing"],
        }


def build_propagator_from_frame(
    h5_path: str | Path,
    *,
    namelist_path: str | Path | None = None,
    acknowledge_pending_codes: bool = False,
    **kwargs,
) -> LoadedPropagator:
    """Build the B2 propagator on one recorded ``jorek*.h5`` frame.

    Frame-level smoke only: recorded frames are a cadence of ~10 fluid steps,
    so this is *not* a baseline or gate for Phase C/D (ADR-0018 decision 4/10).
    Boundary flags are read from the namelist value chain; pending codes are
    registered in ``boundary_report``.
    """

    import h5py

    flags = read_namelist_bc_flags(namelist_path)
    sheath = read_namelist_sheath_parameters(namelist_path)
    if flags["values"]["mach_one_bnd_integral"] is not None:
        # The namelist flag wins over the preset default inside the sheath
        # parameters (both are registered in the context notes).
        from dataclasses import replace

        sheath = replace(
            sheath, mach_one_bnd_integral=bool(flags["values"]["mach_one_bnd_integral"])
        )
    frame = sc.load_frame_geometry(h5_path, with_values=True)
    codes = frame.boundary_code
    if flags["values"]["use_simple_bnd_types"]:
        codes = fold_simple_boundary_codes(codes)
    psi_gradient = sc.nodal_gradient(frame, field=0)
    with h5py.File(h5_path, "r") as handle:
        f0 = float(np.asarray(handle["F0"][()]).reshape(-1)[0]) if "F0" in handle else 1.0
    context = jb.build_boundary_context(
        codes,
        frame.pos,
        frame.vertex,
        freeboundary=frame.freeboundary,
        axis_node=frame.axis_node,
        psi_gradient=psi_gradient,
        no_mach1_bc=bool(flags["values"]["no_mach1_bc"]),
        sheath=sheath,
        grid_to_wall=bool(flags["values"]["grid_to_wall"]),
        f0=f0,
    )
    geometry = frame.stencil()
    b_hat_nodes = b_hat_from_gradients(psi_gradient, frame.pos[:, 0], f0)
    operators = build_term_operators(
        geometry,
        b_hat_nodes=b_hat_nodes,
        terms=TERM_NAMES,
        metric_weight=kwargs.pop("metric_weight", None),
    )
    propagator = FieldPropagator(
        geometry,
        boundary=context,
        operators=operators,
        b_hat_nodes=b_hat_nodes,
        boundary_codes=codes,
        acknowledge_pending_codes=acknowledge_pending_codes,
        **kwargs,
    )
    return LoadedPropagator(
        frame=frame,
        geometry=geometry,
        context=context,
        propagator=propagator,
        namelist_flags=flags,
        sheath=sheath,
        boundary_report=propagator.boundary_report,
    )


__all__ = [
    "FILM_GROUPS",
    "EXCLUDED_GROUPS",
    "CONFIRMED_BOUNDARY_CODES",
    "PENDING_BOUNDARY_CODES",
    "SIMPLE_BOUNDARY_FOLD",
    "fold_simple_boundary_codes",
    "read_namelist_bc_flags",
    "read_namelist_sheath_parameters",
    "check_boundary_codes",
    "validate_condition_groups",
    "scramble_group",
    "ShallowEncoder",
    "ShallowDecoder",
    "FiLMHead",
    "LTCFieldStep",
    "FieldPropagator",
    "LoadedPropagator",
    "build_propagator_from_frame",
]
