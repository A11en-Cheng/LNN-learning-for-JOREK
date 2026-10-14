"""Self-generated analytic shard source and persistent marker feedback.

ADR-0020 separates the two objects. The analytic arm retains the existing
substepped Parks/NGS fragment predictor. The marker arm accepts the shard
state/ablation already produced by the preceding fluid event; it holds shards
fixed while particles sample the two AVAILABLE field endpoints, preserving
cloud, full charge probabilities, velocity history and previous feedback.
Shard updates belong after the next fluid solve, not inside particle substeps.

MarkerChain uses the small reference kernel through an explicitly approximate
2D/bilinear/lumped adapter. It has no trainable physical parameters and detaches
reference physics; learned residual heads remain differentiable and bounded.
GRU memory is learning memory only. No future records are read. For the exact
event ordering driver with replaceable physics callbacks see marker_state's
ReferenceRollout/reference_particle_then_fluid. This is component support,
not a validated full fluid-source closed loop or exact nine-field coupling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from src.models.fem_sampling import FemMeshSampler
from src.models.marker_state import MarkerRuntime
from src.models.fragment_encoder import FragmentSetEncoder
from src.models.source_contract import (
    N_SOURCE_FIELDS,
    SOURCE_LAYOUT,
    SOURCE_OBJECT_ANALYTIC_SHARD,
    validate_shared_source,
)
from src.physics.adas_closure import ADAS_DIR, Z_MAX, _bilinear, parse_adf11
from src.physics.spi_model import (
    EL_CHG,
    MU_ZERO,
    NonlocalAblationTable,
    erode_radius,
    parks_ablation_neon,
    particle_density_si,
    peloton_rocket_acceleration,
    temperature_ev,
)

__all__ = [
    "MEMORY_CHANNELS",
    "INSTANT_CHANNELS",
    "MARKER_FEATURE_DIM",
    "SHARD_LAYOUTS",
    "SOURCE_CHANNEL_NAMES",
    "AugmentedState",
    "MarkerChain",
    "MarkerChainConfig",
    "NeonTables",
    "SourceModule",
    "advance_augmented_source",
    "default_neon_tables",
]

SOURCE_CHANNEL_NAMES: tuple[str, ...] = ("dE_ion", "rad_sink", "Z_eff", "Z_imp", "N_imp")
INSTANT_CHANNELS: tuple[int, ...] = (2, 3, 4)  # Z_eff, Z_imp, N_imp
MEMORY_CHANNELS: tuple[int, ...] = (0, 1)  # dE_ion, rad_sink

#: Backward-compatible re-export of the shared source-object kinds (frozen in
#: :mod:`src.models.source_contract`, ADR-0020 decision 6).
SOURCE_OBJECT_MARKER_FEEDBACK = "marker_feedback"

#: JOREK ``GAMMA`` (model502 adiabatic index) used by the marker export path:
#: ``E_norm = 1 / (MU_ZERO * (GAMMA - 1))`` (particles/examples/marker_spi_source.f90:261).
GAMMA = 5.0 / 3.0
E_NORM = 1.0 / (MU_ZERO * (GAMMA - 1.0))

#: Two registered shard-array orders exist in the repository and they differ;
#: mixing them silently moves ``radius``/``abl``/``velocity`` onto the wrong
#: axes (measured: a 1e27 ablation blow-up when the A' loader order is read
#: with the v1 constants).  The layout is therefore explicit.
SHARD_LAYOUTS: dict[str, tuple[str, ...]] = {
    # ``src/data/step_dataset.py`` / A' export order (native loader stacks this).
    "a_prime": (
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
    ),
    # ``src/data/fragment_state.py`` / v1 order (the index constants of
    # ``src/physics/spi_model.py`` refer to this one).
    "fragment_state": (
        "R",
        "Z",
        "phi",
        "phi_init",
        "Vel_R",
        "Vel_Z",
        "Vel_RxZ",
        "radius",
        "abl",
        "species",
        "vol",
        "psi",
        "grad_psi",
        "vol_drift",
        "psi_drift",
        "grad_psi_drift",
    ),
}

_SHARD_ALIASES: dict[str, str] = {
    "R": "spi_R_arr",
    "Z": "spi_Z_arr",
    "phi": "spi_phi_arr",
    "phi_init": "spi_phi_init_arr",
    "Vel_R": "spi_Vel_R_arr",
    "Vel_Z": "spi_Vel_Z_arr",
    "Vel_RxZ": "spi_Vel_RxZ_arr",
    "radius": "spi_radius_arr",
    "abl": "spi_abl_arr",
    "species": "spi_species_arr",
    "psi": "spi_psi_arr",
}


def _shard_indices(layout: str) -> dict[str, int]:
    if layout not in SHARD_LAYOUTS:
        raise ValueError(
            f"unknown shard layout {layout!r}; use one of {sorted(SHARD_LAYOUTS)}"
        )
    names = SHARD_LAYOUTS[layout]
    canonical = {_SHARD_ALIASES.get(name, name): index for index, name in enumerate(names)}
    missing = [key for key in _SHARD_ALIASES.values() if key not in canonical]
    if missing:
        raise ValueError(f"shard layout {layout!r} misses {missing}")
    return {short: canonical[key] for short, key in _SHARD_ALIASES.items()}


# --------------------------------------------------------------------------- #
# ADAS neon tables (charge states, ionisation potentials, ADF01 cooling)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class NeonTables:
    """Static ADAS-derived neon data used by the analytic backbone.

    All arrays are plain numpy: they are physics constants, not trainable
    parameters, so gradients never flow through them (the closure is evaluated
    with ``detach()``).
    """

    scd: tuple[np.ndarray, np.ndarray, np.ndarray]
    acd: tuple[np.ndarray, np.ndarray, np.ndarray]
    plt: tuple[np.ndarray, np.ndarray, np.ndarray]
    ionization_potentials_ev: np.ndarray  # [Z_MAX] cumulative barrier per state
    source: str = "adas"

    @property
    def z_max(self) -> int:
        return int(self.ionization_potentials_ev.size)


def _parse_ionization_potentials(path: Path) -> np.ndarray:
    """Read ``ion_ne.dat`` (one ionisation potential in eV per charge state)."""
    values: list[float] = []
    for line in Path(path).read_text().splitlines():
        tokens = line.split()
        if len(tokens) < 2:
            continue
        try:
            float(tokens[0])
            values.append(float(tokens[1]))
        except ValueError:
            continue
    if not values:
        raise ValueError(f"{path}: no ionisation potentials found")
    return np.asarray(values, dtype=np.float64)


def _surrogate_neon_tables(z_max: int = Z_MAX) -> NeonTables:
    """Documented surrogate used when the ADAS files are unavailable.

    The ionisation potentials are the physical neon values (they are short and
    public); the *rates* fall back to a monotone power-law charge-state ramp and
    a zero radiation coefficient.  Registered in the report as a degraded mode:
    never used silently, the returned ``source`` field is ``"surrogate"``.
    """
    potentials = np.asarray(
        [21.5645, 40.963, 63.423, 97.190, 126.247, 157.934, 207.271, 239.097, 1195.808, 1362.199],
        dtype=np.float64,
    )[:z_max]
    empty = (
        np.linspace(8.0, 15.0, 24),
        np.linspace(-0.7, 4.2, 30),
        np.zeros((z_max, 30, 24), dtype=np.float64),
    )
    return NeonTables(
        scd=empty,
        acd=empty,
        plt=empty,
        ionization_potentials_ev=potentials,
        source="surrogate",
    )


@lru_cache(maxsize=4)
def load_neon_tables(adas_dir: str | Path = ADAS_DIR) -> NeonTables:
    """Load + cache the ADAS neon tables; raise when the files are missing."""
    adas_dir = Path(adas_dir)
    scd = parse_adf11(adas_dir / "scd96_ne.dat")
    acd = parse_adf11(adas_dir / "acd96_ne.dat")
    plt = parse_adf11(adas_dir / "plt96_ne.dat")
    potentials = _parse_ionization_potentials(adas_dir / "ion_ne.dat")
    return NeonTables(scd=scd, acd=acd, plt=plt, ionization_potentials_ev=potentials)


def default_neon_tables(
    adas_dir: str | Path = ADAS_DIR, *, allow_surrogate: bool = False
) -> NeonTables:
    """ADAS tables when available; the documented surrogate otherwise."""
    try:
        return load_neon_tables(adas_dir)
    except Exception:  # pragma: no cover - environment dependent
        if not allow_surrogate:
            raise
        return _surrogate_neon_tables()


def _neon_closure(
    tables: NeonTables,
    te_ev: torch.Tensor,
    ne_si: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coronal charge-state closure per shard.

    Returns ``(zbar, z2bar, e_ion_j_per_atom, rad_w_per_atom)`` where the last
    two are the per-ablated-atom ionisation cost [J] and radiated power [W].
    ``te_ev``/``ne_si`` are ``[B, S]`` tensors; the closure runs in numpy on
    detached values (deterministic tables).
    """
    te = te_ev.detach().cpu().numpy().reshape(-1)
    ne = ne_si.detach().cpu().numpy().reshape(-1)
    n = te.shape[0]
    if tables.source == "surrogate":
        # Documented degraded mode: coronal ramp + zero radiation.
        zbar = np.clip(3.0 * np.log10(np.clip(te, 1.0, None) / 10.0), 0.0, Z_MAX)
        z2bar = zbar**2
        cumulative = np.concatenate([[0.0], np.cumsum(tables.ionization_potentials_ev)])
        idx = np.clip(zbar, 0, Z_MAX).astype(np.int64)
        e_ion = cumulative[idx]
        rad = np.zeros_like(zbar)

        def _to_surrogate(tensor: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(tensor, dtype=te_ev.dtype, device=te_ev.device).reshape(
                te_ev.shape
            )

        return _to_surrogate(zbar), _to_surrogate(z2bar), _to_surrogate(e_ion * EL_CHG), _to_surrogate(rad)
    ne_axis, te_axis, scd_coeff = tables.scd
    _, _, acd_coeff = tables.acd
    ne_axis_plt, te_axis_plt, plt_coeff = tables.plt
    # ADAS axes: log10(ne [cm^-3]) and log10(Te [eV]).
    ne_cm3 = np.clip(ne * 1e-6, 1e-30, None)
    ne_log10 = np.log10(ne_cm3)
    te_log10 = np.log10(np.clip(te, 1e-6, None))
    scd = np.zeros((Z_MAX, n), dtype=np.float64)
    acd = np.zeros((Z_MAX, n), dtype=np.float64)
    for z in range(Z_MAX):
        scd[z] = 10 ** _bilinear(scd_coeff[z], ne_axis, te_axis, ne_log10, te_log10)
        acd[z] = 10 ** _bilinear(acd_coeff[z], ne_axis, te_axis, ne_log10, te_log10)
    # Coronal balance (forward sweep): S_{z-1} f_{z-1} = R_z f_z.
    fractions = np.zeros((Z_MAX + 1, n), dtype=np.float64)
    fractions[0] = 1.0
    for z in range(1, Z_MAX + 1):
        ratio = np.clip(scd[z - 1] / (acd[z - 1] + 1e-300), 0.0, 1e30)
        fractions[z] = fractions[z - 1] * ratio
    fractions /= np.maximum(fractions.sum(axis=0, keepdims=True), 1e-300)
    z_grid = np.arange(Z_MAX + 1, dtype=np.float64)[:, None]
    zbar = (z_grid * fractions).sum(axis=0)
    z2bar = (z_grid**2 * fractions).sum(axis=0)
    cumulative = np.concatenate([[0.0], np.cumsum(tables.ionization_potentials_ev)])
    e_ion_ev = (fractions * cumulative[:, None]).sum(axis=0)
    # ADF01 PLT: total radiated power coefficient L_z [W cm^3] per ion per
    # electron; radiated power per impurity ion = n_e[cm^-3] * L_z.
    lz = np.zeros(n, dtype=np.float64)
    for state in range(Z_MAX + 1):
        # The ADF01 blocks are the *ionisation stages* 0..Z; the PLT tables in
        # this data set carry Z_MAX blocks (stage 0..Z-1) -- the fully stripped
        # state has no bound electrons and no line radiation.
        if state < Z_MAX:
            coefficient = _bilinear(plt_coeff[state], ne_axis_plt, te_axis_plt, ne_log10, te_log10)
            lz = lz + fractions[state] * (10.0**coefficient)
    rad_w = ne_cm3 * lz
    def _to(tensor: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(tensor, dtype=te_ev.dtype, device=te_ev.device).reshape(te_ev.shape)

    return _to(zbar), _to(z2bar), _to(e_ion_ev * EL_CHG), _to(rad_w)


# --------------------------------------------------------------------------- #
# Reduced-order marker chain (ADR-0020 decision 2 / roadmap_v2 Phase B1)
# --------------------------------------------------------------------------- #
#
# Full particle replay is infeasible on this data set (``part*.h5`` = 1e6 slots,
# 172 MB/frame, ~840 frames/run; see ADR-0020 "降阶输运").  The chain below is
# the registered reduced-order surrogate:
#
#   shard ablation/birth (deterministic) -> ADAS charge state/ionisation energy/
#   radiation (deterministic) -> field-aligned advection + perpendicular
#   diffusion of the marker cloud (deterministic main term) -> bilinear FEM
#   deposition + gather-time average (same normalisation as projections*.h5).
#
# The ``part*.h5`` snapshots (every 10 fluid steps, with ``i_life``/``P_imp``)
# remain the calibration source for the transport closure; the learned bounded
# residual (split heads in :class:`SourceModule`) learns only the non-local /
# closure / transport correction on top of this deterministic chain.

#: Registered reduced-order defaults.  ``D_perp`` is the B2 stability-table
#: value (``d_perp(1) = 5e-6 m^2 / JOREK time``); the displacement cap keeps the
#: explicit marker update bounded at the physical fluid step.
DEFAULT_MARKERS_PER_SHARD = 4
DEFAULT_MARKER_DIFFUSION = 5.0e-6
DEFAULT_MARKER_DISPLACEMENT_CAP = 0.05

#: Number of deterministic marker-cloud summary features consumed by the split
#: heads (always present so the head shapes do not depend on the runtime switch).
MARKER_FEATURE_DIM = 5


@dataclass(frozen=True)
class MarkerChainConfig:
    """Registered configuration of the reduced-order marker chain."""

    markers_per_shard: int = DEFAULT_MARKERS_PER_SHARD
    diffusion_perp: float = DEFAULT_MARKER_DIFFUSION
    displacement_cap: float = DEFAULT_MARKER_DISPLACEMENT_CAP
    diff_diffusive_flux: bool = False


def _adas_rates_np(
    tables: NeonTables, te_ev: torch.Tensor, ne_si: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ADAS SCD/ACD/Lz tables on flattened ``(te, ne)`` query points.

    Returns ``(scd [Z, n], acd [Z, n], lz [Z+1, n], ne_cm3 [n])``.  ``scd[z]``
    is the ``z -> z+1`` ionisation coefficient, ``acd[z]`` the ``z+1 -> z``
    recombination coefficient (both ``cm^3/s`` on the ADAS axes), and ``lz``
    the ADF01 line-radiation coefficient ``[W cm^3]`` per ion/electron.
    """
    te = te_ev.detach().cpu().numpy().reshape(-1).astype(np.float64)
    ne = ne_si.detach().cpu().numpy().reshape(-1).astype(np.float64)
    ne_cm3 = np.clip(ne * 1e-6, 1e-30, None)
    ne_log10 = np.log10(ne_cm3)
    te_log10 = np.log10(np.clip(te, 1e-6, None))
    ne_axis, te_axis, scd_coeff = tables.scd
    _, _, acd_coeff = tables.acd
    scd = np.stack(
        [10.0 ** _bilinear(scd_coeff[z], ne_axis, te_axis, ne_log10, te_log10) for z in range(Z_MAX)]
    )
    acd = np.stack(
        [10.0 ** _bilinear(acd_coeff[z], ne_axis, te_axis, ne_log10, te_log10) for z in range(Z_MAX)]
    )
    ne_axis_plt, te_axis_plt, plt_coeff = tables.plt
    lz = np.zeros((Z_MAX + 1, te.size), dtype=np.float64)
    for state in range(Z_MAX):
        lz[state] = 10.0 ** _bilinear(
            plt_coeff[state], ne_axis_plt, te_axis_plt, ne_log10, te_log10
        )
    return scd, acd, lz, ne_cm3


def _coronal_fractions_np(scd: np.ndarray, acd: np.ndarray) -> np.ndarray:
    """Stationary ADAS coronal balance ``[n, Z+1]`` (same sweep as JOREK)."""

    n = scd.shape[1]
    fractions = np.zeros((Z_MAX + 1, n), dtype=np.float64)
    fractions[0] = 1.0
    for z in range(1, Z_MAX + 1):
        ratio = np.clip(scd[z - 1] / (acd[z - 1] + 1e-300), 0.0, 1e30)
        fractions[z] = fractions[z - 1] * ratio
    fractions /= np.maximum(fractions.sum(axis=0, keepdims=True), 1e-300)
    return fractions.T


def _golden_spiral_offsets(
    n_markers: int, *, dtype: torch.dtype = torch.float64, device: torch.device | None = None
) -> torch.Tensor:
    """Deterministic unit-disk marker offsets (golden-angle spiral, no RNG)."""

    index = torch.arange(n_markers, dtype=dtype, device=device) + 0.5
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    angle = index * golden_angle
    radius = torch.sqrt(index / n_markers)
    return torch.stack([radius * torch.cos(angle), radius * torch.sin(angle)], dim=-1)


class MarkerChain(nn.Module):
    """Deterministic reduced-order marker chain over one source window.

    Consumes the per-substep trace collected by :class:`SourceModule` (shard
    state, local ablation, sub-step interpolated fields) and returns the
    five-channel marker feedback in the recorded ``projections*.h5``
    normalisation, plus the pooled marker features used by the memory head.
    The chain itself has **no trainable parameters**; the learnable transport /
    closure / non-local residual is the bounded correction head of
    :class:`SourceModule` (ADR-0020 decision 2).
    """

    def __init__(
        self,
        sampler: FemMeshSampler,
        *,
        tables: NeonTables,
        config: MarkerChainConfig | None = None,
        shard_layout: str = "a_prime",
    ) -> None:
        super().__init__()
        self.sampler = sampler
        self.tables = tables
        self.config = config or MarkerChainConfig()
        if self.config.markers_per_shard < 1:
            raise ValueError("markers_per_shard must be >= 1")
        self.shard_index = _shard_indices(shard_layout)
        self.register_buffer(
            "birth_offsets",
            _golden_spiral_offsets(self.config.markers_per_shard),
        )

    # -- geometry / field helpers -----------------------------------------

    def _b_hat_poloidal(self, psi_gradient: torch.Tensor) -> torch.Tensor:
        """Poloidal projection of ``b_hat`` from ``(dpsi/dR, dpsi/dZ)``.

        ``B_pol = (1/R)(dpsi/dZ, -dpsi/dR)`` (the same convention as the B2
        operators); the toroidal ``F0/R`` component is dropped in the 2D
        reduced order (registered in ADR-0020 §3).
        """

        dpsi_dR = psi_gradient[..., 0]
        dpsi_dZ = psi_gradient[..., 1]
        b_r = dpsi_dZ
        b_z = -dpsi_dR
        norm = torch.sqrt(b_r**2 + b_z**2).clamp_min(1e-30)
        return torch.stack([b_r / norm, b_z / norm], dim=-1)

    # -- forward -----------------------------------------------------------

    def forward(self, trace, *, dt_source_jorek, t_norm, central_density,
                marker_state, u_start, u_end, central_mass=1.):
        from src.models.reduced_marker import run_reduced_chain
        return run_reduced_chain(self, trace, dt_source_jorek=dt_source_jorek,
            t_norm=t_norm, central_density=central_density,
            marker_state=marker_state, u_start=u_start, u_end=u_end, central_mass=central_mass)


# --------------------------------------------------------------------------- #
# Module
# --------------------------------------------------------------------------- #


class SourceModule(nn.Module):
    """Self-generated five-channel source over one sampling window.

    Parameters
    ----------
    sampler:
        Element search + shape functions of the run mesh (deterministic).
    n_substeps:
        Sub-steps used for the window integrals and the fragment evolution.
        The field path is linear in the window: ``u(lambda) = (1-lambda) u_{k-1}
        + lambda u_k``.
    field_path:
        ``"linear"`` (marker sampling window: the ablated markers advance in
        the interpolated ``u_{k-1} -> u_k`` field), ``"start"`` (JOREK's
        ``update_spi`` evaluates with the window-start frame ``u_{k-1}``) or
        ``"end"`` (window-end frame ``u_k``).  Registered alternatives; the
        default is ``"linear"``.
    use_memory / use_correction:
        Architecture switches.  The delivered default is analytic-only
        (``use_correction=False``); the bounded correction and the memory head
        are the Phase B1 ablation arms.
    """

    def __init__(
        self,
        sampler: FemMeshSampler,
        *,
        n_substeps: int = 4,
        field_path: str = "linear",
        shard_layout: str = "a_prime",
        use_memory: bool = True,
        use_correction: bool = False,
        use_marker_transport: bool = False,
        marker_config: MarkerChainConfig | None = None,
        correction_bound: float = 2.0,
        memory_dim: int = 16,
        hidden: int = 64,
        embed_dim: int = 32,
        neon_tables: NeonTables | None = None,
        allow_surrogate: bool = True,
        state_mean: torch.Tensor | None = None,
        state_std: torch.Tensor | None = None,
        shard_mean: torch.Tensor | None = None,
        shard_std: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if field_path not in {"linear", "start", "end"}:
            raise ValueError(f"unknown field_path {field_path!r}")
        if n_substeps < 1:
            raise ValueError("n_substeps must be >= 1")
        if correction_bound <= 1.0:
            raise ValueError("correction_bound must be > 1 (log-space clamp)")
        self.sampler = sampler
        self.n_substeps = int(n_substeps)
        self.field_path = field_path
        self.shard_layout = shard_layout
        self.shard_index = _shard_indices(shard_layout)
        self.use_memory = bool(use_memory)
        self.use_correction = bool(use_correction)
        self.use_marker_transport = bool(use_marker_transport)
        self.marker_config = marker_config or MarkerChainConfig()
        self.memory_dim = int(memory_dim)
        self.register_buffer("correction_bound_log", torch.tensor(math.log(correction_bound)))
        self.tables = neon_tables if neon_tables is not None else default_neon_tables(
            allow_surrogate=allow_surrogate
        )
        self.table_source = self.tables.source

        self.encoder = FragmentSetEncoder(16, hidden, embed_dim)
        # pooled features: shard embedding + local fields (4) + source summary (5) + inside (1)
        # + deterministic marker-cloud summary (MARKER_FEATURE_DIM; zeros when
        # the reduced-order marker transport is disabled).
        pooled_dim = embed_dim + 4 + 5 + 1 + MARKER_FEATURE_DIM
        self.instant_head = nn.Sequential(
            nn.Linear(pooled_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(INSTANT_CHANNELS)),
        )
        self.memory_cell = nn.GRUCell(pooled_dim + 5, memory_dim)
        self.memory_head = nn.Sequential(
            nn.Linear(memory_dim + pooled_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(MEMORY_CHANNELS)),
        )
        for head in (self.instant_head, self.memory_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        self.register_buffer("state_mean", torch.zeros(9) if state_mean is None else torch.as_tensor(state_mean, dtype=torch.float32).view(-1))
        self.register_buffer("state_std", torch.ones(9) if state_std is None else torch.as_tensor(state_std, dtype=torch.float32).view(-1))
        self.register_buffer("shard_mean", torch.zeros(16) if shard_mean is None else torch.as_tensor(shard_mean, dtype=torch.float32).view(-1))
        self.register_buffer("shard_std", torch.ones(16) if shard_std is None else torch.as_tensor(shard_std, dtype=torch.float32).view(-1))
        #: Registered (train-split) proxy calibration: analytic -> projection
        #: channel scale.  ``1`` = no calibration (analytic-only arm).
        self.register_buffer("channel_scale", torch.ones(len(SOURCE_CHANNEL_NAMES)))
        #: per-channel analytic magnitude used to normalise the head features
        #: (set from the training split by the trainer; 1 = unnormalised).
        self.register_buffer("source_norm", torch.ones(len(SOURCE_CHANNEL_NAMES)))
        #: Marker chain (deterministic, no parameters).  Created eagerly so the
        #: birth offsets are registered buffers; only run when the caller
        #: enables ``use_marker_transport``.
        self.marker_chain = MarkerChain(
            sampler,
            tables=self.tables,
            config=self.marker_config,
            shard_layout=shard_layout,
        )

    # ------------------------------------------------------------- utilities

    @torch.no_grad()
    def set_channel_scale(self, scale: torch.Tensor) -> None:
        self.channel_scale.copy_(
            torch.as_tensor(scale, dtype=self.channel_scale.dtype).view(-1)
        )

    @torch.no_grad()
    def set_source_norm(self, norm: torch.Tensor) -> None:
        self.source_norm.copy_(
            torch.as_tensor(norm, dtype=self.source_norm.dtype).view(-1).clamp_min(1e-30)
        )

    @torch.no_grad()
    def set_stats(
        self,
        *,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
        shard_mean: torch.Tensor,
        shard_std: torch.Tensor,
    ) -> None:
        self.state_mean.copy_(torch.as_tensor(state_mean, dtype=self.state_mean.dtype).view(-1))
        self.state_std.copy_(
            torch.as_tensor(state_std, dtype=self.state_std.dtype).view(-1).clamp_min(1e-12)
        )
        self.shard_mean.copy_(torch.as_tensor(shard_mean, dtype=self.shard_mean.dtype).view(-1))
        self.shard_std.copy_(
            torch.as_tensor(shard_std, dtype=self.shard_std.dtype).view(-1).clamp_min(1e-12)
        )

    # ---------------------------------------------------------------- forward

    def apply_heads(
        self,
        analytic: torch.Tensor,
        features: torch.Tensor,
        *,
        memory: torch.Tensor | None = None,
        prev_source: torch.Tensor | None = None,
        use_correction: bool | None = None,
        use_memory: bool | None = None,
        apply_calibration: bool = True,
    ) -> dict:
        """Split heads + bounded correction from cached features.

        ``analytic``: ``[B, N, 5]`` *unscaled* analytic source; the registered
        proxy calibration (``channel_scale``) is applied here; ``features``:
        ``[B, F]`` pooled per-sample features (shard embedding + local field
        means + analytic summary + inside fraction).  Kept separate from
        :meth:`forward` so single-step training can fit the heads on cached
        features without re-running the (deterministic) backbone.
        """
        batch = analytic.shape[0]
        dtype = analytic.dtype
        device = analytic.device
        correction_enabled = self.use_correction if use_correction is None else bool(use_correction)
        memory_enabled = self.use_memory if use_memory is None else bool(use_memory)
        head_dtype = self.instant_head[0].weight.dtype
        features = features.to(head_dtype)
        if apply_calibration:
            scale = self.channel_scale.to(dtype=dtype, device=device).view(1, 1, 5)
            analytic = analytic * scale
        else:
            scale = torch.ones(5, dtype=dtype, device=device)
        if memory is None:
            memory = torch.zeros(batch, self.memory_dim, dtype=head_dtype, device=device)
        if prev_source is None:
            prev_source = torch.zeros(batch, 5, dtype=head_dtype, device=device)
        if memory_enabled:
            memory_out = self.memory_cell(
                torch.cat([features, prev_source.to(head_dtype)], dim=-1), memory.to(head_dtype)
            )
        else:
            memory_out = torch.zeros_like(memory.to(head_dtype))
        if correction_enabled:
            instant_logit = self.instant_head(features)
            memory_logit = self.memory_head(torch.cat([memory_out, features], dim=-1))
            log_scale = torch.zeros(batch, 5, dtype=head_dtype, device=device)
            log_scale[:, list(INSTANT_CHANNELS)] = instant_logit
            log_scale[:, list(MEMORY_CHANNELS)] = memory_logit
            log_scale = log_scale.clamp(
                -self.correction_bound_log, self.correction_bound_log
            ).to(dtype)
        else:
            log_scale = torch.zeros(batch, 5, dtype=dtype, device=device)
        source = analytic * torch.exp(log_scale.unsqueeze(1))
        return {
            "source": source,
            "analytic": analytic,
            "log_scale": log_scale,
            "memory_out": memory_out.to(dtype),
        }

    def forward(
        self,
        u_start: torch.Tensor,
        u_end: torch.Tensor,
        shard_values: torch.Tensor,
        shard_mask: torch.Tensor,
        *,
        dt_source_jorek: torch.Tensor | float,
        t_norm: float,
        central_density: float,
        central_mass: float = 1.0,
        pellet_density: float = 4.30e8,
        pellet_density_bg: float | None = None,
        magnetic_field: float = 0.0,
        magnetic_reduction: bool = False,
        nonlocal_tables: list[NonlocalAblationTable] | None = None,
        psi_axis: torch.Tensor | None = None,
        psi_bnd: torch.Tensor | None = None,
        memory: torch.Tensor | None = None,
        prev_source: torch.Tensor | None = None,
        use_correction: bool | None = None,
        use_marker_transport: bool | None = None,
        marker_state: MarkerRuntime | None = None,
    ) -> dict:
        """Evaluate the window ``[T_{k-1}, T_k]``.

        ``u_start``/``u_end``: ``[B, N, 9]`` raw JOREK fields at the two window
        ends.  ``shard_values``: ``[B, S, 16]`` shard state at the window start
        (``p_{k-1}``); ``shard_mask``: ``[B, S]`` alive mask.  Returns the
        five-channel source ``[B, N, 5]`` (projection units), the evolved shard
        state ``p_k``, the per-shard channel integrals and the budget totals.
        """
        if u_start.ndim != 3 or u_start.shape[-1] != 9:
            raise ValueError(f"u_start must be [B, N, 9], got {tuple(u_start.shape)}")
        if u_end.shape != u_start.shape:
            raise ValueError("u_start and u_end must have the same shape")
        batch, n_nodes, _ = u_end.shape
        n_shards = shard_values.shape[1]
        dtype = u_end.dtype
        device = u_end.device
        dt = torch.as_tensor(dt_source_jorek, dtype=dtype, device=device)
        if dt.ndim == 0:
            dt = dt.expand(batch)
        dt_si = dt * t_norm
        dt_sub = dt_si / self.n_substeps
        correction_enabled = self.use_correction if use_correction is None else bool(use_correction)
        marker_enabled = (
            self.use_marker_transport
            if use_marker_transport is None
            else bool(use_marker_transport)
        )
        if marker_enabled and marker_state is None:
            raise ValueError("marker transport requires persistent marker_state; missing initial state")
        marker_trace: list[dict] | None = [] if marker_enabled else None

        values = shard_values.to(dtype=dtype).clone()
        alive = shard_mask.to(device=device)
        idx = self.shard_index
        i_R = idx["R"]
        i_Z = idx["Z"]
        i_phi = idx["phi"]
        i_phi_init = idx["phi_init"]
        i_velR = idx["Vel_R"]
        i_velZ = idx["Vel_Z"]
        i_velRXZ = idx["Vel_RxZ"]
        i_radius = idx["radius"]
        i_abl = idx["abl"]
        i_species = idx["species"]
        i_psi = idx["psi"]
        alive = alive & (values[..., i_radius] > 0.0)

        deposit = values.new_zeros(batch, n_nodes, len(SOURCE_CHANNEL_NAMES))
        per_shard = values.new_zeros(
            batch, n_shards, len(SOURCE_CHANNEL_NAMES)
        )  # N, Z, Zeff, E, R order below; reordered when returned
        ablation_last = values.new_zeros(batch, n_shards)
        local_sum = values.new_zeros(batch, 4)
        inside_sum = values.new_zeros(batch, 1)
        inside_count = float(n_shards) * self.n_substeps
        # Accumulators in SOURCE_CHANNEL_NAMES order:
        # 0 dE_ion [J], 1 rad_sink [J], 2 Z_eff, 3 Z_imp, 4 N_imp [atoms]
        for step in range(self.n_substeps):
            lam = (step + 0.5) / self.n_substeps
            if self.field_path == "end":
                lam = 1.0
            elif self.field_path == "start":
                lam = 0.0
            u_local = (1.0 - lam) * u_start + lam * u_end
            points = values[..., :2]
            location = self.sampler.locate(points)
            sampled = self.sampler.sample(u_local.to(dtype), location)
            local_rho = sampled[..., 4]
            local_rho_imp = sampled[..., 7]
            local_te = sampled[..., 8]
            local_psi = sampled[..., 0]
            n_si = particle_density_si(local_rho, central_density)
            n_imp_si = particle_density_si(local_rho_imp, central_density)
            ne_si = (n_si - n_imp_si).clamp_min(0.0)
            te_ev = temperature_ev(local_te, central_density)
            radius = values[..., i_radius]
            species = values[..., i_species]
            magnetic = torch.full_like(te_ev, float(magnetic_field))
            ablation = parks_ablation_neon(
                radius,
                te_ev,
                ne_si,
                species,
                magnetic_field=magnetic,
                magnetic_reduction=magnetic_reduction,
            )
            if nonlocal_tables is not None:
                if psi_axis is None or psi_bnd is None:
                    raise ValueError("nonlocal_tables requires psi_axis and psi_bnd")
                te_nl = torch.zeros_like(te_ev)
                ne_nl = torch.zeros_like(ne_si)
                inside_nl = torch.zeros_like(te_ev, dtype=torch.bool)
                for row, table in enumerate(nonlocal_tables):
                    row_te, row_ne, row_inside = table.surface_state(
                        values[row, :, i_psi], psi_axis[row], psi_bnd[row], central_density
                    )
                    te_nl[row] = row_te.to(dtype).to(device)
                    ne_nl[row] = row_ne.to(dtype).to(device)
                    inside_nl[row] = row_inside.to(device)
                ablation_nl = parks_ablation_neon(
                    radius, te_nl, ne_nl, species, magnetic_field=magnetic,
                    magnetic_reduction=magnetic_reduction,
                )
                ablation = torch.where(
                    inside_nl, torch.maximum(ablation, ablation_nl), ablation
                )
            alive_f = alive.to(dtype)
            ablation = ablation * alive_f
            if marker_enabled:
                # JOREK allocates the complete source-window births from the
                # shards/ablation set by the preceding fluid/shard event.
                ablation = values[..., i_abl].clamp_min(0.) * species.clamp_min(0.) * alive_f
            if marker_trace is not None and step == 0:
                # Substep-start state for the reduced-order marker chain
                # (ADR-0020 decision 2): the trace never contains a future
                # frame -- only the window-interpolated field and past/current
                # shard state.
                marker_trace.append(
                    {
                        "u_local": u_local,
                        "shards": values.clone(),
                        "alive": alive.clone(),
                        "ablation": ablation,
                        "sampled": sampled,
                    }
                )

            zbar, z2bar, e_ion_j, rad_w = _neon_closure(self.tables, te_ev, ne_si)
            # Integrands in SOURCE_CHANNEL_NAMES order per shard.
            per_step = torch.stack(
                [
                    ablation * e_ion_j * dt_sub.view(batch, 1),  # dE_ion [J]
                    ablation * rad_w * dt_sub.view(batch, 1),  # rad_sink [J]
                    ablation * z2bar * dt_sub.view(batch, 1),  # Z_eff
                    ablation * zbar * dt_sub.view(batch, 1),  # Z_imp
                    ablation * dt_sub.view(batch, 1),  # N_imp [atoms]
                ],
                dim=-1,
            )
            per_shard = per_shard + per_step
            deposit = deposit + self.sampler.deposit(per_step, location)
            ablation_last = ablation
            local_sum = local_sum + torch.stack(
                [local_rho.mean(1), local_rho_imp.mean(1), local_te.mean(1), local_psi.mean(1)],
                dim=-1,
            )
            inside_sum = inside_sum + location.inside.to(dtype).sum(dim=1, keepdim=True)

            if marker_enabled:
                # Particle sampling must not update shards again. The next
                # fluid event's update_shards callback owns this operation.
                continue
            # ---- deterministic fragment evolution (historical analytic backbone) ----
            delta_phi = values[..., i_phi] - values[..., i_phi_init]
            a_r, a_z, a_rxz = peloton_rocket_acceleration(
                te_ev,
                ne_si,
                radius,
                species,
                delta_phi=delta_phi,
                pellet_density=pellet_density,
                pellet_density_bg=pellet_density_bg,
                central_mass=central_mass,
            )
            density = torch.where(
                species > 0.5,
                torch.full_like(species, float(pellet_density)),
                torch.full_like(species, float(pellet_density_bg if pellet_density_bg is not None else pellet_density)),
            )
            columns = [values[..., index] for index in range(values.shape[-1])]
            # JOREK update_spi order: advance position with the current velocity,
            # erode the radius with the current-step ablation, then apply the
            # rocket acceleration (models/pellet_module.f90:280-330, 760-800).
            seconds = dt_sub.view(batch, 1)
            columns[i_R] = values[..., i_R] + values[..., i_velR] * seconds
            columns[i_Z] = values[..., i_Z] + values[..., i_velZ] * seconds
            columns[i_phi] = values[..., i_phi] + values[..., i_velRXZ] * seconds
            columns[i_radius] = erode_radius(
                columns[i_radius], ablation, density * 1e20, seconds
            )
            columns[i_velR] = values[..., i_velR] + a_r * seconds
            columns[i_velZ] = values[..., i_velZ] + a_z * seconds
            columns[i_velRXZ] = values[..., i_velRXZ] + a_rxz * seconds
            columns[i_abl] = ablation
            updated = torch.stack(columns, dim=-1)
            updated = torch.where(alive.unsqueeze(-1), updated, torch.zeros_like(updated))
            values = updated
            alive = alive & (values[..., i_radius] > 0.0)

        measure = self.sampler.node_measure().to(device=device, dtype=dtype)
        floor = max(1e-9, float(measure.mean().abs()) * 1e-6)
        node_physical = deposit / measure.clamp_min(floor).view(1, -1, 1)  # [B, N, 5]

        # ---- unit conversion to the recorded projection normalisation ----
        n_norm = float(central_density) * 1e20
        analytic = node_physical.clone()
        analytic[..., list(INSTANT_CHANNELS)] = (
            node_physical[..., list(INSTANT_CHANNELS)] / n_norm
        )
        analytic[..., list(MEMORY_CHANNELS)] = (
            node_physical[..., list(MEMORY_CHANNELS)] * (t_norm / E_NORM)
        )
        scale = self.channel_scale.to(dtype=dtype, device=device).clone()
        # ---- reduced-order marker chain (deterministic main term, ADR-0020) --
        marker: dict | None = None
        base = analytic
        source_object = SOURCE_OBJECT_ANALYTIC_SHARD
        if marker_trace is not None:
            marker = self.marker_chain(
                marker_trace,
                dt_source_jorek=dt,
                t_norm=t_norm,
                central_density=central_density,
                marker_state=marker_state, u_start=u_start, u_end=u_end, central_mass=central_mass,
            )
            validate_shared_source(
                marker["source"], what="marker feedback S", n_nodes=n_nodes
            )
            base = marker["source"]
            source_object = SOURCE_OBJECT_MARKER_FEEDBACK

        # ---- split heads + bounded multiplicative correction ----
        head_dtype = self.instant_head[0].weight.dtype
        shard_norm = (
            torch.nan_to_num(values, nan=0.0) - self.shard_mean.to(dtype)
        ) / self.shard_std.to(dtype)
        shard_norm = torch.where(alive.unsqueeze(-1), shard_norm, torch.zeros_like(shard_norm))
        embedding = self.encoder(shard_norm.to(head_dtype), alive)
        inside_fraction = (inside_sum / max(inside_count, 1.0)).to(dtype)
        scaled_analytic = analytic * self.channel_scale.to(dtype).view(1, 1, 5)
        source_feature = torch.log1p(
            scaled_analytic.abs().mean(dim=1) / self.source_norm.to(dtype).view(1, 5)
        )
        feature_parts = [
            embedding,
            local_sum / self.n_substeps,
            source_feature,
            inside_fraction,
        ]
        if marker is not None:
            # Cross-step h_marker input: deterministic cloud summaries feed the
            # bounded residual heads (transport / closure / non-local).
            feature_parts.append(marker["features"].to(dtype))
        else:
            feature_parts.append(
                torch.zeros(batch, MARKER_FEATURE_DIM, dtype=dtype, device=device)
            )
        pooled = torch.cat(feature_parts, dim=-1).to(head_dtype)
        if marker is not None and prev_source is None:
            prev_source = torch.as_tensor(marker_state.feedback.values, dtype=dtype, device=device).mean(0, keepdim=True)
        heads = self.apply_heads(
            base,
            pooled,
            memory=memory,
            prev_source=prev_source,
            use_correction=correction_enabled,
            apply_calibration=marker is None,
        )
        source = heads["source"]
        log_scale = heads["log_scale"]
        memory_out = heads["memory_out"]
        if marker is not None:
            # The feedback actually sent to the field solver must also be the
            # feedback seen by particles and delta construction next window.
            physical = source[0].detach().cpu().numpy().copy()
            marker["state"].feedback.values = physical
            marker["state"].feedback.deltas = physical - marker_state.feedback.values

        atoms = per_shard[..., 4].sum(dim=1)
        ionization_energy = per_shard[..., 0].sum(dim=1)
        radiation_energy = per_shard[..., 1].sum(dim=1)
        if marker is not None:
            # The marker arm's budget comes from its own cloud (emitted atoms
            # and deposited ionisation/radiation energy), not from the analytic
            # backbone (ADR-0020 decision 2).
            cloud = marker["cloud"]
            cloud_alive = cloud["alive"].to(dtype)
            atoms = (cloud["weights"] * cloud_alive).sum(dim=1)
            ionization_energy = (cloud["energy"] * cloud_alive).sum(dim=1)
            radiation_energy = (cloud["rad"] * cloud_alive).sum(dim=1)
        return {
            "source": source,  # [B, N, 5] shared S interface (ADR-0020)
            "source_object": source_object,
            "analytic": analytic,
            "marker_feedback": None if marker is None else marker["source"],
            "marker": marker,
            "correction_log_scale": log_scale,
            "per_shard_channels": per_shard,  # [B, S, 5] (dE_ion, rad_sink, Z_eff, Z_imp, N_imp)
            "shards": {
                "values": values,
                "alive": alive,
                "ablation": ablation_last,
            },
            "budget": {
                "atoms": atoms,
                "ionization_energy": ionization_energy,
                "radiation_energy": radiation_energy,
            },
            "sampler_features": {
                "local_mean": local_sum / self.n_substeps,  # [B, 4]
                "source_feature": source_feature,  # [B, 5]
                "inside_fraction": inside_fraction,  # [B, 1]
            },
            "memory_out": memory_out.to(dtype),
            "memory_used": (
                torch.zeros(batch, self.memory_dim, dtype=dtype, device=device)
                if memory is None
                else memory.to(dtype)
            ),
            "sampler_inside_fraction": inside_fraction.mean(),
            "channel_scale": scale,
            "table_source": self.table_source,
        }


# --------------------------------------------------------------------------- #
# Augmented state (u, p_shard, h_marker) and the staggered source step
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class AugmentedState:
    """Reduced source-window state; t_jorek is the marker EVENT clock.

    u is the known start field, u_previous retains prior field history; marker
    state separately stores physical particle time, feedback and field times.
    p_shard is supplied by the preceding fluid/shard event. h_marker is the
    compatibility name for optional learned h_residual, not physical memory.
    ReferenceRollout carries the full fluid event clock and plan cursor.
    """

    u: torch.Tensor
    p_shard: torch.Tensor
    p_shard_mask: torch.Tensor
    h_marker: torch.Tensor
    step_index: int
    t_jorek: float
    marker_state: MarkerRuntime | None = None
    u_previous: torch.Tensor | None = None
    previous_t_jorek: float | None = None

    @property
    def h_residual(self):
        """Learning memory only; h_marker retained as a compatibility name."""
        return self.h_marker


def advance_augmented_source(
    module: SourceModule,
    state: AugmentedState,
    u_next: torch.Tensor,
    *,
    dt_source_jorek: torch.Tensor | float,
    t_norm: float,
    central_density: float,
    t_next: float | None = None,
    **kwargs: object,
) -> tuple[AugmentedState, dict]:
    """Advance particles between two already available field endpoints.

    u_next is the available solved end field (not a request to read a future
    frame). The reference event driver handles Source -> next fluid solve ->
    shard update; this convenience wrapper handles only the source window.
    """

    if t_next is not None and abs(float(t_next) - state.t_jorek - float(torch.as_tensor(dt_source_jorek).reshape(-1)[0])) > 1e-8:
        raise ValueError("t_next disagrees with source window duration")
    marker_state = state.marker_state
    if marker_state is not None:
        from copy import deepcopy
        marker_state = deepcopy(marker_state)
        expected = state.t_jorek * t_norm
        if abs(marker_state.clock.sim_time_si - expected) > 1e-14:
            raise ValueError("augmented state/marker event clock mismatch")
        marker_state.clock.field_start_si = expected
        marker_state.clock.field_end_si = expected + float(torch.as_tensor(dt_source_jorek).reshape(-1)[0])*t_norm
    out = module(
        state.u,
        u_next,
        state.p_shard,
        state.p_shard_mask,
        dt_source_jorek=dt_source_jorek,
        t_norm=t_norm,
        central_density=central_density,
        memory=state.h_residual,
        marker_state=marker_state,
        **kwargs,
    )
    next_time = state.t_jorek + float(
        dt_source_jorek.reshape(-1)[0]
        if isinstance(dt_source_jorek, torch.Tensor)
        else dt_source_jorek
    )
    updated = AugmentedState(
        u=u_next,
        p_shard=out["shards"]["values"],
        p_shard_mask=out["shards"]["alive"],
        h_marker=out["memory_out"],
        step_index=state.step_index + 1,
        t_jorek=next_time if t_next is None else float(t_next),
        marker_state=None if out["marker"] is None else out["marker"]["state"],
        u_previous=state.u, previous_t_jorek=state.t_jorek,
    )
    return updated, out
