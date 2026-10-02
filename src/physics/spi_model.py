"""Differentiable analytic backbone of the JOREK SPI model.

Ports the parts of ``models/pellet_module.f90`` used by the EHL-2 runs:

* ablation model 3 (Parks/NGS scaling, neon branch) with the local electron
  density built the way JOREK does it in ``use_marker`` mode;
* rocket-effect model 6 (PELOTON JET scaling);
* spherical shard radius erosion and position/velocity advance.

The functions are pure torch ops so they can be used as the analytic baseline
inside the learned source/dynamics modules.  The *nonlocal* ablation branch is
ported from the frozen ``nonlocal_array.dat`` snapshot that ``int3d_new`` writes
once per run (``nl_avg_Te(1) == 0`` guard in ``core/mod_jorek_timestepping.f90``):
JOREK interpolates the shard psi into that table and takes
``max(local, nonlocal)`` per shard, which is what ``NonlocalAblationTable`` and
the ``nonlocal_table`` argument of :func:`advance_shards_analytic` reproduce.

Units follow ``type_SPI``: positions in m, radius in m, ablation in atom/s,
species in [0, 1].  Field inputs are JOREK-normalised unless stated otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

EL_CHG = 1.602176565e-19  # C
MU_ZERO = 4.0 * math.pi * 1e-7  # H/m
K_BOLTZ = 1.380649e-23  # J/K
MOLE_NUMBER = 6.02214076e23  # 1/mol
AMU = 1.66053906660e-27  # kg

# type_SPI field indices as stored by src/data/fragment_state.py.
SHARD_R = 0
SHARD_Z = 1
SHARD_PHI = 2
SHARD_RADIUS = 7
SHARD_ABL = 8
SHARD_SPECIES = 9
SHARD_PSI = 11


@dataclass(frozen=True)
class NonlocalAblationTable:
    """Frozen ``nonlocal_array.dat`` snapshot (one per run).

    Columns written by ``diagnostics/mod_integrals3D.f90``:
    index, ``psi_n``, surface-averaged electron temperature in JOREK units
    (``<p_e>/<n_e>``, the same quantity as the local ``T_e`` channel) and
    surface-averaged electron density in JOREK units.
    """

    psi_n: torch.Tensor
    avg_te: torch.Tensor
    avg_ne: torch.Tensor

    @classmethod
    def from_arrays(
        cls,
        psi_n: np.ndarray,
        avg_te: np.ndarray,
        avg_ne: np.ndarray,
    ) -> "NonlocalAblationTable":
        return cls(
            psi_n=torch.as_tensor(np.asarray(psi_n), dtype=torch.float64),
            avg_te=torch.as_tensor(np.asarray(avg_te), dtype=torch.float64),
            avg_ne=torch.as_tensor(np.asarray(avg_ne), dtype=torch.float64),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "NonlocalAblationTable":
        rows = np.loadtxt(path, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] < 4:
            raise ValueError(f"{path}: expected >=4 columns, got {rows.shape}")
        return cls.from_arrays(rows[:, 1], rows[:, 2], rows[:, 3])

    def __len__(self) -> int:
        return int(self.psi_n.numel())

    def nearest_index(self, psi_n: torch.Tensor) -> torch.Tensor:
        """JOREK's ``minloc`` + "step toward the axis" lookup, in psi_n space.

        Fortran (``models/pellet_module.f90``)::

            i_loc = minloc(abs(nl_Psi - Psi_spi), dim=1)
            if (abs(Psi_spi-ES%psi_axis) < abs(nl_Psi(i_loc)-ES%psi_axis)) i_loc = i_loc - 1

        Both distances are affine in ``psi_n`` with the same positive scale, so
        the comparison and the argmin are equivalent in ``psi_n`` space (the
        frozen table's psi grid uses the axis/boundary of the snapshot frame;
        the measured span drift over the SPI phase is <0.1%, see the progress
        log).
        """
        shard = psi_n.reshape(-1)
        distance = (self.psi_n.reshape(-1, 1) - shard.reshape(1, -1)).abs()
        index = distance.argmin(dim=0)
        step_toward_axis = shard.abs() < self.psi_n.reshape(-1)[index].abs()
        return torch.where(step_toward_axis, index - 1, index).clamp_(0, len(self) - 1)

    def surface_state(
        self,
        psi_shard: torch.Tensor,
        psi_axis: float | torch.Tensor,
        psi_bnd: float | torch.Tensor,
        central_density: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Nonlocal ``(T_e [eV], n_e [m^-3], inside-LCFS mask)`` per shard."""
        if not isinstance(psi_shard, torch.Tensor):
            psi_shard = torch.as_tensor(np.asarray(psi_shard), dtype=torch.float64)
        psi_n = (psi_shard - psi_axis) / (psi_bnd - psi_axis)
        work = psi_n.to(dtype=self.psi_n.dtype, device=self.psi_n.device)
        index = self.nearest_index(work)
        te_ev = (self.avg_te[index] / (EL_CHG * MU_ZERO * central_density * 1e20)).to(
            dtype=psi_shard.dtype, device=psi_shard.device
        )
        ne_si = (self.avg_ne[index] * 1e20 * central_density).to(
            dtype=psi_shard.dtype, device=psi_shard.device
        )
        return te_ev, ne_si, (psi_n <= 1.0)


def particle_density_si(field: torch.Tensor, central_density: float | torch.Tensor) -> torch.Tensor:
    """JOREK normalised density -> SI number density [m^-3]."""
    return field * 1e20 * central_density


def temperature_ev(te_jorek: torch.Tensor, central_density: float | torch.Tensor) -> torch.Tensor:
    """JOREK normalised electron temperature -> eV (WITH_TiTe branch)."""
    return te_jorek / (EL_CHG * MU_ZERO * central_density * 1e20)


def _species_mass_ratio(species: torch.Tensor, *, neon: float = 1.0 / 20.0) -> torch.Tensor:
    """``m_i/m_imp`` used by JOREK (central_mass = 1 -> Ne ratio 1/20)."""
    return torch.where(species > 0.5, torch.full_like(species, neon), torch.full_like(species, 0.5))


def parks_ablation_neon(
    radius_m: torch.Tensor,
    te_ev: torch.Tensor,
    electron_density_si: torch.Tensor,
    species: torch.Tensor,
    *,
    magnetic_field: torch.Tensor | None = None,
    magnetic_reduction: bool = False,
) -> torch.Tensor:
    """JOREK ``spi_abl_model == 3`` neon branch [atom/s].

    ``species`` is the neon fraction (1 = pure neon, 0 = pure D2).  The
    magnetic reduction is disabled by default because the EHL-2 namelists leave
    ``spi_abl_mag_reduction`` at its ``.false.`` default.
    """
    radius_cm = radius_m * 1e2
    coefficient = 27.0837 + torch.tan(1.48709 * (1.0 - species) / (1.0 + species))
    mass_term = 20.183 * species + 2.0141 * (1.0 - species)
    ablation = (
        coefficient
        * MOLE_NUMBER
        * (radius_cm / 0.2).clamp_min(0.0) ** (4.0 / 3.0)
        * (electron_density_si * 1e-20).clamp_min(0.0) ** (1.0 / 3.0)
        * (te_ev / 2e3).clamp_min(0.0) ** (5.0 / 3.0)
        / mass_term.clamp_min(1e-12)
    )
    ablation = ablation.clamp_min(0.0)
    if magnetic_reduction and magnetic_field is not None:
        b0 = magnetic_field.abs()
        factor = torch.where(b0 > 2.0, (2.0 / b0.clamp_min(1e-12)) ** 0.843, torch.ones_like(b0))
        ablation = ablation * factor
    return ablation


def peloton_rocket_acceleration(
    te_ev: torch.Tensor,
    electron_density_si: torch.Tensor,
    radius_m: torch.Tensor,
    species: torch.Tensor,
    *,
    delta_phi: torch.Tensor,
    pellet_density: float,
    pellet_density_bg: float | None = None,
    central_mass: float = 1.0,
    hydrogen_mass_modifier: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """JOREK ``spi_rkt_model == 6`` (PELOTON JET) acceleration [m/s^2].

    Returns ``(a_R, a_Z, a_RxZ)``.  ``delta_phi = phi - phi_init`` selects the
    direction of the asymmetric pressure.
    """
    if pellet_density_bg is None:
        pellet_density_bg = pellet_density
    pure = species > 0.5
    mixed = (species > 0.0) & (species < 1.0)
    density = torch.where(
        pure,
        torch.full_like(species, float(pellet_density)),
        torch.where(
            mixed,
            1.0
            / (
                (1.0 - species) / max(pellet_density_bg, 1e-30)
                + species / max(pellet_density, 1e-30)
            ),
            torch.full_like(species, float(pellet_density_bg)),
        ),
    )
    mass_ratio = _species_mass_ratio(species)
    mass_eff = species * central_mass / mass_ratio + (1.0 - species) * central_mass * hydrogen_mass_modifier
    fragment_mass = (
        (4.0 * math.pi / 3.0)
        * radius_m.clamp_min(1e-5) ** 3
        * mass_eff
        * AMU
        * density
        * 1e20
    ).clamp_min(1e-30)
    pressure_asymmetry = (
        1.0e5
        * (te_ev / 2e3).clamp_min(0.0) ** 1.26
        * (electron_density_si / 1e20).clamp_min(0.0) ** 0.4
    )
    thrust = pressure_asymmetry * math.pi * radius_m.clamp_min(0.0) ** 2 / fragment_mass
    a_r = thrust * torch.cos(delta_phi)
    a_z = torch.zeros_like(thrust)
    a_rxz = thrust * torch.sin(delta_phi)
    return a_r, a_z, a_rxz


def erode_radius(
    radius_m: torch.Tensor,
    ablation_atom_per_s: torch.Tensor,
    density_si: torch.Tensor,
    dt_seconds: torch.Tensor | float,
) -> torch.Tensor:
    """Spherical shard erosion ``r -= dt * abl / (4 pi r^2 rho)``, clipped at 0."""
    decrement = (
        torch.as_tensor(dt_seconds)
        * ablation_atom_per_s.clamp_min(0.0)
        / (4.0 * math.pi * radius_m.clamp_min(1e-12) ** 2 * density_si.clamp_min(1e-30))
    )
    return (radius_m - decrement).clamp_min(0.0)


def advance_shards_analytic(
    shard_values: torch.Tensor,
    shard_mask: torch.Tensor,
    *,
    local_rho: torch.Tensor,
    local_rho_imp: torch.Tensor,
    local_te: torch.Tensor,
    central_density: float,
    central_mass: float = 1.0,
    dt_jorek: torch.Tensor | float = 0.0,
    t_norm: float = 1.0,
    tstep_jorek: torch.Tensor | float = 0.0,
    pellet_density: float = 4.30e8,
    pellet_density_bg: float | None = None,
    magnetic_field: torch.Tensor | None = None,
    magnetic_reduction: bool = False,
    nonlocal_table: NonlocalAblationTable | None = None,
    psi_shard: torch.Tensor | None = None,
    psi_axis: float | torch.Tensor | None = None,
    psi_bnd: float | torch.Tensor | None = None,
    ablation_model: int = 3,
    rocket_model: int = 6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One analytic SPI step for a batch of shard sets.

    ``shard_values`` is ``[S, F]`` (or ``[B, S, F]``) and ``local_*`` are the
    field values sampled at the shard positions.  Returns the updated shard
    values and the ablation rate [atom/s] per shard.  ``dt_jorek`` is the frame
    interval in JOREK units (multiplied by ``t_norm`` for SI); pass
    ``tstep_jorek`` when the caller works at the internal-step granularity.

    When ``nonlocal_table`` is given, the shard psi (``psi_shard``, JOREK units,
    with the frame's ``psi_axis``/``psi_bnd``) is looked up in the frozen
    ``nonlocal_array`` and the per-shard rate becomes ``max(local, nonlocal)``
    inside the LCFS, exactly as ``pellet_module.f90`` does.
    """
    if ablation_model != 3:
        raise NotImplementedError(f"only spi_abl_model == 3 is ported, got {ablation_model}")
    if rocket_model != 6:
        raise NotImplementedError(f"only spi_rkt_model == 6 is ported, got {rocket_model}")
    values = shard_values.clone()
    radius = values[..., SHARD_RADIUS]
    species = values[..., SHARD_SPECIES]

    n_si = particle_density_si(local_rho, central_density)
    n_imp_si = particle_density_si(local_rho_imp, central_density)
    # JOREK use_marker branch: Z_imp is a placeholder zero, so
    # beta_imp = mu_imp * Z_imp - 1 = -1.
    ne_si = (n_si - n_imp_si).clamp_min(0.0)
    te_ev = temperature_ev(local_te, central_density)

    ablation = parks_ablation_neon(
        radius,
        te_ev,
        ne_si,
        species,
        magnetic_field=magnetic_field,
        magnetic_reduction=magnetic_reduction,
    )
    if nonlocal_table is not None:
        if psi_shard is None or psi_axis is None or psi_bnd is None:
            raise ValueError("nonlocal_table requires psi_shard, psi_axis and psi_bnd")
        te_ev_nl, ne_si_nl, inside = nonlocal_table.surface_state(
            psi_shard, psi_axis, psi_bnd, central_density
        )
        ablation_nl = parks_ablation_neon(
            radius,
            te_ev_nl,
            ne_si_nl,
            species,
            magnetic_field=magnetic_field,
            magnetic_reduction=magnetic_reduction,
        )
        ablation = torch.where(inside, torch.maximum(ablation, ablation_nl), ablation)
    if pellet_density_bg is None:
        pellet_density_bg = pellet_density
    pure = species > 0.5
    density = torch.where(
        pure,
        torch.full_like(species, float(pellet_density)),
        torch.full_like(species, float(pellet_density_bg)),
    )

    # Position advance uses the current velocity (JOREK order).
    seconds = torch.as_tensor(dt_jorek) * t_norm
    values[..., SHARD_R] = values[..., SHARD_R] + values[..., 4] * seconds
    values[..., SHARD_Z] = values[..., SHARD_Z] + values[..., 5] * seconds
    values[..., SHARD_PHI] = values[..., SHARD_PHI] + values[..., 6] * seconds

    # Radius erosion uses the previous ablation rate in JOREK; use the freshly
    # computed one here so the step is self-consistent.
    values[..., SHARD_RADIUS] = erode_radius(radius, ablation, density * 1e20, seconds)

    delta_phi = values[..., SHARD_PHI] - values[..., 3]
    a_r, a_z, a_rxz = peloton_rocket_acceleration(
        te_ev,
        ne_si,
        values[..., SHARD_RADIUS],
        species,
        delta_phi=delta_phi,
        pellet_density=pellet_density,
        pellet_density_bg=pellet_density_bg,
        central_mass=central_mass,
    )
    seconds = torch.as_tensor(tstep_jorek if tstep_jorek is not None else dt_jorek) * t_norm
    values[..., 4] = values[..., 4] + a_r * seconds
    values[..., 5] = values[..., 5] + a_z * seconds
    values[..., 6] = values[..., 6] + a_rxz * seconds
    values[..., SHARD_ABL] = ablation
    dead = values[..., SHARD_RADIUS] <= 0.0
    values[dead] = 0.0
    mask = shard_mask & ~dead
    return values, ablation * mask.to(ablation.dtype)

__all__ = [
    "AMU",
    "EL_CHG",
    "K_BOLTZ",
    "MOLE_NUMBER",
    "MU_ZERO",
    "NonlocalAblationTable",
    "SHARD_ABL",
    "SHARD_PHI",
    "SHARD_PSI",
    "SHARD_R",
    "SHARD_RADIUS",
    "SHARD_SPECIES",
    "SHARD_Z",
    "advance_shards_analytic",
    "erode_radius",
    "parks_ablation_neon",
    "particle_density_si",
    "peloton_rocket_acceleration",
    "temperature_ev",
]
