"""Fragment state ``p(t)`` extraction for the SPI-coupled Model 502 surrogate.

Two levels exist in the EHL-2 data:

* **shard level** (primary): the ``spi_*`` datasets in ``jorek<index>.h5``.
  They describe one spherical shard of a shattered pellet per row
  (``type_SPI`` in JOREK: R, Z, phi, velocities, radius, ablation rate,
  species fraction, deposited volume, psi, |grad psi| and drift variants).
  Shards exist only after the injection time ``t_ns``; before that
  ``n_spi_tot == 0`` and ``read_shard_state`` returns ``None``.
* **marker level** (secondary): per-group reductions of ``part<index>.h5``.
  Markers are stored per particle (``x = (R, Z, phi)``, ``v = (v_R, v_Z,
  v_phi)``, weight, life index); files hold up to 1e6 particles per group, so
  only a stride-sampled summary is produced here.

Units follow the JOREK ``type_SPI`` declarations: positions in m, angles in
radian, velocities in m/s, radius in m, ablation in atom/s, species as a
fraction.  ``psi``/``grad_psi`` are kept in the JOREK field normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


# (field name, HDF5 dataset in jorek*.h5, unit)
SHARD_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("R", "spi_R_arr", "m"),
    ("Z", "spi_Z_arr", "m"),
    ("phi", "spi_phi_arr", "rad"),
    ("phi_init", "spi_phi_init_arr", "rad"),
    ("Vel_R", "spi_Vel_R_arr", "m/s"),
    ("Vel_Z", "spi_Vel_Z_arr", "m/s"),
    ("Vel_RxZ", "spi_Vel_RxZ_arr", "m/s"),
    ("radius", "spi_radius_arr", "m"),
    ("abl", "spi_abl_arr", "atom/s"),
    ("species", "spi_species_arr", "1"),
    ("vol", "spi_vol_arr", "JOREK units"),
    ("psi", "spi_psi_arr", "JOREK psi"),
    ("grad_psi", "spi_grad_psi_arr", "JOREK psi per m"),
    ("vol_drift", "spi_vol_arr_drift", "JOREK units"),
    ("psi_drift", "spi_psi_arr_drift", "JOREK psi"),
    ("grad_psi_drift", "spi_grad_psi_arr_drift", "JOREK psi per m"),
)
SHARD_FIELD_NAMES: tuple[str, ...] = tuple(field for field, _, _ in SHARD_FIELDS)


@dataclass(frozen=True)
class ShardState:
    """One frame of shard-level fragment state.

    ``values`` is ``[n_spi, len(SHARD_FIELDS)]`` in ``SHARD_FIELD_NAMES`` order.
    ``alive`` marks shards with a positive radius (JOREK sets the radius to 0
    once a shard is fully ablated; its other fields are then stale).
    Missing optional datasets (drift variants) are filled with NaN.
    """

    values: np.ndarray
    alive: np.ndarray
    fields: tuple[str, ...]
    n_spi: int
    frame_path: Path
    plasmoid_in_domain: np.ndarray | None = None

    @property
    def n_alive(self) -> int:
        return int(self.alive.sum())


@dataclass(frozen=True)
class MarkerGroupSummary:
    """Stride-sampled summary of one particle group in a ``part*.h5`` file."""

    group: str
    type_name: str
    charge: float
    mass: float
    n_particles: int
    n_alive: int
    total_weight: float
    mean_R: float
    mean_Z: float
    mean_phi: float
    mean_VR: float
    mean_VZ: float
    mean_Vphi: float
    stride: int


@dataclass(frozen=True)
class MarkerSummary:
    """Aggregate of all particle groups of one marker frame."""

    groups: tuple[MarkerGroupSummary, ...]
    n_alive: int
    total_weight: float
    frame_path: Path


def read_shard_state(state_path: str | Path) -> ShardState | None:
    """Read shard-level ``p(t)`` from one ``jorek<index>.h5`` frame.

    Returns ``None`` when the frame predates the SPI injection
    (``n_spi_tot == 0``) or when the shard bookkeeping is absent.
    """
    path = Path(state_path)
    with h5py.File(path, "r") as handle:
        if "n_spi_tot" not in handle:
            return None
        n_spi = int(np.ravel(handle["n_spi_tot"][()])[0])
        if n_spi <= 0:
            return None

        columns: list[np.ndarray] = []
        for _, dataset_name, _ in SHARD_FIELDS:
            if dataset_name not in handle:
                columns.append(np.full(n_spi, np.nan, dtype=np.float64))
                continue
            raw = np.asarray(handle[dataset_name][()], dtype=np.float64).reshape(-1)
            if raw.shape[0] != n_spi:
                raise ValueError(
                    f"{path.name}: dataset '{dataset_name}' has {raw.shape[0]} rows, "
                    f"expected n_spi_tot = {n_spi}"
                )
            columns.append(raw)

        values = np.stack(columns, axis=1)
        radius = values[:, SHARD_FIELD_NAMES.index("radius")]
        alive = np.isfinite(radius) & (radius > 0.0)

        plasmoid = None
        if "plasmoid_in_domain_arr" in handle:
            raw = np.asarray(handle["plasmoid_in_domain_arr"][()]).reshape(-1)
            if raw.shape[0] == n_spi:
                plasmoid = raw.astype(np.int32)

    return ShardState(
        values=values,
        alive=alive,
        fields=SHARD_FIELD_NAMES,
        n_spi=n_spi,
        frame_path=path,
        plasmoid_in_domain=plasmoid,
    )


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        return float(values.mean()) if values.size else float("nan")
    return float(np.dot(values, weights) / total)


def read_marker_summary(
    marker_path: str | Path,
    *,
    stride: int = 10,
) -> MarkerSummary:
    """Stride-sampled marker summary for one ``part<index>.h5`` frame.

    The alive count is exact (``i_life > 0`` over the full array); the position,
    velocity and weight means are computed on every ``stride``-th particle.
    """
    if stride < 1:
        raise ValueError("stride must be >= 1")
    path = Path(marker_path)
    groups: list[MarkerGroupSummary] = []
    with h5py.File(path, "r") as handle:
        if "groups" not in handle:
            return MarkerSummary(groups=(), n_alive=0, total_weight=0.0, frame_path=path)
        for group_name in sorted(handle["groups"].keys()):
            group = handle["groups"][group_name]
            if "i_life" not in group:
                continue
            life = np.asarray(group["i_life"][()]).reshape(-1)
            n_alive = int((life > 0).sum())

            life_sampled = life[::stride]
            mask = life_sampled > 0
            if not mask.any():
                continue

            x = np.asarray(group["x"][::stride], dtype=np.float64)
            v = np.asarray(group["v"][::stride], dtype=np.float64)
            weight = np.asarray(group["weight"][::stride], dtype=np.float64).reshape(-1)
            x, v, weight = x[mask], v[mask], weight[mask]

            type_raw = group["type"][()] if "type" in group else b""
            if isinstance(type_raw, bytes):
                type_name = type_raw.decode(errors="replace").strip()
            else:
                type_name = str(np.ravel(type_raw)[0])
            charge = float(np.ravel(group["Z"][()])[0]) if "Z" in group else float("nan")
            mass = float(np.ravel(group["mass"][()])[0]) if "mass" in group else float("nan")

            groups.append(
                MarkerGroupSummary(
                    group=group_name,
                    type_name=type_name,
                    charge=charge,
                    mass=mass,
                    n_particles=int(life.size),
                    n_alive=n_alive,
                    total_weight=_weighted_mean(weight, np.ones_like(weight)) * weight.size,
                    mean_R=_weighted_mean(x[:, 0], weight),
                    mean_Z=_weighted_mean(x[:, 1], weight),
                    mean_phi=_weighted_mean(x[:, 2], weight),
                    mean_VR=_weighted_mean(v[:, 0], weight),
                    mean_VZ=_weighted_mean(v[:, 1], weight),
                    mean_Vphi=_weighted_mean(v[:, 2], weight),
                    stride=stride,
                )
            )

    n_alive = sum(group.n_alive for group in groups)
    total_weight = sum(group.total_weight for group in groups)
    return MarkerSummary(
        groups=tuple(groups),
        n_alive=n_alive,
        total_weight=float(total_weight),
        frame_path=path,
    )


__all__ = [
    "SHARD_FIELDS",
    "SHARD_FIELD_NAMES",
    "MarkerGroupSummary",
    "MarkerSummary",
    "ShardState",
    "read_marker_summary",
    "read_shard_state",
]
