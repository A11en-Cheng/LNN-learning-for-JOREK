"""Synthetic JOREK/part/projection frames used by the Phase-1 tests.

The files mimic the real EHL-2 layout closely enough for the parser,
``collect_frames`` and the fragment readers: numeric frame names, the
``values``/``x``/``boundary`` layout, the ``spi_*`` shard arrays and one
``particle_marker`` group in ``part*.h5``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

NODES = 6
N_ELEMENTS = 2
SHARD_DATASETS = (
    "spi_R_arr",
    "spi_Z_arr",
    "spi_phi_arr",
    "spi_phi_init_arr",
    "spi_Vel_R_arr",
    "spi_Vel_Z_arr",
    "spi_Vel_RxZ_arr",
    "spi_radius_arr",
    "spi_abl_arr",
    "spi_species_arr",
    "spi_vol_arr",
    "spi_psi_arr",
    "spi_grad_psi_arr",
)


def write_frame(
    run_dir: str | Path,
    index: int,
    *,
    t_now: float,
    n_spi: int = 0,
    eta: float = 1e-10,
    visco: float = 1e-5,
    t_norm: float = 4.085775e-07,
    marker_particles: int = 8,
    marker_alive: int = 4,
    write_projections: bool = True,
) -> None:
    """Write one synthetic ``jorek``/``part``/``projections`` frame triple."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    values = np.zeros((9, 4, 1, NODES), dtype=np.float64)
    values[:, 0, 0, :] = np.linspace(-1.0, 1.0, NODES)[None, :]
    coords = np.zeros((2, 4, 1, NODES), dtype=np.float64)
    coords[0, 0, 0, :] = np.linspace(1.0, 2.0, NODES)
    coords[1, 0, 0, :] = np.linspace(-0.5, 0.5, NODES)

    with h5py.File(run_dir / f"jorek{index:06d}.h5", "w") as handle:
        handle.create_dataset("values", data=values)
        handle.create_dataset("deltas", data=np.zeros_like(values))
        handle.create_dataset("x", data=coords)
        handle.create_dataset("boundary", data=np.zeros(NODES, dtype=np.int32))
        # Element geometry consumed by ``load_run_topology`` (and therefore by
        # every FEM/graph feature).  Same two synthetic quadrilaterals as the
        # ``contain_node`` block below, but in the *real* restart contract:
        # ``size`` [4, 4, n_elem] and 1-based ``vertex`` [4, n_elem].
        handle.create_dataset("size", data=np.ones((4, 4, N_ELEMENTS)))
        handle.create_dataset(
            "vertex",
            data=np.array(
                [
                    [1, 3],
                    [2, 4],
                    [3, 5],
                    [4, 6],
                ],
                dtype=np.int64,
            ),
        )
        # Two quadrilateral elements covering the six synthetic nodes.
        handle.create_dataset(
            "contain_node",
            data=np.array(
                [
                    [1, 2, 3, 4, 0],
                    [3, 4, 5, 6, 0],
                ],
                dtype=np.int32,
            ),
        )
        for key, value in (
            ("n_tor", 1),
            ("n_dim", 2),
            ("n_var", 9),
            ("jorek_model", 502),
        ):
            handle.create_dataset(key, data=np.array([value]))
        handle.create_dataset("t_now", data=np.array([t_now]))
        handle.create_dataset("tstep", data=np.array([1.0]))
        handle.create_dataset("central_density", data=np.array([0.8]))
        handle.create_dataset("central_mass", data=np.array([1.0]))
        handle.create_dataset("F0", data=np.array([1.0]))
        handle.create_dataset("R_domm", data=np.array([-3.3]))
        handle.create_dataset("sqrt_mu0_rho0", data=np.array([t_norm]))
        handle.create_dataset("sqrt_mu0_over_rho0", data=np.array([1.0 / t_norm]))
        handle.create_dataset("t_norm", data=np.array([t_norm]))
        handle.create_dataset("eta", data=np.array([eta]))
        handle.create_dataset("visco", data=np.array([visco]))
        handle.create_dataset("visco_par", data=np.array([1e-5]))
        handle.create_dataset("n_spi_tot", data=np.array([n_spi]))
        if n_spi > 0:
            for offset, name in enumerate(SHARD_DATASETS):
                base = np.arange(1, n_spi + 1, dtype=np.float64) + 0.1 * offset
                if name == "spi_radius_arr":
                    base = np.full(n_spi, 1e-4)
                    base[-1:] = 0.0
                if name == "spi_species_arr":
                    base = np.ones(n_spi)
                handle.create_dataset(name, data=base)

    with h5py.File(run_dir / f"part{index:06d}.h5", "r+") if (run_dir / f"part{index:06d}.h5").exists() else h5py.File(
        run_dir / f"part{index:06d}.h5", "w"
    ) as handle:
        group = handle.require_group("groups/001")
        if "x" in group:
            del group["x"]
        x = np.zeros((marker_particles, 3), dtype=np.float64)
        v = np.zeros((marker_particles, 3), dtype=np.float64)
        weight = np.full(marker_particles, 1.0)
        life = np.zeros(marker_particles, dtype=np.int32)
        life[:marker_alive] = np.arange(1, marker_alive + 1)
        x[:, 0] = np.linspace(1.5, 1.7, marker_particles)
        x[:, 1] = 0.01
        x[:, 2] = 0.02
        v[:, 0] = -300.0
        group.create_dataset("x", data=x)
        group.create_dataset("v", data=v)
        group.create_dataset("weight", data=weight)
        group.create_dataset("i_life", data=life)
        group.create_dataset("t_birth", data=np.zeros(marker_particles, dtype=np.float32))
        group.create_dataset("Z", data=np.array([10.0]))
        group.create_dataset("mass", data=np.array([20.1797]))
        group.create_dataset("type", data=np.array([b"particle_marker"]))

    if write_projections:
        projections = np.zeros((5, 4, 1, NODES), dtype=np.float64)
        for channel in range(5):
            projections[channel, 0, 0, :] = 0.001 * (channel + 1)
        with h5py.File(run_dir / f"projections{index:06d}.h5", "w") as handle:
            handle.create_dataset("values", data=projections)
            handle.create_dataset("t_now", data=np.array([t_now * t_norm]))


def write_namelist(run_dir: str | Path, *, t_ns: float | None = None, extra: str = "") -> None:
    run_dir = Path(run_dir)
    lines = ["&jorek", "  n_inj = 1", "  n_spi = 12", "  spi_abl_model = 3", "  spi_rkt_model = 6"]
    if t_ns is not None:
        lines.append(f"  t_ns = {t_ns}")
    if extra:
        lines.append(extra)
    lines.append("/")
    (run_dir / "jorek_namelist_spitest").write_text("\n".join(lines) + "\n", encoding="utf-8")
