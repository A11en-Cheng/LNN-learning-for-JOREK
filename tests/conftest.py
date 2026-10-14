"""Shared test fixtures for the Model 502 data contract."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

# Archived v1 tests (GNN stack / teacher-source interpolation / v1 physics loss)
# live in tests/legacy/. They are reference-only: not collected, not baselines.
collect_ignore = ["legacy"]


def write_model502_frame(
    path,
    *,
    t_now=0.0,
    t_norm=None,
    n_tor=1,
    n_dim=2,
    harmonics=1,
    rho_values=None,
    n_nodes=6,
):
    """Write a minimal Model 502 shaped HDF5 frame with literal field values."""
    with h5py.File(path, "w") as f:
        f["n_var"] = 9
        f["n_tor"] = n_tor
        f["n_dim"] = n_dim
        f["jorek_model"] = 502
        f["t_now"] = t_now
        if t_norm is not None:
            f["t_norm"] = t_norm
        f["boundary"] = np.arange(n_nodes, dtype=np.int64)
        f["contain_node"] = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=np.int32
        ).T
        f["x"] = np.zeros((2, 4, 1, n_nodes), dtype=np.float64)
        f["x"][0, 0, 0, :] = np.arange(n_nodes)
        f["x"][1, 0, 0, :] = np.arange(n_nodes) * 2
        # Minimal element geometry: ``load_run_topology`` (and therefore every
        # FEM/graph feature) needs ``size`` and ``vertex`` to exist, exactly as
        # the real restart files have them.
        n_elem = max(1, n_nodes - 1)
        f["size"] = np.ones((4, 4, n_elem), dtype=np.float64)
        f["vertex"] = np.tile(
            np.arange(1, 5, dtype=np.int64)[:, None], (1, n_elem)
        )
        values = np.zeros((9, 4, harmonics, n_nodes), dtype=np.float64)
        for channel in range(9):
            values[channel, 0, 0, :] = float(channel)
        if rho_values is not None:
            values[4, 0, 0, :] = rho_values
        f["values"] = values


def write_projection_frame(path, *, t_seconds, n_nodes=6, channels=5):
    """Write a minimal ``projections*.h5`` frame.

    ``values`` uses the recorded layout ``[channels, dof, harmonics, nodes]``
    with the marker-projection channel order; ``t_now`` is in physical seconds.
    """
    with h5py.File(path, "w") as f:
        f["t_now"] = t_seconds
        f["n_dim"] = 2
        f["n_tor"] = 1
        f["n_nodes"] = n_nodes
        values = np.zeros((channels, 4, 1, n_nodes), dtype=np.float64)
        for channel in range(channels):
            values[channel, 0, 0, :] = float(channel)
        f["values"] = values


def write_particle_frame(path, *, time_seconds, groups=None):
    """Write a minimal ``part*.h5`` frame with ``/time`` in physical seconds."""
    with h5py.File(path, "w") as f:
        f["time"] = time_seconds
        for group, datasets in (groups or {}).items():
            handle = f.create_group(f"groups/{group}")
            for name, values in datasets.items():
                handle[name] = values


def write_macroscopic_times(path, times):
    """Write a minimal ``macroscopic_vars.dat`` with ``@times`` records."""
    lines = ["@jorek_model:   502"]
    lines.extend(f"@times: {int(index)} {float(value):.12E}" for index, value in times.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# A' fine-grained export fixtures (report/data-export-spec-v3.md)
# --------------------------------------------------------------------------- #

SHARD_NAMES = (
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
)

DEFAULT_CONDITION = {
    "names": ("n_spi", "v0", "ang", "eta"),
    "values": (251.0, 400.0, 0.349, 1.443e-10),
    "units": ("1", "m/s", "rad", "-"),
    "groups": ("pellet_mass", "pellet_velocity", "pellet_angle", "excluded_transport"),
    "provenance": ("namelist", "namelist", "namelist", "h5"),
    "film_eligible": (1, 1, 1, 0),
}


def write_export_step(
    path,
    *,
    index: int,
    t_interval_start: float,
    t_interval_end: float,
    t_source_start: float,
    t_source_end: float,
    t_norm: float = 2.0e-4,
    layout: str = "node_dof0",
    n_nodes: int = 6,
    cadence: str = "fluid_step",
    substeps: int = 1,
    source_object: str = "inj_source",
    boundary=None,
    condition=None,
    n_spi_start: int = 3,
    n_spi_end: int = 4,
    state_offset: float = 0.0,
    seconds_keys: bool = True,
    u_after_override_first_channel: float | None = None,
):
    """Write one ``step_{n:06d}.h5`` file in the frozen A' layout."""
    path = Path(path)
    with h5py.File(path, "w") as handle:
        handle.attrs["@step_index"] = np.int64(index)
        handle.attrs["@cadence"] = cadence.encode("utf-8")
        time = handle.create_group("time")
        time["interval_start_jorek"] = np.float64(t_interval_start)
        time["interval_end_jorek"] = np.float64(t_interval_end)
        time["source_interval_start_jorek"] = np.float64(t_source_start)
        time["source_interval_end_jorek"] = np.float64(t_source_end)
        if seconds_keys:
            time["interval_start_seconds"] = np.float64(t_interval_start * t_norm)
            time["interval_end_seconds"] = np.float64(t_interval_end * t_norm)
            time["source_interval_start_seconds"] = np.float64(t_source_start * t_norm)
            time["source_interval_end_seconds"] = np.float64(t_source_end * t_norm)
        time["t_norm"] = np.float64(t_norm)
        time["substeps"] = np.int64(substeps)
        time["substep_group_id"] = np.int64(index)
        handle["provenance/source_object"] = np.bytes_(source_object)
        handle["provenance/jorek_version"] = np.bytes_("test-rcs")
        handle["provenance/model_id"] = np.int32(502)
        handle["provenance/run"] = np.bytes_(path.parent.parent.name)

        state = handle.create_group("state")
        if layout == "full_dof":
            shape = (9, 4, 1, n_nodes)
        else:
            shape = (9, n_nodes)
        u_before = np.zeros(shape, dtype=np.float64)
        u_after = np.zeros(shape, dtype=np.float64)
        for channel in range(9):
            u_before[channel, ...] = float(channel) + state_offset
            u_after[channel, ...] = float(channel) + state_offset
        if u_after_override_first_channel is not None:
            if layout == "full_dof":
                u_after[0, 0, 0, 0] = u_after_override_first_channel
            else:
                u_after[0, 0] = u_after_override_first_channel
        state["u_before"] = u_before
        state["u_after"] = u_after

        source = np.zeros((5, 4, 1, n_nodes) if layout == "full_dof" else (5, n_nodes))
        for channel in range(5):
            source[channel, ...] = float(channel) + state_offset
        handle["source/S_used"] = source

        if boundary is not None:
            mesh = handle.create_group("mesh")
            mesh["boundary"] = np.asarray(boundary, dtype=np.int32)
            mesh["freeboundary"] = np.zeros(n_nodes, dtype=np.int8)
            mesh["vertex"] = np.tile(np.arange(1, 5, dtype=np.int32)[:, None], (1, 1))
        for side, count in (("start", n_spi_start), ("end", n_spi_end)):
            group = handle.create_group(f"shards/{side}")
            for name in SHARD_NAMES:
                values = np.arange(count, dtype=np.float64) + 1.0
                if name == "spi_species_arr":
                    values = np.ones(count, dtype=np.float64)
                group[name] = values
            handle[f"shards/n_spi_tot_{side}"] = np.int32(count)
        if condition is not None:
            group = handle.create_group("condition")
            for key, values in condition.items():
                if key in {"names", "units", "groups", "provenance"}:
                    group[key] = np.asarray([str(item).encode("utf-8") for item in values])
                else:
                    group[key] = np.asarray(values)
    return path


def write_export_v3_run(
    root,
    *,
    indices=(10, 11, 12),
    dt_step: float = 0.5,
    dt_source: float | None = None,
    t0: float = 100.0,
    t_norm: float = 2.0e-4,
    layout: str = "node_dof0",
    n_nodes: int = 6,
    cadence: str = "fluid_step",
    boundary=None,
    condition=DEFAULT_CONDITION,
    state_offset: float = 0.0,
    seconds_keys: bool = True,
    manifest_extra: dict | None = None,
    u_after_override: dict[int, float] | None = None,
):
    """Write ``<root>/export_v3`` with a manifest and one file per index."""
    root = Path(root)
    export = root / "export_v3"
    steps = export / "steps"
    steps.mkdir(parents=True, exist_ok=True)
    dt_source = dt_step if dt_source is None else dt_source
    for index in indices:
        t_start = t0 + (index - indices[0]) * dt_step
        write_export_step(
            steps / f"step_{index:06d}.h5",
            index=index,
            t_interval_start=t_start,
            t_interval_end=t_start + dt_step,
            t_source_start=t_start - dt_source,
            t_source_end=t_start,
            t_norm=t_norm,
            layout=layout,
            n_nodes=n_nodes,
            cadence=cadence,
            boundary=boundary,
            condition=condition,
            state_offset=state_offset,
            seconds_keys=seconds_keys,
            u_after_override_first_channel=(u_after_override or {}).get(index),
        )
    manifest = {
        "run": root.name,
        "layout": layout,
        "cadence": cadence,
        "source_object": "inj_source",
        "n_nodes": n_nodes,
        "jorek_version": "test-rcs",
        "model_id": 502,
    }
    manifest.update(manifest_extra or {})
    (export / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return export


# --------------------------------------------------------------------------- #
# Native stepwise run fixtures (src/data/native_run.py)
# --------------------------------------------------------------------------- #


def write_native_frame(
    path,
    *,
    index: int,
    t_now: float,
    t_norm: float = 2.0e-4,
    tstep: float = 0.5,
    n_nodes: int = 6,
    n_spi: int = 3,
    boundary=None,
    values_offset: float = 0.0,
):
    """Write one native ``jorek{n}.h5`` frame (recorded layout)."""
    with h5py.File(path, "w") as handle:
        handle["t_now"] = np.float64(t_now)
        handle["t_norm"] = np.float64(t_norm)
        handle["tstep"] = np.float64(tstep)
        handle["index_now"] = np.int32(index)
        handle["n_var"] = np.int32(9)
        handle["n_dim"] = np.int32(2)
        handle["n_tor"] = np.int32(1)
        handle["values"] = np.asarray(
            [
                np.full((4, 1, n_nodes), float(channel) + values_offset)
                for channel in range(9)
            ],
            dtype=np.float64,
        )
        handle["x"] = np.zeros((2, 4, 1, n_nodes), dtype=np.float64)
        n_elem = max(1, n_nodes - 1)
        handle["size"] = np.ones((4, 4, n_elem), dtype=np.float64)
        handle["vertex"] = np.tile(
            np.arange(1, 5, dtype=np.int32)[:, None], (1, n_elem)
        )
        handle["contain_node"] = np.ones((5, n_elem), dtype=np.int32)
        handle["neighbours"] = np.ones((4, n_elem), dtype=np.int32)
        handle["boundary"] = np.asarray(
            boundary if boundary is not None else [0] * n_nodes, dtype=np.int32
        )
        handle["freeboundary"] = np.bytes_("F")
        handle["n_spi_tot"] = np.int32(n_spi)
        if n_spi > 0:
            for name in SHARD_NAMES:
                values = np.arange(n_spi, dtype=np.float64) + 1.0
                if name == "spi_species_arr":
                    values = np.ones(n_spi, dtype=np.float64)
                handle[name] = values


def write_native_projection(path, *, t_seconds: float, n_nodes: int = 6):
    """Write one native ``projections{n}.h5`` frame (``t_now`` in seconds)."""
    with h5py.File(path, "w") as handle:
        handle["t_now"] = np.float64(t_seconds)
        handle["n_dim"] = np.int32(2)
        handle["n_nodes"] = np.int32(n_nodes)
        source = np.zeros((5, 4, 1, n_nodes), dtype=np.float64)
        for channel in range(5):
            source[channel, 0, 0, :] = float(channel) + 0.5
        handle["values"] = source


def write_native_run(
    root,
    *,
    indices=(10, 11, 12, 13),
    dt: float = 0.5,
    t0: float = 100.0,
    t_norm: float = 2.0e-4,
    n_nodes: int = 6,
    n_spi: int = 3,
    boundary=None,
    duplicate_frames=(),
    projection_lag: dict | None = None,
    part_lag: dict | None = None,
    part_indices=None,
    leading_projection: bool = True,
    restart_file=None,
    namelist: str | None = None,
):
    """Write a native stepwise run directory (``jorek``/``projections``/``part``).

    ``duplicate_frames`` lists indices whose ``t_now`` repeats the previous
    frame (the code-restart zero-dt artifacts); ``projection_lag`` /
    ``part_lag`` inject the ADR-0019 restart offsets for negative tests.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    indices = list(indices)
    projection_lag = projection_lag or {}
    part_lag = part_lag or {}
    # Fluid-step time table for the whole numbered range (so frames removed by
    # dedup still have a table entry), including the source-window start for the
    # first pair.  Times advance cumulatively: a duplicate frame repeats the
    # previous time and the *next* real step continues from there (exactly the
    # code-restart pattern recorded in the v400 run).
    times = {indices[0] - 1: t0 - dt}
    for index in range(indices[0], indices[-1] + 1):
        if index in duplicate_frames:
            times[index] = times[index - 1]
        else:
            times[index] = times[index - 1] + dt
    for position, index in enumerate(indices):
        following = indices[position + 1] if position + 1 < len(indices) else None
        tstep = (times[following] - times[index]) if following is not None else dt
        write_native_frame(
            root / f"jorek{index:06d}.h5",
            index=index,
            t_now=times[index],
            t_norm=t_norm,
            tstep=tstep,
            n_nodes=n_nodes,
            n_spi=n_spi,
            boundary=boundary,
        )
        write_native_projection(
            root / f"projections{index:06d}.h5",
            t_seconds=(times[index] - projection_lag.get(index, 0.0)) * t_norm,
            n_nodes=n_nodes,
        )
    for event_index in duplicate_frames:
        startup_projection = root / f"projections{event_index - 1:06d}.h5"
        if startup_projection.exists():
            with h5py.File(startup_projection, "a") as handle:
                handle["values"][...] = 0.0
    if leading_projection:
        write_native_projection(
            root / f"projections{indices[0] - 1:06d}.h5",
            t_seconds=times[indices[0] - 1] * t_norm,
            n_nodes=n_nodes,
        )
    if part_indices is None:
        part_indices = indices
    for index in part_indices:
        path = root / f"part{index:05d}.h5"
        time_seconds = (
            times[index - 1] - part_lag.get(index, 0.0)
        ) * t_norm
        write_particle_frame(
            path,
            time_seconds=time_seconds,
            groups={"001": {"weight": [1.0, 2.0]}},
        )
    write_macroscopic_times(root / "macroscopic_vars.dat", times)
    if restart_file is not None:
        lines = [
            f"/remote/run/jorek{index:06d}.h5" for index in restart_file
        ]
        (root / "available_restart_files.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
    if namelist is not None:
        (root / "jorek_namelist_spitest").write_text(namelist, encoding="utf-8")
    return root
