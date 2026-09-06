"""Shared test fixtures for the Model 502 data contract."""

from __future__ import annotations

import h5py
import numpy as np


def write_model502_frame(
    path,
    *,
    t_now=0.0,
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
        f["boundary"] = np.arange(n_nodes, dtype=np.int64)
        f["contain_node"] = np.array(
            [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]], dtype=np.int32
        ).T
        f["x"] = np.zeros((2, 4, 1, n_nodes), dtype=np.float64)
        f["x"][0, 0, 0, :] = np.arange(n_nodes)
        f["x"][1, 0, 0, :] = np.arange(n_nodes) * 2
        values = np.zeros((9, 4, harmonics, n_nodes), dtype=np.float64)
        for channel in range(9):
            values[channel, 0, 0, :] = float(channel)
        if rho_values is not None:
            values[4, 0, 0, :] = rho_values
        f["values"] = values

