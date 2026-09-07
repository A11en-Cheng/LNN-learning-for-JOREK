"""Smoke tests for derived-field reconstruction and residuals."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.jorek_h5_parser import JorekDataParser
from src.physics.model502_physics import model502_residuals, reconstruct_derived


def test_reconstruction_is_finite_on_real_frame():
    path = "data/EHL-2/scan_1d22_ang12_massloss_neon/jorek002710.h5"
    if not __import__("pathlib").Path(path).exists():
        pytest.skip("real data not present")
    graph = JorekDataParser(path).parse_to_graph()
    z_j, omega = reconstruct_derived(
        graph.x[:, 0], graph.x[:, 1], graph.pos, graph.edge_index, graph.edge_attr
    )
    assert torch_isfinite(z_j).all()
    assert torch_isfinite(omega).all()


def test_residuals_are_finite_on_real_frame():
    path = "data/EHL-2/scan_1d22_ang12_massloss_neon/jorek002710.h5"
    if not __import__("pathlib").Path(path).exists():
        pytest.skip("real data not present")
    graph = JorekDataParser(path).parse_to_graph()
    residuals = model502_residuals(graph.x, graph.x, graph, dt=10.0)
    assert all(np.isfinite(v) for v in residuals.values())
    assert set(residuals) >= {
        "induction", "vorticity", "density", "ion_pressure",
        "electron_pressure", "parallel_momentum", "impurity",
    }


def torch_isfinite(tensor):
    import torch

    return torch.isfinite(tensor)

