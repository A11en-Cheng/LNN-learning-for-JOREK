from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.data.jorek_h5_parser import JorekDataParser
from src.data.model502_schema import NUM_SOURCE_FIELDS, NUM_STATE_FIELDS


def _write_minimal_model502(path: Path, source_layout: str):
    with h5py.File(path, "w") as h5:
        h5["x"] = np.array([[[[1.0, 2.0, 3.0]]], [[[0.0, 0.5, 1.0]]]])
        h5["values"] = np.arange(NUM_STATE_FIELDS * 3, dtype=np.float32).reshape(NUM_STATE_FIELDS, 1, 1, 3)
        h5["contain_node"] = np.array([[1], [2], [3]], dtype=np.int32)
        h5["boundary"] = np.array([0, 0, 1], dtype=np.int32)
        h5["t_now"] = np.array([0.1], dtype=np.float64)
        if source_layout == "vector":
            h5["sources"] = np.ones((NUM_SOURCE_FIELDS, 1, 1, 3), dtype=np.float32)
        elif source_layout == "partial":
            h5["S_rho"] = np.ones(3, dtype=np.float32)


def test_parser_reads_complete_vector_source_contract(tmp_path):
    path = tmp_path / "model502.h5"
    _write_minimal_model502(path, "vector")
    graph = JorekDataParser(str(path), require_source=True).parse_to_graph()
    assert graph.x.shape == (3, NUM_STATE_FIELDS)
    assert graph.source.shape == (3, NUM_SOURCE_FIELDS)
    assert bool(graph.source_available)


def test_parser_rejects_partial_source_contract(tmp_path):
    path = tmp_path / "partial_model502.h5"
    _write_minimal_model502(path, "partial")
    with pytest.raises(ValueError, match="Partial Model 502 source record"):
        JorekDataParser(str(path)).parse_to_graph()
