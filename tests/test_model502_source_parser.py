"""Protocol-v2 source contract: marker projections + background sources."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from src.data.jorek_h5_parser import JorekDataParser
from src.data.model502_schema import (
    MARKER_PROJECTION_FIELD_NAMES,
    NUM_BACKGROUND_SOURCE_FIELDS,
    NUM_MARKER_PROJECTION_FIELDS,
    NUM_STATE_FIELDS,
)


T_NORM = 4.0e-7


def _write_frame(path: Path, *, t_now: float, source_layout: str = "none"):
    nodes = 3
    with h5py.File(path, "w") as h5:
        h5["x"] = np.array([[[[1.0, 2.0, 3.0]]], [[[0.0, 0.5, 1.0]]]])
        h5["values"] = np.arange(NUM_STATE_FIELDS * 4 * nodes, dtype=np.float32).reshape(
            NUM_STATE_FIELDS, 4, 1, nodes
        )
        h5["contain_node"] = np.array([[1], [2], [3]], dtype=np.int32)
        h5["boundary"] = np.array([0, 0, 1], dtype=np.int32)
        h5["t_now"] = np.array([t_now], dtype=np.float64)
        h5["t_norm"] = np.array([T_NORM], dtype=np.float64)
        h5["n_tor"] = 1
        h5["n_dim"] = 2
        h5["jorek_model"] = 502
        if source_layout == "vector":
            h5["sources"] = np.ones((NUM_BACKGROUND_SOURCE_FIELDS, 1, 1, nodes), dtype=np.float32)
        elif source_layout == "partial":
            h5["S_rho"] = np.ones(nodes, dtype=np.float32)


def _write_projection(path: Path, *, t_now: float, harmonics: int = 1, nodes: int = 3):
    with h5py.File(path, "w") as p:
        p["n_var"] = NUM_MARKER_PROJECTION_FIELDS
        p["t_now"] = np.array([t_now], dtype=np.float64)
        p["values"] = np.arange(
            NUM_MARKER_PROJECTION_FIELDS * 4 * harmonics * nodes, dtype=np.float32
        ).reshape(NUM_MARKER_PROJECTION_FIELDS, 4, harmonics, nodes)


def test_parser_reads_background_sources_into_second_channel(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    _write_frame(frame, t_now=10.0, source_layout="vector")

    graph = JorekDataParser(str(frame)).parse_to_graph()

    assert graph.x.shape == (3, NUM_STATE_FIELDS)
    assert graph.background_source.shape == (3, NUM_BACKGROUND_SOURCE_FIELDS)
    assert not hasattr(graph, "source")
    assert not bool(graph.source_available)


def test_parser_rejects_partial_background_record(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    _write_frame(frame, t_now=10.0, source_layout="partial")

    with pytest.raises(ValueError, match="Partial Model 502 background source record"):
        JorekDataParser(str(frame)).parse_to_graph()


def test_parser_reads_marker_projection_with_time_alignment(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    _write_frame(frame, t_now=10.0)
    _write_projection(tmp_path / "projections00001.h5", t_now=10.0 * T_NORM)

    graph = JorekDataParser(str(frame), require_source=True).parse_to_graph()

    assert graph.source.shape == (3, NUM_MARKER_PROJECTION_FIELDS)
    assert graph.source_fields == MARKER_PROJECTION_FIELD_NAMES
    assert bool(graph.source_available)


def test_parser_requires_projection_when_require_source(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    _write_frame(frame, t_now=10.0)

    with pytest.raises(ValueError, match="marker projection"):
        JorekDataParser(str(frame), require_source=True).parse_to_graph()


def test_parser_rejects_misaligned_projection_time(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    _write_frame(frame, t_now=10.0)
    _write_projection(tmp_path / "projections00001.h5", t_now=10.0 * T_NORM * 2.0)

    with pytest.raises(ValueError, match="time"):
        JorekDataParser(str(frame)).parse_to_graph()


def test_parser_rejects_multi_harmonic_projection(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    _write_frame(frame, t_now=10.0)
    _write_projection(tmp_path / "projections00001.h5", t_now=10.0 * T_NORM, harmonics=2)

    with pytest.raises(ValueError, match="harmonic"):
        JorekDataParser(str(frame)).parse_to_graph()


def test_parser_rejects_projection_with_wrong_node_count(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    _write_frame(frame, t_now=10.0)
    _write_projection(tmp_path / "projections00001.h5", t_now=10.0 * T_NORM, nodes=5)

    with pytest.raises(ValueError, match="node"):
        JorekDataParser(str(frame)).parse_to_graph()


@pytest.mark.skipif(
    not Path("data/EHL-2/scan_1d22_ang12_massloss_neon").exists(),
    reason="real EHL-2 data not present",
)
def test_real_frame_loads_marker_projections():
    frame = Path("data/EHL-2/scan_1d22_ang12_massloss_neon/jorek002710.h5")

    graph = JorekDataParser(str(frame), require_source=True).parse_to_graph()

    assert graph.source.shape == (4125, NUM_MARKER_PROJECTION_FIELDS)
    assert graph.source_fields == MARKER_PROJECTION_FIELD_NAMES
    assert bool(graph.source_available)
    assert graph.source.isfinite().all()
