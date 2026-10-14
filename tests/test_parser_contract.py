"""Parser contract tests for the locked EHL-2 2D shape.

These tests pin the seam agreed in the spec: the parser consumes Model 502 HDF5
frames and returns canonical ``[nodes, 9]`` state tensors.

The v1 ``JorekSequenceDataset`` alignment case was replaced in Phase A by the
ADR-0019 paired loader tests in ``tests/test_paired_dataset.py``.
"""

from __future__ import annotations

import pytest

from conftest import write_model502_frame
from src.data.jorek_h5_parser import JorekDataParser


def test_parser_rejects_multi_harmonic_values(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    write_model502_frame(frame, harmonics=2)

    with pytest.raises(ValueError, match="harmonic"):
        JorekDataParser(str(frame)).parse_to_graph()


def test_parser_accepts_single_harmonic_and_returns_canonical_state(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    write_model502_frame(frame)

    graph = JorekDataParser(str(frame)).parse_to_graph()

    assert graph.x.shape == (6, 9)
    # The fixture writes raw channel c = c, and the H5 channel order is the
    # canonical order (identity mapping), so psi=0.0 and Z_j=2.0.
    assert graph.x[0, 0].item() == pytest.approx(0.0)
    assert graph.x[0, 2].item() == pytest.approx(2.0)
