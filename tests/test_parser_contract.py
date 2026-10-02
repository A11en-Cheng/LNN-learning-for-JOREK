"""Parser/dataset contract tests for the locked EHL-2 2D shape.

These tests pin the seam agreed in the spec: the parser consumes Model 502 HDF5
frames and returns canonical ``[nodes, 9]`` state tensors, and the sequence
dataset only treats strictly named, marker-aligned frames as samples.
"""

from __future__ import annotations

import h5py
import pytest

from conftest import write_model502_frame
from src.data.dataset import JorekSequenceDataset
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


def test_dataset_uses_strict_names_and_marker_alignment(tmp_path):
    write_model502_frame(tmp_path / "jorek00001.h5")
    write_model502_frame(tmp_path / "jorek00002.h5")
    write_model502_frame(tmp_path / "jorek00003.h5")
    (tmp_path / "jorek_restart.h5").write_bytes(b"not a frame")
    for index in (2, 3):
        with h5py.File(tmp_path / f"part{index:05d}.h5", "w") as f:
            f["time"] = [float(index)]

    aligned = JorekSequenceDataset(root_dir=str(tmp_path), seq_len=1, aligned_only=True)
    all_frames = JorekSequenceDataset(root_dir=str(tmp_path), seq_len=1, aligned_only=False)

    assert [p.rsplit("/", 1)[-1] for p in aligned.sorted_files] == [
        "jorek00002.h5",
        "jorek00003.h5",
    ]
    assert len(all_frames.sorted_files) == 3
    assert len(aligned) == 2
