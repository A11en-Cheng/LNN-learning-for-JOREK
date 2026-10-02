"""Tests for the Phase-1 fragment state readers."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.fragment_state import (
    SHARD_FIELD_NAMES,
    read_marker_summary,
    read_shard_state,
)
from tests.synthetic_frames import write_frame


def test_shard_state_reads_arrays_and_alive_mask(tmp_path):
    write_frame(tmp_path, 0, t_now=10.0, n_spi=3)
    state = read_shard_state(tmp_path / "jorek000000.h5")
    assert state is not None
    assert state.n_spi == 3
    assert state.values.shape == (3, len(SHARD_FIELD_NAMES))
    # The synthetic writer zeroes the last radius, i.e. one dead shard.
    assert state.n_alive == 2
    # Optional drift arrays are absent in the synthetic frame -> NaN.
    vol_drift = state.values[:, SHARD_FIELD_NAMES.index("vol_drift")]
    assert np.isnan(vol_drift).all()


def test_shard_state_is_none_before_injection(tmp_path):
    write_frame(tmp_path, 0, t_now=10.0, n_spi=0)
    assert read_shard_state(tmp_path / "jorek000000.h5") is None


def test_marker_summary_counts_alive_and_weights(tmp_path):
    write_frame(tmp_path, 0, t_now=10.0, marker_particles=8, marker_alive=4)
    summary = read_marker_summary(tmp_path / "part000000.h5", stride=1)
    assert summary.n_alive == 4
    assert len(summary.groups) == 1
    group = summary.groups[0]
    assert group.charge == pytest.approx(10.0)
    assert group.mass == pytest.approx(20.1797)
    assert np.isfinite(group.mean_R)
    assert group.mean_VR == pytest.approx(-300.0)
    assert summary.total_weight == pytest.approx(4.0)
