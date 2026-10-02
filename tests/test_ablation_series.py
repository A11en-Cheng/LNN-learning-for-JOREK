"""Tests for the aggregate SPI ablation series reader."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.ablation_series import frame_totals, read_ablation_series


def _write_series(path: Path, *, layout: str = "steps_first") -> None:
    steps = 5
    times = np.arange(steps, dtype=np.float64) * 10.0
    cumulative = np.cumsum(np.arange(1, steps + 1, dtype=np.float64))
    rate = np.arange(1, steps + 1, dtype=np.float64) * 100.0
    with h5py.File(path, "w") as handle:
        handle.create_dataset("xtime", data=times)
        if layout == "steps_first":
            handle.create_dataset("xtime_spi_ablation", data=cumulative.reshape(-1, 1))
            handle.create_dataset("xtime_spi_ablation_rate", data=rate.reshape(-1, 1))
        else:  # per-shard style layout (2 shards, steps)
            handle.create_dataset(
                "xtime_spi_ablation", data=np.stack([cumulative / 2, cumulative / 2])
            )
            handle.create_dataset(
                "xtime_spi_ablation_rate", data=np.stack([rate / 2, rate / 2])
            )
        handle.create_dataset("index_now", data=np.array([steps]))


def test_series_queries_and_increment(tmp_path):
    path = tmp_path / "jorek000000.h5"
    _write_series(path)
    series = read_ablation_series(path)
    assert series is not None
    assert series.times.tolist() == [0.0, 10.0, 20.0, 30.0, 40.0]
    assert series.cumulative.tolist() == [1.0, 3.0, 6.0, 10.0, 15.0]
    assert series.increment(0.0, 20.0) == pytest.approx(5.0)
    assert series.cumulative_at(25.0) == pytest.approx(6.0)


def test_per_shard_layout_is_summed(tmp_path):
    path = tmp_path / "jorek000000.h5"
    _write_series(path, layout="shards_first")
    series = read_ablation_series(path)
    assert series is not None
    assert series.cumulative[-1] == pytest.approx(15.0)
    assert series.rate_total[-1] == pytest.approx(500.0)


def test_frame_totals_return_last_entry(tmp_path):
    path = tmp_path / "jorek000000.h5"
    _write_series(path)
    totals = frame_totals(path)
    assert totals == pytest.approx((40.0, 15.0, 500.0))


def test_missing_ablation_series_returns_none(tmp_path):
    path = tmp_path / "jorek000000.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("t_now", data=np.array([1.0]))
    assert read_ablation_series(path) is None


def test_real_frame_totals_match_shard_sum():
    frame = Path("data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek003000.h5")
    if not frame.exists():
        pytest.skip("real EHL-2 frame not available")
    totals = frame_totals(frame)
    assert totals is not None
    with h5py.File(frame, "r") as handle:
        shard_sum = float(np.sum(handle["spi_abl_arr"][:]))
    assert totals[2] == pytest.approx(shard_sum, rel=1e-6)
