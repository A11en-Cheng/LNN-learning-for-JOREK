"""ADR-0019 paired loader tests (``src/data/paired_dataset.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import (
    write_macroscopic_times,
    write_model502_frame,
    write_particle_frame,
    write_projection_frame,
)
from src.data.output_contract import ContractError, RESTART_BOUNDARY_INDICES
from src.data.paired_dataset import PairedRunDataset

DT_FLUID = 0.5
T_NORM = 2.0e-4
T0 = 100.0
INDICES = (10, 12, 14, 16)

RUN1 = Path("data/EHL-2/scan_1d22_ang12_massloss_neon")
real_data = pytest.mark.skipif(
    not RUN1.exists(), reason="EHL-2 run data not available in this environment"
)


def base_times() -> dict[int, float]:
    return {index: T0 + (index - 10) * DT_FLUID for index in range(9, 17)}


def make_run(
    tmp_path: Path,
    *,
    indices=INDICES,
    times: dict[int, float] | None = None,
    projection_lag: dict[int, float] | None = None,
) -> Path:
    times = dict(times if times is not None else base_times())
    projection_lag = projection_lag or {}
    for index in indices:
        t_state = times[index]
        write_model502_frame(tmp_path / f"jorek{index:06d}.h5", t_now=t_state, t_norm=T_NORM)
        write_projection_frame(
            tmp_path / f"projections{index:06d}.h5",
            t_seconds=(t_state - projection_lag.get(index, 0.0)) * T_NORM,
        )
        write_particle_frame(
            tmp_path / f"part{index:05d}.h5",
            time_seconds=(t_state - DT_FLUID) * T_NORM,
            groups={"001": {"weight": [1.0, 2.0, 3.0]}},
        )
    write_macroscopic_times(tmp_path / "macroscopic_vars.dat", times)
    return tmp_path


def test_synthetic_pairs_follow_adr_0019_semantics(tmp_path):
    make_run(tmp_path)

    dataset = PairedRunDataset(tmp_path)

    assert len(dataset) == 3
    assert dataset.pair_indices() == (10, 12, 14)
    sample = dataset[0]
    assert sample.index == 10
    assert sample.u_before.shape == (6, 9)
    assert sample.source.shape == (6, 5)
    # The fixture writes raw channel c = c, so the identity mapping is visible.
    assert sample.u_before[0, 2] == pytest.approx(2.0)
    assert sample.source[0, 3] == pytest.approx(3.0)
    assert sample.u_after[0, 8] == pytest.approx(8.0)
    assert sample.t_before == pytest.approx(100.0)
    assert sample.t_after == pytest.approx(101.0)
    assert sample.dt_step == pytest.approx(1.0)
    assert sample.dt_source == pytest.approx(DT_FLUID)
    assert sample.source_window == pytest.approx((99.5, 100.0))
    assert sample.field_interval == pytest.approx((100.0, 101.0))
    assert sample.p_start.index == 10
    assert sample.p_start.path.name == "part00010.h5"
    assert sample.p_start.time_seconds == pytest.approx(99.5 * T_NORM)
    assert sample.p_start.groups == ("001",)
    assert sample.p_start.total_weight("001") == pytest.approx(6.0)


def test_dataset_fails_fast_on_non_monotonic_time(tmp_path):
    times = base_times()
    times[12] = times[10]
    make_run(tmp_path, times=times)

    with pytest.raises(ContractError, match="strictly increasing"):
        PairedRunDataset(tmp_path)


def test_dataset_fails_fast_on_missing_projection(tmp_path):
    make_run(tmp_path)
    (tmp_path / "projections000012.h5").unlink()

    with pytest.raises(ContractError, match="partial marker pairing"):
        PairedRunDataset(tmp_path)


def test_dataset_drops_registered_restart_boundary(tmp_path):
    make_run(tmp_path, projection_lag={12: DT_FLUID})

    dataset = PairedRunDataset(tmp_path, restart_indices=(12,))

    assert dataset.dropped_restart == (12,)
    assert dataset.pair_indices() == (10, 14)


def test_dataset_index_window(tmp_path):
    make_run(tmp_path)

    dataset = PairedRunDataset(tmp_path, index_window=(12, 16))

    assert dataset.pair_indices() == (12, 14)


@real_data
def test_real_pairs_match_recorded_times():
    dataset = PairedRunDataset(RUN1, index_window=(7360, 7380))

    assert dataset.pair_indices() == (7360, 7370)
    sample = dataset[0]
    assert dataset.n_nodes == 4125
    assert sample.u_before.shape == (4125, 9)
    assert sample.source.shape == (4125, 5)
    assert np.isfinite(sample.u_before).all()
    assert np.isfinite(sample.source).all()
    # Frame dumps are 10 fluid steps apart in this run.
    assert sample.dt_step == pytest.approx(0.5, abs=1e-6)
    assert sample.dt_source == pytest.approx(0.05, abs=1e-6)
    assert sample.source_window[1] == pytest.approx(sample.t_before)
    assert sample.p_start.time_seconds == pytest.approx(
        (sample.t_before - sample.dt_source) * sample.t_norm, abs=1e-9
    )


@real_data
def test_real_restart_boundary_is_dropped_by_loader():
    dataset = PairedRunDataset(RUN1, index_window=(3540, 3560))

    assert dataset.dropped_restart == (3550,)
    assert dataset.pair_indices() == (3540,)


def test_registry_lists_the_documented_restart_boundaries():
    assert RESTART_BOUNDARY_INDICES["scan_1d22_ang12_massloss_neon"] == (
        3550,
        4150,
        4950,
        5330,
        6600,
        7350,
    )
    assert RESTART_BOUNDARY_INDICES["scan_1d22_ang15_5_massloss_neon_v400"] == (5050,)
