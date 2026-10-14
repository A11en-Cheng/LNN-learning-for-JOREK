"""ADR-0019 recorded-output timing contract (``src/data/output_contract.py``).

Synthetic fixtures pin the failure modes the loader must catch; one real-data
test per behaviour runs only when the EHL-2 symlink target is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import (
    write_macroscopic_times,
    write_model502_frame,
    write_particle_frame,
    write_projection_frame,
)
from src.data import output_contract as oc

DT_FLUID = 0.5
T_NORM = 2.0e-4
T0 = 100.0
INDICES = (10, 12, 14, 16)

RUN1 = Path("data/EHL-2/scan_1d22_ang12_massloss_neon")
real_data = pytest.mark.skipif(
    not RUN1.exists(), reason="EHL-2 run data not available in this environment"
)


def base_times() -> dict[int, float]:
    """One fluid step every ``DT_FLUID``; frames are dumped every 2 steps."""
    return {index: T0 + (index - 10) * DT_FLUID for index in range(9, 17)}


def make_run(
    tmp_path: Path,
    *,
    indices=INDICES,
    times: dict[int, float] | None = None,
    projection_lag: dict[int, float] | None = None,
    part_lag: dict[int, float] | None = None,
    macro: bool = True,
    t_norm: float = T_NORM,
) -> Path:
    times = dict(times if times is not None else base_times())
    projection_lag = projection_lag or {}
    part_lag = part_lag or {}
    for index in indices:
        t_state = times[index]
        write_model502_frame(
            tmp_path / f"jorek{index:06d}.h5", t_now=t_state, t_norm=t_norm
        )
        write_projection_frame(
            tmp_path / f"projections{index:06d}.h5",
            t_seconds=(t_state - projection_lag.get(index, 0.0)) * t_norm,
        )
        write_particle_frame(
            tmp_path / f"part{index:05d}.h5",
            time_seconds=(t_state - DT_FLUID - part_lag.get(index, 0.0)) * t_norm,
            groups={"001": {"weight": [1.0, 2.0, 3.0]}},
        )
    if macro:
        write_macroscopic_times(tmp_path / "macroscopic_vars.dat", times)
    return tmp_path


def test_synthetic_run_satisfies_contract(tmp_path):
    make_run(tmp_path)

    report = oc.check_run_contract(tmp_path)

    assert report.step_table_source == "macroscopic_vars.dat"
    assert report.n_state_frames == 4
    assert report.n_marker_frames == 4
    assert report.pair_indices == (10, 12, 14)
    assert report.dropped_restart == ()
    assert report.issues == ()
    first = report.pair_records[0]
    assert first.index_next == 12
    assert first.t_state == pytest.approx(100.0)
    assert first.t_next == pytest.approx(101.0)
    assert first.dt_step == pytest.approx(1.0)
    assert first.dt_source == pytest.approx(DT_FLUID)


def test_unregistered_projection_offset_is_fatal(tmp_path):
    make_run(tmp_path, projection_lag={12: DT_FLUID})

    with pytest.raises(oc.ContractError, match="offset"):
        oc.check_run_contract(tmp_path)


def test_registered_restart_boundary_is_dropped(tmp_path):
    make_run(tmp_path, projection_lag={12: DT_FLUID})

    report = oc.check_run_contract(tmp_path, restart_indices=(12,))

    assert report.dropped_restart == (12,)
    assert report.pair_indices == (10, 14)
    assert [issue.kind for issue in report.issues] == [
        "restart_boundary_projection_offset"
    ]
    assert report.issues[0].index == 12


def test_registered_restart_boundary_can_be_fatal(tmp_path):
    make_run(tmp_path, projection_lag={12: DT_FLUID})

    with pytest.raises(oc.ContractError, match="restart"):
        oc.check_run_contract(tmp_path, restart_indices=(12,), policy="error")


def test_registered_restart_boundary_must_match_one_fluid_step(tmp_path):
    # A registered index with an offset of half a fluid step is still a hard
    # error: registration excuses the known offset, not arbitrary offsets.
    make_run(tmp_path, projection_lag={12: DT_FLUID / 2})

    with pytest.raises(oc.ContractError, match="does not match"):
        oc.check_run_contract(tmp_path, restart_indices=(12,))


def test_part_time_must_equal_state_minus_fluid_step(tmp_path):
    # Shift every part file back by one extra fluid step: this is the failure
    # mode of "part = T_k - dump_dt" instead of "part = T_k - dt_source".
    make_run(tmp_path, part_lag={index: DT_FLUID for index in INDICES})

    with pytest.raises(oc.ContractError, match="part time"):
        oc.check_run_contract(tmp_path)


def test_partial_marker_pair_is_fatal(tmp_path):
    make_run(tmp_path)
    (tmp_path / "projections000014.h5").unlink()

    with pytest.raises(oc.ContractError, match="partial marker pairing"):
        oc.check_run_contract(tmp_path)


def test_unforced_frame_without_markers_is_allowed(tmp_path):
    make_run(tmp_path, indices=(10, 12, 14))
    write_model502_frame(tmp_path / "jorek000016.h5", t_now=103.0, t_norm=T_NORM)

    report = oc.check_run_contract(tmp_path)

    assert report.n_state_frames == 4
    assert report.n_marker_frames == 3
    assert report.pair_indices == (10, 12, 14)


def test_missing_step_table_is_reported_not_hidden(tmp_path):
    make_run(tmp_path, macro=False)

    report = oc.check_run_contract(tmp_path)

    assert report.step_table_source == "absent"
    assert {issue.kind for issue in report.issues} == {"part_window_unverified"}
    assert report.pair_indices == (10, 12, 14)


def test_explicit_step_times_make_part_assertion_strict(tmp_path):
    make_run(tmp_path, macro=False)

    report = oc.check_run_contract(tmp_path, step_times=base_times())

    assert report.step_table_source == "explicit"
    assert report.issues == ()


def test_non_monotonic_state_times_are_fatal(tmp_path):
    times = base_times()
    times[12] = times[10]
    make_run(tmp_path, times=times)

    with pytest.raises(oc.ContractError, match="strictly increasing"):
        oc.check_run_contract(tmp_path)


def test_parse_macroscopic_times(tmp_path):
    path = tmp_path / "macroscopic_vars.dat"
    write_macroscopic_times(path, {0: 1.0, 1: 1.5, 2: 2.25})

    assert oc.parse_macroscopic_times(path) == {
        0: pytest.approx(1.0),
        1: pytest.approx(1.5),
        2: pytest.approx(2.25),
    }


def test_assert_strictly_increasing_helper():
    oc.assert_strictly_increasing([1.0, 2.0, 3.0], what="test")
    with pytest.raises(oc.ContractError):
        oc.assert_strictly_increasing([1.0, 1.0], what="test")


@real_data
def test_real_run_window_satisfies_contract():
    indices = list(range(2700, 2791, 10))

    report = oc.check_run_contract(RUN1, indices=indices)

    assert report.step_table_source == "macroscopic_vars.dat"
    # projections start at 2700 but part files start at 2710: the leading
    # projection-only frame is a legitimate coverage edge, not a gap.
    assert report.n_marker_frames == len(indices) - 1
    assert report.pair_indices == tuple(indices[1:-1])
    assert report.dropped_restart == ()
    assert [issue.kind for issue in report.issues] == ["unpaired_boundary_frame"]
    assert report.issues[0].index == 2700


@real_data
def test_real_restart_boundary_is_registered_and_dropped():
    report = oc.check_run_contract(RUN1, indices=[3540, 3550, 3560])

    assert report.dropped_restart == (3550,)
    assert report.pair_indices == (3540,)
    offset_issue = next(
        issue
        for issue in report.issues
        if issue.kind == "restart_boundary_projection_offset"
    )
    assert offset_issue.index == 3550
    assert "-1.0" in offset_issue.detail
