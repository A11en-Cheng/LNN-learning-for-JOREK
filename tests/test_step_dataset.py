"""A' fine-grained export loader tests (``src/data/step_dataset.py``).

Synthetic fixtures pin the fail-fast contract; the real-export test is
skip-if-absent because the generation side has not delivered an A' run yet.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import (
    DEFAULT_CONDITION,
    SHARD_NAMES,
    write_export_step,
    write_export_v3_run,
    write_macroscopic_times,
    write_model502_frame,
    write_native_run,
    write_particle_frame,
    write_projection_frame,
)
from src.data.output_contract import ContractError
from src.data.step_dataset import StepRunDataset, detect_run_layout, resolve_export_root

T_NORM = 2.0e-4
T0 = 100.0


def test_synthetic_node_dof0_contract(tmp_path):
    export = write_export_v3_run(
        tmp_path,
        indices=(10, 11, 12),
        boundary=[0] * 6,
    )

    dataset = StepRunDataset(export, require_single_step=True)

    assert len(dataset) == 3
    assert dataset.indices == (10, 11, 12)
    assert dataset.pair_indices() == (10, 11)
    assert dataset.layout == "node_dof0"
    assert dataset.report.dt_step_median == pytest.approx(0.5)
    assert dataset.report.dt_source_median == pytest.approx(0.5)
    sample = dataset[0]
    assert sample.u_before.shape == (6, 9)
    assert sample.source.shape == (6, 5)
    assert sample.u_after.shape == (6, 9)
    assert sample.u_before[0, 2] == pytest.approx(2.0)
    assert sample.source[0, 3] == pytest.approx(3.0)
    assert sample.shard_start.n_total == 3
    assert sample.shard_end.n_total == 4
    assert set(sample.shard_start.arrays) == set(SHARD_NAMES)
    assert sample.source_window == pytest.approx((99.5, 100.0))
    assert sample.field_interval == pytest.approx((100.0, 100.5))
    assert sample.condition is not None
    assert sample.condition.groups[:2] == ("pellet_mass", "pellet_velocity")
    assert sample.t_norm == pytest.approx(T_NORM)


def test_synthetic_full_dof_layout(tmp_path):
    export = write_export_v3_run(tmp_path, layout="full_dof", boundary=[0] * 6)

    dataset = StepRunDataset(export, require_single_step=True)
    sample = dataset[0]

    assert dataset.layout == "full_dof"
    assert sample.u_before.shape == (6, 9)
    assert sample.source.shape == (6, 5)


def test_require_single_step_rejects_frame_level_record(tmp_path):
    export = write_export_v3_run(
        tmp_path,
        indices=(10, 12),
        dt_step=1.0,
        dt_source=0.5,
        boundary=[0] * 6,
        manifest_extra={"gap_indices": [11]},
    )

    with pytest.raises(ContractError, match="require_single_step"):
        StepRunDataset(export, require_single_step=True)


def test_allow_multi_step_mode_registers_frame_level_record(tmp_path):
    export = write_export_v3_run(
        tmp_path,
        indices=(10, 12),
        dt_step=1.0,
        dt_source=0.5,
        boundary=[0] * 6,
        manifest_extra={"gap_indices": [11]},
    )

    dataset = StepRunDataset(export, require_single_step=False)

    assert dataset.report.require_single_step is False
    assert dataset.report.dt_step_median == pytest.approx(1.0)
    assert dataset.report.dt_source_median == pytest.approx(0.5)


def test_source_window_must_end_at_interval_start(tmp_path):
    export = write_export_v3_run(
        tmp_path, indices=(10, 11), boundary=[0] * 6, seconds_keys=False
    )
    step = export / "steps" / "step_000010.h5"
    with h5py.File(step, "r+") as handle:
        handle["time/source_interval_end_jorek"][()] = 99.75

    with pytest.raises(ContractError, match="source window end"):
        StepRunDataset(export)


def test_missing_shard_group_is_fatal(tmp_path):
    export = write_export_v3_run(tmp_path, indices=(10, 11))
    step = export / "steps" / "step_000010.h5"
    with h5py.File(step, "r+") as handle:
        del handle["shards/start/spi_abl_arr"]

    with pytest.raises(ContractError, match="spi_abl_arr"):
        StepRunDataset(export)


def test_shard_length_mismatch_is_fatal(tmp_path):
    export = write_export_v3_run(tmp_path, indices=(10, 11))
    step = export / "steps" / "step_000010.h5"
    with h5py.File(step, "r+") as handle:
        del handle["shards/n_spi_tot_start"]
        handle["shards/n_spi_tot_start"] = np.int32(99)

    with pytest.raises(ContractError, match="n_spi_tot_start"):
        StepRunDataset(export)


def test_empty_shard_arrays_are_allowed(tmp_path):
    # Pre-injection window ends legitimately have n_spi_tot=0; the A' file must
    # still carry the group keys (length-0 arrays), which the loader accepts.
    export = write_export_v3_run(tmp_path, indices=(10, 11), boundary=[0] * 6)
    step = export / "steps" / "step_000010.h5"
    with h5py.File(step, "r+") as handle:
        for side in ("start", "end"):
            for name in SHARD_NAMES:
                del handle[f"shards/{side}/{name}"]
                handle[f"shards/{side}/{name}"] = np.zeros(0, dtype=np.float64)
            del handle[f"shards/n_spi_tot_{side}"]
            handle[f"shards/n_spi_tot_{side}"] = np.int32(0)

    dataset = StepRunDataset(export)

    sample = dataset[0]
    assert sample.shard_start.n_total == 0
    assert sample.shard_end.n_total == 0
    assert sample.shard_start["spi_R_arr"].shape == (0,)


def test_state_shapes_must_match(tmp_path):
    export = write_export_v3_run(tmp_path, indices=(10, 11))
    step = export / "steps" / "step_000010.h5"
    with h5py.File(step, "r+") as handle:
        del handle["state/u_after"]
        handle["state/u_after"] = np.zeros((9, 5), dtype=np.float64)

    with pytest.raises(ContractError, match="node count|nodes"):
        StepRunDataset(export)


def test_unregistered_boundary_code_is_fatal(tmp_path):
    boundary = [0] * 5 + [99]
    export = write_export_v3_run(tmp_path, boundary=boundary)

    with pytest.raises(ContractError, match="unregistered boundary codes"):
        StepRunDataset(export)


def test_missing_boundary_is_registered_warning(tmp_path):
    export = write_export_v3_run(tmp_path, indices=(10, 11))

    dataset = StepRunDataset(export)

    kinds = {issue.kind for issue in dataset.issues}
    assert "boundary_missing" in kinds
    assert all(issue.kind != "condition_missing" for issue in dataset.issues)


def test_illegal_condition_group_is_fatal(tmp_path):
    condition = dict(DEFAULT_CONDITION)
    condition["groups"] = ("pellet_mass", "transport", "pellet_angle", "excluded_transport")
    export = write_export_v3_run(tmp_path, boundary=[0] * 6, condition=condition)

    with pytest.raises(ContractError, match="vocabulary"):
        StepRunDataset(export)


def test_excluded_condition_cannot_be_film_eligible(tmp_path):
    condition = dict(DEFAULT_CONDITION)
    condition["film_eligible"] = (1, 1, 1, 1)
    export = write_export_v3_run(tmp_path, boundary=[0] * 6, condition=condition)

    with pytest.raises(ContractError, match="FiLM-eligible"):
        StepRunDataset(export)


def test_missing_condition_is_registered_warning(tmp_path):
    export = write_export_v3_run(tmp_path, condition=None, boundary=[0] * 6)

    dataset = StepRunDataset(export)

    kinds = {issue.kind for issue in dataset.issues}
    assert "condition_missing" in kinds


def test_condition_required_mode_is_fatal(tmp_path):
    export = write_export_v3_run(tmp_path, condition=None, boundary=[0] * 6)

    with pytest.raises(ContractError, match="condition"):
        StepRunDataset(export, condition_required=True)


def test_non_monotonic_interval_start_is_fatal(tmp_path):
    export = write_export_v3_run(tmp_path, indices=(10, 11), seconds_keys=False)
    step = export / "steps" / "step_000011.h5"
    with h5py.File(step, "r+") as handle:
        handle["time/interval_start_jorek"][()] = T0  # equal -> not strictly increasing
        handle["time/source_interval_end_jorek"][()] = T0
        handle["time/source_interval_start_jorek"][()] = T0 - 0.5
        handle["time/interval_end_jorek"][()] = T0 + 0.5

    with pytest.raises(ContractError, match="strictly increasing"):
        StepRunDataset(export)


def test_chain_break_between_consecutive_steps_is_fatal(tmp_path):
    export = write_export_v3_run(tmp_path, indices=(10, 11), seconds_keys=False)
    step = export / "steps" / "step_000011.h5"
    with h5py.File(step, "r+") as handle:
        handle["time/interval_start_jorek"][()] = 100.75
        handle["time/source_interval_end_jorek"][()] = 100.75
        handle["time/interval_end_jorek"][()] = 101.25

    with pytest.raises(ContractError, match="chain break"):
        StepRunDataset(export)


def test_registered_gap_is_skipped(tmp_path):
    export = write_export_v3_run(
        tmp_path, indices=(10, 11, 13), manifest_extra={"gap_indices": [12]}
    )

    dataset = StepRunDataset(export)

    assert dataset.indices == (10, 11, 13)
    assert any(issue.kind == "registered_gap" for issue in dataset.issues)
    assert dataset.pair_indices() == (10,)


def test_manifest_requires_frozen_keys(tmp_path):
    export = write_export_v3_run(tmp_path, indices=(10,))
    manifest = json.loads((export / "manifest.json").read_text(encoding="utf-8"))
    del manifest["source_object"]
    (export / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ContractError, match="source_object"):
        StepRunDataset(export)


def test_manifest_budget_layout_must_match_step_files(tmp_path):
    export = write_export_v3_run(
        tmp_path, indices=(10, 11), boundary=[0] * 6, manifest_extra={"budget_layout": "slice"}
    )
    step = export / "steps" / "step_000010.h5"
    with h5py.File(step, "r+") as handle:
        handle["xtime"] = np.asarray([0.05, 0.10, 0.15, 0.20])
        handle["xtime_spi_ablation"] = np.asarray([0.0, 1.0, 2.0, 3.0])

    with pytest.raises(ContractError, match="budget_layout='slice'"):
        StepRunDataset(export)


def test_manifest_history_layout_rejects_slice_files(tmp_path):
    export = write_export_v3_run(
        tmp_path,
        indices=(10, 11),
        boundary=[0] * 6,
        manifest_extra={"budget_layout": "history"},
    )
    for index in (10, 11):
        step = export / "steps" / f"step_{index:06d}.h5"
        with h5py.File(step, "r+") as handle:
            handle["xtime"] = np.asarray([0.05])
            handle["xtime_spi_ablation"] = np.asarray([1.0])

    with pytest.raises(ContractError, match="budget_layout='history'"):
        StepRunDataset(export)


def test_resolve_export_root_accepts_run_dir_or_export_dir(tmp_path):
    export = write_export_v3_run(tmp_path, indices=(10,))

    assert resolve_export_root(tmp_path) == export
    assert resolve_export_root(export) == export


def test_linkage_check_detects_inconsistent_duplicate_fields(tmp_path):
    export = write_export_v3_run(
        tmp_path,
        indices=(10, 11, 12),
        boundary=[0] * 6,
        u_after_override={10: 123.0},
    )

    dataset = StepRunDataset(export)
    dataset.check_linkage(1)  # step 11 -> 12 is consistent
    with pytest.raises(ContractError, match="u_after"):
        dataset.check_linkage(0)


# --------------------------------------------------------------------------- #
# Raw recorded run cross-check (ADR-0019)
# --------------------------------------------------------------------------- #


def make_raw_run(
    tmp_path: Path,
    *,
    indices,
    dt_fluid: float = 0.5,
    projection_lag: dict[int, float] | None = None,
    part_lag: dict[int, float] | None = None,
) -> Path:
    raw = tmp_path / "raw_run"
    raw.mkdir(parents=True, exist_ok=True)
    projection_lag = projection_lag or {}
    part_lag = part_lag or {}
    times = {index: T0 + (index - indices[0]) * dt_fluid for index in indices}
    for index in indices:
        t_state = times[index]
        write_model502_frame(raw / f"jorek{index:06d}.h5", t_now=t_state, t_norm=T_NORM)
        write_projection_frame(
            raw / f"projections{index:06d}.h5",
            t_seconds=(t_state - projection_lag.get(index, 0.0)) * T_NORM,
        )
        write_particle_frame(
            raw / f"part{index:05d}.h5",
            time_seconds=(t_state - dt_fluid - part_lag.get(index, 0.0)) * T_NORM,
            groups={"001": {"weight": [1.0, 2.0]}},
        )
    write_macroscopic_times(raw / "macroscopic_vars.dat", times)
    return raw


def test_raw_cross_check_registered_restart_is_dropped(tmp_path):
    indices = (10, 11, 12, 13)
    raw = make_raw_run(tmp_path, indices=indices, projection_lag={12: 0.5})
    export = write_export_v3_run(
        tmp_path, indices=indices, boundary=[0] * 6, manifest_extra={"restart_indices": [12]}
    )

    dataset = StepRunDataset(export, raw_run_dir=raw)

    assert dataset.dropped_restart == (12,)
    assert 12 not in dataset.indices
    assert any(issue.kind == "restart_boundary_projection_offset" for issue in dataset.issues)


def test_raw_cross_check_unregistered_offset_is_fatal(tmp_path):
    indices = (10, 11, 12, 13)
    raw = make_raw_run(tmp_path, indices=indices, projection_lag={12: 0.5})
    export = write_export_v3_run(tmp_path, indices=indices, boundary=[0] * 6)

    with pytest.raises(ContractError, match="offset"):
        StepRunDataset(export, raw_run_dir=raw)


def test_raw_cross_check_part_time_mismatch_is_fatal(tmp_path):
    indices = (10, 11, 12)
    raw = make_raw_run(tmp_path, indices=indices, part_lag={11: 0.5})
    export = write_export_v3_run(tmp_path, indices=indices, boundary=[0] * 6)

    with pytest.raises(ContractError, match="part time"):
        StepRunDataset(export, raw_run_dir=raw)


def test_raw_cross_check_rejects_stale_restart_registry(tmp_path):
    indices = (10, 11, 12)
    raw = make_raw_run(tmp_path, indices=indices)
    export = write_export_v3_run(
        tmp_path, indices=indices, boundary=[0] * 6, manifest_extra={"restart_indices": [11]}
    )

    dataset = StepRunDataset(export, raw_run_dir=raw)

    assert any(issue.kind == "restart_registry_stale" for issue in dataset.issues)
    assert dataset.dropped_restart == ()


REAL_RUN_DIRS = tuple(Path("data/EHL-2").glob("*/export_v3/manifest.json")) if Path(
    "data/EHL-2"
).exists() else ()


@pytest.mark.skipif(not REAL_RUN_DIRS, reason="no A' export_v3 run delivered yet")
def test_real_export_v3_run_if_present():
    export_root = REAL_RUN_DIRS[0].parent

    dataset = StepRunDataset(export_root, require_single_step=True)

    assert len(dataset) > 0
    sample = dataset[0]
    assert sample.u_before.shape[1] == 9
    assert sample.source.shape[1] == 5


# --------------------------------------------------------------------------- #
# Native stepwise layout (jorek*/projections*/part* in the run directory)
# --------------------------------------------------------------------------- #


NATIVE_NAMELIST = """
&namelist
  n_spi = 846
  pellet_density = 5.985d8
  d_par = 0.01d0
  zk_i_par = 1.6666d5
  zk_par_max = 1.0d4
  zkpar_t_dependent = .t.
  bc_natural_open = .t.
/
"""


def test_native_layout_detected_and_paired(tmp_path):
    root = write_native_run(tmp_path / "run", indices=(10, 11, 12, 13), n_spi=4)

    kind, resolved = detect_run_layout(root)
    assert kind == "native"
    assert resolved == root

    dataset = StepRunDataset(root, require_single_step=True)

    assert dataset.report.layout_kind == "native"
    assert dataset.layout == "full_dof"
    assert dataset.source_object == "marker_feedback"
    # Native provenance registers the recorded projection as the marker-run true
    # source (ADR-0020 decision 1).
    assert "marker_feedback" in dataset.manifest["provenance"]["source_object"]
    assert dataset.indices == (10, 11, 12)
    assert dataset.report.n_windows == 3
    assert dataset.report.single_step_method == "native_adjacent_frame"
    assert dataset.report.interval_source == "macroscopic_vars.dat"

    sample = dataset[0]
    # jorek{k} = u(T_k): channel c fills value c.
    assert sample.u_before.shape == (6, 9)
    assert sample.u_before[0, 1] == pytest.approx(1.0)
    assert sample.u_after[0, 8] == pytest.approx(8.0)
    assert sample.source.shape == (6, 5)
    assert sample.shard_start.n_total == 4
    assert sample.shard_end.n_total == 4
    assert set(sample.shard_start.arrays) == set(SHARD_NAMES)
    assert sample.field_interval == pytest.approx((100.0, 100.5))
    assert sample.source_window == pytest.approx((99.5, 100.0))
    assert sample.dt_step == pytest.approx(0.5)
    assert sample.dt_source == pytest.approx(0.5)
    # Shard time labels are the *true* frame times: jorek{k} = T_k (field
    # interval start), jorek{k+1} = T_{k+1} (field interval end).  The source
    # window start p_{k-1} is the previous record's shard_start.
    assert sample.shard_start.time_jorek == pytest.approx(sample.t_before)
    assert sample.shard_end.time_jorek == pytest.approx(sample.t_after)
    assert dataset.report.shard_time_semantics == "field_interval_endpoints"
    assert sample.meta.path_after == root / "jorek000011.h5"
    assert sample.meta.path_source == root / "projections000010.h5"
    # Boundary frames and part checks are registered, nothing dropped.
    kinds = {issue.kind for issue in dataset.issues}
    assert "boundary_frame_leading" in kinds
    assert "boundary_frame_trailing" in kinds
    assert dataset.dropped_restart == ()
    assert dataset.dropped_zero_dt == ()
    assert dataset.read_boundary(0).shape == (6,)
    # part{n}.time == T_{n-1} is verified for the recorded part frames.
    assert any(issue.kind == "part_time_checked" for issue in dataset.issues)


def test_native_condition_from_production_namelist(tmp_path):
    root = write_native_run(
        tmp_path / "run", indices=(10, 11), namelist=NATIVE_NAMELIST
    )

    dataset = StepRunDataset(root)
    condition = dataset.read_condition(0)

    assert condition is not None
    lookup = dict(zip(condition.names, condition.values))
    assert lookup["n_spi"] == pytest.approx(846.0)
    assert lookup["D_par"] == pytest.approx(0.01)
    assert lookup["ZK_par_max"] == pytest.approx(1.0e4)
    groups = dict(zip(condition.names, condition.groups))
    assert groups["n_spi"] == "pellet_mass"
    assert groups["D_par"] == "excluded_transport"
    assert groups["time_evol_theta"] == "excluded_numeric"
    film = dict(zip(condition.names, condition.film_eligible))
    assert film["n_spi"] == 1
    assert film["D_par"] == 0
    summary = dataset.condition_summary
    assert summary is not None
    assert summary.namelist_path == root / "jorek_namelist_spitest"
    assert summary.namelist_sha256


def test_native_condition_group_map_covers_parameter_surface():
    from src.data import parameter_channel
    from src.data.native_run import _GROUP_BY_PARAMETER

    missing = [
        spec.name for spec in parameter_channel._SPEC if spec.name not in _GROUP_BY_PARAMETER
    ]
    assert missing == []


def test_native_zero_dt_unregistered_fails_fast_by_default(tmp_path):
    root = write_native_run(
        tmp_path / "run", indices=(10, 11, 12, 13), duplicate_frames=(12,)
    )

    with pytest.raises(ContractError) as error:
        StepRunDataset(root, require_single_step=True)

    message = str(error.value)
    assert "unregistered" in message
    assert "[12]" in message
    assert "restart_events.json" in message


def test_native_zero_dt_error_policy_still_demands_dedup(tmp_path):
    root = write_native_run(
        tmp_path / "run", indices=(10, 11, 12, 13), duplicate_frames=(12,)
    )

    with pytest.raises(ContractError) as error:
        StepRunDataset(root, require_single_step=True, zero_dt_policy="error")

    message = str(error.value)
    assert "zero-dt" in message
    assert "[12]" in message
    assert "dedup" in message
    assert "Affected pairs (dt_step=0): [11]" in message


def test_native_registered_restart_event_is_accepted(tmp_path):
    """ADR-0019 addendum: registered dt=0 events pass; cross-boundary pair and
    the lagging projections{n-1} are skipped, the event frame stays usable."""
    root = write_native_run(
        tmp_path / "run",
        indices=(10, 11, 12, 13),
        duplicate_frames=(12,),
        projection_lag={11: 0.5},
    )

    dataset = StepRunDataset(
        root, restart_events=[12], require_single_step=True
    )

    report = dataset.report
    assert report.zero_dt_policy == "registered"
    assert report.zero_dt_indices == (12,)
    assert report.restart_events_registered == (12,)
    assert report.restart_cross_boundary_skipped == (11,)
    assert report.restart_lagging_projection_skipped == (11,)
    assert report.restart_degenerate_source_windows == ()
    assert report.shard_time_semantics == "field_interval_endpoints"
    # The cross-boundary record 11 -> 12 is gone; the event frame 12 keeps the
    # S_12 = projections{12} window with positive duration from the startup event.
    assert dataset.indices == (10, 12)
    assert dataset.dropped_restart == (11,)
    sample = dataset[1]
    assert sample.index == 12
    assert sample.dt_source == pytest.approx(0.5)
    assert sample.dt_step == pytest.approx(0.5)
    kinds = {issue.kind for issue in dataset.issues}
    assert "restart_events_registered" in kinds
    assert "restart_cross_boundary_skipped" in kinds
    assert "restart_lagging_projection_skipped" in kinds
    assert "restart_particle_window" in kinds


def test_native_registered_restart_lagging_projection_is_verified(tmp_path):
    """A registered event whose projections{n-1} has an *unexpected* offset fails."""
    root = write_native_run(
        tmp_path / "run",
        indices=(10, 11, 12, 13),
        duplicate_frames=(12,),
        projection_lag={11: 0.25},  # not the registered -dt_source
    )

    with pytest.raises(ContractError, match="lagging projection|unregistered restart"):
        StepRunDataset(root, restart_events=[12], require_single_step=True)


def test_native_zero_dt_list_policy_enumerates_and_drops(tmp_path):
    root = write_native_run(
        tmp_path / "run", indices=(10, 11, 12, 13), duplicate_frames=(12,)
    )

    dataset = StepRunDataset(root, zero_dt_policy="list", require_single_step=True)

    assert dataset.report.zero_dt_policy == "list"
    assert dataset.report.zero_dt_indices == (12,)
    assert dataset.report.zero_dt_pairs == (11,)
    assert dataset.dropped_zero_dt == (11,)
    # The zero-length record 11->12 is gone; the remaining records stay.
    assert dataset.indices == (10, 12)
    kinds = {issue.kind for issue in dataset.issues}
    assert "zero_dt_frames_listed" in kinds
    assert "zero_dt_record_dropped" in kinds
    assert "zero_dt_source_window" in kinds
    # The record after the duplicate keeps a positive window and is verified.
    sample = dataset[1]
    assert sample.index == 12
    assert sample.dt_step == pytest.approx(0.5)


def test_native_deduped_gap_is_accepted(tmp_path):
    """Plan-1 regeneration: the duplicate frame is removed and the gap registered."""
    root = write_native_run(
        tmp_path / "run",
        indices=(10, 11, 12, 14, 15),
        duplicate_frames=(13,),
    )

    dataset = StepRunDataset(
        root, gap_indices=[13], require_single_step=True, zero_dt_policy="error"
    )

    assert dataset.indices == (10, 11, 12, 14)
    assert dataset.report.zero_dt_indices == ()
    sample = dataset[2]  # record 12 -> 14 spans the registered dedup gap
    assert sample.index == 12
    assert sample.meta.index_next == 14
    assert sample.dt_step == pytest.approx(0.5)
    assert sample.dt_source == pytest.approx(0.5)


def test_native_unregistered_index_gap_is_fatal(tmp_path):
    root = write_native_run(tmp_path / "run", indices=(10, 11, 13, 14))

    with pytest.raises(ContractError, match="registered as gaps"):
        StepRunDataset(root, require_single_step=False)


def test_native_require_single_step_rejects_non_zero_dt_gap(tmp_path):
    root = write_native_run(tmp_path / "run", indices=(10, 11, 13, 14))

    with pytest.raises(ContractError, match="dt != 0"):
        StepRunDataset(root, gap_indices=[12], require_single_step=True)


def test_native_frame_level_record_is_registered_when_relaxed(tmp_path):
    root = write_native_run(tmp_path / "run", indices=(10, 11, 13, 14))

    dataset = StepRunDataset(root, gap_indices=[12], require_single_step=False)

    kinds = {issue.kind for issue in dataset.issues}
    assert "multi_step_record" in kinds
    assert dataset.report.require_single_step is False


def test_native_projection_offset_must_be_registered(tmp_path):
    root = write_native_run(
        tmp_path / "run",
        indices=(10, 11, 12, 13),
        projection_lag={12: 0.5},
    )

    with pytest.raises(ContractError, match="offset"):
        StepRunDataset(root, restart_file=None)


def test_native_registered_restart_offset_is_dropped(tmp_path):
    root = write_native_run(
        tmp_path / "run",
        indices=(10, 11, 12, 13),
        projection_lag={12: 0.5},
        restart_file=(11, 12),
    )

    dataset = StepRunDataset(root, policy="drop")

    assert dataset.dropped_restart == (12,)
    assert 12 not in dataset.indices
    kinds = {issue.kind for issue in dataset.issues}
    assert "restart_boundary_projection_offset" in kinds
    assert dataset.report.restart_registry_size == 2
    checks = dataset.restart_report["checks"]
    assert {row["index"] for row in checks} == {11, 12}


def test_native_part_time_mismatch_is_fatal(tmp_path):
    root = write_native_run(
        tmp_path / "run", indices=(10, 11, 12), part_lag={11: 0.25}
    )

    with pytest.raises(ContractError, match="part"):
        StepRunDataset(root)


def test_native_boundary_frames_are_registered_not_silent(tmp_path):
    root = write_native_run(tmp_path / "run", indices=(10, 11, 12))

    dataset = StepRunDataset(root)
    report = dataset.report

    assert report.boundary_frames_leading == (9,)
    assert report.boundary_frames_trailing == (12,)
    kinds = {issue.kind for issue in dataset.issues}
    assert "boundary_frame_leading" in kinds
    assert "boundary_frame_trailing" in kinds


def test_native_negative_dt_is_always_fatal(tmp_path):
    root = write_native_run(tmp_path / "run", indices=(10, 11, 12))
    with h5py.File(root / "jorek000012.h5", "r+") as handle:
        handle["t_now"][()] = 99.0  # T_12 < T_11
    write_macroscopic_times(
        root / "macroscopic_vars.dat",
        {9: 99.5, 10: 100.0, 11: 100.5, 12: 99.0},
    )

    with pytest.raises(ContractError, match="negative dt|cannot be repaired|not monotonic"):
        StepRunDataset(root, zero_dt_policy="list")


def test_native_parameter_surface_names_are_deduped(tmp_path):
    root = write_native_run(tmp_path / "run", indices=(10, 11))

    condition = StepRunDataset(root).read_condition(0)

    duplicated = [name for name in condition.names if name.startswith("bc_natural_open")]
    assert len(duplicated) == 2
    assert duplicated[1].endswith("#2")
    assert len(set(condition.names)) == len(condition.names)


REAL_NATIVE_RUN = Path(
    "data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner"
)


@pytest.mark.skipif(not REAL_NATIVE_RUN.is_dir(), reason="native A' run not present")
def test_real_native_run_if_present():
    """Light real-data check: time table, zero-dt events, one indexed window."""
    from src.data.output_contract import parse_macroscopic_times

    times = parse_macroscopic_times(REAL_NATIVE_RUN / "macroscopic_vars.dat")
    zero_dt = [
        index
        for index in sorted(times)
        if index - 1 in times and abs(times[index] - times[index - 1]) <= 1e-8
    ]
    assert zero_dt == [2704, 5681, 6441, 7221, 8611, 9291]

    dataset = StepRunDataset(
        REAL_NATIVE_RUN,
        index_window=(2701, 2706),
        zero_dt_policy="list",
        require_single_step=True,
    )
    assert dataset.report.layout_kind == "native"
    assert dataset.indices == (2701, 2702, 2704, 2705)
    sample = dataset[0]
    assert sample.u_before.shape == (4125, 9)
    assert sample.source.shape == (4125, 5)
    assert sample.shard_start.n_total == 846

    # The production run is not regenerated: with the registry in
    # data/restart_events.json the default 'registered' policy accepts the event
    # frame while dropping the cross-boundary pair and lagging projection.
    registered = StepRunDataset(
        REAL_NATIVE_RUN, index_window=(2701, 2706), require_single_step=True
    )
    report = registered.report
    assert report.zero_dt_policy == "registered"
    assert report.zero_dt_indices == (2704,)
    # The registered set is the full run-level registry; ``zero_dt_indices``
    # reports the events detected inside the checked window.
    assert report.restart_events_registered == (2704, 5681, 6441, 7221, 8611, 9291)
    assert report.restart_cross_boundary_skipped == (2703,)
    assert report.restart_lagging_projection_skipped == (2703,)
    assert report.restart_degenerate_source_windows == ()
    assert registered.indices == (2701, 2702, 2704, 2705)
    event_sample = registered[2]
    assert event_sample.index == 2704
    assert event_sample.dt_source == pytest.approx(5.0)
    assert event_sample.shard_start.time_jorek == pytest.approx(event_sample.t_before)
