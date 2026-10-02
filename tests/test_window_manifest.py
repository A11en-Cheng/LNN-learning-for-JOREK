"""Tests for the Phase-1 window corpus, blocked split and LOO folds (ADR-0012)."""

from __future__ import annotations

from pathlib import Path

from src.data.window_manifest import (
    assign_blocked_split,
    build_interp_manifest,
    build_windows,
    collect_frames,
    injection_time,
    loo_folds,
    segment_frames,
)
from tests.synthetic_frames import write_frame, write_namelist


def _write_run(run_dir: Path, n_frames: int, *, t_ns: float | None, eta_jump: int | None = None) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    write_namelist(run_dir, t_ns=t_ns)
    for index in range(n_frames):
        t_now = index * 10.0
        n_spi = 0 if (t_ns is None or t_now < t_ns) else 5
        eta = 1e-10
        if eta_jump is not None and index >= eta_jump:
            eta = 1e-8
        write_frame(run_dir, index, t_now=t_now, n_spi=n_spi, eta=eta)


def test_injection_time_parsing():
    assert injection_time("  t_ns = 8380 ! injection\n") == 8380.0
    assert injection_time("  t_ns = 1.5d3\n") == 1500.0
    assert injection_time("no injection here") is None


def test_segments_cut_at_injection_and_material_change(tmp_path):
    run_dir = tmp_path / "run_a"
    _write_run(run_dir, 40, t_ns=105.0, eta_jump=20)
    frames = collect_frames(run_dir)
    segments = segment_frames("run_a", frames, t_ns=105.0)
    assert [(segment.start, segment.end) for segment in segments] == [(0, 11), (11, 20), (20, 40)]
    assert [segment.has_fragments for segment in segments] == [False, True, True]
    assert "injection" in segments[1].boundaries or "fragments" in segments[1].boundaries
    assert "material" in segments[2].boundaries


def test_frame_numbering_gap_is_a_segment_boundary(tmp_path):
    run_dir = tmp_path / "run_gap"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_namelist(run_dir, t_ns=None)
    for index in list(range(4)) + list(range(7, 12)):
        write_frame(run_dir, index, t_now=index * 10.0)
    frames = collect_frames(run_dir)
    segments = segment_frames("run_gap", frames, t_ns=None)
    boundaries = [(segment.start_index, segment.end_index) for segment in segments]
    assert boundaries == [(0, 3), (7, 11)]


def test_time_reversal_is_a_segment_boundary(tmp_path):
    """A restart/backup append must not produce a window with negative dt (#61)."""
    run_dir = tmp_path / "run_rev"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_namelist(run_dir, t_ns=None)
    times = [0.0, 10.0, 20.0, 30.0, 40.0, 5.0, 15.0, 25.0]
    for index, t_now in enumerate(times):
        write_frame(run_dir, index, t_now=t_now)
    frames = collect_frames(run_dir)
    segments = segment_frames("run_rev", frames, t_ns=None)
    assert [(segment.start, segment.end) for segment in segments] == [(0, 5), (5, 8)]
    assert "time_reversal" in segments[1].boundaries
    # No generated window may contain a non-increasing interval.
    for window in build_windows(segments, burn_in=2, horizon=1):
        for a, b in zip(window.frame_indices, window.frame_indices[1:]):
            assert frames[b].t_now > frames[a].t_now


def test_window_generation_and_blocked_embargo(tmp_path):
    run_dir = tmp_path / "run_b"
    _write_run(run_dir, 60, t_ns=None)
    frames = collect_frames(run_dir)
    segments = segment_frames("run_b", frames, t_ns=None)
    windows = build_windows(segments, burn_in=2, horizon=1)
    assert len(windows) == 57  # 60 - (2+1+1) + 1
    assign_blocked_split(windows, segments, val_fraction=0.25, block_windows=8)

    span = 4
    train = [window for window in windows if window.inner == "train"]
    val = [window for window in windows if window.inner == "val"]
    dropped = [window for window in windows if window.inner == "dropped"]
    assert train and val and dropped
    assert max(window.start + span for window in train) <= min(window.start for window in val) - span
    assert not (set(w.start for w in train) & set(w.start for w in val))


def test_loo_folds_cover_all_runs():
    folds = loo_folds(["a", "b", "c"])
    assert len(folds) == 3
    assert {fold["held_out_run"] for fold in folds} == {"a", "b", "c"}
    for fold in folds:
        assert fold["held_out_run"] not in fold["train_runs"]
        assert len(fold["train_runs"]) == 2


def test_build_manifest_end_to_end(tmp_path):
    _write_run(tmp_path / "run_a", 40, t_ns=105.0, eta_jump=20)
    _write_run(tmp_path / "run_b", 35, t_ns=None)
    manifest = build_interp_manifest(tmp_path, burn_in=2, horizon=1, run_glob="run_*")
    assert manifest["schema"] == "model502-interp-manifest-v1"
    assert manifest["summary"]["n_runs"] == 2
    assert manifest["summary"]["n_windows"] > 0
    assert len(manifest["folds"]) == 2
    for window in manifest["windows"]:
        assert window["inner"] in {"train", "val", "dropped"}
        assert len(window["frame_indices"]) == window["burn_in"] + window["horizon"] + 1
