"""Behaviour tests for EHL-2 data discovery, alignment and the manifest builder.

These tests pin the real naming contract observed in ``data/EHL-2``:
state frames are ``jorek<index>.h5``, marker frames are ``part<index>.h5``, and a
frame is valid for training only when the same integer index exists in both
families.  Special names such as ``jorek_restart.h5`` must never be treated as
frames.  All assertions below go through the public functions of
``src.data.data_manifest``.
"""

from __future__ import annotations

import hashlib
import h5py
import numpy as np
import pytest

from conftest import write_model502_frame
from src.data.data_manifest import (
    STATE_FRAME_RE,
    aligned_frames,
    build_manifest,
    discover_runs,
    is_2d_run,
    mesh_summary,
)


NODES = 6


def _write_marker(path, index):
    with h5py.File(path, "w") as f:
        f["time"] = [0.0]
        f["groups/001/weight"] = np.array([1.0, 2.0])


def _make_run(tmp_path, name, *, indices, marker_indices, n_tor=1, n_dim=2):
    run_dir = tmp_path / name
    run_dir.mkdir()
    for index in indices:
        write_model502_frame(
            run_dir / f"jorek{index:05d}.h5",
            t_now=float(index),
            n_tor=n_tor,
            n_dim=n_dim,
        )
    for index in marker_indices:
        _write_marker(run_dir / f"part{index:05d}.h5", index)
    return run_dir


def test_state_frame_re_ignores_special_names():
    assert STATE_FRAME_RE.fullmatch("jorek002710.h5")
    assert STATE_FRAME_RE.fullmatch("jorek008900.h5")
    assert not STATE_FRAME_RE.fullmatch("jorek_restart.h5")
    assert not STATE_FRAME_RE.fullmatch("jorek_grid.h5")
    assert not STATE_FRAME_RE.fullmatch("jorek_equil_rz.h5")


def test_aligned_frames_requires_matching_part_file(tmp_path):
    run_dir = _make_run(
        tmp_path, "run_a", indices=[1, 2, 3], marker_indices=[2, 3]
    )
    (run_dir / "jorek_restart.h5").write_bytes(b"placeholder")
    (run_dir / "part_restart.h5").write_bytes(b"placeholder")

    frames = aligned_frames(run_dir)

    assert [frame.index for frame in frames] == [2, 3]
    assert all(f.state_path.exists() and f.marker_path.exists() for f in frames)


def test_is_2d_run_reads_metadata_from_first_valid_frame(tmp_path):
    run2d = _make_run(tmp_path, "run2d", indices=[1, 2], marker_indices=[1, 2], n_tor=1, n_dim=2)
    run3d = _make_run(tmp_path, "run3d", indices=[1, 2], marker_indices=[1, 2], n_tor=11, n_dim=3)

    assert is_2d_run(run2d)
    assert not is_2d_run(run3d)


def test_discover_runs_lists_immediate_directories_sorted(tmp_path):
    _make_run(tmp_path, "b_run", indices=[1], marker_indices=[1])
    _make_run(tmp_path, "a_run", indices=[1], marker_indices=[1])
    (tmp_path / "not_a_run.txt").write_text("x")

    runs = discover_runs(tmp_path)

    assert [run.name for run in runs] == ["a_run", "b_run"]


def test_build_manifest_split_is_sequence_level_and_stats_are_train_only(tmp_path):
    # Two frames in the train run: rho values 0 and e-1, so after the
    # log1p(clamp) transform they are exactly [0, 1] -> mean 0.5, std 0.5.
    run_train = _make_run(tmp_path, "t_run", indices=[1, 2], marker_indices=[1, 2])
    run_val = _make_run(tmp_path, "v_run", indices=[1], marker_indices=[1])
    write_model502_frame(
        run_train / "jorek00001.h5", t_now=1.0, rho_values=np.array([0.0] * NODES)
    )
    write_model502_frame(
        run_train / "jorek00002.h5",
        t_now=2.0,
        rho_values=np.array([np.e - 1.0] * NODES),
    )
    # Give the val run wildly different rho so train-only stats are observable.
    write_model502_frame(
        run_val / "jorek00001.h5", t_now=1.0, rho_values=np.array([999.0] * NODES)
    )

    manifest = build_manifest(tmp_path, seed=7, val_frac=0.5, test_frac=0.0)

    by_name = {run["name"]: run for run in manifest["runs"]}
    splits = {by_name["t_run"]["split"], by_name["v_run"]["split"]}
    assert splits == {"train", "val"}
    train_name = next(run["name"] for run in manifest["runs"] if run["split"] == "train")
    stats = manifest["stats"]
    assert stats["schema"] == "model502-state-v1"
    rho = stats["fields"]["rho"]
    if train_name == "t_run":
        # t_run rho after log1p is exactly [0, 1] across its two frames.
        assert rho["mean"] == pytest.approx(0.5)
        assert rho["std"] == pytest.approx(0.5)
    else:
        # v_run rho is the constant 999 -> log1p(999), std zero; this also
        # proves the train-only split excludes t_run's [0, 1] values.
        transformed = np.log1p(999.0)
        assert rho["mean"] == pytest.approx(transformed)
        assert rho["std"] == pytest.approx(0.0)
    assert rho["n_negative"] == 0
    psi = stats["fields"]["psi"]
    assert psi["mean"] == pytest.approx(2.0)  # canonical index 0 takes raw channel 2
    assert psi["std"] == pytest.approx(0.0)
    condition = stats["condition"]
    assert condition["names"]
    assert len(condition["mean"]) == len(condition["names"]) == len(condition["std"])


def test_build_manifest_is_seed_deterministic_and_hashes_layout(tmp_path):
    _make_run(tmp_path, "a", indices=[1, 2], marker_indices=[1, 2])
    _make_run(tmp_path, "b", indices=[1, 2], marker_indices=[1, 2])
    _make_run(tmp_path, "c", indices=[1, 2], marker_indices=[1, 2])

    first = build_manifest(tmp_path, seed=11, val_frac=0.34, test_frac=0.33)
    second = build_manifest(tmp_path, seed=11, val_frac=0.34, test_frac=0.33)

    assert first["dataset_id"] == second["dataset_id"]
    assert [r["name"] for r in first["runs"]] == [r["name"] for r in second["runs"]]
    assert [r["split"] for r in first["runs"]] == [r["split"] for r in second["runs"]]
    digest = hashlib.sha256()
    layout_files = sorted(list(tmp_path.rglob("jorek*.h5")) + list(tmp_path.rglob("part*.h5")))
    for path in layout_files:
        digest.update(str(path.relative_to(tmp_path)).encode())
        digest.update(str(path.stat().st_size).encode())
    assert first["dataset_id"] == digest.hexdigest()


def test_build_manifest_reports_mesh_consistency(tmp_path):
    _make_run(tmp_path, "a", indices=[1, 2], marker_indices=[1, 2])
    _make_run(tmp_path, "b", indices=[1, 2], marker_indices=[1, 2])

    consistent = build_manifest(tmp_path, seed=3, val_frac=0.34, test_frac=0.33)
    assert consistent["mesh"] == {"consistent": True, "n_grids": 1}

    # Give run b a different mesh than run a: same shape, different coordinates.
    with h5py.File(tmp_path / "b" / "jorek00001.h5", "r+") as f:
        f["x"][0, 0, 0, :] += 100.0

    inconsistent = build_manifest(tmp_path, seed=3, val_frac=0.34, test_frac=0.33)
    assert inconsistent["mesh"]["consistent"] is False
    assert inconsistent["mesh"]["n_grids"] == 2


@pytest.mark.skipif(
    not (lambda p: p.exists())(
        __import__("pathlib").Path("data/EHL-2/scan_1d22_ang12_massloss_neon")
    ),
    reason="real EHL-2 data not present",
)
def test_real_data_alignment_and_2d_detection():
    from pathlib import Path

    run_dir = Path("data/EHL-2/scan_1d22_ang12_massloss_neon")
    frames = aligned_frames(run_dir)
    assert frames[0].index == 2710
    assert frames[-1].index == 8900
    assert len(frames) == 620
    assert is_2d_run(run_dir)
    ntor11 = Path("data/EHL-2/ntor11_1d22_v400_neon_earlier")
    assert not is_2d_run(ntor11)


@pytest.mark.skipif(
    not (lambda p: p.exists())(
        __import__("pathlib").Path("data/EHL-2/scan_1d22_ang12_massloss_neon")
    ),
    reason="real EHL-2 data not present",
)
def test_real_data_eligible_runs_share_one_grid():
    from pathlib import Path

    summary = mesh_summary(Path("data/EHL-2"))

    assert summary["consistent"] is True
    assert summary["n_grids"] == 1
    assert len(summary["runs"]) == 10
    assert all("ntor11" not in run["name"] for run in summary["runs"])
