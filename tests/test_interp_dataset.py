"""Tests for the Phase-1 interpolation window dataset and collate."""

from __future__ import annotations

from pathlib import Path

import torch

from src.data.interp_dataset import (
    InterpolationWindowDataset,
    RunFrameStore,
    collate_windows,
    compute_transform_stats,
    to_device,
)
from src.data.window_manifest import build_interp_manifest
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


def _manifest(tmp_path: Path) -> dict:
    _write_run(tmp_path / "run_a", 40, t_ns=105.0, eta_jump=20)
    return build_interp_manifest(tmp_path, burn_in=2, horizon=1, run_glob="run_*")


def test_dataset_item_shapes_and_loss_mask(tmp_path):
    manifest = _manifest(tmp_path)
    dataset = InterpolationWindowDataset(manifest, tmp_path, normalize=False)
    assert len(dataset) > 0
    item = dataset[0]
    num_frames = item["u"].shape[0]
    assert num_frames == 4  # burn_in 2 + horizon 1 + 1
    assert item["u"].shape[1:] == (6, 9)
    assert item["dt"].shape == (num_frames - 1,)
    assert item["condition"].shape[0] == num_frames
    assert item["source"].shape[1:] == (6, 5)
    assert item["source_available"].all()
    assert item["loss_mask"].tolist() == [False, False, True, True]
    assert item["topology"].pos.shape == (6, 2)


def test_preinjection_window_has_no_shards(tmp_path):
    manifest = _manifest(tmp_path)
    dataset = InterpolationWindowDataset(manifest, tmp_path, normalize=False)
    pre = [
        index
        for index, record in enumerate(dataset.windows)
        if record["segment"] == 0 and record["run"] == "run_a"
    ]
    assert pre, "expected a pre-injection window in the synthetic run"
    item = dataset[pre[0]]
    assert int(item["shard_mask"].sum()) == 0


def test_normalization_uses_train_only_stats(tmp_path):
    manifest = _manifest(tmp_path)
    store = RunFrameStore(tmp_path / "run_a").load()
    stats = compute_transform_stats({"run_a": store})
    assert stats["n_frames"] == 40
    assert stats["n_state_values"] == 40 * 6
    dataset = InterpolationWindowDataset(manifest, tmp_path, normalize=True, stats=stats)
    item = dataset[0]
    assert torch.isfinite(item["u"]).all()
    # The normalised tensor must differ from the raw tensor for the log fields.
    assert not torch.allclose(item["u"][0, :, 4], item["u_raw"][0, :, 4])


def test_fold_filter_excludes_held_out_run(tmp_path):
    _write_run(tmp_path / "run_a", 40, t_ns=105.0, eta_jump=20)
    _write_run(tmp_path / "run_b", 35, t_ns=None)
    manifest = build_interp_manifest(tmp_path, burn_in=2, horizon=1, run_glob="run_*")
    dataset = InterpolationWindowDataset(manifest, tmp_path, fold=0, normalize=False)
    assert dataset.held_out_run is not None
    assert all(record["run"] != dataset.held_out_run for record in dataset.windows)
    assert len(dataset) < len(manifest["windows"])


def test_collate_pads_shards_and_keeps_topology(tmp_path):
    manifest = _manifest(tmp_path)
    dataset = InterpolationWindowDataset(manifest, tmp_path, normalize=False)
    batch = collate_windows([dataset[0], dataset[1]])
    assert batch["u"].shape == (2, 4, 6, 9)
    assert batch["shard_values"].shape[0] == 2
    assert batch["shard_mask"].shape[:2] == (2, 4)
    moved = to_device(batch, "cpu")
    assert moved["topology"].pos.shape == (6, 2)
    assert moved["runs"] == [dataset.windows[0]["run"], dataset.windows[1]["run"]]


def test_disk_cache_roundtrip(tmp_path):
    manifest = _manifest(tmp_path)
    cache_dir = tmp_path / "cache"
    first = RunFrameStore(tmp_path / "run_a", cache_dir=cache_dir).load()
    assert (cache_dir / "run_a.pt").exists()
    second = RunFrameStore(tmp_path / "run_a", cache_dir=cache_dir).load()
    assert torch.equal(first.u, second.u)
    assert torch.equal(first.time, second.time)
    dataset = InterpolationWindowDataset(
        manifest, tmp_path, normalize=False, cache_dir=cache_dir
    )
    assert dataset[0]["u"].shape[1:] == (6, 9)
