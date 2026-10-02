"""Tests for coefficient-aware temporal segmentation and slice sampling."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from src.data.data_manifest import aligned_frames
from src.data.segmented_sampler import build_run_segments, sample_slice


RUN = "scan_1d22_ang12_massloss_neon"
CACHE = "/tmp/ehl2_cache"

pytestmark = pytest.mark.skipif(
    not Path(CACHE).exists(),
    reason=(
        "frame cache was removed with the retired nine-field workflow; this sampler is "
        "superseded by src/data/window_manifest.py (ADR-0012)"
    ),
)


def test_build_segments_preserves_long_raw_segments():
    frames = aligned_frames(Path("data/EHL-2") / RUN)
    if not frames:
        pytest.skip("real EHL-2 data not present")
    segments = build_run_segments(RUN, frames, CACHE, min_length=25)
    assert segments
    assert all(segment.length >= 25 for segment in segments)
    assert all(
        segment.times[i] < segment.times[i + 1]
        for segment in segments
        for i in range(len(segment.times) - 1)
    )


def test_sampled_slice_is_monotonic_and_target_is_last_kept():
    frames = aligned_frames(Path("data/EHL-2") / RUN)
    if not frames:
        pytest.skip("real EHL-2 data not present")
    segments = build_run_segments(RUN, frames, CACHE, min_length=25)
    rng = random.Random(0)
    sample = sample_slice(
        segments,
        rng,
        min_frames=8,
        max_frames=16,
        remove_count=3,
    )
    assert len(sample.kept_frame_indices) >= 5
    assert sample.target_index == sample.kept_frame_indices[-1]
    assert all(a < b for a, b in zip(sample.times, sample.times[1:]))
    assert all(dt > 0 for dt in sample.dts[1:])
    assert not set(sample.removed_frame_indices) & set(sample.kept_frame_indices)
    assert sample.reset_state is True
