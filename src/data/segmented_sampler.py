"""Coefficient-aware temporal segmentation and segment-local slice sampling.

This module is read-only with respect to the training loop.  It builds temporal
segments from the zero-dimensional condition channel, drops segments shorter
than the configured minimum, and samples non-uniform subsequences inside one
segment.  Removed interior frames are omitted from the input rather than used
as reconstruction targets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch

from src.data.data_manifest import aligned_frames


# Flattened condition-vector indices that directly enter the assembled field
# equations: eta, visco, visco_par, visco_par_par, ZK_i/e_par, D_perp,
# ZK_i/e_perp, time_evol_theta/zeta, F0 and R_domm.
MATERIAL_INDICES: tuple[int, ...] = tuple(list(range(0, 26)) + [33, 34])


@dataclass(frozen=True)
class TemporalSegment:
    run: str
    start: int
    end: int
    frame_indices: tuple[int, ...]
    times: tuple[float, ...]

    @property
    def length(self) -> int:
        return len(self.frame_indices)


@dataclass(frozen=True)
class SegmentedSlice:
    run: str
    segment_start: int
    segment_end: int
    kept_frame_indices: tuple[int, ...]
    removed_frame_indices: tuple[int, ...]
    target_index: int
    times: tuple[float, ...]
    dts: tuple[float, ...]
    reset_state: bool = True


def _rel_change(a: float, b: float) -> float:
    denom = abs(a) + abs(b) + 1e-12
    return abs(a - b) / denom


def _boundary_indices(conditions: list[torch.Tensor], threshold: float) -> list[int]:
    boundaries: list[int] = []
    for index in range(1, len(conditions)):
        prev = conditions[index - 1]
        curr = conditions[index]
        change = max(
            _rel_change(float(prev[idx]), float(curr[idx]))
            for idx in MATERIAL_INDICES
        )
        if change > threshold:
            boundaries.append(index)
    return boundaries


def _segments_from_boundaries(
    frame_indices: list[int],
    times: list[float],
    boundaries: list[int],
) -> list[TemporalSegment]:
    cuts = [0] + list(boundaries) + [len(frame_indices)]
    segments: list[TemporalSegment] = []
    for i in range(len(cuts) - 1):
        start, end = cuts[i], cuts[i + 1]
        segments.append(
            TemporalSegment(
                run="",
                start=start,
                end=end,
                frame_indices=tuple(frame_indices[start:end]),
                times=tuple(times[start:end]),
            )
        )
    return segments


def build_run_segments(
    run: str,
    frames: list,
    cache_root: str | Path,
    *,
    rel_threshold: float = 0.05,
    min_length: int = 25,
) -> list[TemporalSegment]:
    """Build kept temporal segments for one run."""
    cache_root = Path(cache_root)
    frame_indices: list[int] = []
    times: list[float] = []
    conditions: list[torch.Tensor] = []

    for frame in frames:
        cached = torch.load(
            cache_root / run / f"{frame.index}.pt",
            map_location="cpu",
            weights_only=False,
        )
        frame_indices.append(frame.index)
        times.append(float(cached["time"][0].item()))
        conditions.append(cached["condition"].numpy())

    boundaries = _boundary_indices(conditions, rel_threshold)
    segments = _segments_from_boundaries(frame_indices, times, boundaries)
    kept = [segment for segment in segments if segment.length >= min_length]
    return [
        TemporalSegment(
            run=run,
            start=segment.start,
            end=segment.end,
            frame_indices=segment.frame_indices,
            times=segment.times,
        )
        for segment in kept
    ]


def build_all_segments(
    manifest_path: str | Path,
    cache_root: str | Path,
    *,
    rel_threshold: float = 0.05,
    min_length: int = 25,
) -> dict[str, list[TemporalSegment]]:
    """Build segments for all train/val runs in the split manifest."""
    import json

    manifest = json.loads(Path(manifest_path).read_text())
    output: dict[str, list[TemporalSegment]] = {}
    for run_meta in manifest["runs"]:
        if run_meta["split"] not in {"train", "val"}:
            continue
        run = run_meta["name"]
        frames = aligned_frames(Path("data/EHL-2") / run)
        output[run] = build_run_segments(
            run,
            frames,
            cache_root,
            rel_threshold=rel_threshold,
            min_length=min_length,
        )
    return output


def sample_slice(
    segments: list[TemporalSegment],
    rng: random.Random,
    *,
    min_frames: int = 5,
    max_frames: int | None = None,
    remove_count: int = 0,
) -> SegmentedSlice:
    """Sample one non-uniform subsequence from a randomly selected segment.

    The first and last kept frames are always retained.  ``remove_count``
    interior frames are removed from the input sequence and are not part of the
    prediction target.
    """
    if not segments:
        raise ValueError("no segments available")
    segment = rng.choice(segments)
    upper = segment.length if max_frames is None else min(max_frames, segment.length)
    lower = min(min_frames, upper)
    if lower < 2:
        lower = 2
    slice_length = rng.randint(lower, upper)
    start = rng.randint(0, segment.length - slice_length)
    local_indices = list(range(start, start + slice_length))

    removable = local_indices[1:-1]
    remove_count = max(0, min(remove_count, len(removable)))
    removed = sorted(rng.sample(removable, remove_count)) if remove_count else []
    removed_set = set(removed)
    kept_local = [idx for idx in local_indices if idx not in removed_set]

    kept_abs = tuple(segment.frame_indices[idx] for idx in kept_local)
    removed_abs = tuple(segment.frame_indices[idx] for idx in removed)
    kept_times = tuple(segment.times[idx] for idx in kept_local)
    dts = [0.0]
    dts.extend(
        kept_times[i] - kept_times[i - 1]
        for i in range(1, len(kept_times))
    )

    return SegmentedSlice(
        run=segment.run,
        segment_start=segment.start,
        segment_end=segment.end,
        kept_frame_indices=kept_abs,
        removed_frame_indices=removed_abs,
        target_index=kept_abs[-1],
        times=kept_times,
        dts=tuple(dts),
        reset_state=True,
    )
