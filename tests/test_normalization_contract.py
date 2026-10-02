"""Normalisation round-trip contract (audit guard).

The network lives in ``z-score(symlog(...))`` space while every physics residual
must be evaluated in raw JOREK units.  The route is only sound if
``inverse_transform_state`` \circ ``destandardize`` is an exact inverse of
``standardize`` \circ ``transform_state``.  These tests pin that contract down,
including the 4 symlog channels where an un-inverted transform would silently
feed ``symlog(rho)`` into the density equation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.model502_schema import (
    FIELD_NAMES,
    SYMLOG_FIELD_INDICES,
    NUM_STATE_FIELDS,
    destandardize,
    inverse_transform_state,
    standardize,
    transform_state,
)


def test_transform_roundtrip_is_exact_for_positive_state() -> None:
    torch.manual_seed(0)
    raw = torch.rand(64, NUM_STATE_FIELDS, dtype=torch.float64) * 5.0
    restored = inverse_transform_state(transform_state(raw))
    torch.testing.assert_close(restored, raw, rtol=0, atol=1e-12)


def test_transform_roundtrip_is_exact_for_negative_state() -> None:
    """symlog must keep the sign: the old clamp made negatives unrecoverable."""
    torch.manual_seed(0)
    raw = torch.randn(64, NUM_STATE_FIELDS, dtype=torch.float64) * 1e-2
    restored = inverse_transform_state(transform_state(raw))
    torch.testing.assert_close(restored, raw, rtol=0, atol=1e-15)


def test_transform_roundtrip_survives_zscore() -> None:
    torch.manual_seed(0)
    raw = torch.rand(64, NUM_STATE_FIELDS, dtype=torch.float64) * 5.0
    transformed = transform_state(raw)
    mean = transformed.mean(dim=0)
    std = transformed.std(dim=0)
    normalised = standardize(transformed, mean, std, name="state")
    restored = inverse_transform_state(
        destandardize(normalised, mean, std, name="state")
    )
    torch.testing.assert_close(restored, raw, rtol=0, atol=1e-12)


def test_symlog_fields_are_the_positive_channels() -> None:
    # Guards against a re-ordering of FIELD_NAMES silently changing which
    # channels get symlog (the psi/Z_j swap class of bug).
    assert {FIELD_NAMES[index] for index in SYMLOG_FIELD_INDICES} == {
        "rho",
        "T_i",
        "rho_imp",
        "T_e",
    }


@pytest.mark.parametrize("position", [0, 5])
def test_dataset_normalisation_matches_raw_frames(position: int) -> None:
    """End-to-end: a cached frame's normalised tensor must de-normalise back.

    With symlog the round trip is exact everywhere, including the noise-level
    negative values JOREK writes at boundary nodes (1.7% of rho_imp before).
    """
    manifest_path = Path(".scratch/n/manifest.json")
    cache = Path(".scratch/n/frames")
    if not manifest_path.exists() or not cache.exists():
        pytest.skip("frame cache not available")
    from src.data.interp_dataset import (
        InterpolationWindowDataset,
        RunFrameStore,
        compute_transform_stats,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fold = manifest["folds"][0]
    stores = {
        name: RunFrameStore(Path(manifest["data_root"]) / name, cache_dir=cache).load()
        for name in fold["train_runs"]
    }
    stats = compute_transform_stats(stores, runs=fold["train_runs"])
    dataset = InterpolationWindowDataset(
        manifest, manifest["data_root"], runs=[fold["held_out_run"]], inner="val",
        stats=stats, cache_dir=cache, with_shards=False,
    )
    item = dataset[min(position, len(dataset) - 1)]
    mean = torch.tensor(stats["field_mean"])
    std = torch.tensor(stats["field_std"])
    restored = inverse_transform_state(
        destandardize(item["u"].reshape(-1, NUM_STATE_FIELDS), mean, std, name="u")
    )
    raw = item["u_raw"].reshape(-1, NUM_STATE_FIELDS).float()
    restored = restored.float()
    # float32 cache: the symlog/symexp pair loses a few ULPs on the O(50) fields.
    torch.testing.assert_close(restored, raw, rtol=1e-5, atol=1e-5)
