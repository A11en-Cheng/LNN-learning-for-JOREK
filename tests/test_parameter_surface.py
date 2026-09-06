"""Machine-checkable cross-validation of the Model 502 parameter surface.

The report ``report/model502-parameter-surface.json`` is the single source of
truth for which zero-dimensional parameters the nine-field assembly needs,
which of them JOREK already exports into ``jorek*.h5``, which must be added to
``mod_export_restart.f90``, and which are absorbed by the network.  These tests
keep the JSON honest: every exported key must exist in a real frame, and the
human-readable Markdown must mention every parameter.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "report" / "model502-parameter-surface.json"
MARKDOWN_PATH = ROOT / "report" / "model502-parameter-surface.md"
REAL_FRAME = ROOT / "data" / "EHL-2" / "scan_1d22_ang12_massloss_neon" / "jorek002710.h5"

ALLOWED_STATUS = {"exported", "to-export", "absorbed", "source-channel", "derived"}


def _surface() -> dict:
    assert JSON_PATH.exists(), "report/model502-parameter-surface.json is missing"
    surface = json.loads(JSON_PATH.read_text())
    assert surface["schema"].startswith("model502-parameter-surface")
    return surface


def test_parameter_entries_are_wellformed():
    surface = _surface()
    names = [entry["name"] for entry in surface["parameters"]]
    assert len(names) == len(set(names)), "duplicate parameter names"
    for entry in surface["parameters"]:
        assert entry["name"]
        assert entry["status"] in ALLOWED_STATUS
        assert entry["jorek_variable"]
        if entry["status"] == "exported":
            assert entry["h5_key"], "exported parameters must name their h5 key"


def test_markdown_mentions_every_parameter():
    surface = _surface()
    markdown = MARKDOWN_PATH.read_text()
    missing = [entry["name"] for entry in surface["parameters"] if entry["name"] not in markdown]
    assert not missing, f"parameters missing from the Markdown report: {missing}"


def test_exported_keys_exist_in_real_frame():
    if not REAL_FRAME.exists():
        pytest.skip("real EHL-2 data not present")
    surface = _surface()
    with h5py.File(REAL_FRAME, "r") as handle:
        for entry in surface["parameters"]:
            if entry["status"] == "exported":
                assert entry["h5_key"] in handle, (
                    f"'{entry['name']}' claims h5 key '{entry['h5_key']}' but the real "
                    "frame does not contain it"
                )

