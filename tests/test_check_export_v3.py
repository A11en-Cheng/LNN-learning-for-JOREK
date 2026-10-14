"""End-to-end tests for ``tools/check_export_v3.py``."""

from __future__ import annotations

import json

import h5py
import numpy as np

from tools import check_export_v3 as cli

from conftest import write_export_v3_run, write_native_run


def test_cli_ok_on_synthetic_export(tmp_path):
    export = write_export_v3_run(tmp_path, indices=(10, 11, 12), boundary=[0] * 6)
    out = tmp_path / "report.json"

    code = cli.main(["--run-dir", str(tmp_path), "--quiet", "--out", str(out)])

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["contract"]["n_steps"] == 3
    assert report["contract"]["require_single_step"] is True
    assert report["boundary"]["available"] is True
    assert report["condition"]["available"] is True
    assert report["shards"]["n_spi_start"]["median"] == 3
    assert report["budget"]["available"] is False
    assert export.exists()


def test_cli_requires_single_step_by_default(tmp_path):
    write_export_v3_run(
        tmp_path,
        indices=(10, 12),
        dt_step=1.0,
        dt_source=0.5,
        boundary=[0] * 6,
        manifest_extra={"gap_indices": [11]},
    )

    code = cli.main(["--run-dir", str(tmp_path), "--quiet"])
    relaxed = cli.main(["--run-dir", str(tmp_path), "--quiet", "--allow-multi-step"])

    assert code == 1
    assert relaxed == 0


def test_cli_budget_from_raw_dir(tmp_path):
    write_export_v3_run(tmp_path, indices=(10, 11), boundary=[0] * 6)
    raw = tmp_path / "raw"
    raw.mkdir()
    with h5py.File(raw / "jorek000004.h5", "w") as handle:
        handle["t_now"] = np.float64(0.2)
        handle["xtime"] = np.asarray([0.05, 0.10, 0.15, 0.20])
        handle["xtime_spi_ablation"] = np.asarray([0.0, 2.0, 5.0, 9.0])[:, None]
        handle["xtime_spi_ablation_rate"] = np.asarray([0.0, 2.0, 5.0, 9.0])[:, None]
        handle["xtime_Ne_imp"] = np.asarray([1.0e20] * 4)
        handle["xtime_E_ion"] = np.asarray([1.0, 3.0, 6.0, 10.0])
        handle["xtime_E_ion_power"] = np.asarray([-1.0e8] * 4)
        handle["xtime_radiation"] = np.asarray([0.0, 1.0, 2.0, 4.0])
        handle["xtime_rad_power"] = np.asarray([1.0e8] * 4)
        handle["t_norm"] = np.float64(2.0e-4)
    out = tmp_path / "report.json"

    code = cli.main(
        [
            "--run-dir",
            str(tmp_path),
            "--raw-run-dir",
            str(raw),
            "--quiet",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["budget"]["available"] is True
    assert report["budget"]["ablation"]["total_atoms"] == 9.0
    assert report["budget"]["ionization_radiation"]["layout"] == "history"
    assert report["alignment"]["checked"] is True
    assert report["alignment"]["jorek_frames_found"] == 0  # budget-only raw dir


# --------------------------------------------------------------------------- #
# Native run mode (raw jorek*/projections*/part* directory)
# --------------------------------------------------------------------------- #


def test_cli_native_run_mode(tmp_path):
    root = write_native_run(
        tmp_path / "run", indices=(10, 11, 12, 13), n_spi=2, restart_file=(10, 12)
    )
    out = tmp_path / "native.json"

    code = cli.main(
        [
            "--run-dir",
            str(root),
            "--quiet",
            "--no-budget",
            "--out",
            str(out),
        ]
    )

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["layout_kind"] == "native"
    assert report["contract"]["n_steps"] == 3
    assert report["contract"]["n_windows"] == 3
    assert report["contract"]["single_step_method"] == "native_adjacent_frame"
    assert report["contract"]["interval_source"] == "macroscopic_vars.dat"
    assert report["contract"]["zero_dt"]["indices"] == []
    assert report["manifest"]["layout_kind"] == "native"
    assert report["alignment"]["mode"] == "native_inline"
    assert report["restart"]["registry"]["n"] == 2
    assert report["restart"]["n_registry_checked"] == 2
    assert {row["index"] for row in report["restart"]["checks"]} == {10, 12}
    assert report["shards"]["n_spi_start"]["median"] == 2
    assert report["boundary"]["available"] is True
    assert report["condition"]["available"] is True
    assert report["condition"]["n_entries"] == [61]
    assert report["files"]["n_files"] == 7  # 4 jorek + 3 projections (u_after reuses jorek)
    assert report["linkage"]["checked"] == 2
    assert report["budget"]["available"] is False


def test_cli_native_zero_dt_policies(tmp_path):
    root = write_native_run(
        tmp_path / "run",
        indices=(10, 11, 12, 13),
        duplicate_frames=(12,),
        projection_lag={11: 0.5},
    )

    # Default (registered) without a registry: the unregistered dt=0 frame is
    # fail-fast.
    unregistered_code = cli.main(["--run-dir", str(root), "--quiet", "--no-budget"])
    error_code = cli.main(
        [
            "--run-dir",
            str(root),
            "--quiet",
            "--no-budget",
            "--zero-dt-policy",
            "error",
        ]
    )
    list_code = cli.main(
        [
            "--run-dir",
            str(root),
            "--quiet",
            "--no-budget",
            "--zero-dt-policy",
            "list",
        ]
    )
    registered_code = cli.main(
        [
            "--run-dir",
            str(root),
            "--quiet",
            "--no-budget",
            "--zero-dt-policy",
            "registered",
            "--restart-events",
            "12",
        ]
    )

    assert unregistered_code == 1
    assert error_code == 1
    assert list_code == 0
    assert registered_code == 0


def test_cli_native_budget_summary(tmp_path):
    root = write_native_run(tmp_path / "run", indices=(10, 11, 12))
    for index, (ablation, radiation) in ((11, (2.0, 1.0)), (12, (5.0, 2.0))):
        with h5py.File(root / f"jorek{index:06d}.h5", "r+") as handle:
            handle["xtime"] = np.asarray([0.05, 0.10, 0.15])
            handle["xtime_spi_ablation"] = np.asarray([0.0, ablation - 3.0, ablation])
            handle["xtime_spi_ablation_rate"] = np.asarray([0.0, 2.0, 5.0])
            handle["xtime_Ne_imp"] = np.full(3, 1.0e20)
            handle["xtime_E_ion"] = np.asarray([1.0, 3.0, 6.0])
            handle["xtime_E_ion_power"] = np.full(3, -1.0e8)
            handle["xtime_radiation"] = np.asarray([0.0, radiation - 1.0, radiation])
            handle["xtime_rad_power"] = np.full(3, 1.0e8)
    out = tmp_path / "report.json"

    code = cli.main(
        [
            "--run-dir",
            str(root),
            "--quiet",
            "--out",
            str(out),
            "--budget-step",
            "12",
        ]
    )

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    budget = report["budget"]
    assert budget["available"] is True
    assert budget["source"] == "native_run_dir"
    assert budget["native_layout"]["n_jorek_frames"] == 3
    assert budget["step_budget"]["step_index"] == 12
    assert budget["step_budget"]["ablated_atoms_step"] == 3.0
    assert budget["step_budget"]["read_convention"].startswith("xtime* tail")
