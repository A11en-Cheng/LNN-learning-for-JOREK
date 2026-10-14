"""Budget reader tests (``src/data/budget.py``), synthetic data only."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.data import budget


def write_budget_history(path, *, cumulative=(0.0, 2.0, 5.0, 9.0), tstep=0.05):
    n = len(cumulative)
    xtime = np.arange(1, n + 1, dtype=np.float64) * tstep
    rate = np.asarray(cumulative, dtype=np.float64)
    with h5py.File(path, "w") as handle:
        handle["xtime"] = xtime
        handle["xtime_spi_ablation"] = np.asarray(cumulative, dtype=np.float64)[:, None]
        handle["xtime_spi_ablation_rate"] = rate[:, None]
        handle["xtime_Ne_imp"] = np.full(n, 1.0e20)
        handle["xtime_E_ion"] = np.asarray([1.0, 3.0, 6.0, 10.0])
        handle["xtime_E_ion_power"] = np.full(n, -1.0e8)
        handle["xtime_radiation"] = np.asarray([0.0, 1.0, 2.0, 4.0])
        handle["xtime_rad_power"] = np.full(n, 1.0e8)
        handle["t_norm"] = 2.0e-4
        handle["tstep"] = tstep
        handle["index_now"] = np.int32(n)
    return path


def test_ablation_budget_history(tmp_path):
    path = write_budget_history(tmp_path / "jorek000004.h5")

    summary = budget.ablation_budget(path)

    assert summary["source"] == "jorek_history"
    assert summary["layout"] == "history"
    assert summary["total_atoms"] == pytest.approx(9.0)
    assert summary["last_step_atoms"] == pytest.approx(4.0)
    assert summary["rate_integral_atoms"] == pytest.approx(
        np.sum(np.asarray([0.0, 2.0, 5.0, 9.0]) * 0.05 * 2.0e-4)
    )


def test_detect_series_layout_slice_and_history(tmp_path):
    history = write_budget_history(tmp_path / "jorek000004.h5")
    assert budget.detect_series_layout(budget.read_series(history)) == "history"

    slice_path = tmp_path / "step_000004.h5"
    with h5py.File(slice_path, "w") as handle:
        handle["xtime"] = np.asarray([1.0])
        handle["xtime_spi_ablation"] = np.asarray([9.0])
    assert budget.detect_series_layout(budget.read_series(slice_path)) == "slice"


def test_read_abl_history(tmp_path):
    path = tmp_path / "abl_history.dat"
    header = "\n".join(
        ["       time", "abl_rate_inj", "abl_number_inj", "total_abl_rate", "total_abl_number"]
    )
    rows = [
        "1 0.010 " + " ".join(["1.0e20"] * 2 + ["0.0"] * 6 + ["2.0e20", "9.0", "0.0", "0.0"]),
        "2 0.020 " + " ".join(["2.0e20"] * 2 + ["0.0"] * 6 + ["4.0e20", "18.0", "0.0", "0.0"]),
    ]
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    history = budget.read_abl_history(path)

    assert history.n_inj == 2
    assert history.abl_tot.tolist() == [9.0, 18.0]
    assert history.total_ablated_atoms == pytest.approx(18.0)
    assert history.to_dict()["n_rows"] == 2


def test_global_closure_residuals(tmp_path):
    path = tmp_path / "macroscopic_vars.dat"
    lines = [
        '@energy_conservation: %"time" "-dEtotdt" "Sum"',
        "@energy_conservation: 0.01 10.0 9.0",
        "@energy_conservation: 0.02 12.0 12.5",
        '@dparticles_dt: %"time" "Total" "Ions" "Neutrals"',
        "@dparticles_dt: 0.01 5.0 3.0 2.0",
        "@dparticles_dt: 0.02 8.0 6.0 3.0",
        '@particlecontent: %"time" "Total main ions" "Inside" "Outside" "Neutrals" "Impurities"',
        "@particlecontent: 0.01 10.0 6.0 4.0 0.0 2.0",
        '@bnd_particle_fluxes: %"time" "Dpar" "Dperp" "Vpar" "Vperp" "neutral"',
        "@bnd_particle_fluxes: 0.01 1.0 2.0 3.0 4.0 5.0",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    closure = budget.global_closure(tmp_path)

    assert closure["available"] is True
    assert closure["energy_conservation"]["residual"]["last"] == pytest.approx(-0.5)
    assert closure["energy_conservation"]["residual"]["max_abs"] == pytest.approx(1.0)
    assert closure["dparticles_dt"]["residual"]["last"] == pytest.approx(-1.0)
    assert closure["particlecontent"]["last"]["total_impurities"] == pytest.approx(2.0)
    assert closure["bnd_particle_fluxes"]["sum_last"] == pytest.approx(15.0)


def test_closure_stats_handles_nan(tmp_path):
    path = tmp_path / "macroscopic_vars.dat"
    lines = [
        '@dparticles_dt: %"time" "Total" "Ions" "Neutrals"',
        "@dparticles_dt: 0.01 1.0 0.5 0.5",
        "@dparticles_dt: 0.02 nan 0.5 0.5",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    closure = budget.global_closure(tmp_path)

    residual = closure["dparticles_dt"]["residual"]
    assert residual["n_nonfinite"] == 1


def test_projection_integrals_unit_square(tmp_path):
    path = tmp_path / "projections000001.h5"
    with h5py.File(path, "w") as handle:
        values = np.ones((5, 4, 1, 4), dtype=np.float64)
        handle["values"] = values
        x = np.zeros((2, 4, 1, 4), dtype=np.float64)
        x[0, 0, 0, :] = [0.0, 1.0, 1.0, 0.0]
        x[1, 0, 0, :] = [0.0, 0.0, 1.0, 1.0]
        handle["x"] = x
        handle["vertex"] = np.asarray([[1], [2], [3], [4]], dtype=np.int32)

    result = budget.projection_integrals(path)

    assert result["ok"] is True
    assert result["measure_total"] == pytest.approx(1.0)
    assert result["channel_integrals"]["N_imp"] == pytest.approx(1.0)
    assert result["channel_integrals"]["dE_ion"] == pytest.approx(1.0)


def test_summarize_budget_absent_is_graceful(tmp_path):
    summary = budget.summarize_budget(tmp_path)

    assert "error" in summary["ablation"] or summary["ablation"]["source"] is None
    assert summary["impurity_particle_budget"]["owner"] == "Phase B1"
    assert "gate" in summary


# --------------------------------------------------------------------------- #
# Native layout (jorek*/projections*/part* in one run directory)
# --------------------------------------------------------------------------- #


def write_native_jorek(path, *, index, cumulative, time_step=0.05, t_norm=2.0e-4):
    """Write a minimal native ``jorek{index}.h5`` with full xtime histories."""
    n = index
    with h5py.File(path, "w") as handle:
        handle["t_now"] = np.float64(index * time_step)
        handle["t_norm"] = np.float64(t_norm)
        handle["tstep"] = np.float64(time_step)
        handle["index_now"] = np.int32(index)
        handle["values"] = np.zeros((9, 4, 1, 6), dtype=np.float64)
        handle["xtime"] = np.arange(1, n + 1, dtype=np.float64) * time_step
        handle["xtime_spi_ablation"] = np.asarray(cumulative, dtype=np.float64)
        handle["xtime_spi_ablation_rate"] = np.full(n, 1.0e20)
        handle["xtime_Ne_imp"] = np.full(n, 2.0e20)
        handle["xtime_E_ion"] = np.linspace(0.0, 1.0, n)
        handle["xtime_E_ion_power"] = np.full(n, -3.0e8)
        handle["xtime_radiation"] = np.linspace(0.0, 2.0, n)
        handle["xtime_rad_power"] = np.full(n, 4.0e8)
    return path


def test_read_series_tail_is_tail_only(tmp_path):
    path = write_native_jorek(
        tmp_path / "jorek000004.h5", index=4, cumulative=[0.0, 1.0, 2.0, 5.0]
    )

    tail = budget.read_series_tail(path)

    assert tail["xtime"] == pytest.approx(0.20)
    assert tail["xtime_spi_ablation"] == pytest.approx(5.0)
    assert tail["xtime_Ne_imp"] == pytest.approx(2.0e20)


def test_native_step_budget_differences_and_tails(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_native_jorek(run / "jorek000003.h5", index=3, cumulative=[0.0, 1.0, 2.0])
    write_native_jorek(run / "jorek000004.h5", index=4, cumulative=[0.0, 1.0, 2.0, 5.0])

    step = budget.native_step_budget(run, 4)

    assert step["available"] is True
    assert step["layout"] == "native"
    assert step["read_convention"].startswith("xtime* tail element only")
    assert step["ablated_atoms_step"] == pytest.approx(3.0)
    assert step["electrons_injected_step"] == pytest.approx(2.0e20 * 0.05 * 2.0e-4)
    assert step["ionization_energy_step"] == pytest.approx(-3.0e8 * 0.05)
    assert step["radiated_energy_step"] == pytest.approx(4.0e8 * 0.05)
    assert step["tstep_jorek"] == pytest.approx(0.05)


def test_native_step_budget_registers_abl_history(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_native_jorek(run / "jorek000003.h5", index=3, cumulative=[0.0, 1.0, 2.0])
    write_native_jorek(run / "jorek000004.h5", index=4, cumulative=[0.0, 1.0, 2.0, 5.0])
    rows = [
        "1 3.0 " + " ".join(["1.0e20"] * 2 + ["0.0"] * 6 + ["2.0e20", "2.0", "0.0", "0.0"]),
        "2 4.0 " + " ".join(["2.0e20"] * 2 + ["0.0"] * 6 + ["4.0e20", "9.0", "0.0", "0.0"]),
    ]
    (run / "abl_history.dat").write_text(
        "header\ntime\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )

    step = budget.native_step_budget(run, 4)

    assert step["abl_history"]["abl_tot"] == pytest.approx(9.0)
    assert step["abl_history"]["ablated_atoms_step"] == pytest.approx(7.0)


def test_summarize_budget_registers_native_layout(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write_native_jorek(run / "jorek000003.h5", index=3, cumulative=[0.0, 1.0, 2.0])
    write_native_jorek(run / "jorek000004.h5", index=4, cumulative=[0.0, 1.0, 2.0, 5.0])
    (run / "macroscopic_vars.dat").write_text(
        "\n".join(
            [
                '@energy_conservation: %"time" "-dEtotdt" "Sum"',
                "@energy_conservation: 0.01 10.0 9.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = budget.summarize_budget(run)

    assert summary["native_layout"]["n_jorek_frames"] == 2
    assert summary["native_layout"]["index_first"] == 3
    assert summary["native_layout"]["per_step_reader"].startswith("native_step_budget")
    assert summary["ablation"]["source"] == "jorek_history"
    assert summary["global_closure"]["available"] is True


def test_restart_segmented_budget(tmp_path):
    """ADR-0020 §5: the cumulative series resets at code restarts; budgets are
    segmented and the per-step value at the boundary uses the rate series."""
    run = tmp_path / "run"
    run.mkdir()
    write_native_jorek(
        run / "jorek000005.h5",
        index=5,
        cumulative=[0.0, 1.0, 2.0, 0.5, 1.5],  # reset at row 3
    )
    (run / "restart_events.json").write_text(
        '{"restart_events": [5]}', encoding="utf-8"
    )

    detected = budget.detect_restart_events(run)
    assert detected["events"] == [5]

    reconciliation = budget.ablation_series_reconciliation(run / "jorek000005.h5")
    assert reconciliation["reset_indices"] == [3]
    assert [segment["atoms"] for segment in reconciliation["restart_segments"]] == [
        2.0,
        1.0,
    ]
    assert reconciliation["segmented_total_atoms"] == pytest.approx(3.0)
    assert reconciliation["cumulative_last"] == pytest.approx(1.5)
    assert reconciliation["registered_restart_events"]["events"] == [5]

    step = budget.native_step_budget(run, 5)
    assert step["restart_segment_boundary"] is True
    assert step["restart_segment_reason"] == "registered_event"
    assert step["ablated_atoms_step"] == pytest.approx(1.0e20 * 0.05 * 2.0e-4)
    assert step["ablated_atoms_step_source"].startswith("xtime_spi_ablation_rate")

    summary = budget.summarize_budget(run)
    assert summary["restart_segmentation"]["events"] == [5]
    assert summary["restart_segmentation"]["segmented_total_atoms"] == pytest.approx(3.0)
