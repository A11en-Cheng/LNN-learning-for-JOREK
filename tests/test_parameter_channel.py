"""Behaviour tests for the zero-dimensional parameter channel (#0005)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data.parameter_channel import (
    build_condition_vector,
    derive_time_evol_params,
    rescale_zeta,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_frame_with_scalars(path):
    with h5py.File(path, "w") as f:
        for key, value in {
            "eta": 1.443e-10,
            "visco": 1e-5,
            "visco_par": 4e-4,
            "tstep": 0.5,
            "t_now": 100.0,
            "central_density": 1.0,
            "central_mass": 1.0,
            "F0": 1.0,
            "R_domm": 1.0,
            "sqrt_mu0_rho0": 1.0,
            "t_norm": 1.0,
        }.items():
            f[key] = value


def test_derive_time_evol_params_matches_jorek_logic():
    assert derive_time_evol_params("Gears") == pytest.approx((1.0, 0.5))
    assert derive_time_evol_params("Crank-Nicholson") == pytest.approx((0.5, 0.0))
    assert derive_time_evol_params("implicit Euler") == pytest.approx((1.0, 0.0))
    with pytest.raises(ValueError):
        derive_time_evol_params("unknown scheme")


def test_rescale_zeta_replicates_variable_dt_assembly():
    assert rescale_zeta(zeta=0.5, tstep=2.0, tstep_prev=6.0) == pytest.approx(0.25)
    assert rescale_zeta(zeta=0.5, tstep=2.0, tstep_prev=2.0) == pytest.approx(0.5)


def test_build_condition_vector_fills_missing_with_jorek_defaults(tmp_path):
    frame = tmp_path / "jorek00001.h5"
    _write_frame_with_scalars(frame)
    namelist = """
&in1
  ZK_i_par = 8.1d2
  ZK_e_par = 8.1d3
  D_perp(1) = 5.d-6
  D_perp(2) = 0.d0
  D_perp(3) = 0.13d0
  D_perp(4) = 0.01d0
  D_perp(5) = 0.92d0
  D_perp(6) = 0.d0
  time_evol_scheme= "Gears"
/
"""
    with h5py.File(frame, "r") as handle:
        condition = build_condition_vector(handle, namelist_text=namelist)

    index = {name: position for position, name in enumerate(condition["names"])}
    vector = condition["vector"]
    provenance = condition["provenance"]
    assert vector[index["eta"]] == pytest.approx(1.443e-10)
    assert provenance["eta"] == "h5"
    assert vector[index["ZK_i_par"]] == pytest.approx(810.0)
    assert provenance["ZK_i_par"] == "namelist"
    assert vector[index["visco_par_par"]] == pytest.approx(0.0)
    assert provenance["visco_par_par"] == "default"
    assert vector[index["time_evol_theta"]] == pytest.approx(1.0)
    assert vector[index["time_evol_zeta"]] == pytest.approx(0.5)
    assert provenance["time_evol_zeta"] == "derived"
    assert vector[index["D_perp[1]"]] == pytest.approx(5e-6)
    assert provenance["D_perp[6]"] == "namelist"
    assert vector[index["T_0"]] == pytest.approx(1e-6)
    assert provenance["T_0"] == "default"
    assert np.isfinite(vector).all()


def test_condition_spec_covers_exported_and_to_export_surface_entries():
    real_frame = ROOT / "data" / "EHL-2" / "scan_1d22_ang12_massloss_neon" / "jorek002710.h5"
    if not real_frame.exists():
        pytest.skip("real EHL-2 data not present")
    surface = json.loads(
        (ROOT / "report" / "model502-parameter-surface.json").read_text()
    )
    required = {
        entry["name"]
        for entry in surface["parameters"]
        if entry["status"] in {"exported", "to-export"} and entry["role"] == "condition"
    }
    # Arrays flatten to [1..6]; derived theta/zeta map from time_evol_scheme.
    array_entries = {"D_perp", "ZK_i_perp", "ZK_e_perp"}
    covered = set()
    with h5py.File(real_frame, "r") as handle:
        condition = build_condition_vector(handle, namelist_text="")
    for name in condition["names"]:
        base = name.split("[", 1)[0]
        covered.add(base)
    assert covered >= required - array_entries
    assert all(f"{entry}[1]" in condition["names"] for entry in array_entries)


@pytest.mark.skipif(
    not (ROOT / "data" / "EHL-2" / "scan_1d22_ang12_massloss_neon").exists(),
    reason="real EHL-2 data not present",
)
def test_real_frame_and_namelist_values():
    run_dir = ROOT / "data" / "EHL-2" / "scan_1d22_ang12_massloss_neon"
    namelist_text = (run_dir / "jorek_namelist_spitest").read_text(errors="replace")
    with h5py.File(run_dir / "jorek002710.h5", "r") as handle:
        condition = build_condition_vector(handle, namelist_text=namelist_text)
        h5_visco_par = float(handle["visco_par"][()].item())
    index = {name: position for position, name in enumerate(condition["names"])}
    vector = condition["vector"]
    assert vector[index["eta"]] == pytest.approx(1.443e-10)
    assert vector[index["ZK_i_par"]] == pytest.approx(1.6666e5)
    assert vector[index["ZK_e_par"]] == pytest.approx(4.1037e6)
    assert vector[index["visco_par"]] == pytest.approx(h5_visco_par)
    assert vector[index["visco_par_par"]] == pytest.approx(0.0)
    assert condition["provenance"]["visco_par_par"] == "default"
    assert vector[index["time_evol_theta"]] == pytest.approx(1.0)
    assert vector[index["time_evol_zeta"]] == pytest.approx(0.5)
