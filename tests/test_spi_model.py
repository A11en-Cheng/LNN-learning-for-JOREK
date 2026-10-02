"""Tests for the analytic JOREK SPI backbone (ablation model 3 / rocket model 6)."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from src.physics.spi_model import (
    EL_CHG,
    MOLE_NUMBER,
    MU_ZERO,
    NonlocalAblationTable,
    advance_shards_analytic,
    erode_radius,
    parks_ablation_neon,
    peloton_rocket_acceleration,
    temperature_ev,
)


def test_temperature_conversion_matches_jorek_normalisation():
    te = torch.tensor([1.0])
    value = temperature_ev(te, central_density=0.8)
    expected = 1.0 / (1.602176565e-19 * 4 * math.pi * 1e-7 * 0.8 * 1e20)
    assert torch.allclose(value, torch.tensor([expected]), rtol=1e-6)


def test_parks_ablation_neon_matches_hand_computation():
    radius = torch.tensor([1e-4])
    te_ev = torch.tensor([1e3])
    ne_si = torch.tensor([8e19])
    species = torch.tensor([1.0])
    ablation = parks_ablation_neon(radius, te_ev, ne_si, species)
    expected = (
        27.0837
        * MOLE_NUMBER
        * ((1e-4 * 1e2 / 0.2) ** (4.0 / 3.0))
        * ((8e19 * 1e-20) ** (1.0 / 3.0))
        * ((1e3 / 2e3) ** (5.0 / 3.0))
        / 20.183
    )
    assert torch.allclose(ablation, torch.tensor([expected]), rtol=1e-6)
    assert float(ablation) == float(ablation.clamp_min(0.0))


def test_parks_ablation_scales_and_clamps():
    base = parks_ablation_neon(
        torch.tensor([1e-4]), torch.tensor([1e3]), torch.tensor([8e19]), torch.tensor([1.0])
    )
    doubled = parks_ablation_neon(
        torch.tensor([2e-4]), torch.tensor([1e3]), torch.tensor([8e19]), torch.tensor([1.0])
    )
    assert abs(float(doubled / base) - 2.0 ** (4.0 / 3.0)) < 1e-4
    zero_radius = parks_ablation_neon(
        torch.tensor([0.0]), torch.tensor([1e3]), torch.tensor([8e19]), torch.tensor([1.0])
    )
    assert float(zero_radius) == 0.0
    cold = parks_ablation_neon(
        torch.tensor([1e-4]), torch.tensor([0.0]), torch.tensor([8e19]), torch.tensor([1.0])
    )
    assert float(cold) == 0.0


def test_peloton_rocket_direction_and_zero_axial_force():
    te_ev = torch.tensor([1e3])
    ne_si = torch.tensor([8e19])
    radius = torch.tensor([1e-4])
    species = torch.tensor([1.0])
    a_r, a_z, a_rxz = peloton_rocket_acceleration(
        te_ev, ne_si, radius, species, delta_phi=torch.tensor([0.0]), pellet_density=4.3e8
    )
    assert float(a_r) > 0.0
    assert float(a_z) == 0.0
    assert abs(float(a_rxz)) < 1e-6
    perpendicular = peloton_rocket_acceleration(
        te_ev, ne_si, radius, species, delta_phi=torch.tensor([math.pi / 2]), pellet_density=4.3e8
    )
    assert abs(float(perpendicular[0])) < abs(float(perpendicular[2])) * 1e-5


def test_radius_erosion_and_clip():
    radius = torch.tensor([1e-4])
    ablation = torch.tensor([4.3482e21])
    density = torch.tensor([4.3e28])
    eroded = erode_radius(radius, ablation, density, 4.1e-6)
    expected = 1e-4 - 4.1e-6 * 4.3482e21 / (4 * math.pi * (1e-4) ** 2 * 4.3e28)
    assert torch.allclose(eroded, torch.tensor([expected]), rtol=1e-4)
    aggressive = erode_radius(radius, torch.tensor([1e40]), density, 1.0)
    assert float(aggressive) == 0.0


def test_analytic_step_advances_and_keeps_positivity():
    values = torch.zeros(2, 16)
    values[0, :] = torch.tensor(
        [1.7, 0.0, 0.0, 0.0, -300.0, 0.0, 0.0, 1e-4, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    values[1, :] = values[0, :]
    values[1, 7] = 0.0  # dead shard
    mask = torch.tensor([True, False])
    updated, ablation = advance_shards_analytic(
        values,
        mask,
        local_rho=torch.tensor([0.8, 0.8]),
        local_rho_imp=torch.tensor([0.0, 0.0]),
        # Realistic EHL-2 electron temperature: ~0.03 JOREK units ~ 1.9 keV.
        local_te=torch.tensor([0.03, 0.03]),
        central_density=0.8,
        dt_jorek=10.0,
        t_norm=4.085775e-07,
        tstep_jorek=10.0,
        pellet_density=4.3e8,
    )
    expected_shift = -300.0 * 10.0 * 4.085775e-07
    assert torch.allclose(updated[0, 0], torch.tensor(1.7 + expected_shift), rtol=1e-5)
    assert float(updated[0, 7]) > 0.0
    assert float(updated[1, 7]) == 0.0
    assert float(ablation[1]) == 0.0
    assert torch.isfinite(updated).all()


def _write_table(path: Path, values: list[tuple[float, float, float]]) -> NonlocalAblationTable:
    lines = [
        f"{i:5d} {psi_n:14.6e} {te:14.6e} {ne:14.6e}"
        for i, (psi_n, te, ne) in enumerate(values, start=1)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return NonlocalAblationTable.from_file(path)


def test_nonlocal_table_parses_and_converts_to_si(tmp_path):
    table = _write_table(
        tmp_path / "nonlocal_array.dat",
        [(0.0, 3.10758e-2, 1.32382e1), (0.5, 2.0e-2, 1.0e1), (1.0, 1.0e-2, 5.0e0)],
    )
    assert len(table) == 3
    te_ev, ne_si, inside = table.surface_state(
        torch.tensor([0.5]), psi_axis=0.5, psi_bnd=0.1, central_density=0.8
    )
    expected_te = 3.10758e-2 / (EL_CHG * MU_ZERO * 0.8 * 1e20)
    expected_ne = 1.32382e1 * 1e20 * 0.8
    assert torch.allclose(te_ev, torch.tensor([expected_te], dtype=te_ev.dtype), rtol=1e-6)
    assert torch.allclose(ne_si, torch.tensor([expected_ne], dtype=ne_si.dtype), rtol=1e-6)
    assert bool(inside[0])


def test_nonlocal_lookup_uses_nearest_index_and_axis_tiebreak(tmp_path):
    table = _write_table(
        tmp_path / "nonlocal_array.dat",
        [(0.0, 1.0, 1.0), (0.25, 2.0, 2.0), (0.5, 3.0, 3.0), (0.75, 4.0, 4.0)],
    )
    # 0.30 is nearest to index 1 (0.25); the shard lies farther from the axis
    # than the table entry, so JOREK keeps it.
    assert int(table.nearest_index(torch.tensor([0.30]))[0]) == 1
    # 0.24 is nearest to index 1 as well, but now the table entry lies farther
    # from the axis than the shard -> step one index toward the axis.
    assert int(table.nearest_index(torch.tensor([0.24]))[0]) == 0
    # Outside the LCFS is flagged but still yields a finite lookup.
    _, _, inside = table.surface_state(
        torch.tensor([1.5]), psi_axis=0.0, psi_bnd=1.0, central_density=0.8
    )
    assert not bool(inside[0])


def test_analytic_step_takes_max_with_nonlocal_branch(tmp_path):
    table = _write_table(
        tmp_path / "nonlocal_array.dat",
        [(0.0, 1.0e-2, 1.0e1), (0.5, 1.0e-1, 1.0e2), (1.0, 1.0e-2, 1.0e1)],
    )
    values = torch.zeros(2, 16)
    values[:, 7] = 1e-4
    values[:, 9] = 1.0
    mask = torch.tensor([True, True])
    common = dict(
        local_rho=torch.tensor([0.8, 0.8]),
        local_rho_imp=torch.tensor([0.0, 0.0]),
        local_te=torch.tensor([0.03, 0.03]),
        central_density=0.8,
        dt_jorek=0.0,
        tstep_jorek=0.0,
    )
    local_only, _ = advance_shards_analytic(values, mask, **common)
    combined, ablation = advance_shards_analytic(
        values,
        mask,
        nonlocal_table=table,
        # Shard 0 sits at psi_n = 0.8 -> JOREK's axis-side lookup lands on the
        # hot table row (index 1) and must be raised by the nonlocal branch;
        # shard 1 is outside the LCFS (psi_n = 1.125) and keeps the local rate.
        psi_shard=torch.tensor([0.18, 0.05]),
        psi_axis=0.5,
        psi_bnd=0.1,
        **common,
    )
    assert float(ablation[0]) > float(local_only[0, 8])
    assert float(combined[1, 8]) == float(local_only[1, 8])
