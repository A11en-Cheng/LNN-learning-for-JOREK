"""Behaviour tests for the minimal Fortran namelist reader."""

from __future__ import annotations

import pytest

from src.data.namelist import parse_namelist


def test_parse_scalars_strings_and_d_exponents():
    text = """
&in1
  eta = 1.443d-10 ! eta_Spitzer
  visco = 4.d-4
  time_evol_scheme= "Gears"
  T_max_eta = 1.d99
/
"""
    values = parse_namelist(text)

    assert values["eta"] == pytest.approx(1.443e-10)
    assert values["visco"] == pytest.approx(4e-4)
    assert values["time_evol_scheme"] == "Gears"
    assert values["t_max_eta"] == pytest.approx(1e99)


def test_parse_indexed_array_assignments():
    text = """
  D_perp(1) = 5.d-6
  D_perp(2) = 0.d0
  D_perp(3) = 0.13d0
  D_perp(4) = 0.01d0
  D_perp(5) = 0.92d0
  D_perp(6) = 0.d0
"""
    values = parse_namelist(text)

    assert values["d_perp"] == pytest.approx([5e-6, 0.0, 0.13, 0.01, 0.92, 0.0])


def test_parse_array_constructor_and_range_assignment():
    text = """
  ZK_i_perp(1:5) = (/ 1.d-5, 0.d0, 0.d0, 99.d0, 99.d0 /)
"""
    values = parse_namelist(text)

    assert values["zk_i_perp"] == pytest.approx([1e-5, 0.0, 0.0, 99.0, 99.0])


def test_parse_multiple_assignments_on_one_line_and_inline_comments():
    text = "eta = 1.d0 , visco = 2.d0 ! viscosity\nzk_i_par = 8.1d2 !d-3\n"
    values = parse_namelist(text)

    assert values["eta"] == pytest.approx(1.0)
    assert values["visco"] == pytest.approx(2.0)
    assert values["zk_i_par"] == pytest.approx(810.0)


def test_ignores_block_markers_headers_and_junk():
    text = "*  namelist produced by eqdsk2jorek\n&in1\nEFITD\n/"
    values = parse_namelist(text)

    assert values == {}

