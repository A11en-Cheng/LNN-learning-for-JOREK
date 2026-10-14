"""Actual Fortran oracle: no training, global numerical solve deliberately excluded."""
from pathlib import Path

import numpy as np
import pytest

from tests.native_nearby_oracle import (JOREK_ROOT, compile_oracle, geometry, grid,
                                        native_mesh, run_oracle)


def python_locator():
    path = Path(__file__).resolve().parents[1] / 'src/models/native_nearby_locator.py'
    if not path.exists():
        pytest.skip('main agent has not created native_nearby_locator.py')
    from src.models.native_nearby_locator import NativeNearbyLocator
    return NativeNearbyLocator


@pytest.fixture(scope='session')
def oracle(tmp_path_factory):
    if not (JOREK_ROOT / 'grids/grid_utils/mod_find_rz_nearby.f90').exists():
        pytest.skip('JOREK_ROOT checkout is required')
    import shutil
    if shutil.which('gfortran') is None:
        pytest.skip('gfortran is required')
    return compile_oracle(tmp_path_factory.mktemp('native-nearby'))


CASES = [
    ('samecell', False, False, [.3, .4], 0, [.7, .6], 1),
    ('curved_crossing', True, False, [.8, .4], 1, [.2, .45], 2),
    ('reversededge', True, True, [.8, .4], 1, [.8, .55], 2),
    ('multicell', False, False, [.8, .4], 2, [.2, .4], 3),
    ('corner', False, False, [.5, .5], 0, [1., 1.], -1),
    ('lost', False, False, [.3, .4], 0, [.3, -.2], -1),
    ('lost_after_crossing', False, False, [.3, .4], 2, [1.2, .4], -3),
]


@pytest.mark.parametrize('name,curved,reverse,start,target_e,target_st,expected', CASES)
def test_actual_fortran_analytic_cases(oracle, name, curved, reverse, start, target_e, target_st, expected):
    p, vertex, nb = grid(curved=curved, reversed_edge=reverse)
    target = geometry(p, target_e, target_st)[0]
    result = run_oracle(oracle, p, vertex, nb, 1, start, target)
    assert result.element == expected
    assert result.flag == (-1 if expected < 0 else 0)
    assert result.fallback_calls == 0
    np.testing.assert_array_equal(result.target, target)  # intent(in), even on loss
    boundary = {'corner': [1., 1.], 'lost': [.3, 0.],
                'lost_after_crossing': [1., .4]}.get(name)
    np.testing.assert_allclose(result.st, boundary if expected < 0 else target_st, rtol=0, atol=1e-12)


@pytest.mark.parametrize('st,side', [([0., 0.], 4), ([1., 0.], 2),
                                     ([1., 1.], 3), ([0., 1.], 3)])
def test_diagonal_corner_side_selection(oracle, st, side):
    p, vertex, nb = grid(cells=1)
    nb[:, 0] = [-11, -12, -13, -14]
    result = run_oracle(oracle, p, vertex, nb, 1, st, [2., 0.], mode=1)
    assert result.element == -10-side
    np.testing.assert_array_equal(result.st, st)


@pytest.mark.parametrize('neighbor,expected_calls', [(0, 0), (-1, 1), (-7, 1)])
def test_zero_lost_negative_global_failed_seam(oracle, neighbor, expected_calls):
    p, vertex, nb = grid(cells=1)
    nb[0, 0] = neighbor
    result = run_oracle(oracle, p, vertex, nb, 1, [.3, .4], [2.3, -.2])
    assert (result.element, result.flag, result.fallback_calls) == (-1, -1, expected_calls)
    np.testing.assert_allclose(result.st, [.3, 0.], rtol=0, atol=1e-12)


def test_nonreciprocal_positive_neighbor_is_lost_without_global(oracle):
    p, vertex, nb = grid(cells=2)
    nb[3, 1] = 0
    result = run_oracle(oracle, p, vertex, nb, 1, [.8, .4], [3.2, .4])
    assert (result.element, result.flag, result.fallback_calls) == (-1, -1, 0)


@pytest.mark.parametrize('old', [0, -1, 4])
def test_invalid_hint_calls_global_directly(oracle, old):
    p, vertex, nb = grid()
    result = run_oracle(oracle, p, vertex, nb, old, [.3, .4], [2.7, .6], fallback=(1, 0, .7, .6))
    assert (result.element, result.flag, result.fallback_calls) == (1, 0, 1)
    np.testing.assert_allclose(result.st, [.7, .6], rtol=0, atol=1e-12)


@pytest.mark.parametrize('sign', [1., -1.])
def test_signed_determinant_floor_and_eight_iteration_exhaustion(oracle, sign):
    p, vertex, nb = grid(cells=1)
    # det=+/-1e-10: the floor contracts s error by .99 per iteration.
    p[0, 0, 1, 0] = sign*1e-10
    target = geometry(p, 0, [.7, .4])[0]
    result = run_oracle(oracle, p, vertex, nb, 1, [.3, .4], target,
                        fallback=(1, 0, .123, .456))
    assert (result.element, result.flag, result.fallback_calls) == (1, 3, 1)
    np.testing.assert_array_equal(result.st, [.123, .456])


@pytest.mark.parametrize('target_e,expected,flag,calls', [(7, 8, 0, 0), (8, 9, 3, 1)])
def test_eight_rounds_include_crossings_not_eight_rounds_per_element(oracle, target_e, expected, flag, calls):
    p, vertex, nb = grid(cells=10)
    target = geometry(p, target_e, [.2, .4])[0]
    result = run_oracle(oracle, p, vertex, nb, 1, [.3, .4], target,
                        fallback=(9, 0, .2, .4))
    assert (result.element, result.flag, result.fallback_calls) == (expected, flag, calls)
    np.testing.assert_allclose(result.st, [.2, .4], rtol=0, atol=1e-12)


@pytest.mark.parametrize('sign', [1., -1.])
def test_floor_convergence_on_exactly_eighth_round(oracle, sign):
    p, vertex, nb = grid(cells=1)
    p[0, 0, 0, 0] = 0.
    p[0, 0, 1, 0] = sign*1e-10
    start = [.3, .4]
    target = geometry(p, 0, start)[0] + [sign*1.08e-12, 0.]
    result = run_oracle(oracle, p, vertex, nb, 1, start, target)
    assert (result.element, result.flag, result.fallback_calls) == (1, 0, 0)
    # After seven rounds the residual is still >1e-12; eighth is below.
    expected_s = .3+.0108*(1-.99**8)
    np.testing.assert_allclose(result.st, [expected_s, .4], rtol=0, atol=1e-12)


def test_nan_residual_successful_fallback_has_flag_two(oracle):
    p, vertex, nb = grid(cells=1)
    p[0, 0, 0, 0] = np.nan
    result = run_oracle(oracle, p, vertex, nb, 1, [.3, .4], [2.7, .6],
                        fallback=(1, 0, .7, .6))
    assert (result.element, result.flag, result.fallback_calls) == (1, 2, 1)


@pytest.mark.parametrize('old_side', range(4))
@pytest.mark.parametrize('new_side', range(4))
@pytest.mark.parametrize('reverse', [False, True])
def test_all_edge_coordinate_transforms_actual_source(oracle, old_side, new_side, reverse):
    p, vertex, nb = grid(cells=2)
    nb[:] = 0
    nb[old_side, 0] = 2; nb[new_side, 1] = 1
    vertex[:, 0] = [1, 2, 3, 4]
    vertex[:, 1] = [5, 6, 7, 8]
    endpoints = vertex[[old_side, (old_side+1) % 4], 0]
    vertex[[new_side, (new_side+1) % 4], 1] = endpoints[::-1] if reverse else endpoints
    along = .27
    old = ([along, 0.], [1., along], [1-along, 1.], [0., 1-along])[old_side]
    x = 1-along if reverse else along
    expected = ([x, 0.], [1., x], [1-x, 1.], [0., 1-x])[new_side]
    result = run_oracle(oracle, p, vertex, nb, 1, old, [0., 0.], mode=1)
    assert result.element == 2
    np.testing.assert_allclose(result.st, expected, rtol=0, atol=1e-12)


def test_native_geometry_matches_independent_polynomial():
    p, vertex, _ = grid(curved=True, reversed_edge=True)
    mesh = native_mesh(p, vertex)
    for e in range(3):
        for st in ([.17, .29], [.81, .63], [0., 1.]):
            actual = mesh._location_geometry(e, np.array(st))
            expected = geometry(p, e, st)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize('name,curved,reverse,start,target_e,target_st,expected', CASES)
def test_python_matches_actual_fortran(oracle, name, curved, reverse, start, target_e, target_st, expected):
    # Do not import a production module until the main agent creates it.
    NativeNearbyLocator = python_locator()
    p, vertex, nb = grid(curved=curved, reversed_edge=reverse)
    mesh = native_mesh(p, vertex)
    target = geometry(p, target_e, target_st)[0]
    reference = run_oracle(oracle, p, vertex, nb, 1, start, target)
    locator = NativeNearbyLocator(mesh, nb)
    positions = np.array([[*target, .37]])
    saved = positions.copy()
    elm, st = locator.locate_batch(positions, element_hints=np.array([1]),
                                    local_hints=np.array([start], dtype=float))
    assert int(elm[0]) == reference.element
    np.testing.assert_allclose(st[0], reference.st, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(positions, saved)
    assert bool(elm[0] < 0) == (reference.flag == -1)
    assert int(locator.last_ifail[0]) == reference.flag
    assert locator.last_stats['global_fallback_points'] == 0
    assert isinstance(locator.last_stats, dict)


def test_negative_neighbor_failure_is_explicitly_different_global_seam(oracle):
    NativeNearbyLocator = python_locator()
    p, vertex, nb = grid(cells=1)
    nb[0, 0] = -7
    target = np.array([2.3, -.2])
    reference = run_oracle(oracle, p, vertex, nb, 1, [.3, .4], target)
    locator = NativeNearbyLocator(native_mesh(p, vertex), nb)
    elm, st = locator.locate_batch(np.array([[*target, 0.]]),
                                    element_hints=np.array([1]), local_hints=np.array([[.3, .4]]))
    # Actual JOREK translates failed crossing fallback into boundary loss;
    # the chosen Python seam terminates with its original global 0/99 policy.
    assert (reference.element, reference.flag, reference.fallback_calls) == (-1, -1, 1)
    assert int(elm[0]) == 0
    assert int(locator.last_ifail[0]) == 99
    assert locator.last_stats['negative_neighbour_points'] == 1
    assert locator.last_stats['global_fallback_points'] == 1


def test_malformed_saved_topology_is_explicit_constructor_difference(oracle):
    NativeNearbyLocator = python_locator()
    p, vertex, nb = grid(cells=2)
    nb[3, 1] = 0
    reference = run_oracle(oracle, p, vertex, nb, 1, [.8, .4], [3.2, .4])
    assert (reference.element, reference.flag, reference.fallback_calls) == (-1, -1, 0)
    with pytest.raises(ValueError, match='reciprocal'):
        NativeNearbyLocator(native_mesh(p, vertex), nb)


def test_positive_floor_exhaustion_matches_controlled_fallback_flag(oracle, monkeypatch):
    NativeNearbyLocator = python_locator()
    p, vertex, nb = grid(cells=1)
    p[0, 0, 1, 0] = 1e-10
    target = geometry(p, 0, [.7, .4])[0]
    reference = run_oracle(oracle, p, vertex, nb, 1, [.3, .4], target,
                            fallback=(1, 0, .123, .456))
    mesh = native_mesh(p, vertex)
    calls = []
    def controlled_global(x):
        calls.append(x.copy())
        return np.ones(len(x), dtype=np.int32), np.tile([.123, .456], (len(x), 1))
    monkeypatch.setattr(mesh, 'locate', controlled_global)
    locator = NativeNearbyLocator(mesh, nb)
    elm, st = locator.locate_batch(np.array([[*target, 0.]]),
                                    element_hints=np.array([1]), local_hints=np.array([[.3, .4]]))
    assert len(calls) == reference.fallback_calls == 1
    assert int(elm[0]) == reference.element
    assert int(locator.last_ifail[0]) == reference.flag == 3
    assert locator.last_stats['newton_particle_rounds'] == 8
    np.testing.assert_allclose(st[0], reference.st, rtol=0, atol=1e-12)
