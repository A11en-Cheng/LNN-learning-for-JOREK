import numpy as np
import pytest

from src.models.native_location_candidates import location_candidates


def oldloop(candidates, centers, bounds, point, hint=-1):
    ordered = sorted(candidates, key=lambda e: np.linalg.norm(centers[e] - point))
    if 0 <= hint < len(centers):
        ordered = [hint] + [e for e in ordered if e != hint]
    result = []
    for e in ordered:
        lower, upper = bounds[e]
        if np.any(point < lower - 1e-9) or np.any(point > upper + 1e-9):
            continue
        result.append(e)
    return result


def test_randomized_order_matches_independent_oldloop():
    rng = np.random.default_rng(98121)
    for _ in range(500):
        n = int(rng.integers(1, 150))
        centers = rng.normal(size=(n, 2))
        widths = rng.uniform(0, 3, size=(n, 2))
        bounds = np.stack((centers - widths, centers + widths), axis=1)
        point = rng.normal(size=2)
        candidates = rng.integers(0, n, size=int(rng.integers(0, 200))).tolist()
        hint = int(rng.integers(-1, n + 2))
        before = candidates.copy()
        assert location_candidates(candidates, centers, bounds, point, hint) == oldloop(
            candidates, centers, bounds, point, hint)
        assert candidates == before


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('side', [0, 1])
def test_exact_boundary_comparisons(axis, side):
    centers = np.zeros((1, 2))
    bounds = np.array([[[-1., -2.], [1., 2.]]])
    edge = bounds[0, side, axis] + (-1e-9 if side == 0 else 1e-9)
    for value in (edge, np.nextafter(edge, -np.inf), np.nextafter(edge, np.inf)):
        point = np.zeros(2)
        point[axis] = value
        expected = oldloop([0], centers, bounds, point)
        assert location_candidates([0], centers, bounds, point) == expected
        assert location_candidates([], centers, bounds, point, 0) == expected


def test_ties_hint_duplicates_and_outside_hint():
    centers = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
    bounds = np.tile(np.array([[-2., -2.], [2., 2.]]), (4, 1, 1))
    point = np.zeros(2)
    candidates = [2, 0, 2, 1]
    assert location_candidates(candidates, centers, bounds, point) == candidates
    assert location_candidates(candidates, centers, bounds, point, 2) == [2, 0, 1]
    assert location_candidates(candidates, centers, bounds, point, 3) == [3, 2, 0, 2, 1]
    bounds[3] = [[3., 3.], [4., 4.]]
    assert location_candidates(candidates, centers, bounds, point, 3) == candidates


def test_mutated_and_different_inputs_have_no_cached_selection():
    centers = np.array([[0., 0.], [1., 0.]])
    bounds = np.tile(np.array([[-2., -2.], [2., 2.]]), (2, 1, 1))
    point = np.zeros(2)
    assert location_candidates([0, 1], centers, bounds, point) == [0, 1]
    centers[0] = [3., 0.]
    assert location_candidates([0, 1], centers, bounds, point) == [1, 0]
    bounds[1] = [[4., 4.], [5., 5.]]
    assert location_candidates([0, 1], centers, bounds, point) == [0]
    point[:] = 4.5
    assert location_candidates([0, 1], centers, bounds, point) == [1]
    assert location_candidates([], centers.copy(), bounds.copy(), np.zeros(2)) == []


def test_work_signal_only_survivors_use_scalar_norm(monkeypatch):
    centers = np.zeros((100, 2))
    bounds = np.tile(np.array([[3., 3.], [4., 4.]]), (100, 1, 1))
    bounds[[17, 82]] = [[-1., -1.], [1., 1.]]
    calls = []
    original = np.linalg.norm

    def counted(value):
        calls.append(value.copy())
        return original(value)

    monkeypatch.setattr(np.linalg, 'norm', counted)
    assert location_candidates(list(range(100)), centers, bounds, np.zeros(2)) == [17, 82]
    assert len(calls) == 2
    assert all(value.shape == (2,) for value in calls)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_nonfinite_fallback_preserves_nan_sort(dtype):
    centers = np.array([[2., 0.], [np.nan, 0.], [1., 0.]], dtype=dtype)
    bounds = np.tile(np.array([[-3., -3.], [3., 3.]], dtype=dtype), (3, 1, 1))
    bounds[1, 0, 0] = np.nan
    point = np.zeros(2, dtype=dtype)
    for candidates in ([0, 1, 2], [2, 1, 0], []):
        for hint in (-1, 1):
            assert location_candidates(candidates, centers, bounds, point, hint) == oldloop(
                candidates, centers, bounds, point, hint)


def test_invalid_candidate_keeps_sort_exception():
    centers = np.zeros((2, 2))
    bounds = np.zeros((2, 2, 2))
    with pytest.raises(IndexError):
        location_candidates([2], centers, bounds, np.zeros(2))
