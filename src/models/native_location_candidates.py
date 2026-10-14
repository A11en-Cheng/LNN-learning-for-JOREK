"""Exact, call-local bounding-box prefilter for scalar native location."""

import numpy as np


def location_candidates(candidates, centers, bounds, point, hint=-1):
    """Return bbox-eligible elements in legacy stable distance/hint order.

    ``point`` is R/Z, ``bounds`` has shape (elements, 2, 2), and ``hint``
    is a zero-based element or -1. Work/storage scale with the candidate
    list, not the mesh. No geometry or selection is cached across calls.
    Ordinary float64 arrays use batched bbox comparisons but retain the
    exact scalar norm key. Other inputs take the original sort-first path
    to preserve nonfinite ordering and exception behavior.
    """
    def legacy():
        ordered = sorted(candidates, key=lambda e: np.linalg.norm(centers[e] - point))
        if 0 <= hint < len(centers):
            ordered = [hint] + [e for e in ordered if e != hint]
        eligible = []
        for e in ordered:
            lower, upper = bounds[e]
            if np.any(point < lower - 1e-9) or np.any(point > upper + 1e-9):
                continue
            eligible.append(e)
        return eligible

    if not (
        type(centers) is np.ndarray and type(bounds) is np.ndarray
        and type(point) is np.ndarray
        and centers.dtype == bounds.dtype == point.dtype == np.dtype('float64')
        and centers.ndim == 2 and centers.shape[1] == 2
        and bounds.shape == (len(centers), 2, 2) and point.shape == (2,)
        and isinstance(hint, (int, np.integer))
        and all(isinstance(e, (int, np.integer)) and 0 <= e < len(centers)
                for e in candidates)
    ):
        return legacy()
    valid_hint = 0 <= hint < len(centers)
    checked = candidates + ([hint] if valid_hint else [])
    selected_bounds = bounds[checked]
    if not (np.isfinite(point).all() and np.isfinite(centers[checked]).all()
            and np.isfinite(selected_bounds).all()):
        return legacy()
    outside = np.any(point < selected_bounds[:, 0] - 1e-9, axis=1)
    outside |= np.any(point > selected_bounds[:, 1] + 1e-9, axis=1)
    survivors = [e for e, rejected in zip(candidates, outside) if not rejected]
    ordered = sorted(survivors, key=lambda e: np.linalg.norm(centers[e] - point))
    if valid_hint:
        ordered = [e for e in ordered if e != hint]
        if not outside[-1]:
            ordered.insert(0, hint)
    return ordered
