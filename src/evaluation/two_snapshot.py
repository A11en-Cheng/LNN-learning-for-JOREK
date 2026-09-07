"""Two-snapshot least-squares identification (ticket #0013, cross-check only)."""

from __future__ import annotations

import numpy as np


def least_squares_identify(regressors: np.ndarray, residual: np.ndarray):
    """Estimate the scalar(s) ``theta`` in ``residual ~= regressors @ theta``.

    This is a trajectory-constrained least-squares estimator: ``regressors``
    are the equation terms constructed from two consecutive snapshots and
    ``residual`` is the left-over time-evolution term.  ADR-0006 restricts its
    role to cross-checking, never the training path.
    """
    regressors = np.atleast_2d(regressors)
    residual = np.asarray(residual, dtype=np.float64)
    if regressors.shape[1] != residual.shape[0]:
        raise ValueError("regressors must have one row per residual entry")
    theta, *_ = np.linalg.lstsq(regressors.T, residual, rcond=None)
    return theta

