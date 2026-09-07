"""Synthetic identification sanity check for #0013."""

from __future__ import annotations

import numpy as np

from src.evaluation.two_snapshot import least_squares_identify


def test_recovers_known_scalar_from_two_snapshots():
    rng = np.random.default_rng(0)
    theta_true = np.array([2.5, -0.4])
    regressors = rng.normal(size=(2, 200))
    residual = regressors.T @ theta_true + rng.normal(scale=1e-3, size=200)

    theta = least_squares_identify(regressors, residual)

    assert np.allclose(theta, theta_true, atol=5e-2)


def test_rejects_wrong_shape():
    with __import__("pytest").raises(ValueError, match="one row"):
        least_squares_identify(np.ones((3, 10)), np.zeros(9))

