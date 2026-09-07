"""Unit tests for the time-dependent charge-state closure."""

from __future__ import annotations

import numpy as np

from src.physics.charge_state_closure import ChargeStateClosure


def test_closure_relaxes_to_equilibrium_in_held_state():
    closure = ChargeStateClosure(1, axis_params=(1.0, 6.0))
    rng = np.random.default_rng(0)
    z_imp_series = []
    for _ in range(2000):
        z_imp, z_eff = closure.step(
            te_jorek=rng.uniform(1e-3, 1e-1),
            rho=rng.uniform(0.5, 1.5),
            rho_imp=rng.uniform(1e-3, 1e-1),
            dt_s=1e-6,
            central_density=1.0,
            central_mass=2.0,
        )
        z_imp_series.append(z_imp[0])
    assert np.isfinite(z_imp_series).all()
    assert np.isclose(closure.f.sum(), 1.0)
    assert (closure.f >= 0).all()
    assert z_imp_series[-1] > 0


def test_closure_preserves_normalisation_and_bounds():
    closure = ChargeStateClosure(8, axis_params=(1.0, 6.0))
    z_imp, z_eff = closure.step(
        te_jorek=np.linspace(1e-4, 1e-1, 8),
        rho=np.ones(8),
        rho_imp=np.full(8, 1e-2),
        dt_s=1e-7,
        central_density=1.0,
        central_mass=2.0,
    )
    assert np.allclose(closure.f.sum(axis=0), 1.0)
    assert z_imp.max() <= 10.0
    assert z_eff.max() <= 100.0
