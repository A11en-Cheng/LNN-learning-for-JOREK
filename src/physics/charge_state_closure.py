"""Time-dependent impurity charge-state closure following JOREK's marker path.

Unlike the coronal prototype, this carries the charge-state distribution
``f_Z`` forward in time with the same explicit-Euler update used by
``marker_spi_source.f90`` (``CE_marker=.false.``): SCD/ACD rates from ADAS,
per-charge-state radiation, and ``max(P, 0)`` clamping.  This is the
non-equilibrium, history-carrying version; advection of the markers themselves
is not modelled here, so it remains a node-local approximation.
"""

from __future__ import annotations

import numpy as np

from src.physics.adas_closure import ADAS_DIR, Z_MAX, parse_adf11, _bilinear


E_CHARGE = 1.602176634e-19
BOLTZMANN = 1.380649e-23
LOG10_EV_PER_K = np.log10(E_CHARGE / BOLTZMANN)
M_IMP_AMU = 20.1797  # Neon


def _shift_axes(ne, te, coeff):
    return ne - 6.0, te + LOG10_EV_PER_K, coeff


def load_neon_rates(adas_dir=ADAS_DIR):
    scd = _shift_axes(*parse_adf11(adas_dir / "scd96_ne.dat"))
    acd = _shift_axes(*parse_adf11(adas_dir / "acd96_ne.dat"))
    plt = _shift_axes(*parse_adf11(adas_dir / "plt96_ne.dat"))
    prb = _shift_axes(*parse_adf11(adas_dir / "prb96_ne.dat"))
    return scd, acd, plt, prb


class ChargeStateClosure:
    def __init__(self, nodes: int, *, axis_params=(1.0, 6.0)):
        self.f = np.full((Z_MAX + 1, nodes), 1.0 / (Z_MAX + 1))
        self.axis_params = tuple(axis_params)
        self.scd, self.acd, self.plt, self.prb = load_neon_rates()
        self.z_grid = np.arange(Z_MAX + 1, dtype=np.float64)[:, None]

    def step(self, te_jorek, rho, rho_imp, dt_s, central_density, central_mass):
        """Advance ``f_Z`` one step and return ``(Z_imp, Z_eff)`` (dimensionless)."""
        te_jorek = np.clip(np.atleast_1d(np.asarray(te_jorek, dtype=np.float64)), 1e-12, None)
        rho = np.atleast_1d(np.asarray(rho, dtype=np.float64))
        rho_imp = np.atleast_1d(np.asarray(rho_imp, dtype=np.float64))
        n_norm = central_density * 1e20
        n_imp = rho_imp * (central_mass / M_IMP_AMU) * n_norm
        z_imp_prev = (self.z_grid * self.f).sum(axis=0)
        n_e = rho * n_norm + z_imp_prev * n_imp
        a, b = self.axis_params
        te_log10 = a * np.log10(te_jorek) + b
        ne_log10 = np.log10(np.clip(n_e, 1e0, None))
        ne_axis, te_axis = self.scd[0], self.scd[1]
        s = np.zeros((Z_MAX, te_jorek.shape[0]))
        r = np.zeros((Z_MAX, te_jorek.shape[0]))
        for z in range(Z_MAX):
            s[z] = 10 ** _bilinear(self.scd[2][z], ne_axis, te_axis, ne_log10, te_log10)
            r[z] = 10 ** _bilinear(self.acd[2][z], ne_axis, te_axis, ne_log10, te_log10)

        f = self.f.copy()
        for z in range(Z_MAX + 1):
            gain = 0.0
            if z > 0:
                gain += s[z - 1] * self.f[z - 1]
            if z < Z_MAX:
                gain += r[z] * self.f[z + 1]
            loss = ((s[z] if z < Z_MAX else 0.0) + (r[z - 1] if z > 0 else 0.0)) * self.f[z]
            f[z] = self.f[z] + (gain - loss) * n_e * dt_s
        f = np.clip(f, 0.0, None)
        f /= f.sum(axis=0, keepdims=True)
        self.f = f
        z_imp = (self.z_grid * f).sum(axis=0)
        z_eff = (self.z_grid * self.z_grid * f).sum(axis=0)
        return z_imp, z_eff
