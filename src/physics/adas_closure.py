"""Coronal-equilibrium source-moment closure from ADAS tables (ticket #0014).

The prototype reconstructs the impurity charge moments ``Z_imp`` and ``Z_eff``
from the native state using ADAS SCD/ACD rate tables solved in coronal
equilibrium.  Because the JOREK-normalized temperature axis is not directly the
ADAS ``log10(Te/K)`` axis, the axis mapping is calibrated on the training split
against the recorded projections.  ``dE_ion`` (transient ionisation energy) is
deliberately left external per ADR-0005; ``rad_sink`` is computable but omitted
from the first validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


ADAS_DIR = Path("/home/ac_desktop/adas/adas_data")
Z_MAX = 10  # Neon


def parse_adf11(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(ne_log10 [n_ne], te_log10 [n_te], coeff [n_states, n_te, n_ne])``.

    Coefficients are stored in log10 in the ADF11 files.  ``n_states`` is the
    first integer of the header; SCD blocks index ionisation ``Z -> Z+1`` for
    ``Z = 0..n_states-1`` and ACD blocks index recombination ``Z -> Z-1`` for
    ``Z = 1..n_states``.
    """
    tokens = Path(path).read_text().split()
    n_states, n_ne, n_te = int(tokens[0]), int(tokens[1]), int(tokens[2])
    cursor = 5
    def _next_numeric_run(start, count):
        values = []
        index = start
        while len(values) < count:
            try:
                values.append(float(tokens[index]))
            except ValueError:
                pass
            index += 1
        return np.array(values, dtype=np.float64), index

    ne, cursor = _next_numeric_run(cursor, n_ne)
    te, cursor = _next_numeric_run(cursor, n_te)
    blocks = []
    for _ in range(n_states):
        block, cursor = _next_numeric_run(cursor, n_te * n_ne)
        block = block.reshape(n_te, n_ne)
        blocks.append(block)
    return ne, te, np.stack(blocks)


def _bilinear(coeff: np.ndarray, ne_axis: np.ndarray, te_axis: np.ndarray,
              ne_query, te_query):
    ne_query = np.clip(np.asarray(ne_query, dtype=np.float64), ne_axis[0], ne_axis[-1])
    te_query = np.clip(np.asarray(te_query, dtype=np.float64), te_axis[0], te_axis[-1])
    i = np.searchsorted(ne_axis, ne_query) - 1
    j = np.searchsorted(te_axis, te_query) - 1
    i = np.clip(i, 0, len(ne_axis) - 2)
    j = np.clip(j, 0, len(te_axis) - 2)
    dn = (ne_query - ne_axis[i]) / (ne_axis[i + 1] - ne_axis[i])
    dt = (te_query - te_axis[j]) / (te_axis[j + 1] - te_axis[j])
    c00 = coeff[j, i]
    c01 = coeff[j, i + 1]
    c10 = coeff[j + 1, i]
    c11 = coeff[j + 1, i + 1]
    return c00 * (1 - dn) * (1 - dt) + c01 * dn * (1 - dt) + c10 * (1 - dn) * dt + c11 * dn * dt


def coronal_moments(te_log10, ne_log10, scd, acd):
    """Solve the coronal charge-state balance and return ``(Z_imp, Z_eff)``.

    ``te_log10``/``ne_log10`` are node vectors on the ADAS axes; ``scd`` and
    ``acd`` are parsed ADF11 arrays.  Stationary balance is
    ``S_{Z-1} f_{Z-1} - (S_Z + R_Z) f_Z + R_{Z+1} f_{Z+1} = 0``.
    """
    te_log10 = np.asarray(te_log10, dtype=np.float64)
    ne_log10 = np.full_like(te_log10, np.asarray(ne_log10, dtype=np.float64))
    ne_axis, te_axis = scd[0], scd[1]
    scd_coeff = scd[2]
    acd_coeff = acd[2]
    s = np.zeros((Z_MAX, te_log10.shape[0]))  # S_Z for Z=0..9
    r = np.zeros((Z_MAX, te_log10.shape[0]))  # R_Z for Z=1..10 (r[0] unused)
    for z in range(Z_MAX):
        s[z] = 10 ** _bilinear(scd_coeff[z], ne_axis, te_axis, ne_log10, te_log10)
    for z in range(1, Z_MAX + 1):
        r[z - 1] = 10 ** _bilinear(acd_coeff[z - 1], ne_axis, te_axis, ne_log10, te_log10)

    f = np.zeros((Z_MAX + 1, te_log10.shape[0]))
    # forward sweep of the tridiagonal system for the charge fractions
    f[0] = 1.0
    for z in range(1, Z_MAX + 1):
        f[z] = f[z - 1] * (s[z - 1] / (r[z - 1] + 1e-300))
    f /= f.sum(axis=0, keepdims=True)
    z_grid = np.arange(Z_MAX + 1, dtype=np.float64)[:, None]
    z_imp = (z_grid * f).sum(axis=0)
    z_eff = (z_grid * z_grid * f).sum(axis=0)
    return z_imp, z_eff


def fit_te_axis(te_jorek, z_imp_target, z_eff_target, scd, acd):
    """Calibrate the ``log10(Te_ADAS) = a*log10(Te_JOREK) + b`` mapping."""
    te_jorek = np.clip(np.asarray(te_jorek, dtype=np.float64), 1e-12, None)

    def residual(params):
        a, b = params
        z_imp, z_eff = coronal_moments(
            a * np.log10(te_jorek) + b, 13.0, scd, acd,
        )
        return np.concatenate([z_imp - z_imp_target, z_eff - z_eff_target])

    result = least_squares(residual, x0=(1.0, 6.0), max_nfev=60)
    return result.x, result


def load_neon_tables(adas_dir: Path = ADAS_DIR):
    scd = parse_adf11(adas_dir / "scd96_ne.dat")
    acd = parse_adf11(adas_dir / "acd96_ne.dat")
    return scd, acd
