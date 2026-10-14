"""Charge-table interpolation with query work shared only within one ADAS table."""

import numpy as np


def interpolate_charge_table(table, log_ne, log_te):
    """Return linear ADAS coefficients, shape ``(N, charge_states)``.

    ``table`` is ``(log_density_axis, log_temperature_axis, log_coefficients)``
    with coefficients ordered (charge, temperature, density). Inputs are paired
    one-dimensional log10 queries, in cm^-3 and eV. Axes must be increasing and
    contain at least two points, as in the provider's existing ADAS contract.
    No unit conversion or charge remapping is performed here.

    Each call uses this table's own axes; nothing is cached across tables or
    source steps. Rowwise arithmetic and exponentiation preserve the original
    scalar-row bilinear evaluation order (including clipping at endpoints).
    """
    ne_axis, te_axis, coefficients = table
    ne_query = np.clip(np.asarray(log_ne, dtype=np.float64), ne_axis[0], ne_axis[-1])
    te_query = np.clip(np.asarray(log_te, dtype=np.float64), te_axis[0], te_axis[-1])
    i = np.searchsorted(ne_axis, ne_query) - 1
    j = np.searchsorted(te_axis, te_query) - 1
    i = np.clip(i, 0, len(ne_axis) - 2)
    j = np.clip(j, 0, len(te_axis) - 2)
    dn = (ne_query - ne_axis[i]) / (ne_axis[i + 1] - ne_axis[i])
    dt = (te_query - te_axis[j]) / (te_axis[j + 1] - te_axis[j])
    result = []
    for row in coefficients:
        c00 = row[j, i]
        c01 = row[j, i + 1]
        c10 = row[j + 1, i]
        c11 = row[j + 1, i + 1]
        log_value = (c00 * (1 - dn) * (1 - dt) + c01 * dn * (1 - dt)
                     + c10 * (1 - dn) * dt + c11 * dn * dt)
        result.append(10.**log_value)
    return np.stack(result, axis=1)
