"""Weak-form perpendicular-momentum (vorticity) residual, JOREK-aligned.

Transcribed from ``models/model502/mod_elt_matrix.f90``:

* LHS (matrix side, ``amat_22``): ``(1 + zeta) * int (R rho_hat) grad v . grad u``
  with ``rho_hat = R^2 rho`` - the operator :func:`p1_scalar_stiffness` builds it;
* RHS (``rhs_ij_2``): convection ``-0.5 vv2 {v, r0_hat}``,
  advection ``-r0_hat R^2 w {v,u}``, ``+{v,psi} Z_j``,
  viscosity ``-visco_T R grad v . grad w``, pressure ``+R^2 (v_s p_t - v_t p_s)``.

The pressure uses JOREK's two-temperature closure with the *impurity mass ratio*
(``mod_elt_matrix.f90:1369-1403``), which is what makes the pressure and jxB
terms cancel (weak relative residual 0.37 on the recorded data, versus 0.94 for
the strong forms we used before).

Not yet implemented (listed, not silently dropped): the electron-inertia
``tauIC`` terms (need the electron-inertia variable ``pi``), the ``TG_num2``
stabilisation (available separately via ``fem_weak``), and the four terms coming
from ``-(dt rho + div(rho v)) v`` (need the half-step increments).
"""

from __future__ import annotations

import numpy as np


def pressure_fields(rho, rho_imp, ti, te, z_imp, m_i_over_m_imp=0.049639):
    """JOREK pressures and the two densities (``alpha`` mass-ratio closure)."""
    alpha_i = m_i_over_m_imp - 1.0
    alpha_e = m_i_over_m_imp * np.clip(z_imp, 1.0, None) - 1.0
    n_i = rho + rho_imp * alpha_i
    n_e = rho + rho_imp * alpha_e
    return n_i * np.clip(ti, 1e-12, None) + n_e * np.clip(te, 1e-12, None), n_i, n_e


def p1_triangles(vertex: np.ndarray, pos: np.ndarray):
    """Per-triangle ``(nodes, grad, area, centroid)`` for the quad mesh."""
    out = []
    for e in range(vertex.shape[1]):
        nodes = vertex[:, e]
        coords = pos[nodes]
        for tri in ((0, 1, 2), (0, 2, 3)):
            idx = np.asarray(tri)
            x1, y1 = coords[idx[0]]
            x2, y2 = coords[idx[1]]
            x3, y3 = coords[idx[2]]
            area2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            if abs(area2) < 1e-30:
                continue
            b = np.array([y2 - y3, y3 - y1, y1 - y2])
            c = np.array([x3 - x2, x1 - x3, x2 - x1])
            out.append(
                (nodes[idx], np.stack([b, c], axis=1) / area2,
                 abs(area2) / 2.0, coords[idx].mean(axis=0))
            )
    return out


def weak_vorticity_rhs(state, du_dt, vertex, pos, visco=2.4e-3, zeta=0.5,
                       m_i_over_m_imp=0.049639, triangles=None,
                       return_terms=False, return_lhs=False):
    """Assemble the weak LHS and RHS of the perpendicular-momentum equation.

    ``state`` is ``[nodes, 9]`` in physical units, ``du_dt`` the recorded (or
    model) time derivative of the potential.  Returns ``(lhs, rhs)`` -
    both weak (volume-integrated) nodal vectors - optionally with the per-term
    breakdown.
    """
    n_nodes = pos.shape[0]
    if triangles is None:
        triangles = p1_triangles(vertex, pos)
    rho = np.clip(state[:, 4], 1e-12, None)
    psi, uu, zj, om = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    rimp = np.clip(state[:, 7], 1e-12, None)
    pressure, _n_i, _n_e = pressure_fields(rho, rimp, state[:, 5], state[:, 8], zj)

    names = ("convection", "advection", "jxb", "viscosity", "pressure")
    terms = {name: np.zeros(n_nodes) for name in names}
    lhs = np.zeros(n_nodes)
    for nodes, grad, area, centre in triangles:
        r_c = max(float(centre[0]), 1e-8)
        u_c, om_c = uu[nodes].mean(), om[nodes].mean()
        zj_c, rho_c = zj[nodes].mean(), rho[nodes].mean()
        r0_hat = r_c ** 2 * rho_c
        grad_u = grad.T @ uu[nodes]
        grad_psi = grad.T @ psi[nodes]
        grad_om = grad.T @ om[nodes]
        grad_p = grad.T @ pressure[nodes]
        grad_rho_hat = grad.T @ (r_c ** 2 * rho[nodes])
        vv2 = r_c ** 2 * float((grad_u ** 2).sum())
        gd = grad.T @ du_dt[nodes]
        for i in range(3):
            gi = grad[i]
            cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
            terms["convection"][nodes[i]] += -0.5 * vv2 * float(cross(gi, grad_rho_hat)) * area
            terms["advection"][nodes[i]] += -r0_hat * r_c ** 2 * om_c * float(cross(gi, grad_u)) * area
            terms["jxb"][nodes[i]] += float(cross(gi, grad_psi)) * zj_c * area
            terms["viscosity"][nodes[i]] += -visco * r_c * float(gi @ grad_om) * area
            terms["pressure"][nodes[i]] += r_c ** 2 * float(cross(gi, grad_p)) * area
            lhs[nodes[i]] += (1.0 + zeta) * r_c * r0_hat * float(gi @ gd) * area
    rhs = sum(terms.values())
    if return_terms:
        return lhs, rhs, terms
    if return_lhs:
        return lhs, rhs
    return rhs
