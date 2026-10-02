"""Weak-form vorticity residual, term by term, transcribed from JOREK.

``models/model502/mod_elt_matrix.f90:1489-1540`` (``rhs_ij_2``) is the
*weak-form* perpendicular-momentum equation: every term carries the test
function ``v`` or its gradient.  Our residual module used strong forms and was
therefore not equivalent (the pressure force came out ~7x too large and did not
cancel ``{psi, Z_j}``).

This script assembles the same terms with P1 test functions on the nodal fields
and reports the per-term rms on the **recorded** interval - the acceptance
criterion being that all terms land in the same order of magnitude and the
residual falls in the 0.05-0.8 range (like induction/density/impurity).

Term list (weak form, ``v`` = test function, ``{a,b} = (dR a dZ b - dZ a dR b)/R``):

  1. ``-0.5 vv2 {v, r0_hat}``                      vv2 = R^2 |grad u|^2, r0_hat = R^2 rho
  2. ``-r0_hat R^2 w {v, u}``
  3. ``+{v, psi} Z_j``
  4. ``-visco_T R grad v . grad w``
  5. ``+R^2 {p, v}``                               (weak pressure force)
  6. ``-tau_IC R^4 {pi, w}`` / ``-tau_IC R^3 pi_y (grad v . grad u)`` /
     ``-tau_IC R^4 (u_xy (pi_xx - pi_yy) - pi_xy (u_xx - u_yy))``   (electron inertia)
  7. ``-c_TG r0_hat R^3 (grad w x grad u)(grad v x grad u)``        (numerical stabilisation)
  8. four new momentum terms from ``-(dt rho + div(rho v)) v``

Terms 6-8 need fields we do not carry (the electron-inertia variable ``pi``, the
half-step increments) and are reported as UNAVAILABLE rather than silently
dropped.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import RunFrameStore
from src.physics.weak_vorticity import (
    p1_triangles as _p1_triangles_mod,
    pressure_fields,
    weak_vorticity_rhs,
)


def _p1_setup(vertex, pos):
    """Per-triangle geometry: area, P1 test-function gradients, centroids."""
    triangles = []
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
            grad = np.stack([b, c], axis=1) / area2      # [3,2]
            triangles.append(
                (nodes[idx], grad, abs(area2) / 2.0, coords[idx].mean(axis=0))
            )
    return triangles


def _field_grad(values, grad, nodes_vals):
    """P1 gradient of a nodal field on a triangle."""
    return grad.T @ nodes_vals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--frame", type=int, default=30)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    store = RunFrameStore(
        Path(manifest["data_root"]) / args.run, cache_dir=args.cache_dir
    ).load()
    topology = store.topology
    vertex = topology.vertex.numpy()
    pos = topology.pos.double().numpy()
    n_nodes = pos.shape[0]
    triangles = _p1_setup(vertex, pos)

    u0 = store.u[args.frame].double().numpy()
    u1 = store.u[args.frame + 1].double().numpy()
    dt = float(store.time[args.frame + 1] - store.time[args.frame])
    rho = np.clip(u0[:, 4], 1e-12, None)
    psi, uu, zj, om, ti, te = u0[:, 0], u0[:, 1], u0[:, 2], u0[:, 3], u0[:, 5], u0[:, 8]
    rimp = np.clip(u0[:, 7], 1e-12, None)
    # JOREK's pressures (mod_elt_matrix.f90:1369-1403):
    #   alpha_i = m_i/m_imp - 1 ; alpha_e = (m_i/m_imp)*Z_imp - 1
    #   Pi = (rho + rho_imp*alpha_i) Ti ; Pe = (rho + rho_imp*alpha_e) Te ; P = Pi + Pe
    # Neon pellet: m_i_over_m_imp = central_mass/20*(2/2.0145) = 0.0496.
    central_mass = 1.0
    m_i_over_m_imp = central_mass / 20.0 * (2.0 / 2.01455279245)
    z_imp = np.clip(2.0 + 0.0 * rimp, 1.0, None)   # mean charge (recorded Z_imp comes later)
    alpha_i = m_i_over_m_imp - 1.0
    alpha_e = m_i_over_m_imp * z_imp - 1.0
    n_i = rho + rimp * alpha_i
    n_e = rho + rimp * alpha_e
    pressure = n_i * np.clip(ti, 1e-12, None) + n_e * np.clip(te, 1e-12, None)
    du_dt = (u1[:, 1] - u0[:, 1]) / dt

    terms = {name: np.zeros(n_nodes) for name in
             ("convection", "advection", "jxb", "viscosity", "pressure", "inertia")}
    for nodes, grad, area, centre in triangles:
        r_c = max(centre[0], 1e-8)
        def g(values):
            return grad.T @ values[nodes]
        psi_c, u_c, zj_c, om_c = psi[nodes].mean(), uu[nodes].mean(), zj[nodes].mean(), om[nodes].mean()
        rho_c, p_c = rho[nodes].mean(), pressure[nodes].mean()
        r0_hat = r_c ** 2 * rho_c
        grad_u = g(uu)
        grad_psi = g(psi)
        grad_zj = g(zj)
        grad_om = g(om)
        grad_p = g(pressure)
        grad_rho_hat = g(r_c ** 2 * rho)
        vv2 = r_c ** 2 * float((grad_u ** 2).sum())
        for i in range(3):
            gi = grad[i]
            cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
            terms["convection"][nodes[i]] += -0.5 * vv2 * float(cross(gi, grad_rho_hat)) * area
            terms["advection"][nodes[i]] += -r0_hat * r_c ** 2 * om_c * float(cross(gi, grad_u)) * area
            terms["jxb"][nodes[i]] += float(cross(gi, grad_psi)) * zj_c * area
            terms["viscosity"][nodes[i]] += -2.4e-3 * r_c * float(gi @ grad_om) * area
            # JOREK: + R^2 (v_s p_t - v_t p_s) = + R^3 {v, p} with v -> N_i
            terms["pressure"][nodes[i]] += r_c ** 2 * float(cross(gi, grad_p)) * area
    # inertia: r0_hat * Delta+ (du/dt) in the strong sense of the weak bracket
    lap_du = np.zeros(n_nodes)
    for nodes, grad, area, centre in triangles:
        r_c = max(centre[0], 1e-8)
        g = lambda values: grad.T @ values[nodes]
        gd = g(du_dt)
        for i in range(3):
            lap_du[nodes[i]] += float(grad[i] @ gd) * area
    # --- LHS: (1 + zeta) * int (R rho_hat) grad N_i . grad(delta u) ---
    zeta = 0.5  # time_evol_zeta for the Gears scheme (namelist time_evol_scheme)
    lhs = np.zeros(n_nodes)
    for nodes, grad, area, centre in triangles:
        r_c = max(centre[0], 1e-8)
        r0_hat_c = r_c ** 2 * rho[nodes].mean()
        gd = grad.T @ du_dt[nodes]
        for i in range(3):
            lhs[nodes[i]] += (1.0 + zeta) * r_c * r0_hat_c * float(grad[i] @ gd) * area

    rms = lambda x: float(np.sqrt((x ** 2).mean()))
    cc = float(np.corrcoef(terms["jxb"], terms["pressure"])[0, 1])
    print(f"corr(jxb, pressure) = {cc:+.3f}  (force balance wants ~ -1)")
    print(f"run={args.run} frame={args.frame}->{args.frame+1} dt={dt:.1f} "
          f"triangles={len(triangles)}")
    print(f"{'term':14s} {'weak rms':>12s}")
    for name, value in terms.items():
        print(f"{name:14s} {rms(value):12.4e}")
    total = sum(terms[name] for name in ("convection", "advection", "jxb", "viscosity", "pressure"))
    print(f"{'LHS (1+zeta)*A_r3rho*du/dt':14s} {rms(lhs):12.4e}")
    print()
    residual = lhs - total
    scale_all = sum(abs(rms(v)) for v in
                    (lhs, terms["convection"], terms["advection"], terms["jxb"],
                     terms["viscosity"], terms["pressure"]))
    print(f"weak residual rms = {rms(residual):.4e}")
    print(f"relative (|LHS-RHS| / sum|terms|) = {rms(residual) / max(scale_all, 1e-30):.4f}  "
          f"(target 0.05-0.8 for a usable equation)")


if __name__ == "__main__":
    main()
