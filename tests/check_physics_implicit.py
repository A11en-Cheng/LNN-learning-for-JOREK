"""Implicit physics evolution with the verified P1 weak-form operators (ADR-0017).

The explicit roll-out (``check_physics_rollout.py``) blows up on the energy
equations because the capped parallel conduction is still stiff (kappa_par = 1e4,
stability limit dtau < h^2/(2 kappa) ~ 5e-9).  This script advances the stiff
conduction **implicitly** with the operators verified in
``tests/check_weak_laplacian.py``:

* ``A``      - isotropic P1 stiffness (``lap T ~= -(A T)/m``);
* ``A_par``  - field-aligned P1 stiffness (``(b.grad)^2 T ~= -(A_par T)/m``);

with the backward-Euler system  ``(I + (dtau/n)[k_perp A/m + (k_par-k_perp)
A_par/m]) T_new = T + (dtau/n)(explicit terms)``, solved directly (sparse LU).
Everything else (advection, exchange, sources) stays explicit, and the
recorded sources of frame k+1 are used as the teacher.

Example:

    ./.torch-cuda-env/bin/python tests/check_physics_implicit.py --frames 2 --sub-steps 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import RunFrameStore
from src.data.model502_schema import FIELD_NAMES
from src.physics.fem_weak import (
    boundary_mask,
    build_p1_advection,
    build_p1_anisotropic_stiffness,
    triangle_stiffness,
    build_p1_laplacian,
    build_p1_parallel_laplacian,
)
from src.physics.weak_vorticity import p1_triangles, weak_vorticity_rhs
from src.physics.model502_full_loss import (
    model502_full_residuals,
    _effective_par_conduction,
    compute_gradient,
    get_magnetic_field,
    get_velocity_field,
)
from src.physics.grad_ops import compute_divergence
from tests.check_physics_rollout import rhs, smooth_floor
from tests.train_cfc_interpolation import _condition_from_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--start", type=int, default=30)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument(
        "--sub-steps", type=int, default=-1,
        help="fluid sub-steps per recorded frame; -1 (default) uses dt/tstep from the "
        "H5 (tstep is the actual JOREK fluid step: 0.1 for the 50-step runs, 0.2 for "
        "the 10-step run), so the CFL is resolved the way JOREK itself does",
    )
    parser.add_argument("--t-floor", type=float, default=1e-3)
    args = parser.parse_args()

    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve
    from src.physics.fem_weak import boundary_mask as _boundary_mask

    def _assemble_scalar_stiffness(vertex, pos, coeff_c, cluster):
        """``A_ij = int c(x) grad N_i . grad N_j`` with the coefficient per cluster."""
        n_clusters = int(cluster.max()) + 1
        rows, cols, vals = [], [], []
        for e in range(vertex.shape[1]):
            nodes = cluster[vertex[:, e]]
            coords = pos[vertex[:, e]]
            c_e = float(np.mean(coeff_c[nodes]))
            for tri in ((0, 1, 2), (0, 2, 3)):
                idx = np.asarray(tri)
                a_e, _ = triangle_stiffness(coords[idx])
                used = nodes[idx]
                for i in range(3):
                    for j in range(3):
                        rows.append(used[i])
                        cols.append(used[j])
                        vals.append(c_e * a_e[i, j])
        return sp.coo_matrix((vals, (rows, cols)), shape=(n_clusters, n_clusters)).tocsr()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    store = RunFrameStore(
        Path(manifest["data_root"]) / args.run, cache_dir=args.cache_dir
    ).load()
    topology = store.topology
    edge_index, edge_attr = topology.edge_index, topology.edge_attr.double()
    vertex = topology.vertex.numpy()
    pos = topology.pos.double().numpy()
    r_axis = topology.pos.double()[:, 0].clamp_min(1e-8)

    a_iso, mass, cluster = build_p1_laplacian(vertex, pos)
    boundary_nodes = boundary_mask(vertex)
    weak_triangles = p1_triangles(vertex, pos)
    a_iso = a_iso.tocsc()
    m_inv = 1.0 / np.clip(mass, 1e-30, None)
    cl = torch.as_tensor(cluster, dtype=torch.long)
    n_clusters = a_iso.shape[0]

    wall_clusters = np.unique(cluster[boundary_nodes])

    def pin_wall(op, rhs, keep):
        """Dirichlet rows on the wall/limiter (JOREK: apply_dirichlet_all).

        Without this the implicit conduction/advection operators carry the
        boundary flux term of the weak Laplacian (``A @ 1 != 0`` on boundary
        nodes) and, with kappa_par/h^2 dominating, drain the field to zero.
        """
        op = op.tolil()
        for c in wall_clusters:
            op.rows[c] = [c]
            op.data[c] = [1.0]
            rhs[c] = keep[c]
        return op.tocsc()

    def to_c(values: torch.Tensor) -> np.ndarray:
        out = torch.zeros(n_clusters, dtype=torch.float64)
        cnt = torch.zeros(n_clusters, dtype=torch.float64)
        out.index_add_(0, cl, values.double())
        cnt.index_add_(0, cl, torch.ones_like(values.double()))
        return (out / cnt.clamp_min(1.0)).numpy()

    print(f"run={args.run} clusters={n_clusters} intervals={args.frames} "
          f"sub_steps={'auto (dt/tstep)' if args.sub_steps < 0 else args.sub_steps}")
    results: dict[str, list[float]] = {}
    for index in range(args.start, args.start + args.frames):
        state = store.u[index].double().clone()
        target = store.u[index + 1].double()
        dt = float(store.time[index + 1] - store.time[index])
        condition = _condition_from_frame(store.condition[index].numpy())
        source = store.source[index + 1].double().clamp_min(0.0)
        graph = Data(
            x=state.float(), pos=topology.pos.double(), edge_index=edge_index,
            edge_attr=edge_attr.double(), source=source,
        )
        f0 = float(condition[33])
        zk_i_par, zk_e_par = float(condition[4]), float(condition[5])
        zk_i_perp, zk_e_perp = float(condition[12]), float(condition[18])
        zk_i_neg, zk_e_neg = float(condition[53]), float(condition[52])
        unstable = False
        tstep = float(condition[29]) if float(condition[29]) > 0 else dt
        sub_steps = int(round(dt / tstep)) if args.sub_steps < 0 else args.sub_steps
        with torch.no_grad():
            dtau = dt / max(sub_steps, 1)
            for step_index in range(sub_steps):
                d = rhs(state, graph, condition, source)
                state = state.clone()
                state[:, 0] = state[:, 0] + dtau * d["psi"]
                state[:, 4] = state[:, 4] + dtau * d["rho"]
                state[:, 7] = state[:, 7] + dtau * d["rho_imp"]
                n_i = state[:, 4] + state[:, 7]
                n_e = state[:, 4] + (state[:, 2].clamp_min(1.0) - 1.0) * state[:, 7]
                grad_psi = compute_gradient(state[:, 0], edge_index, edge_attr)
                b_field = get_magnetic_field(grad_psi, r_axis.unsqueeze(1), f0)
                b_hat = (b_field / b_field.norm(dim=1).clamp_min(1e-8).unsqueeze(1))[:, 0:2]
                a_par, _, _ = build_p1_parallel_laplacian(
                    vertex, pos, b_hat.detach().cpu().numpy(), cluster
                )
                a_par = a_par.tocsc()
                # --- implicit advection (+ conduction for the energies) ---
                v_field = get_velocity_field(
                    grad_psi, b_field, state[:, 6].unsqueeze(1), r_axis.unsqueeze(1)
                )
                v_pol = v_field[:, 0:2]
                div_v = compute_divergence(v_pol, edge_index, edge_attr)
                v_np = v_pol.detach().cpu().numpy()
                div_v_c = to_c(div_v)
                m_adv, _, _ = build_p1_advection(vertex, pos, v_np, cluster, div_v=div_v_c)
                m_adv = m_adv.tocsc()
                d_perp = float(condition[6])
                d_perp_imp = float(condition[11])
                for slot, key, extra_diff in (
                    (4, "rho", d_perp), (7, "rho_imp", d_perp_imp),
                ):
                    rhs_c = to_c(state[:, slot] + dtau * d[key])
                    op = (
                        sp.identity(n_clusters, format="csc")
                        - sp.diags(dtau * m_inv) @ m_adv
                        + sp.diags(dtau * extra_diff * m_inv) @ a_iso
                    )
                    op = pin_wall(op, rhs_c, to_c(state[:, slot]))
                    state[:, slot] = torch.as_tensor(
                        np.asarray(spsolve(op, rhs_c))[cluster]
                    )
                for slot, kap0, kap_neg, t0, tmin, k_perp in (
                    (5, zk_i_par, zk_i_neg, float(condition[27]), float(condition[46]), zk_i_perp),
                    (8, zk_e_par, zk_e_neg, float(condition[28]), float(condition[47]), zk_e_perp),
                ):
                    n_field = n_i if slot == 5 else n_e
                    key = "n_i_T_i" if slot == 5 else "n_e_T_e"
                    k_par = _effective_par_conduction(
                        state[:, slot], t0, kap0, kap_neg,
                        float(condition[44]), tmin, float(condition[45]) > 0.5,
                    )
                    coeff = dtau / n_field.clamp_min(1e-30)
                    coeff_c = to_c(coeff)
                    k_par_c = to_c(k_par)
                    grad_t = compute_gradient(state[:, slot], edge_index, edge_attr)
                    lap_t = compute_divergence(grad_t, edge_index, edge_attr)
                    g_par = b_hat[:, 0] * grad_t[:, 0] + b_hat[:, 1] * grad_t[:, 1]
                    lap_par_t = compute_divergence(g_par.unsqueeze(1) * b_hat, edge_index, edge_attr)
                    explicit_cond = k_par * lap_par_t + k_perp * (lap_t - lap_par_t)
                    rhs_c = to_c(
                        state[:, slot]
                        + torch.as_tensor(coeff_c)[cluster] * (d[key] - explicit_cond)
                    )
                    op = (
                        sp.identity(n_clusters, format="csc")
                        - sp.diags(coeff_c * m_inv) @ m_adv
                        + sp.diags(coeff_c * m_inv) @ (k_perp * a_iso)
                        + sp.diags(coeff_c * m_inv)
                        @ sp.diags(np.maximum(k_par_c - k_perp, 0.0))
                        @ a_par
                    )
                    op = pin_wall(op, rhs_c, to_c(state[:, slot]))
                    solved = spsolve(op, rhs_c)
                    state[:, slot] = smooth_floor(
                        torch.as_tensor(np.asarray(solved)[cluster]), args.t_floor
                    )
                # parallel momentum: implicit advection (+ visco_par diffusion)
                f_par = state[:, 4] * state[:, 6]
                visco_par = float(condition[2])
                explicit_visc = visco_par * compute_divergence(
                    compute_gradient(state[:, 6], edge_index, edge_attr),
                    edge_index, edge_attr,
                )
                visco_c = to_c(visco_par / state[:, 4].clamp_min(1e-30))
                rhs_c = to_c(
                    f_par + dtau * (d["rho_v_par"] - explicit_visc)
                    + torch.as_tensor(to_c(dtau / state[:, 4].clamp_min(1e-30)))[cluster]
                    * explicit_visc
                )
                op = (
                    sp.identity(n_clusters, format="csc")
                    - sp.diags(dtau * m_inv) @ m_adv
                    + sp.diags(visco_c * m_inv) @ a_iso
                )
                op = pin_wall(op, rhs_c, to_c(f_par))
                solved_v = np.asarray(spsolve(op, rhs_c))
                state[:, 6] = (
                    torch.as_tensor(solved_v)[cluster] / state[:, 4].clamp_min(1e-30)
                )
                # --- elliptic solve for dPhi/dt (vorticity equation) ---
                # L_Phi[f] = div(r * rho_hat * grad f) with rho_hat = r^2 rho,
                # i.e. div(r^3 rho grad f) = RHS_vort  ->  weak form
                # -int (r^3 rho) grad N_i . grad f = int N_i RHS_vort.
                # --- weak-form RHS of the perpendicular momentum equation ---
                # (transcribed from mod_elt_matrix.f90:1489-1510; the terms depend
                # only on the current state, so no d/dt placeholder is needed)
                state_np = state.detach().cpu().numpy()
                _lhs_unused, rhs_weak = weak_vorticity_rhs(
                    state_np, np.zeros(len(pos)), vertex, pos,
                    visco=visco_par, triangles=weak_triangles, return_lhs=True,
                )
                rhs_phi = to_c(torch.as_tensor(rhs_weak))
                # LHS: (1 + zeta) * int (R rho_hat) grad N_i . grad du/dt
                zeta_evol = 0.5   # time_evol_zeta (Gears)
                a_phi = (1.0 + zeta_evol) * _assemble_scalar_stiffness(
                    vertex, pos, to_c(r_axis.pow(3) * state[:, 4]), cluster
                )
                wall_c = np.unique(cluster[boundary_nodes])
                op_phi = a_phi.tolil()
                for c in wall_c:
                    op_phi.rows[c] = [c]
                    op_phi.data[c] = [1.0]
                    rhs_phi[c] = 0.0
                du_dt = np.asarray(spsolve(op_phi.tocsc(), rhs_phi))
                state[:, 1] = state[:, 1] + dtau * torch.as_tensor(du_dt)[cluster]
                # omega = Delta+ Phi = lap Phi + dR Phi / R  (weak lap + graph gradient)
                lap_phi = -(
                    a_iso @ to_c(state[:, 1])
                ) / np.clip(np.asarray([mass[c] for c in range(n_clusters)]), 1e-30, None)
                grad_phi = compute_gradient(state[:, 1], edge_index, edge_attr)
                state[:, 3] = torch.as_tensor(lap_phi + to_c(grad_phi[:, 0] / r_axis.clamp_min(1e-8)))[cluster]
                # Z_j = Delta* psi = lap psi - dR psi / R
                lap_psi = -(a_iso @ to_c(state[:, 0])) / np.clip(
                    np.asarray([mass[c] for c in range(n_clusters)]), 1e-30, None
                )
                grad_psi2 = compute_gradient(state[:, 0], edge_index, edge_attr)
                state[:, 2] = torch.as_tensor(
                    lap_psi - to_c(grad_psi2[:, 0] / r_axis.clamp_min(1e-8))
                )[cluster]

                if not torch.isfinite(state).all():
                    unstable = True
                    finite = torch.isfinite(state).all(dim=0)
                    bad = [FIELD_NAMES[i] for i, ok in enumerate(finite.tolist()) if not ok]
                    print(f"    diverged at sub-step {step_index}: non-finite {bad}")
                    for key, value in d.items():
                        print(f"      |d({key})/dt| rms={float(value.pow(2).mean().sqrt()):.3e} "
                              f"max={float(value.abs().max()):.3e}")
                    print(f"      state rms per field: "
                          f"{[round(float(state[:,i].pow(2).mean().sqrt()),3) for i in range(9)]}")
                    break
        if unstable:
            print(f"  frame {index}->{index+1}: UNSTABLE")
            continue
        delta_phys = state - store.u[index].double()
        delta_true = target - store.u[index].double()
        line = []
        for field, name in enumerate(FIELD_NAMES):
            err = float(((delta_phys[:, field] - delta_true[:, field]) ** 2).sum())
            ref = float((delta_true[:, field] ** 2).sum())
            r2 = 1.0 - err / max(ref, 1e-30)
            results.setdefault(name, []).append(r2)
            line.append(f"{name}={r2:+.2f}")
        print(f"  frame {index}->{index+1} (dt={dt:.1f}): " + " ".join(line))
    print()
    print(f"{'field':9s} {'implicit physics dR2':>21s}")
    for name in FIELD_NAMES:
        if name in results:
            print(f"{name:9s} {float(np.median(results[name])):21.3f}")


if __name__ == "__main__":
    main()
