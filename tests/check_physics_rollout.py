"""Independent physics-rollout test (no neural network in the loop).

ADR-0017: before any physics residual is allowed into the CfC training loop, the
physics must be shown to *predict* the next recorded frame from the current one.
This script does exactly that:

1. take the **recorded** frame ``k`` (values + the gradients that come from the
   Hermite dofs) and the frame's own zeroth-order coefficients;
2. advance the state with the model-502 right-hand sides only - no network, no
   data loss - from ``t_k`` to ``t_{k+1}`` with ``--sub-steps`` explicit steps;
3. compare with the **recorded** frame ``k+1`` and report, per field,
   ``dR2_phys = 1 - ||Δ_phys - Δ_true||^2 / ||Δ_true||^2``
   (0 = persistence, 1 = perfect, negative = worse than doing nothing), which is
   the same metric we use for the CfC, so the two are directly comparable.

Deliberate simplifications of this first version (must stay listed here):

* the algebraic fields ``Z_j`` and ``omega`` (and ``Phi``, whose vorticity
  equation has a differential LHS) are **frozen** at their frame-k values;
  recomputing them needs an elliptic solve that is not wired yet;
* the particle/energy **sources** are the recorded projections of frame ``k+1``
  (teacher), i.e. not self-generated;
* the residual module has no particle source term in the density/impurity
  equations and the energy equations are the simplified strong form - the
  equations here are a *mirror* of ``model502_full_loss`` and carry the same
  limitations.

Example:

    ./.torch-cuda-env/bin/python tests/check_physics_rollout.py --frames 3 --sub-steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import RunFrameStore
from src.data.model502_schema import FIELD_NAMES
from src.physics.model502_full_loss import (
    _clamp_physical,
    _source_controls,
    compute_divergence,
    compute_gradient,
    get_magnetic_field,
    get_velocity_field,
    poisson_bracket,
)
from tests.train_cfc_interpolation import _condition_from_frame


def rhs(state: torch.Tensor, graph, condition: torch.Tensor, source: torch.Tensor):
    """``d/dt`` of psi, rho, rho_imp, n_i T_i, n_e T_e, rho v_par`` (mirror of
    ``model502_full_loss.model502_full_residuals`` - keep the two in sync)."""
    edge_index, edge_attr = graph.edge_index, graph.edge_attr
    pos = graph.pos
    r = pos[:, 0].clamp_min(1e-8)
    dtype = torch.float64
    state = _clamp_physical(state).to(dtype)
    pos = pos.to(dtype)
    edge_attr = edge_attr.to(dtype)

    eta = float(condition[0]); visco_par = float(condition[2])
    zk_i_par = float(condition[4]); zk_e_par = float(condition[5])
    d_perp = float(condition[6]); d_perp_imp = float(condition[11])
    zk_i_perp = float(condition[12]); zk_e_perp = float(condition[18])
    f0 = float(condition[33])

    psi, u, z_j, omega = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    rho, ti, v_par, rho_imp, te = (
        state[:, 4].clamp_min(1e-12), state[:, 5].clamp_min(1e-12),
        state[:, 6], state[:, 7].clamp_min(1e-12), state[:, 8].clamp_min(1e-12),
    )
    d_ion, rad_sink, _z_eff, z_imp, _n_imp = _source_controls(
        graph, None, state.device, dtype
    )
    z_imp = z_imp.to(dtype).clamp_min(1.0)
    d_ion, rad_sink = d_ion.to(dtype), rad_sink.to(dtype)

    grad_psi = compute_gradient(psi, edge_index, edge_attr)
    grad_u = compute_gradient(u, edge_index, edge_attr)
    grad_rho = compute_gradient(rho, edge_index, edge_attr)
    grad_rho_imp = compute_gradient(rho_imp, edge_index, edge_attr)
    grad_ti = compute_gradient(ti, edge_index, edge_attr)
    grad_te = compute_gradient(te, edge_index, edge_attr)
    grad_vpar = compute_gradient(v_par, edge_index, edge_attr)
    b_field = get_magnetic_field(grad_psi, r.unsqueeze(1), f0)
    v_field = get_velocity_field(grad_u, b_field, v_par.unsqueeze(1), r.unsqueeze(1))
    v_pol = v_field[:, 0:2]
    b_norm = b_field.norm(dim=1).clamp_min(1e-8)
    b_hat = b_field / b_norm.unsqueeze(1)
    b_pol = b_hat[:, 0:2]

    n_i = rho + rho_imp
    n_e = rho + (z_imp - 1.0) * rho_imp
    pressure = n_i * ti + n_e * te
    grad_p = compute_gradient(pressure, edge_index, edge_attr)

    lap_rho = compute_divergence(compute_gradient(rho, edge_index, edge_attr), edge_index, edge_attr)
    lap_rimp = compute_divergence(compute_gradient(rho_imp, edge_index, edge_attr), edge_index, edge_attr)
    lap_ti = compute_divergence(compute_gradient(ti, edge_index, edge_attr), edge_index, edge_attr)
    lap_te = compute_divergence(compute_gradient(te, edge_index, edge_attr), edge_index, edge_attr)
    lap_vpar = compute_divergence(compute_gradient(v_par, edge_index, edge_attr), edge_index, edge_attr)
    div_v = compute_divergence(v_pol, edge_index, edge_attr)

    par = lambda g: b_hat[:, 0] * g[:, 0] + b_hat[:, 1] * g[:, 1]
    div_par = lambda x: compute_divergence(
        par(compute_gradient(x, edge_index, edge_attr)).unsqueeze(1) * b_pol,
        edge_index, edge_attr,
    )
    div_par_ti, div_par_te, div_par_vpar = div_par(ti), div_par(te), div_par(v_par)
    exchange = (n_e * n_e / n_i.clamp_min(1e-12)) * 1e-2 * (ti - te)
    ohmic = eta * z_j * z_j / (r * r)

    dpsi = eta * z_j - r * poisson_bracket(grad_u, grad_psi)
    drho = -compute_divergence(rho.unsqueeze(1) * v_pol, edge_index, edge_attr) \
        + d_perp * lap_rho + d_perp_imp * lap_rimp
    drho_imp = -compute_divergence(rho_imp.unsqueeze(1) * v_pol, edge_index, edge_attr) \
        + d_perp_imp * lap_rimp
    d_ni_ti = -(
        compute_divergence((n_i * ti).unsqueeze(1) * v_field[:, 0:2], edge_index, edge_attr)
        + n_i * ti * div_v
    ) + zk_i_par * div_par_ti + zk_i_perp * (lap_ti - div_par_ti) - exchange
    d_ne_te = -(
        compute_divergence((n_e * te).unsqueeze(1) * v_field[:, 0:2], edge_index, edge_attr)
        + n_e * te * div_v
    ) + zk_e_par * div_par_te + zk_e_perp * (lap_te - div_par_te) \
        - ohmic + rad_sink - d_ion + exchange
    d_rho_vpar = -rho * v_par * div_v - par(grad_p) \
        + visco_par * lap_vpar
    return {
        "psi": dpsi, "rho": drho, "rho_imp": drho_imp,
        "n_i_T_i": d_ni_ti, "n_e_T_e": d_ne_te, "rho_v_par": d_rho_vpar,
    }




def smooth_floor(value: torch.Tensor, threshold: float, coef=(0.5, 0.5)):
    """Mirror of JOREK's ``corr_neg_temp`` / ``corr_neg_dens``.

    Below ``L1+L2`` the value becomes ``L1 + L2*exp((v-(L1+L2))/L2)``: an
    exponentially smooth positive floor that - unlike a hard clamp - stays
    differentiable.  JOREK applies it to the fluid temperature/density entering
    the transport coefficients; the marker/source side instead floors charge
    states, populations and source distributions hard at 0 and guards divisions
    (``marker_spi_source.f90``).
    """
    L1 = threshold * coef[0]
    L2 = threshold * coef[1]
    limit = L1 + L2
    return torch.where(
        value < limit, L1 + L2 * torch.exp((value - limit) / max(L2, 1e-30)), value
    )


def conduction(T, b_hat, edge_index, edge_attr, kappa_par, kappa_perp, extra_par=0.0):
    """kappa_par*div_par(grad_par T) + kappa_perp*(lap T - div_par(grad_par T))."""
    grad = compute_gradient(T, edge_index, edge_attr)
    lap = compute_divergence(grad, edge_index, edge_attr)
    g_par = b_hat[:, 0] * grad[:, 0] + b_hat[:, 1] * grad[:, 1]
    l_par = compute_divergence(g_par.unsqueeze(1) * b_hat, edge_index, edge_attr)
    return (kappa_par + extra_par) * l_par + kappa_perp * (lap - l_par)


def cg_solve(matvec, b, iterations=400, tol=1e-7):
    """Preconditioner-free conjugate gradient (the implicit operator is SPD)."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = (r * r).sum()
    rs0 = rs.clamp_min(1e-300)
    used = 0
    for _ in range(iterations):
        used += 1
        if float(rs / rs0) < tol * tol:
            break
        ap = matvec(p)
        alpha = rs / (p * ap).sum().clamp_min(1e-300)
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = (r * r).sum()
        p = r + (rs_new / rs.clamp_min(1e-300)) * p
        rs = rs_new
    return x, used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--frames", type=int, default=3, help="how many intervals to test")
    parser.add_argument("--start", type=int, default=30)
    parser.add_argument("--sub-steps", type=int, default=20)
    parser.add_argument(
        "--fields",
        default="psi",
        help="which fields to advance: 'psi' (induction only; explicitly integrable) or "
        "'all' (adds rho/rho_imp/T_i/T_e/v_par - these are STIFF and blow up with an "
        "explicit scheme, which is the point of the ADR-0017 test)",
    )
    parser.add_argument("--cg-iterations", type=int, default=400)
    parser.add_argument("--t-floor", type=float, default=1e-3,
                        help="smooth negative-temperature floor (JOREK corr_neg_temp)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    store = RunFrameStore(
        Path(manifest["data_root"]) / args.run, cache_dir=args.cache_dir
    ).load()
    topology = store.topology
    from torch_geometric.data import Data

    edge_index, edge_attr = topology.edge_index, topology.edge_attr.double()
    results: dict[str, list[float]] = {}
    print(f"run={args.run} intervals={args.frames} sub_steps={args.sub_steps}")
    print("(Z_j, omega, Phi are FROZEN at frame k; sources are the recorded teacher)")
    for index in range(args.start, args.start + args.frames):
        if index + 1 >= store.u.shape[0]:
            break
        state_k = store.u[index].double()
        state_next = store.u[index + 1].double()
        dt = float(store.time[index + 1] - store.time[index])
        condition = _condition_from_frame(store.condition[index].numpy())
        f0 = float(condition[33])
        zk_i_par, zk_e_par = float(condition[4]), float(condition[5])
        zk_i_perp, zk_e_perp = float(condition[12]), float(condition[18])
        r_axis = topology.pos[:, 0].double().clamp_min(1e-8)
        last_cg_iters = 0
        source = store.source[index + 1].double()
        # Mirror ``marker_spi_source.f90``: the projection channels are floored
        # at 0 (JOREK does this for charge states, populations and the source
        # distribution; the recorded projections do contain negatives).
        source = source.clamp_min(0.0)
        graph = Data(
            x=state_k.float(), pos=topology.pos.double(), edge_index=edge_index,
            edge_attr=edge_attr.double(), source=source,
        )
        with torch.no_grad():
            state = state_k.clone()
            dtau = dt / args.sub_steps
            unstable = False
            for step_index in range(args.sub_steps):
                d = rhs(state, graph, condition, source)
                state = state.clone()
                state[:, 0] = state[:, 0] + dtau * d["psi"]
                if args.fields == "all":
                    # IMEX split: advection/exchange/sources explicit, the stiff
                    # conduction (kappa_par ~ 1e5-1e6) implicit.  JOREK solves the
                    # same kind of matrix every step.
                    state[:, 4] = state[:, 4] + dtau * d["rho"]
                    state[:, 7] = state[:, 7] + dtau * d["rho_imp"]
                    n_i = state[:, 4] + state[:, 7]
                    n_e = state[:, 4] + (state[:, 2].clamp_min(1.0) - 1.0) * state[:, 7]
                    grad_psi = compute_gradient(state[:, 0], edge_index, edge_attr)
                    b_field = get_magnetic_field(grad_psi, r_axis.unsqueeze(1), f0)
                    b_hat = (b_field / b_field.norm(dim=1).clamp_min(1e-8).unsqueeze(1))[:, 0:2]
                    for slot, kap_par, kap_perp in (
                        (5, zk_i_par, zk_i_perp), (8, zk_e_par, zk_e_perp),
                    ):
                        n_field = n_i if slot == 5 else n_e
                        rhs_key = "n_i_T_i" if slot == 5 else "n_e_T_e"
                        explicit_cond = conduction(
                            state[:, slot], b_hat, edge_index, edge_attr, kap_par, kap_perp
                        )
                        coeff = dtau / n_field.clamp_min(1e-30)
                        rhs_field = state[:, slot] + coeff * (d[rhs_key] - explicit_cond)

                        def matvec(x, coeff=coeff, kap_par=kap_par, kap_perp=kap_perp):
                            return x - coeff * conduction(
                                x, b_hat, edge_index, edge_attr, kap_par, kap_perp
                            )

                        solved, iters = cg_solve(
                            matvec, rhs_field, iterations=args.cg_iterations
                        )
                        state[:, slot] = smooth_floor(solved, args.t_floor)
                        last_cg_iters = iters
                    rho_vpar = state[:, 4] * state[:, 6] + dtau * d["rho_v_par"]
                    state[:, 6] = rho_vpar / state[:, 4].clamp_min(1e-30)
                if not torch.isfinite(state).all():
                    unstable = True
                    finite = torch.isfinite(state).all(dim=0)
                    bad = [FIELD_NAMES[i] for i, ok in enumerate(finite.tolist()) if not ok]
                    print(f"    diverged at sub-step {step_index}: first non-finite fields {bad}")
                    for key, value in d.items():
                        print(f"      |d({key})/dt| rms={float(value.pow(2).mean().sqrt()):.3e} "
                              f"max={float(value.abs().max()):.3e}")
                    break
            delta_phys = state - state_k
        if args.fields == "all" and not unstable:
            print(f"    (implicit CG iterations used: {last_cg_iters})")
        if unstable:
            print(f"  frame {index}->{index+1} (dt={dt:.1f}): UNSTABLE (explicit step "
                  f"exceeds the stability limit of the stiff terms; needs JOREK's "
                  f"implicit solve)")
            continue
        delta_true = state_next - state_k
        scale = float((delta_true**2).sum())
        err = float(((delta_phys - delta_true) ** 2).sum())
        line = []
        for field, name in enumerate(FIELD_NAMES):
            err_f = float(((delta_phys[:, field] - delta_true[:, field]) ** 2).sum())
            true_f = float((delta_true[:, field] ** 2).sum())
            r2 = 1.0 - err_f / max(true_f, 1e-30)
            results.setdefault(name, []).append(r2)
            line.append(f"{name}={r2:+.2f}")
        print(f"  frame {index}->{index+1} (dt={dt:.1f}): " + " ".join(line))
    print()
    print(f"{'field':9s} {'physics dR2 (median)':>20s}")
    for name in FIELD_NAMES:
        if name in results:
            print(f"{name:9s} {float(np.median(results[name])):20.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
