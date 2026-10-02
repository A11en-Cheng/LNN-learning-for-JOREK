"""Re-verification of pitfall #21 (second-order operators) on the fixed code base.

The original #21 evidence (FEM ``Delta* psi`` vs the recorded ``Z_j`` channel,
``corr = 0.05``, rms ratio ~1e4; geometry ``|lap(R)|`` up to 5e-2) was collected
before the loss/parameterisation bugs were found, so it is re-measured here:

1. **geometry self-check** - ``lap(R)`` and ``lap(Z)`` must vanish; a non-zero
   value means the chain rule or the element assembly is wrong, not the physics;
2. **definition check** - ``Delta* psi = lap psi - psi_R/R`` vs the recorded
   ``Z_j`` channel and ``Delta+ Phi = lap Phi + Phi_R/R`` vs the recorded
   ``omega`` channel, for both the FEM dof-based Laplacian and the graph-LSQ
   ``div(grad(...))`` the physics module currently uses;
3. degenerate-corner statistics, so we know how much of the mesh the operator
   actually covers.

Example:

    ./.torch-cuda-env/bin/python tests/check_fem_second_order.py
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
from src.physics.fem_derivatives import build_laplacian_map, nodal_laplacian_torch
from src.physics.model502_full_loss import delta_plus, delta_star


def _stats(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
    a = a.float()
    b = b.float()
    if mask is not None:
        a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.pow(2).mean().sqrt() * b.pow(2).mean().sqrt()).clamp_min(1e-30)
    return {
        "corr": float((a * b).mean() / denom),
        "rms_a": float(a.pow(2).mean().sqrt()),
        "rms_b": float(b.pow(2).mean().sqrt()),
        "relL2": float((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt().clamp_min(1e-30)),
        "n": int(a.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--jac-threshold", type=float, default=1e-5)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    store = RunFrameStore(
        Path(manifest["data_root"]) / args.run, cache_dir=args.cache_dir
    ).load()
    store.load()
    if store.topology is None:
        from src.data.interp_dataset import load_run_topology

        store.topology = load_run_topology(Path(manifest["data_root"]) / args.run)
    topology = store.topology
    pos = topology.pos.double()
    r = pos[:, 0].clamp_min(1e-8)
    vertex = topology.vertex
    assert vertex is not None and topology.x_dofs is not None and topology.size is not None

    frame = min(args.frame, store.u_dofs.shape[0] - 1)
    u_dofs = store.u_dofs[frame].double()
    u = store.u[frame].double()
    print(f"run={args.run} frame={frame} ({store.frames[frame].state_path.name}) "
          f"nodes={u.shape[0]} elements={vertex.shape[1]}")

    lap_map = build_laplacian_map(
        topology.size.numpy(), vertex.numpy(), topology.x_dofs.numpy(),
        jac_threshold=args.jac_threshold,
    )
    lap = nodal_laplacian_torch(
        u_dofs, torch.as_tensor(lap_map), vertex, n_nodes=u.shape[0]
    )
    covered = (lap.abs().sum(dim=1) != 0)
    print(f"nodes with a non-degenerate corner contribution: "
          f"{int(covered.sum())}/{covered.numel()} ({float(covered.float().mean()):.3%})")

    print("\n--- 1. geometry self-check (lap of the coordinate fields must be 0) ---")
    for name, values in (("R", pos[:, 0]), ("Z", pos[:, 1])):
        # Coordinates live in x_dofs as Hermite dofs; rebuild the same operator by
        # applying the map to a coordinate dof vector is not exposed, so use the
        # graph-free check: lap(R) via the *same* map applied to x dofs.
        x_dofs = topology.x_dofs.double()
        lap_x = nodal_laplacian_torch(
            x_dofs, torch.as_tensor(lap_map), vertex, n_nodes=u.shape[0]
        )[:, 0 if name == "R" else 1]
        sel = covered
        print(f"  lap({name}): mean={float(lap_x[sel].mean()):+.3e} "
              f"rms={float(lap_x[sel].pow(2).mean().sqrt()):.3e} "
              f"max|.|={float(lap_x[sel].abs().max()):.3e}")

    # The geometry test above only exercises the *first-derivative* part of the
    # chain rule (linear fields have zero second derivatives), so it passes even
    # while the quadratic terms are off by ~1e4.  These analytic fields are the
    # real acceptance criterion; their Hermite dofs are built exactly from the
    # coordinate dofs, so no field data is needed.
    print("\n--- 1b. analytic-field operator test (the real criterion) ---")
    x_dofs = topology.x_dofs.double()
    R, Z = x_dofs[:, 0, 0], x_dofs[:, 0, 1]
    analytic = {
        "R       (lap=0)": (x_dofs[:, :, 0], 0.0),
        "Z       (lap=0)": (x_dofs[:, :, 1], 0.0),
        "R*Z     (lap=0)": (
            torch.stack(
                [R * Z, Z * x_dofs[:, 1, 0], R * x_dofs[:, 2, 0],
                 Z * x_dofs[:, 3, 0] + x_dofs[:, 1, 0] * x_dofs[:, 2, 0]], dim=1
            ),
            0.0,
        ),
        "R^2     (lap=4)": (
            torch.stack(
                [R * R, 2 * R * x_dofs[:, 1, 0], 2 * R * x_dofs[:, 2, 0],
                 2 * (x_dofs[:, 1, 0] * x_dofs[:, 2, 0] + R * x_dofs[:, 3, 0])], dim=1
            ),
            4.0,
        ),
    }
    for name, (field_dofs, exact) in analytic.items():
        got = nodal_laplacian_torch(
            field_dofs.unsqueeze(-1), torch.as_tensor(lap_map), vertex, n_nodes=u.shape[0]
        )[:, 0]
        err = float((got - exact).pow(2).mean().sqrt())
        flag = "OK" if err < 1e-6 else "**BROKEN**"
        print(f"  {name:16s} rms={float(got.pow(2).mean().sqrt()):.4e} "
              f"exact={exact:g}  err_rms={err:.3e}  {flag}")

    print("\n--- 2. definition checks (FEM dof Laplacian vs recorded channels) ---")
    edge_index = topology.edge_index
    edge_attr = topology.edge_attr.double()
    zj_recorded = u[:, FIELD_NAMES.index("Z_j")]
    omega_recorded = u[:, FIELD_NAMES.index("omega")]
    from src.physics.grad_ops import compute_gradient

    rows = []
    grad_psi = compute_gradient(u[:, 0].float(), edge_index, edge_attr.float()).double()
    lap_psi_fem = lap[:, 0]
    delta_star_fem = lap_psi_fem - grad_psi[:, 0] / r
    delta_star_graph = delta_star(u[:, 0].float(), pos, edge_index, edge_attr.float())
    grad_phi = compute_gradient(u[:, 1].float(), edge_index, edge_attr.float()).double()
    lap_phi_fem = lap[:, 1]
    delta_plus_fem = lap_phi_fem + grad_phi[:, 0] / r
    delta_plus_graph = delta_plus(u[:, 1].float(), pos, edge_index, edge_attr.float())

    for label, op, target in (
        ("Delta*psi FEM   vs Z_j", delta_star_fem, zj_recorded),
        ("Delta*psi graph vs Z_j", delta_star_graph.double(), zj_recorded),
        ("lap(psi)  FEM   vs Z_j", lap_psi_fem, zj_recorded),
        ("Delta+Phi FEM   vs omega", delta_plus_fem, omega_recorded),
        ("Delta+Phi graph vs omega", delta_plus_graph.double(), omega_recorded),
        ("lap(Phi)  FEM   vs omega", lap_phi_fem, omega_recorded),
    ):
        s = _stats(op, target, covered)
        print(f"  {label:26s} corr={s['corr']:+.4f}  rms(op)={s['rms_a']:.3e}  "
              f"rms(target)={s['rms_b']:.3e}  relL2={s['relL2']:.3f}")
        rows.append((label, s))

    print("\n--- 3. dof-based vs graph-LSQ operators (same field) ---")
    for label, a, b in (
        ("lap(psi): FEM vs graph", lap_psi_fem, delta_star_graph.double() + grad_psi[:, 0] / r),
    ):
        s = _stats(a, b, covered)
        print(f"  {label:26s} corr={s['corr']:+.4f} relL2={s['relL2']:.3f}")


if __name__ == "__main__":
    main()
