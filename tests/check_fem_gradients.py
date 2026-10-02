"""Validate the Hermite-dof gradient against the graph least-squares gradient."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import RunFrameStore
from src.physics.fem_derivatives import nodal_gradient
from src.physics.fem_derivatives import build_gradient_map, nodal_gradient_torch
from src.physics.grad_ops import compute_gradient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--position", type=int, default=30)
    args = parser.parse_args()

    store = RunFrameStore(Path(args.data_root) / args.run).load()
    topology = store.topology
    u_dofs = store.u_dofs[args.position].numpy()
    x_dofs = topology.x_dofs.numpy()
    size = topology.size.numpy()
    vertex = topology.vertex.numpy()
    u = store.u[args.position]

    for field, name in ((0, "psi"), (8, "T_e")):
        t0 = time.time()
        fem = nodal_gradient(u_dofs, x_dofs, size, vertex, field=field)
        t_fem = time.time() - t0
        t0 = time.time()
        lsq = compute_gradient(
            u[:, field].float(), topology.edge_index, topology.edge_attr.float()
        ).double().numpy()
        t_lsq = time.time() - t0
        print(f"{name}: FEM {t_fem*1e3:6.2f} ms vs graph-LSQ {t_lsq*1e3:6.2f} ms")
        for comp, axis in ((0, "d/dR"), (1, "d/dZ")):
            a, b = fem[:, comp], lsq[:, comp]
            corr = float(np.corrcoef(a, b)[0, 1])
            rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
            print(f"   {axis}: corr={corr:8.4f}  relL2(FEM vs LSQ)={rel:8.4f} "
                  f"rms(FEM)={np.sqrt((a**2).mean()):.4e} rms(LSQ)={np.sqrt((b**2).mean()):.4e}")

    # torch/precomputed path: all fields at once, one static geometry map.
    t0 = time.time()
    gradient_map, vertex_t = build_gradient_map(size, vertex, x_dofs)
    t_build = time.time() - t0
    u_dofs_t = store.u_dofs[args.position]
    counts = None  # let nodal_gradient_torch accumulate the per-node element counts
    g_map = torch.as_tensor(gradient_map, dtype=u_dofs_t.dtype)
    vertex_tt = torch.as_tensor(vertex_t)
    for _ in range(3):
        torch_grad = nodal_gradient_torch(
            u_dofs_t, g_map, vertex_tt, n_nodes=u_dofs_t.shape[0], counts=counts
        )
    t0 = time.time()
    torch_grad = nodal_gradient_torch(
        u_dofs_t, g_map, vertex_tt, n_nodes=u_dofs_t.shape[0], counts=counts
    )
    t_torch = time.time() - t0
    ref = torch.as_tensor(fem_all := np.stack(
        [nodal_gradient(u_dofs, x_dofs, size, vertex, field=f) for f in range(9)], axis=-1
    ))
    diff = (torch_grad.double() - ref).abs()
    print(f"torch FEM (9 fields): build map {t_build*1e3:.1f} ms (once/run), "
          f"per-frame {t_torch*1e3:.2f} ms; max|torch-numpy|={float(diff.max()):.3e} "
          f"(ref rms={float(ref.abs().mean()):.3e})")


if __name__ == "__main__":
    main()
