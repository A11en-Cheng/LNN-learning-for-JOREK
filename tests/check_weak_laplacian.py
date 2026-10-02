"""Verify the weak-form (nodal bilinear) Laplacian against the graph Laplacian.

Four checks (see ``src/physics/fem_weak.py``):

1. ``A`` is symmetric and positive semi-definite, and annihilates constants;
2. linear coordinate fields have zero Laplacian (the null space is right);
3. on a *smooth* recorded field the weak Laplacian agrees with the graph-LSQ
   ``div(grad)`` the physics module currently uses;
4. **the point of the exercise**: adding node-level noise to the field amplifies
   the weak Laplacian far less than the graph-LSQ one.

Example:

    ./.torch-cuda-env/bin/python tests/check_weak_laplacian.py
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
from src.physics.fem_weak import boundary_mask, build_p1_laplacian
from src.physics.model502_full_loss import _laplacian


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
    a_mat, mass, cluster = build_p1_laplacian(vertex, pos)
    a_dense = torch.as_tensor(a_mat.toarray(), dtype=torch.float64)
    mass_t = torch.as_tensor(mass, dtype=torch.float64).clamp_min(1e-30)

    n_clusters = a_mat.shape[0]
    cluster_t = torch.as_tensor(cluster, dtype=torch.long)

    def to_cluster(values: torch.Tensor) -> torch.Tensor:
        """Average the nodal values inside each cluster (coincident axis nodes)."""
        out = torch.zeros(n_clusters, dtype=torch.float64)
        cnt = torch.zeros(n_clusters, dtype=torch.float64)
        out.index_add_(0, cluster_t, values.double())
        cnt.index_add_(0, cluster_t, torch.ones_like(values.double()))
        return out / cnt.clamp_min(1.0)

    def weak_lap(values: torch.Tensor) -> torch.Tensor:
        return -(a_dense @ to_cluster(values)) / mass_t

    print(f"run={args.run} nodes={n_nodes} elements={vertex.shape[1]} "
          f"clusters={n_clusters} (merged {n_nodes - n_clusters} coincident nodes)")
    print(f"A: nnz={a_mat.nnz}  symmetric={np.abs((a_mat - a_mat.T)).max() < 1e-12}  "
          f"min eigenvalue >= -1e-10: ", end="")
    eig = np.linalg.eigvalsh(a_mat.toarray()[::7, ::7])  # cheap submatrix check
    print(bool(eig.min() > -1e-8), f"(subsampled min eig {eig.min():.2e})")

    wall = torch.as_tensor(boundary_mask(vertex), dtype=torch.bool)
    boundary = torch.zeros(n_clusters, dtype=torch.bool)
    boundary[cluster_t[wall]] = True
    boundary[cluster_t[topology.boundary.long()]] = True
    interior = ~boundary
    print(f"boundary nodes: {int(boundary.sum())}  interior: {int(interior.sum())}")

    def weak_lap_int(values):
        return weak_lap(values)[interior]

    ones = torch.ones(n_nodes, dtype=torch.float64)
    print(f"lap(constant) rms (interior) = {float(weak_lap_int(ones).pow(2).mean().sqrt()):.3e}")
    for name, field in (("R", pos[:, 0]), ("Z", pos[:, 1])):
        val = weak_lap_int(torch.as_tensor(field, dtype=torch.float64))
        print(f"lap({name}) rms (interior) = {float(val.pow(2).mean().sqrt()):.3e} (should be ~0)")

    psi = store.u[args.frame][:, 0]
    edge_index, edge_attr = topology.edge_index, topology.edge_attr.float()
    graph_full = _laplacian(psi.float(), edge_index, edge_attr).double()
    graph = to_cluster(graph_full)[interior]
    weak = weak_lap_int(psi)
    corr = float(torch.corrcoef(torch.stack([graph.float(), weak.float()]))[0, 1])
    print(f"\nsmooth field (psi): corr(weak, graph-LSQ) = {corr:+.4f}  "
          f"rms weak={float(weak.pow(2).mean().sqrt()):.3e}  "
          f"rms graph={float(graph.pow(2).mean().sqrt()):.3e}")

    torch.manual_seed(0)
    noise_level = 0.01 * float(psi.pow(2).mean().sqrt())
    noisy = psi + noise_level * torch.randn_like(psi)
    graph_noisy = to_cluster(
        _laplacian(noisy.float(), edge_index, edge_attr).double()
    )[interior]
    weak_noisy = weak_lap_int(noisy)
    graph_amp = float((graph_noisy - graph).pow(2).mean().sqrt()) / noise_level
    weak_amp = float((weak_noisy - weak).pow(2).mean().sqrt()) / noise_level
    print(f"noise amplification (||d lap|| / ||noise||): graph-LSQ {graph_amp:.3e} vs "
          f"weak {weak_amp:.3e}  ->  {graph_amp / max(weak_amp, 1e-30):.1f}x lower")


if __name__ == "__main__":
    main()
