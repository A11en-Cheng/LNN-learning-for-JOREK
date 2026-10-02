"""Per-term attribution of the model-502 residuals (uses the term registry).

For one recorded interval this prints, per equation, the residual (relative
norm) together with the rms of every term, on

* the **recorded** trajectory ``u_k -> u_{k+1}``  (what the data itself gives:
  the residual floor and which terms dominate it), and
* optionally the **model** sub-interval trajectory ``u(t*) -> u(t*+delta)`` from
  a CfC checkpoint (which term the model gets wrong, and by how much).

Example:

    ./.torch-cuda-env/bin/python tests/report_residual_terms.py --checkpoint output/cfc_interp/gate6_deriv2/checkpoint.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import (
    InterpolationWindowDataset,
    RunFrameStore,
    collate_windows,
    compute_transform_stats,
    to_device,
)
from src.models.cfc_interp import CfCInterpolator
from src.models.cfc_interp import model_kwargs_from_meta
from src.physics.model502_full_loss import model502_full_residuals
from tests.train_cfc_interpolation import _condition_from_frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dt-scale", type=float, default=50.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(r for r in manifest["folds"] if r["held_out_run"] == args.held_out)
    stats = compute_transform_stats(
        {
            name: RunFrameStore(
                Path(manifest["data_root"]) / name, cache_dir=args.cache_dir
            ).load()
            for name in fold["train_runs"]
        },
        runs=fold["train_runs"],
        with_lap=bool(args.checkpoint),
    )
    dataset = InterpolationWindowDataset(
        manifest, manifest["data_root"], runs=[args.held_out], inner="val",
        stats=stats, cache_dir=args.cache_dir, with_shards=False,
        with_lap=bool(args.checkpoint),
    )
    mean = torch.tensor(stats["field_mean"], device=device)
    std = torch.tensor(stats["field_std"], device=device)
    source_stats = (
        torch.tensor(stats["source_mean"], device=device),
        torch.tensor(stats["source_std"], device=device).clamp_min(1e-12),
    )

    item = dataset[args.window]
    batch = to_device(collate_windows([item]), device)
    u = batch["u"][0]
    dt = batch["dt"][0]
    source = batch["source"][0]
    condition = item["condition"]
    topology = batch["topology"]
    burn_in = int((~batch["loss_mask"][0]).sum().item())
    coeff = _condition_from_frame(condition[burn_in].numpy()).to(device)
    graph = Data(
        x=u[burn_in],
        pos=topology.pos.to(device).double(),
        edge_index=topology.edge_index.to(device),
        edge_attr=topology.edge_attr.to(device).double(),
        condition=coeff,
        source=source[burn_in + 1].to(device).double(),
    )

    recorded = model502_full_residuals(
        u[burn_in + 1].double(), u[burn_in].double(), graph, float(dt[burn_in]),
        stats=(mean, std), source_stats=source_stats, return_terms=True,
    )
    rows = {"recorded": recorded}

    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model = CfCInterpolator(**model_kwargs_from_meta(payload.get("meta") or {}))
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        deriv = batch["u_deriv"][0]
        lap = batch["u_lap"][0]
        with torch.no_grad():
            hidden = None
            for k in range(burn_in):
                _, _, hidden = model.step(u[k], hidden, dt[k], source[k + 1], None,
                                           deriv[k], lap[k])
            latent = model.encode(u[burn_in], deriv[burn_in], lap[burn_in])
            mid_latent, mid_hidden = model.advance(latent, hidden, 0.5 * dt[burn_in],
                                                   source[burn_in + 1], None)
            u_mid = model.decode(mid_latent) + u[burn_in]
            end_latent, _ = model.advance(mid_latent, mid_hidden, 0.5 * dt[burn_in],
                                          source[burn_in + 1], None)
            u_end = model.decode(end_latent) + u[burn_in]
        graph_mid = Data(
            x=u_mid, pos=graph.pos, edge_index=graph.edge_index,
            edge_attr=graph.edge_attr, condition=coeff,
            source=source[burn_in + 1].to(device).double(),
        )
        rows["model"] = model502_full_residuals(
            u_end.double(), u_mid.double(), graph_mid, 0.5 * float(dt[burn_in]),
            stats=(mean, std), source_stats=source_stats, return_terms=True,
        )

    for equation in rows["recorded"]["residuals"]:
        print(f"\n=== {equation} ===")
        header = f"{'term':16s}"
        for tag in rows:
            header += f"{tag:>14s}"
        print(header)
        reference = rows["recorded"]["term_rms"].get(equation, {})
        model_terms = rows.get("model", {}).get("term_rms", {}).get(equation, {})
        for name, value in reference.items():
            line = f"{name:16s}{value:14.4e}"
            if model_terms:
                line += f"{model_terms.get(name, float('nan')):14.4e}"
            print(line)
        res_line = f"{'RESIDUAL':16s}{float(rows['recorded']['residuals'][equation]):14.4f}"
        if "model" in rows:
            res_line += f"{float(rows['model']['residuals'][equation]):14.4f}"
        print(res_line)


if __name__ == "__main__":
    main()
