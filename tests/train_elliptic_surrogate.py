"""Split elliptic surrogates: psi->Z_j (Grad-Shafranov) and u->omega (Laplacian)."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames


def _load(run, index, cache):
    d = torch.load(f"{cache}/{run}/{index}.pt", map_location="cpu")
    return d


class _Head(torch.nn.Module):
    def __init__(self, in_dim=3, hidden=96):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden, heads=2, concat=False, edge_dim=2)
        self.c2 = GATConv(hidden, hidden, heads=2, concat=False, edge_dim=2)
        self.c3 = GATConv(hidden, hidden, heads=2, concat=False, edge_dim=2)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, edge_attr):
        x = F.silu(self.c1(x, edge_index, edge_attr=edge_attr))
        x = F.silu(self.c2(x, edge_index, edge_attr=edge_attr))
        x = F.silu(self.c3(x, edge_index, edge_attr=edge_attr))
        return self.out(x)


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="/tmp/ehl2_cache")
    p.add_argument("--train-frames", type=int, default=80)
    p.add_argument("--val-frames", type=int, default=16)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/model502_baseline/elliptic_surrogate_split")
    args = p.parse_args()

    m = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    tr = [r["name"] for r in m["runs"] if r["split"] == "train"]
    va = [r["name"] for r in m["runs"] if r["split"] == "val"]

    def frames(runs, n, seed):
        rng = random.Random(seed)
        out = []
        for run in runs:
            fs = aligned_frames(Path("data/EHL-2") / run)
            for _ in range(n // len(runs) + 1):
                out.append(_load(run, rng.choice(fs).index, args.cache))
        return out

    train = frames(tr, args.train_frames, 0)
    val = frames(va, args.val_frames, 1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    def batch(d):
        psi = torch.cat([d["x"][:, 0:1], d["pos"]], dim=1)
        u = torch.cat([d["x"][:, 1:2], d["pos"]], dim=1)
        return (
            Data(x=psi, edge_index=d["edge_index"], edge_attr=d["edge_attr"][:, :2]),
            Data(x=u, edge_index=d["edge_index"], edge_attr=d["edge_attr"][:, :2]),
            d["x"][:, 2:3], d["x"][:, 3:4],
        )

    head_psi = _Head().to(device)
    head_u = _Head().to(device)
    opt = torch.optim.AdamW(list(head_psi.parameters()) + list(head_u.parameters()), lr=1e-3)
    for epoch in range(args.epochs):
        total = torch.zeros((), device=device)
        for d in train:
            gpsi, gu, zj, om = batch(d)
            gpsi = gpsi.to(device); gu = gu.to(device); zj = zj.to(device); om = om.to(device)
            pz = head_psi(gpsi.x, gpsi.edge_index, gpsi.edge_attr)
            pw = head_u(gu.x, gu.edge_index, gu.edge_attr)
            total = total + F.mse_loss(pz, zj) + F.mse_loss(pw, om)
        opt.zero_grad(); total.backward(); opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1} loss {total.item():.4e}", flush=True)

    head_psi.eval(); head_u.eval()
    zj_p, zj_t, om_p, om_t = [], [], [], []
    with torch.no_grad():
        for d in val:
            gpsi, gu, zj, om = batch(d)
            gpsi = gpsi.to(device); gu = gu.to(device)
            zj_p.append(head_psi(gpsi.x, gpsi.edge_index, gpsi.edge_attr)); zj_t.append(zj.to(device))
            om_p.append(head_u(gu.x, gu.edge_index, gu.edge_attr)); om_t.append(om.to(device))
    zj_p = torch.cat(zj_p); zj_t = torch.cat(zj_t); om_p = torch.cat(om_p); om_t = torch.cat(om_t)

    def r2(t, pr):
        return 1.0 - ((t - pr) ** 2).sum() / ((t - t.mean()) ** 2).sum()

    report = {"Z_j_r2": float(r2(zj_t, zj_p).item()), "omega_r2": float(r2(om_t, om_p).item())}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    torch.save({"head_psi": head_psi.state_dict(), "head_u": head_u.state_dict()}, out / "model.pt")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

