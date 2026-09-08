"""Grad-Shafranov surrogate: (psi, geometry) -> Z_j via GNN (#0018)."""

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
    def __init__(self, hidden=128):
        super().__init__()
        self.c1 = GATConv(3, hidden, heads=2, concat=False, edge_dim=2)
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
    p.add_argument("--train-frames", type=int, default=120)
    p.add_argument("--val-frames", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="output/model502_baseline/elliptic_surrogate_zj")
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
    model = _Head().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def graph(d):
        x = torch.cat([d["x"][:, 0:1], d["pos"]], dim=1)  # psi, R, Z
        return Data(x=x, edge_index=d["edge_index"], edge_attr=d["edge_attr"][:, :2]), d["x"][:, 2:3]

    for epoch in range(args.epochs):
        total = torch.zeros((), device=device)
        for d in train:
            g, y = graph(d)
            g = g.to(device); y = y.to(device)
            pred = model(g.x, g.edge_index, g.edge_attr)
            total = total + F.mse_loss(pred, y)
        opt.zero_grad(); total.backward(); opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1} loss {total.item():.4e}", flush=True)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for d in val:
            g, y = graph(d)
            g = g.to(device)
            preds.append(model(g.x, g.edge_index, g.edge_attr)); targets.append(y.to(device))
    preds = torch.cat(preds); targets = torch.cat(targets)
    r2 = 1.0 - ((targets - preds) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()
    report = {"Z_j_r2": float(r2.item())}
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    torch.save(model.state_dict(), out / "model.pt")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

