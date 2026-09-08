"""Elliptic surrogate: (psi, u, geometry) -> (Z_j, omega) via GNN (#0018)."""

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
from src.data.jorek_h5_parser import JorekDataParser


def _graph(run, index, cache):
    d = torch.load(f"{cache}/{run}/{index}.pt", map_location="cpu")
    x = d["x"]
    features = torch.cat([x[:, 0:1], x[:, 1:2], d["pos"]], dim=1)  # psi, u, R, Z
    return Data(
        x=features,
        edge_index=d["edge_index"],
        edge_attr=d["edge_attr"][:, :2],
        y=x[:, 2:4],  # Z_j, omega
    )


class _GNN(torch.nn.Module):
    def __init__(self, in_dim=4, hidden=128, out_dim=2):
        super().__init__()
        self.c1 = GATConv(in_dim, hidden, heads=2, concat=False, edge_dim=2)
        self.c2 = GATConv(hidden, hidden, heads=2, concat=False, edge_dim=2)
        self.c3 = GATConv(hidden, hidden, heads=2, concat=False, edge_dim=2)
        self.head = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.silu(self.c1(x, edge_index, edge_attr=edge_attr))
        x = F.silu(self.c2(x, edge_index, edge_attr=edge_attr))
        x = F.silu(self.c3(x, edge_index, edge_attr=edge_attr))
        return self.head(x)


def _r2(t, p):
    t = torch.as_tensor(t)
    p = torch.as_tensor(p)
    return 1.0 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default="/tmp/ehl2_cache")
    parser.add_argument("--train-frames", type=int, default=80)
    parser.add_argument("--val-frames", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="output/model502_baseline/elliptic_surrogate")
    args = parser.parse_args()

    manifest = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    train_runs = [r["name"] for r in manifest["runs"] if r["split"] == "train"]
    val_runs = [r["name"] for r in manifest["runs"] if r["split"] == "val"]

    def sample(runs, n, seed):
        rng = random.Random(seed)
        out = []
        for run in runs:
            frames = aligned_frames(Path("data/EHL-2") / run)
            for _ in range(n // len(runs) + 1):
                f = rng.choice(frames)
                out.append(_graph(run, f.index, args.cache))
        return out

    train = sample(train_runs, args.train_frames, 0)
    val = sample(val_runs, args.val_frames, 1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _GNN(hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for epoch in range(args.epochs):
        model.train()
        loss = torch.zeros((), device=device)
        for g in train:
            g = g.to(device)
            pred = model(g.x, g.edge_index, g.edge_attr)
            loss = loss + F.mse_loss(pred, g.y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 20 == 0:
            print(f"epoch {epoch+1} train {loss.item():.4e}", flush=True)

    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for g in val:
            g = g.to(device)
            preds.append(model(g.x, g.edge_index, g.edge_attr))
            targets.append(g.y)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    report = {
        "Z_j_r2": float(_r2(targets[:, 0], preds[:, 0]).item()),
        "omega_r2": float(_r2(targets[:, 1], preds[:, 1]).item()),
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    torch.save(model.state_dict(), out / "model.pt")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

