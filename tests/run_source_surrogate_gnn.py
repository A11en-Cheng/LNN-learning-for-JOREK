"""GNN source-surrogate upgrade with past-projection history (#0015)."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames
from src.data.jorek_h5_parser import JorekDataParser
from src.data.model502_schema import FIELD_NAMES, standardize, transform_state


def _stats():
    manifest = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    fields = manifest["stats"]["fields"]
    src = manifest["stats"]["source_stats"]
    return (
        torch.tensor([fields[n]["mean"] for n in FIELD_NAMES], dtype=torch.float32),
        torch.tensor([fields[n]["std"] for n in FIELD_NAMES], dtype=torch.float32),
        torch.tensor(src["mean"], dtype=torch.float32),
        torch.tensor(src["std"], dtype=torch.float32),
    )


def _frames(run, n_pairs, seed):
    rng = random.Random(seed)
    state_mean, state_std, src_mean, src_std = _stats()
    frames = aligned_frames(Path("data/EHL-2") / run)
    out = []
    for _ in range(n_pairs):
        start = rng.randrange(1, len(frames))
        prev = JorekDataParser(str(frames[start - 1].state_path), require_source=True).parse_to_graph()
        cur = JorekDataParser(str(frames[start].state_path), require_source=True).parse_to_graph()
        state = standardize(transform_state(cur.x), state_mean, state_std, name="state")
        prev_src = (prev.source - src_mean) / torch.where(src_std > 0, src_std, torch.ones_like(src_std))
        target = (cur.source - src_mean) / torch.where(src_std > 0, src_std, torch.ones_like(src_std))
        features = torch.cat([state, prev_src], dim=1)
        out.append(
            Data(
                x=features,
                edge_index=cur.edge_index,
                edge_attr=cur.edge_attr,
                y=target,
            )
        )
    return out


class _GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=5):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=2, concat=False, edge_dim=3)
        self.conv2 = GATConv(hidden, hidden, heads=2, concat=False, edge_dim=3)
        self.head = torch.nn.Linear(hidden, out_dim)

    def forward(self, x, edge_index, edge_attr):
        x = F.silu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.silu(self.conv2(x, edge_index, edge_attr=edge_attr))
        return self.head(x)


def _r2(target, pred):
    target = np.asarray(target)
    pred = np.asarray(pred)
    return 1.0 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)


def main():
    manifest = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    train_runs = [r["name"] for r in manifest["runs"] if r["split"] == "train"][:3]
    val_runs = [r["name"] for r in manifest["runs"] if r["split"] == "val"]
    train = [g for run in train_runs for g in _frames(run, 4, seed=0)]
    val = [g for run in val_runs for g in _frames(run, 3, seed=1)]

    model = _GNN(in_dim=14)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(80):
        total = torch.zeros(())
        for graph in train:
            pred = model(graph.x, graph.edge_index, graph.edge_attr)
            total = total + F.mse_loss(pred, graph.y)
        opt.zero_grad(); total.backward(); opt.step()
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for graph in val:
            preds.append(model(graph.x, graph.edge_index, graph.edge_attr).numpy())
            targets.append(graph.y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    fields = ["dE_ion", "rad_sink", "Z_eff", "Z_imp", "N_imp"]
    report = {
        "r2": {name: float(_r2(targets[:, i], preds[:, i])) for i, name in enumerate(fields)},
        "note": "GAT 2-layer + past-projection history; node features = 9 state + 5 past projections",
    }
    output = Path("output/model502_baseline/source_surrogate_gnn/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

