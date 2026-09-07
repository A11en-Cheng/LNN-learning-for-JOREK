"""Longer training run for the split-head source surrogate (post-#0016)."""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames
from src.data.jorek_h5_parser import JorekDataParser
from src.data.model502_schema import FIELD_NAMES, standardize, transform_state


def _stats():
    m = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    f = m["stats"]["fields"]; s = m["stats"]["source_stats"]
    return (
        torch.tensor([f[n]["mean"] for n in FIELD_NAMES], dtype=torch.float32),
        torch.tensor([f[n]["std"] for n in FIELD_NAMES], dtype=torch.float32),
        torch.tensor(s["mean"], dtype=torch.float32),
        torch.tensor(s["std"], dtype=torch.float32),
    )


def _collect(runs, n_samples, nodes, hist_len, seed):
    rng = random.Random(seed)
    sm, ss, cm, cs = _stats()
    ix, iy, hx, hy = [], [], [], []
    for run in runs:
        frames = aligned_frames(Path("data/EHL-2") / run)
        for _ in range(n_samples):
            start = rng.randrange(hist_len, len(frames))
            idx = rng.sample(range(4125), nodes)
            g = JorekDataParser(str(frames[start].state_path), require_source=True).parse_to_graph()
            state = standardize(transform_state(g.x), sm, ss, name="state")[idx].numpy()
            target = ((g.source - cm) / torch.where(cs > 0, cs, torch.ones_like(cs)))[idx].numpy()
            past = []
            for off in range(1, hist_len + 1):
                pg = JorekDataParser(str(frames[start - off].state_path), require_source=True).parse_to_graph()
                past.append(
                    ((pg.source - cm) / torch.where(cs > 0, cs, torch.ones_like(cs)))[idx].numpy()
                )
            ix.append(state); iy.append(target[:, [2, 3, 4]])
            hx.append(np.concatenate([state] + past, axis=1)); hy.append(target[:, [0, 1]])
    return np.concatenate(ix), np.concatenate(iy), np.concatenate(hx), np.concatenate(hy)


def _fit(x, y, out_dim, epochs, batch, seed):
    torch.manual_seed(seed)
    x = torch.tensor(x, dtype=torch.float32); y = torch.tensor(y, dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 192), torch.nn.SiLU(),
        torch.nn.Linear(192, 192), torch.nn.SiLU(),
        torch.nn.Linear(192, out_dim),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(epochs):
        idx = torch.randperm(x.shape[0])[:batch]
        opt.zero_grad(); loss = torch.nn.functional.mse_loss(model(x[idx]), y[idx]); loss.backward(); opt.step()
        losses.append(float(loss.item()))
    model.eval()
    return model, losses


def _r2(t, p):
    t = np.asarray(t); p = np.asarray(p)
    return 1.0 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)


def main():
    m = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    tr = [r["name"] for r in m["runs"] if r["split"] == "train"]
    va = [r["name"] for r in m["runs"] if r["split"] == "val"]
    ix, iy, hx, hy = _collect(tr, 40, 500, 3, seed=0)
    vix, viy, vhx, vhy = _collect(va, 12, 500, 3, seed=1)

    inst, inst_loss = _fit(ix, iy, 3, epochs=400, batch=4096, seed=0)
    hist, hist_loss = _fit(hx, hy, 2, epochs=400, batch=4096, seed=1)
    with torch.no_grad():
        pi = inst(torch.tensor(vix, dtype=torch.float32)).numpy()
        ph = hist(torch.tensor(vhx, dtype=torch.float32)).numpy()

    report = {
        "train_samples": int(ix.shape[0]),
        "Z_eff": float(_r2(viy[:, 0], pi[:, 0])),
        "Z_imp": float(_r2(viy[:, 1], pi[:, 1])),
        "N_imp": float(_r2(viy[:, 2], pi[:, 2])),
        "dE_ion": float(_r2(vhy[:, 0], ph[:, 0])),
        "rad_sink": float(_r2(vhy[:, 1], ph[:, 1])),
    }
    out = Path("output/model502_baseline/source_surrogate_long")
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (out / "inst_loss.csv").write_text("\n".join(f"{i},{v}" for i, v in enumerate(inst_loss)))
    (out / "hist_loss.csv").write_text("\n".join(f"{i},{v}" for i, v in enumerate(hist_loss)))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

