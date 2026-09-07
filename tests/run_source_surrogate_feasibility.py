"""Feasibility pre-study: learned source surrogate vs the physical closures (#0012)."""

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
from src.data.model502_schema import standardize, transform_state


def _load_stats():
    manifest = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    fields = manifest["stats"]["fields"]
    from src.data.model502_schema import FIELD_NAMES

    state_mean = torch.tensor([fields[n]["mean"] for n in FIELD_NAMES], dtype=torch.float32)
    state_std = torch.tensor([fields[n]["std"] for n in FIELD_NAMES], dtype=torch.float32)
    src = manifest["stats"]["source_stats"]
    src_mean = torch.tensor(src["mean"], dtype=torch.float32)
    src_std = torch.tensor(src["std"], dtype=torch.float32)
    return state_mean, state_std, src_mean, src_std


def _collect(runs, n_triplets, nodes_per_frame, seed):
    rng = random.Random(seed)
    state_mean, state_std, src_mean, src_std = _load_stats()
    inst, hist, target = [], [], []
    for run in runs:
        frames = aligned_frames(Path("data/EHL-2") / run)
        for _ in range(n_triplets):
            start = rng.randrange(len(frames) - 3)
            graphs = []
            nodes = rng.sample(range(4125), nodes_per_frame)
            for offset in range(3):
                graph = JorekDataParser(
                    str(frames[start + offset].state_path), require_source=True
                ).parse_to_graph()
                state = standardize(
                    transform_state(graph.x), state_mean, state_std, name="state"
                )
                graphs.append((state, graph.source[nodes]))
            x3 = graphs[2][0][nodes].numpy()
            inst.append(x3)
            hist.append(np.concatenate([g[0][nodes].numpy() for g in graphs], axis=1))
            target.append(graphs[2][1].numpy())
    return (
        np.concatenate(inst),
        np.concatenate(hist),
        np.concatenate(target),
    )


def _fit_model(x, y, epochs=250, seed=0):
    torch.manual_seed(seed)
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(x.shape[1], 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, 128),
        torch.nn.SiLU(),
        torch.nn.Linear(128, y.shape[1]),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    for _ in range(epochs):
        idx = torch.randperm(x.shape[0])[:512]
        opt.zero_grad()
        loss = loss_fn(model(x[idx]), y[idx])
        loss.backward()
        opt.step()
    model.eval()
    return model


def _r2(target, pred):
    target = np.asarray(target)
    pred = np.asarray(pred)
    return 1.0 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)


def main():
    manifest = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    train_runs = [r["name"] for r in manifest["runs"] if r["split"] == "train"][:3]
    val_runs = [r["name"] for r in manifest["runs"] if r["split"] == "val"]
    x_inst, x_hist, y = _collect(train_runs, 20, 400, seed=0)
    xv_inst, xv_hist, yv = _collect(val_runs, 8, 400, seed=1)

    src_mean, src_std = _load_stats()[2], _load_stats()[3]
    y = (y - src_mean.numpy()) / src_std.numpy()
    yv = (yv - src_mean.numpy()) / src_std.numpy()

    report = {"fields": ["dE_ion", "rad_sink", "Z_eff", "Z_imp", "N_imp"]}
    for name, x, xv in (("instantaneous", x_inst, xv_inst), ("history3", x_hist, xv_hist)):
        model = _fit_model(x, y, seed=0)
        with torch.no_grad():
            pv = model(torch.tensor(xv, dtype=torch.float32)).numpy()
        report[name] = {
            "r2": [float(_r2(yv[:, i], pv[:, i])) for i in range(y.shape[1])],
        }

    # linear baseline
    lin = []
    for i in range(y.shape[1]):
        coef, *_ = np.linalg.lstsq(
            np.column_stack([x_inst, np.ones(x_inst.shape[0])]), y[:, i], rcond=None
        )
        p = np.column_stack([x_inst, np.ones(x_inst.shape[0])]) @ coef
        # val linear uses same coef
        pv = np.column_stack([xv_inst, np.ones(xv_inst.shape[0])]) @ coef
        lin.append(float(_r2(yv[:, i], pv)))
    report["linear_instantaneous_r2"] = lin
    report["closure_baseline_r2"] = [0.0] * y.shape[1]

    output = Path("output/model502_baseline/source_surrogate/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
