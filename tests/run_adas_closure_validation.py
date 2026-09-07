"""Calibrate and validate the ADAS coronal closure against recorded projections (#0014)."""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames
from src.data.jorek_h5_parser import JorekDataParser
from src.physics.adas_closure import coronal_moments, fit_te_axis, load_neon_tables


N_NORM = 1e20


def _sample(frames, n_frames, n_nodes, seed):
    rng = random.Random(seed)
    te, zeff, zimp = [], [], []
    for frame in rng.sample(frames, min(n_frames, len(frames))):
        graph = JorekDataParser(str(frame.state_path), require_source=True).parse_to_graph()
        projection = graph.source.numpy()  # [N,5]
        idx = rng.sample(range(graph.num_nodes), min(n_nodes, graph.num_nodes))
        te.append(np.clip(graph.x[:, 8].numpy()[idx], 1e-12, None))
        n_imp = np.clip(projection[idx, 4], 1e-12, None)
        zeff.append(projection[idx, 2] / n_imp)
        zimp.append(projection[idx, 3] / n_imp)
    return (
        np.concatenate(te),
        np.concatenate(zimp),
        np.concatenate(zeff),
    )


def _r2(target, pred):
    target = np.asarray(target)
    pred = np.asarray(pred)
    return 1.0 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)


def main():
    manifest = json.loads(Path(".scratch/model502-baseline/manifest.json").read_text())
    train_runs = [r["name"] for r in manifest["runs"] if r["split"] == "train"]
    val_runs = [r["name"] for r in manifest["runs"] if r["split"] == "val"]
    train_frames = [
        frame
        for run in train_runs[:4]
        for frame in aligned_frames(Path("data/EHL-2") / run)
    ]
    val_frames = [
        frame
        for run in val_runs
        for frame in aligned_frames(Path("data/EHL-2") / run)
    ]

    scd, acd = load_neon_tables()
    te_train, zimp_train, zeff_train = _sample(train_frames, 24, 800, seed=0)
    (a, b), fit = fit_te_axis(te_train, zimp_train, zeff_train, scd, acd)

    te_val, zimp_val, zeff_val = _sample(val_frames, 12, 800, seed=1)
    zimp_pred, zeff_pred = coronal_moments(a * np.log10(te_val) + b, 13.0, scd, acd)

    # linear baselines: log Te -> moment
    lx = np.log10(te_train)
    zimp_lr = np.polyfit(lx, zimp_train, 1)
    zeff_lr = np.polyfit(lx, zeff_train, 1)
    zimp_lin = np.polyval(zimp_lr, np.log10(te_val))
    zeff_lin = np.polyval(zeff_lr, np.log10(te_val))

    start = time.perf_counter()
    coronal_moments(a * np.log10(te_val[:4125]) + b, 13.0, scd, acd)
    elapsed = time.perf_counter() - start
    cost_vs_model = elapsed / 0.061  # measured model step on CPU ~61 ms

    report = {
        "axis_params": {"a": float(a), "b": float(b)},
        "z_imp_r2": float(_r2(zimp_val, zimp_pred)),
        "z_eff_r2": float(_r2(zeff_val, zeff_pred)),
        "z_imp_linear_r2": float(_r2(zimp_val, zimp_lin)),
        "z_eff_linear_r2": float(_r2(zeff_val, zeff_lin)),
        "closure_4125_nodes_seconds": elapsed,
        "cost_ratio_vs_model_step": float(cost_vs_model),
        "units_note": "temperature axis calibrated in log10(eV); dE_ion left external",
    }
    output = Path("output/model502_baseline/adas_closure/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
