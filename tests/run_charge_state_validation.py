"""Validate the time-dependent charge-state closure against recorded projections."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_manifest import aligned_frames
from src.data.jorek_h5_parser import JorekDataParser
from src.physics.charge_state_closure import ChargeStateClosure


def _r2(target, pred):
    return 1.0 - np.sum((target - pred) ** 2) / np.sum((target - target.mean()) ** 2)


def main():
    run = Path("data/EHL-2/scan_2d21_ang12_massloss_neon_v300")
    frames = aligned_frames(run)
    indices = list(range(0, 30))
    nodes = np.arange(0, 200)
    recorded_zi, recorded_ze = [], []
    closure_zi, closure_ze = [], []

    closure = ChargeStateClosure(len(nodes), axis_params=(1.0, 6.0))
    for i in indices:
        graph = JorekDataParser(str(frames[i].state_path), require_source=True).parse_to_graph()
        if i + 1 < len(frames):
            next_graph = JorekDataParser(str(frames[i + 1].state_path)).parse_to_graph()
            dt_jorek = float((next_graph.time[0] - graph.time[0]).item())
        else:
            dt_jorek = 10.0
        dt_s = float(dt_jorek * graph.t_norm[0].item())
        te = graph.x[nodes, 8].numpy()
        rho = graph.x[nodes, 4].numpy()
        rho_imp = graph.x[nodes, 7].numpy()
        projection = graph.source.numpy()[nodes]
        n_imp = np.clip(projection[:, 4], 1e-12, None)
        recorded_zi.append(projection[:, 3] / n_imp)
        recorded_ze.append(projection[:, 2] / n_imp)
        zi, ze = closure.step(
            te, rho, rho_imp, dt_s,
            central_density=float(graph.norm_factors.get("central_density", 1.0)),
            central_mass=float(graph.norm_factors.get("central_mass", 1.0)),
        )
        closure_zi.append(zi)
        closure_ze.append(ze)

    recorded_zi = np.concatenate(recorded_zi)
    recorded_ze = np.concatenate(recorded_ze)
    closure_zi = np.concatenate(closure_zi)
    closure_ze = np.concatenate(closure_ze)
    report = {
        "z_imp_r2": float(_r2(recorded_zi, closure_zi)),
        "z_eff_r2": float(_r2(recorded_ze, closure_ze)),
        "note": "time-dependent node-local f_Z, fixed axis (a=1,b=6), no advection",
    }
    output = Path("output/validation/charge_state_closure/report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
