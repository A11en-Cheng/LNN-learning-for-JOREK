"""How much of each field's increment is predictable *locally*?

The CfC route has no spatial coupling at all (no ``edge_index`` anywhere in
``src/models/cfc_interp.py`` / ``lnn_processor.py``): every node evolves from its
own state, its own controls and its own hidden state.  Terms of the JOREK model
that need spatial derivatives -- the Poisson brackets ``{Phi,psi}``, ``{u,psi}``,
advection, ``Delta*psi`` / ``Delta+Phi`` -- are therefore outside the model
class, no matter how long it trains.

This script measures that directly with per-field *linear* regressions of the
recorded increment:

* ``local``      - features = the node's own 9 fields (the best a pointwise
  model can do linearly);
* ``local+grad`` - features += the recorded first-order derivative features
  (read from the HDF5 Hermite dofs), i.e. spatial information.

The reported number is the increment R2 ``1 - ||d_pred - d_true||^2/||d_true||^2``
on the held-out run.  Persistence (predict no change) is 0 by construction.
If a field jumps from ~0 to high when derivatives are added, its dynamics is
transport dominated and a pointwise model cannot learn it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import RunFrameStore
from src.data.model502_schema import FIELD_NAMES, NUM_STATE_FIELDS


def _ridge(features: np.ndarray, target: np.ndarray, lam: float = 1e-6) -> np.ndarray:
    """Least squares with a tiny ridge term (features are standardised)."""
    gram = features.T @ features
    gram += lam * np.eye(gram.shape[0]) * np.trace(gram) / gram.shape[0]
    return np.linalg.solve(gram, features.T @ target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(r for r in manifest["folds"] if r["held_out_run"] == args.held_out)
    root = Path(manifest["data_root"])

    def collect(runs: list[str]):
        local, grad, delta = [], [], []
        for name in runs:
            store = RunFrameStore(root / name, cache_dir=args.cache_dir).load()
            steps = store.u.shape[0]
            state = store.u.numpy().reshape(steps, -1, NUM_STATE_FIELDS)
            deriv = store.derivative_features().numpy().reshape(
                steps, -1, 2 * NUM_STATE_FIELDS
            )
            local.append(state[:-1].reshape(-1, NUM_STATE_FIELDS))
            grad.append(deriv[:-1].reshape(-1, 2 * NUM_STATE_FIELDS))
            delta.append((state[1:] - state[:-1]).reshape(-1, NUM_STATE_FIELDS))
        return np.concatenate(local), np.concatenate(grad), np.concatenate(delta)

    x_local_tr, x_grad_tr, d_tr = collect(fold["train_runs"])
    x_local_te, x_grad_te, d_te = collect([args.held_out])
    print(f"train rows={x_local_tr.shape[0]} test rows={x_local_te.shape[0]}")

    base_mean, base_std = x_local_tr.mean(axis=0), x_local_tr.std(axis=0) + 1e-30
    grad_mean, grad_std = x_grad_tr.mean(axis=0), x_grad_tr.std(axis=0) + 1e-30
    a_tr = (x_local_tr - base_mean) / base_std
    a_te = (x_local_te - base_mean) / base_std
    b_tr = np.concatenate([a_tr, (x_grad_tr - grad_mean) / grad_std], axis=1)
    b_te = np.concatenate([a_te, (x_grad_te - grad_mean) / grad_std], axis=1)
    one_tr = np.ones((a_tr.shape[0], 1))
    one_te = np.ones((a_te.shape[0], 1))
    a_tr, a_te = np.concatenate([a_tr, one_tr], axis=1), np.concatenate([a_te, one_te], axis=1)
    b_tr, b_te = np.concatenate([b_tr, one_tr], axis=1), np.concatenate([b_te, one_te], axis=1)

    print(f"{'field':9s} {'persistence':>11s} {'local':>9s} {'local+grad':>11s} {'gain':>8s}")
    summary = {}
    for index, name in enumerate(FIELD_NAMES):
        target = d_tr[:, index]
        pred_local = a_te @ _ridge(a_tr, target)
        pred_grad = b_te @ _ridge(b_tr, target)
        truth = d_te[:, index]
        scale = float((truth**2).sum())
        r2_local = 1.0 - float(((pred_local - truth) ** 2).sum()) / scale
        r2_grad = 1.0 - float(((pred_grad - truth) ** 2).sum()) / scale
        summary[name] = {
            "persistence_dR2": 0.0,
            "local_dR2": r2_local,
            "local_plus_grad_dR2": r2_grad,
            "gain": r2_grad - r2_local,
        }
        print(f"{name:9s} {0.0:11.3f} {r2_local:9.3f} {r2_grad:11.3f} {r2_grad - r2_local:8.3f}")
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
