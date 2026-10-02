"""Phase-4 step 1: physics-residual floor smoke.

.. note::
   Superseded by :mod:`tests.check_residual_floor_stats` (per-interval distribution
   over every run, coefficients from each interval's own frame, source in physical
   units).  Keep this file only as a single-interval debug view.

Loads the Phase-3 CfC checkpoint and compares the nine-equation residual set on

  (a) the *recorded* trajectory over one supervised interval  -> residual floor
  (b) the CfC sub-interval trajectory at an interpolated point -> physics signal

with the recorded projections as teacher source.  Coefficients come from the
EHL-2 namelist `f0` plus the defaults used inside ``model502_full_loss``
(eta/visco/kappa defaults; the full namelist condition vector is Phase-4.1 work).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch_geometric.data import Data
import h5py
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import InterpolationWindowDataset, compute_transform_stats
from src.data.window_manifest import collect_frames
from src.models.cfc_interp import CfCInterpolator
from src.models.cfc_interp import model_kwargs_from_meta
from src.physics.model502_full_loss import model502_full_residuals

# Defaults taken from the ``_coeff(...)`` calls in model502_full_loss.
CONDITION_DEFAULTS = {
    0: 1e-5,   # eta
    1: 1e-5,   # visco
    2: 1e-5,   # visco_par
    3: 0.0,    # visco_par_par
    4: 1.0,    # ZK_i_par
    5: 1.0,    # ZK_e_par
    6: 1e-5,   # D_perp
    11: 1e-5,  # D_perp_imp
    12: 1e-5,  # ZK_i_perp
    18: 1e-5,  # ZK_e_perp
}


def _condition(frame_condition: np.ndarray) -> torch.Tensor:
    """Map the frame's parameter channel onto the residual module's layout.

    The frame vector has ``eta, visco, visco_par, visco_par_par, ZK_i_par,
    ZK_e_par, D_perp[1..6], ZK_i_perp[1..6], ZK_e_perp[1..6], ..., F0``; the
    residual module expects a single value per coefficient at indices
    ``0..5, 6 (D_perp), 11 (D_perp_imp), 12 (ZK_i_perp), 18 (ZK_e_perp), 33 (F0)``.
    Entries the data does not provide keep the module defaults.
    """
    condition = torch.zeros(34, dtype=torch.float64)
    for index, value in CONDITION_DEFAULTS.items():
        condition[index] = value
    condition[0:6] = torch.as_tensor(frame_condition[0:6], dtype=torch.float64)
    condition[6] = float(frame_condition[6])    # D_perp[1]
    condition[11] = float(frame_condition[6])   # D_perp_imp ~ D_perp (module default)
    condition[12] = float(frame_condition[12])  # ZK_i_perp[1]
    condition[18] = float(frame_condition[18])  # ZK_e_perp[1]
    condition[33] = float(frame_condition[33])  # F0 (h5)
    return condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=".scratch/n/manifest.json")
    parser.add_argument("--cache-dir", default=".scratch/n/frames")
    parser.add_argument("--held-out", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--checkpoint", default="output/cfc_interp/phase3_v3/checkpoint.pt")
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dt-scale", type=float, default=50.0)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument("--rho", type=float, default=0.2, help="delta = rho * dt")
    parser.add_argument(
        "--f0",
        type=float,
        default=None,
        help="override; by default F0 is read from the frame's jorek*.h5",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    fold = next(r for r in manifest["folds"] if r["held_out_run"] == args.held_out)
    stats = compute_transform_stats(
        {name: __import__("src.data.interp_dataset", fromlist=["RunFrameStore"]).RunFrameStore(
            Path(manifest["data_root"]) / name, cache_dir=args.cache_dir).load()
         for name in fold["train_runs"]},
        runs=fold["train_runs"],
    )
    dataset = InterpolationWindowDataset(
        manifest, manifest["data_root"], runs=[args.held_out], inner="val",
        stats=stats, cache_dir=args.cache_dir, with_shards=False,
    )
    item = dataset[args.window]
    topology = item["topology"]
    burn_in_position = int((~item["loss_mask"]).sum().item())
    store = dataset._store(item["run"])
    state_path = store.frames[int(item["frame_indices"][burn_in_position])].state_path
    frame = collect_frames(state_path.parent)[int(item["frame_indices"][burn_in_position])]
    with h5py.File(state_path, "r") as handle:
        f0_h5 = float(np.ravel(handle["F0"][()])[0])
    frame_condition = np.asarray(frame.condition, dtype=np.float64)
    if args.f0 is not None:
        frame_condition = frame_condition.copy()
        frame_condition[33] = float(args.f0)
    print(
        f"frame {state_path.name}: eta={frame_condition[0]:.3e} visco={frame_condition[1]:.4g} "
        f"visco_par={frame_condition[2]:.3g} D_perp[1]={frame_condition[6]:.3g} "
        f"ZK_i_perp[1]={frame_condition[12]:.3g} F0={frame_condition[33]:.9f} (h5={f0_h5:.9f})"
    )
    mean = torch.tensor(stats["field_mean"], dtype=torch.float64)
    std = torch.tensor(stats["field_std"], dtype=torch.float64)

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = CfCInterpolator(**model_kwargs_from_meta(payload.get("meta") or {}))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()

    u = item["u"].to(device)                # normalised recorded states
    dt = item["dt"].to(device)
    source = item["source"].to(device)      # teacher source (raw projections)
    burn_in = int((~item["loss_mask"]).sum().item())
    graph = Data(
        x=u[burn_in],
        pos=topology.pos.to(device).double(),
        edge_index=topology.edge_index.to(device),
        edge_attr=topology.edge_attr.to(device).double(),
        condition=_condition(frame_condition).to(device),
        source=source[burn_in + 1].to(device).double(),
    )

    with torch.no_grad():
        hidden = None
        for k in range(burn_in):
            _, _, hidden = model.step(u[k], hidden, dt[k], source[k + 1])
        # (a) recorded trajectory over the supervised interval
        recorded = model502_full_residuals(
            u[burn_in + 1].double(), u[burn_in].double(), graph, float(dt[burn_in]),
            stats=(mean, std),
        )
        # (b) CfC sub-interval: t* = alpha*dt, then delta = rho*dt
        latent = model.encode(u[burn_in])
        mid_latent, mid_hidden = model.advance(
            latent, hidden, args.alpha * dt[burn_in], source[burn_in + 1], None
        )
        u_mid = model.decode(mid_latent)
        delta = args.rho * float(dt[burn_in])
        end_latent, _ = model.advance(mid_latent, mid_hidden, delta, source[burn_in + 1], None)
        u_end = model.decode(end_latent)
        model_res = model502_full_residuals(
            u_end.double(), u_mid.double(), graph, delta, stats=(mean, std)
        )

    print(f"window={item['run']} burn_in={burn_in} dt={float(dt[burn_in]):.2f} "
          f"alpha={args.alpha} rho={args.rho} delta={delta:.2f}")
    print(f"{'residual':24s} {'recorded':>10s} {'cfc-sub':>10s}")
    for name in recorded:
        print(f"{name:24s} {float(recorded[name]):10.4f} {float(model_res[name]):10.4f}")


if __name__ == "__main__":
    main()
