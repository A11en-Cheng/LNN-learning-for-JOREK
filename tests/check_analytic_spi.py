"""Compare the analytic SPI port with JOREK's recorded shard ablation.

Example:

    .torch-cuda-env/bin/python tests/check_analytic_spi.py \
        --run scan_1d22_ang15_5_massloss_neon_v400 --position 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.interp_dataset import RunFrameStore
from src.models.field_sampling import knn_weights, sample_nodes
from src.physics.spi_model import (
    SHARD_ABL,
    SHARD_RADIUS,
    parks_ablation_neon,
    temperature_ev,
    particle_density_si,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/EHL-2")
    parser.add_argument("--run", default="scan_1d22_ang15_5_massloss_neon_v400")
    parser.add_argument("--position", type=int, default=5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--central-density", type=float, default=0.8)
    parser.add_argument("--f0", type=float, default=-3.308745132)
    parser.add_argument("--r-geo", type=float, default=1.11)
    args = parser.parse_args()

    store = RunFrameStore(Path(args.data_root) / args.run).load()
    if store.u is None or store.topology is None:
        raise SystemExit("frame store failed to load")
    position = min(args.position, len(store) - 1)
    u = store.u[position]
    shards = store.shard_values[position]
    alive = store.shard_alive[position]
    if shards is None or alive is None:
        raise SystemExit("frame has no shard state (pre-injection)")
    alive_idx = torch.nonzero(alive, as_tuple=False).squeeze(-1)
    if alive_idx.numel() == 0:
        raise SystemExit("no alive shards in this frame")
    alive_shards = shards[alive_idx]

    idx, weights = knn_weights(store.topology.pos, alive_shards[:, :2], k=args.k)

    def _sample(channel: int) -> torch.Tensor:
        return sample_nodes(u[:, channel : channel + 1], idx, weights).squeeze(-1)

    local_rho = _sample(4)
    local_rho_imp = _sample(7)
    local_te = _sample(8)
    ne_si = (particle_density_si(local_rho, args.central_density) -
             particle_density_si(local_rho_imp, args.central_density)).clamp_min(0.0)
    te_ev = temperature_ev(local_te, args.central_density)
    analytic = parks_ablation_neon(
        alive_shards[:, SHARD_RADIUS].clamp_min(0.0),
        te_ev,
        ne_si,
        alive_shards[:, 9],
    )
    recorded = alive_shards[:, SHARD_ABL]

    ratio = (analytic / recorded.clamp_min(1e-30)).detach().numpy()
    both = (recorded > 0) & (analytic > 0)
    log_corr = float(np.corrcoef(np.log(analytic[both].detach().numpy()),
                                 np.log(recorded[both].detach().numpy()))[0, 1]) if int(both.sum()) > 2 else float("nan")
    print(f"run={args.run} position={position} alive={int(alive_idx.numel())} n_spi={int(shards.shape[0])}")
    print(f"recorded abl [atom/s] min/med/max = {float(recorded.min()):.3e}/{float(recorded.median()):.3e}/{float(recorded.max()):.3e}")
    print(f"analytic abl [atom/s] min/med/max = {float(analytic.min()):.3e}/{float(analytic.median()):.3e}/{float(analytic.max()):.3e}")
    print(f"analytic/recorded ratio median={float(np.median(ratio)):.3f}  log-corr={log_corr:.3f}")
    print(f"within 2x: {float(np.mean((ratio > 0.5) & (ratio < 2.0))):.2%}")
    print(f"te_ev median={float(te_ev.median()):.1f}  ne_si median={float(ne_si.median()):.3e}  radius median={float(alive_shards[:, SHARD_RADIUS].median()):.3e}")


if __name__ == "__main__":
    main()
