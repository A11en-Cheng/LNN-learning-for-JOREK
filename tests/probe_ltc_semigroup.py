"""Acceptance test for the propagator: identity + semigroup, untrained model.

Protocol (the one requested for the LTC rewrite):

  1. integrate the hidden state from ``t = 0`` to ``t = 1`` -> ``h1``;
  2. integrate ``0 -> 0.3``, feed the state back, integrate ``0.3 -> 1`` -> ``h2``;
  3. ``||h1 - h2||`` must be at floating-point level, not "21% of an update".

Everything runs on a **randomly initialised** model - identity and semigroup are
properties of the parameterisation, not of the fit.  Three levels are reported:

  * ``ltc`` (default: coefficients from the cell input only): the composition is
    algebraically exact, so the defect is pure floating point;
  * ``ltc-state-dependent`` (``--state-dependent``): ``rate``/``target`` also see
    ``h``, so a split re-evaluates them at the intermediate state and the defect
    grows to ``O(ts^2)`` - reported for reference so the trade-off is explicit;
  * ``ncps`` (legacy): gives the historical 21%-of-an-update behaviour.

The augmented source accumulator is checked as well: with a constant decoded
source the trapezoid integral must equal ``S * dt`` exactly, and the integral must
be additive over a split.

Example:

    ./.torch-cuda-env/bin/python tests/probe_ltc_semigroup.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.cfc_interp import CfCInterpolator
from src.models.lnn_processor import JorekLNNProcessor, JorekLTCProcessor
from src.models.ltc_cell import LTCCell


def cell_level(input_dim, hidden, backbone, state_dependent, dt, dtype=torch.float32):
    torch.manual_seed(0)
    cell = LTCCell(input_dim, hidden, backbone_units=backbone,
                   state_dependent=state_dependent).to(dtype).eval()
    x = torch.randn(64, input_dim, dtype=dtype)
    h0 = torch.randn(64, hidden, dtype=dtype)
    with torch.no_grad():
        identity = cell(x, h0, torch.tensor(0.0, dtype=dtype))
        one = cell(x, h0, torch.tensor(dt, dtype=dtype))
        mid = cell(x, h0, torch.tensor(0.3 * dt, dtype=dtype))
        two = cell(x, mid, torch.tensor(0.7 * dt, dtype=dtype))
    update = float((one - h0).norm())
    return {
        "identity": float((identity - h0).norm()),
        "semigroup": float((one - two).norm()),
        "update": update,
    }


def model_level(cell, state_dependent, args, device):
    torch.manual_seed(0)
    model = CfCInterpolator(
        latent_dim=args.latent, hidden_size=args.hidden, dt_scale=args.dt_scale,
        cell=cell, backbone_units=args.backbone, backbone_layers=args.backbone_layers,
        state_dependent=state_dependent, init_decay=args.init_decay,
    ).to(device).eval()
    latent = torch.randn(args.batch, args.latent, device=device)

    def initial_state():
        zeros = torch.zeros(args.batch, args.hidden, device=device)
        # the legacy ncps processor carries an LSTM (h, c) pair
        return (zeros, zeros.clone()) if cell == "ncps" else zeros

    def head(state):
        return state[0] if isinstance(state, tuple) else state

    h0 = initial_state()
    with torch.no_grad():
        def step(h, dt):
            return model.advance(latent, h, dt)[1]

        identity = step(h0, 0.0)
        one = step(h0, args.dt_scale)                      # ts = 1.0
        mid = step(h0, 0.3 * args.dt_scale)
        two = step(mid, 0.7 * args.dt_scale)
        chained = h0
        for _ in range(10):
            chained = step(chained, 0.1 * args.dt_scale)
    one, two, chained, identity = head(one), head(two), head(chained), head(identity)
    reference = head(h0)
    update = float((one - reference).norm())
    return {
        "identity": float((identity - reference).norm()),
        "semigroup": float((one - two).norm()),
        "ten_step": float((one - chained).norm()),
        "update": update,
    }


def source_integral_check(args, device):
    """Constant decoded source -> trapezoid integral == S*dt, and it is additive."""
    torch.manual_seed(0)
    model = CfCInterpolator(
        latent_dim=args.latent, hidden_size=args.hidden, dt_scale=args.dt_scale,
        cell="ltc", backbone_units=args.backbone, backbone_layers=args.backbone_layers,
        predict_source=True, init_decay=args.init_decay,
    ).to(device).eval()
    # freeze the source head on a constant output
    for parameter in model.source_head.parameters():
        parameter.data.zero_()
    model.source_head[-1].bias.data.fill_(0.25)
    state = torch.randn(args.batch, 9, device=device)
    dt = 2.0
    with torch.no_grad():
        out = model.advance_with_source_integral(state, None, dt, num_substeps=5)
        integral = out["source_integral"]
        half = model.advance_with_source_integral(
            state, None, 0.4 * dt, num_substeps=3)["source_integral"]
        rest = model.advance_with_source_integral(
            state, out["hidden"], 0.6 * dt, num_substeps=3)["source_integral"]
    exact = 0.25 * dt
    return {
        "constant": float((integral - exact).abs().max()),
        "additivity": float((half + rest - integral).abs().max()),
        "relative": float((half + rest - integral).abs().max() / max(abs(exact), 1e-12)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--backbone", type=int, default=256)
    parser.add_argument("--backbone-layers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dt-scale", type=float, default=50.0)
    parser.add_argument("--init-decay", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    print("cell level (random init, ts = 1.0):")
    print(f"{'variant':24s} {'||Phi(0)h - h||':>16s} {'||h1 - h2||':>14s} "
          f"{'/||update||':>12s}")
    for label, state_dependent in (("ltc (input-only)", False), ("ltc (state-dep)", True)):
        row = cell_level(args.latent, args.hidden, args.backbone, state_dependent, 1.0)
        print(f"{label:24s} {row['identity']:16.3e} {row['semigroup']:14.3e} "
              f"{row['semigroup'] / max(row['update'], 1e-12):12.3e}")
    row = cell_level(args.latent, args.hidden, args.backbone, False, 1.0, dtype=torch.float64)
    print(f"{'ltc float64':24s} {row['identity']:16.3e} {row['semigroup']:14.3e} "
          f"{row['semigroup'] / max(row['update'], 1e-12):12.3e}")

    print("\nmodel level (CfCInterpolator.advance, dt = dt_scale so ts = 1.0):")
    print(f"{'variant':24s} {'||Phi(0)h - h||':>16s} {'||h1 - h2||':>14s} "
          f"{'||h1 - h10||':>14s} {'/||update||':>12s}")
    variants = (
        ("ltc (input-only)", "ltc", False),
        ("ltc (state-dep)", "ltc", True),
        ("ncps (legacy)", "ncps", False),
    )
    for label, cell, state_dependent in variants:
        row = model_level(cell, state_dependent, args, device)
        print(f"{label:24s} {row['identity']:16.3e} {row['semigroup']:14.3e} "
              f"{row['ten_step']:14.3e} {row['semigroup'] / max(row['update'], 1e-12):12.3e}")

    print("\naugmented source accumulator (constant S = 0.25, dt = 2.0):")
    row = source_integral_check(args, device)
    print(f"  |∫S - S·dt| = {row['constant']:.3e}   "
          f"|∫(0→0.4dt) + ∫(0.4dt→dt) - ∫(0→dt)| = {row['additivity']:.3e} "
          f"(relative {row['relative']:.3e})")


if __name__ == "__main__":
    main()
