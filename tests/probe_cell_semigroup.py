"""Structural test: which cell is (approximately) a time propagator?

The roadmap puts the physics loss at *interpolated* times ``t*`` (ADR-0010).
That only means something if the latent integrator is a time propagator, i.e.

    Phi(t1 + t2) ~= Phi(t2) o Phi(t1)

for the *same* cell parameters.  This script measures the semigroup defect of
the candidate cells on random weights (no training involved - it is a property
of the parameterisation, not of the fit):

  * ``ncps``  - the installed ``ncps.torch.CfCCell`` in ``mode="default"``:
    ``h_new = ff1 * (1 - t) + t * ff2`` with ``t = sigmoid(t_a * ts + t_b)``.
    ``ts -> 0`` does **not** give the identity, so splitting an interval is not
    a refinement (pitfall #43).
  * ``ltc``   - the closed-form LTC/CfC propagator **with the initial-state
    term**: ``h(t) = (h0 - target) * exp(-rate * t) + target`` where
    ``rate = |w_tau| + |tanh(ff1(z))|`` and ``target = tanh(ff2(z))``.
    ``ts -> 0`` is exactly the identity and the map is exactly a semigroup
    whenever ``rate``/``target`` are constant in time (they depend on ``h``, so
    in practice the defect is small but non-zero).

Reported per cell: the median relative defect
``||Phi(a*dt) o Phi((1-a)*dt)(h0) - Phi(dt)(h0)|| / ||Phi(dt)(h0) - h0||``
(= 1 means "the split changes the answer by as much as the update itself") and
the identity defect ``||Phi(0)(h0) - h0|| / ||h0||``.

Example:

    ./.torch-cuda-env/bin/python tests/probe_cell_semigroup.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.lnn_processor import JorekLNNProcessor, JorekLTCProcessor
from src.models.ltc_cell import LTCCell


def _ncps_cell(input_size: int, hidden_size: int):
    from ncps.torch import CfCCell

    torch.manual_seed(0)
    return CfCCell(input_size, hidden_size)


def _step_ncps(cell, x, h, ts):
    out, _ = cell.forward(x, h, torch.as_tensor(float(ts)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--latent", type=int, default=64, help="latent width of the processors")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.1, help="timespan per interval (dt/dt_scale)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch, args.input_dim)
    h0 = torch.randn(args.batch, args.hidden)
    dt = args.dt

    cells = {
        "ncps": (_ncps_cell(args.input_dim, args.hidden), _step_ncps),
        "ltc": (LTCCell(args.input_dim, args.hidden), None),
        "ltc-h": (LTCCell(args.input_dim, args.hidden, state_dependent=True), None),
    }
    print(f"dt={dt} (={dt} * dt_scale in JOREK units) batch={args.batch} hidden={args.hidden}")
    print(f"{'cell':6s} {'identity defect':>17s} {'split defect a=0.5':>20s} "
          f"{'a=0.25':>10s} {'a=0.75':>10s} {'compose 5x':>12s}")
    for name, (cell, step_fn) in cells.items():
        cell.eval()
        step = step_fn if step_fn is not None else (
            lambda c, x_, h_, ts: c(x_, h_, torch.tensor([[ts]]))
        )
        with torch.no_grad():
            zero = step(cell, x, h0, 0.0)
            identity = float((zero - h0).norm() / h0.norm())
            full = step(cell, x, h0, dt)
            update = float((full - h0).norm())
            defects = []
            for alpha in (0.25, 0.5, 0.75):
                mid = step(cell, x, h0, alpha * dt)
                split = step(cell, x, mid, (1.0 - alpha) * dt)
                defects.append(float((split - full).norm()) / max(update, 1e-12))
            chained = h0
            for _ in range(5):
                chained = step(cell, x, chained, dt / 5.0)
            compose = float((chained - full).norm()) / max(update, 1e-12)
        print(
            f"{name:6s} {identity:17.3e} {defects[1]:20.3f} {defects[0]:10.3f} "
            f"{defects[2]:10.3f} {compose:12.3f}"
        )

    # Processor level: this is what the model actually uses (the ncps wrapper
    # additionally runs an LSTM "mixed memory" shortcut before the cell, which
    # ignores the timespan entirely).
    print("\nprocessors (what the model really runs; decoded latent output)")
    print("(identity defect = ||state(ts=0) - state_before|| / ||state(dt) - state_before||)")
    print(f"{'proc':6s} {'state id. defect':>17s} {'split defect a=0.5':>20s} "
          f"{'a=0.25':>10s} {'a=0.75':>10s} {'compose 5x':>12s}")
    torch.manual_seed(args.seed)
    x_latent = torch.randn(args.batch, args.latent)
    controls = torch.randn(args.batch, 5)
    for name, cls in (("ncps", JorekLNNProcessor), ("ltc", JorekLTCProcessor)):
        torch.manual_seed(args.seed)
        processor = cls(latent_dim=args.latent, hidden_size=args.hidden).eval()

        def run(input_latent, state, ts):
            # Matching ``CfCInterpolator.multi_substep``: the interval is encoded
            # once, and only the hidden state is chained across sub-steps.
            out, new_state = processor(input_latent, None, state, time_steps=ts, controls=controls)
            return out, new_state

        with torch.no_grad():
            reference_state = torch.zeros(args.batch, args.hidden)
            _, state0 = run(x_latent, None, 0.0)
            full, state_full = run(x_latent, None, dt)
            update = float((full - x_latent).norm())
            state_update = float((state_full[0] - reference_state).norm())
            identity = float((state0[0] - reference_state).norm()) / max(state_update, 1e-12)
            defects = []
            for alpha in (0.25, 0.5, 0.75):
                _, mid_state = run(x_latent, None, alpha * dt)
                split, _ = run(x_latent, mid_state, (1.0 - alpha) * dt)
                defects.append(float((split - full).norm()) / max(update, 1e-12))
            chained, chained_state = x_latent, None
            for _ in range(5):
                chained, chained_state = run(x_latent, chained_state, dt / 5.0)
            compose = float((chained - full).norm()) / max(update, 1e-12)
        print(
            f"{name:6s} {identity:17.3e} {defects[1]:20.3f} {defects[0]:10.3f} "
            f"{defects[2]:10.3f} {compose:12.3f}"
        )


if __name__ == "__main__":
    main()
