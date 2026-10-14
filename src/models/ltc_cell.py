# NOTE (2026-10-04): reusable component in the new architecture (ADR-0018).
# The docstring's "physics residual at an interpolated time" rationale belongs
# to the v1 route; current semantics are in ADR-0018/0019 + roadmap_v2 Phase B/C.
"""Liquid time-constant (LTC) cell with the exact closed-form step.

The cell no longer predicts the next hidden state.  It predicts the two
coefficients of the LTC ordinary differential equation

    dh/dt = -rate * (h - target),        rate > 0

and applies the **analytic solution** of that equation over the requested
timespan ``ts`` (coefficients frozen inside the step):

    decay     = exp(-rate * ts)
    h(ts)     = h_old * decay + target * (1 - decay)

Structural properties (verified in ``tests/test_ltc_cell.py`` and
``tests/probe_ltc_semigroup.py``):

* ``rate`` is produced by ``softplus`` -> strictly positive -> the update is
  dissipative-by-construction (the state can never be blown up by the rate);
* **identity**: ``ts = 0`` gives ``decay = 1`` and ``h(ts) = h_old`` exactly;
* **semigroup**: ``Phi(t1 + t2) = Phi(t2) o Phi(t1)`` holds to floating-point
  precision *when the coefficients are re-evaluated identically at the split
  point*.  That requires the coefficients to depend only on the cell **input**
  (``state_dependent=False``, the default).  With ``state_dependent=True`` the
  coefficients also see ``h_old``, so a split re-evaluates them at the new
  state and the composition is only accurate to ``O(ts^2)`` - which is exactly
  the 21%-of-an-update defect the ncps cell had, just much smaller.

Why this matters for the project: the roadmap puts a physics residual at an
interpolated time ``t*``.  A residual evaluated on a state that depends on how
the interval happened to be chopped is not a physical quantity, so the
propagator must be (nearly) exact in the split sense.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_softplus(value: float) -> float:
    """``x`` such that ``softplus(x) = value`` (stable for large values)."""
    if value <= 0.0:
        raise ValueError("inverse_softplus requires a positive value")
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class LTCCell(nn.Module):
    """Closed-form leaky-integrator cell, ``rate``/``target`` from a backbone.

    ``init_decay``/``ts_reference`` set the initial time constant: the rate head
    bias is initialised so that one step of ``ts_reference`` (default 0.1, the
    typical ``dt / dt_scale`` of the v400 frames) moves the state
    ``1 - init_decay`` of the way to the target.  With the default 0.5 the cell
    starts at "half-life = one cell call"; leaving the bias at 0 would make the
    cell move only ~7% per call, which is what throttled the first LTC attempt.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        backbone_units: int = 256,
        backbone_layers: int = 2,
        state_dependent: bool = False,
        target_activation: str = "tanh",
        init_decay: float = 0.5,
        ts_reference: float = 0.1,
        rate_eps: float = 1e-6,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.backbone_units = int(backbone_units)
        self.state_dependent = bool(state_dependent)
        self.target_activation = str(target_activation)
        if self.target_activation not in ("tanh", "linear"):
            raise ValueError(
                f"target_activation must be 'tanh' or 'linear', got {target_activation!r}"
            )
        self.rate_eps = float(rate_eps)
        if not 0.0 < float(init_decay) < 1.0:
            raise ValueError("init_decay must be in (0, 1)")
        if ts_reference <= 0.0:
            raise ValueError("ts_reference must be positive")

        in_features = self.input_size + (self.hidden_size if self.state_dependent else 0)
        layers: list[nn.Module] = [nn.Linear(in_features, self.backbone_units), nn.SiLU()]
        for _ in range(max(int(backbone_layers) - 1, 0)):
            layers += [nn.Linear(self.backbone_units, self.backbone_units), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)
        self.rate_net = nn.Linear(self.backbone_units, self.hidden_size)
        self.target_net = nn.Linear(self.backbone_units, self.hidden_size)

        rate_init = -math.log(float(init_decay)) / float(ts_reference)
        nn.init.zeros_(self.rate_net.bias)
        with torch.no_grad():
            self.rate_net.bias.fill_(inverse_softplus(rate_init))

    def coefficients(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Local decay rate (positive) and equilibrium target."""
        features = torch.cat([x, h], dim=1) if self.state_dependent else x
        z = self.backbone(features)
        rate = F.softplus(self.rate_net(z)) + self.rate_eps
        raw_target = self.target_net(z)
        # ``tanh`` is the safe default (bounded state, |h| <= 1 for h0 = 0);
        # ``linear`` restores the unbounded dynamic range that the legacy cell
        # got from its LSTM shortcut, at the cost of needing a divergence check.
        target = torch.tanh(raw_target) if self.target_activation == "tanh" else raw_target
        return rate, target

    def forward(self, x: torch.Tensor, h: torch.Tensor, ts) -> torch.Tensor:
        rate, target = self.coefficients(x, h)
        steps = ts.reshape(-1, 1) if torch.is_tensor(ts) and ts.dim() == 1 else ts
        decay = torch.exp(-rate * steps)
        return h * decay + target * (1.0 - decay)


__all__ = ["LTCCell", "inverse_softplus"]
