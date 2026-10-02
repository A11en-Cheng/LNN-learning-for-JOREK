"""Phase-3 field interpolator: CfC latent dynamics driven by recorded fields.

The model is deliberately small: a per-node encoder, the CfC processor
(:class:`~src.models.lnn_processor.JorekLNNProcessor`) and a per-node decoder.
Continuous-time integration is done through :meth:`step` (single interval) and
:meth:`multi_substep` (random ``S``-substep interpolation of one recorded
interval, ADR-0013): the burn-in warms the hidden state on recorded frames
without loss, then each supervised interval is split into ``S`` sub-intervals
``f_i * dt`` (``sum_i f_i = 1``, sampled per window) whose composition must
reproduce the next recorded frame.  ``two_substep`` is the historical ``S = 2``
special case.

Propagator (2026-10-01, pitfalls #43/#45): the default ``cell="ltc"`` is
:class:`~src.models.ltc_cell.LTCCell`, which predicts a positive ``rate``
(softplus) and an equilibrium ``target`` (tanh) and applies the *exact*
solution of ``dh/dt = -rate * (h - target)``.  ``ts = 0`` is therefore the
identity and splitting an interval is a no-op up to floating point, which is
what makes a physics residual at an interpolated time ``t*`` meaningful.
``cell="ncps"`` (the installed CfC cell, plus its LSTM mixed-memory shortcut)
remains only to re-load the checkpoints trained before that date.

``predict_source=True`` adds the augmented state of the source accumulator:
:meth:`advance_with_source_integral` propagates the interval in ``num_substeps``
equal pieces and accumulates ``A = ∫ S dτ`` with the trapezoid rule, so the
interval *mean* ``A / dt`` can be compared with JOREK's recorded projection.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _checkpoint

from src.data.model502_schema import NUM_MARKER_PROJECTION_FIELDS, NUM_STATE_FIELDS
from src.data.model502_schema import destandardize, inverse_transform_state
from src.models.lnn_processor import JorekLNNProcessor, JorekLTCProcessor
from src.physics.grad_ops import apply_gradient_operator


def make_deriv_fn(
    operator,
    field_mean,
    field_std,
    deriv_mean,
    deriv_std,
    *,
    clamp: float | None = 5.0,
    detach: bool = True,
):
    """Closure: normalised field -> standardised physical-gradient features.

    The sparse gradient operator acts on the *physical* field, so the model
    output is de-standardised first; the resulting gradient is then standardised
    with the train-split statistics (pitfalls #22).
    """

    def deriv_fn(u_norm: torch.Tensor) -> torch.Tensor:
        raw = inverse_transform_state(
            destandardize(u_norm, field_mean, field_std, name="u")
        )
        grad = apply_gradient_operator(operator, raw.float())
        flat = grad.reshape(grad.shape[0], -1)
        flat = (flat - deriv_mean) / deriv_std
        # Model-generated gradients are ~350x the recorded ones early in
        # training (pitfalls #24); clamp hard and keep them out of the graph.
        if clamp is not None:
            flat = flat.clamp(-clamp, clamp)
        if detach:
            flat = flat.detach()
        return flat

    return deriv_fn


def sample_split_fractions(
    batch: int,
    sub_steps: int,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
    mode: str = "random",
    min_fraction: float = 0.0,
) -> torch.Tensor:
    """Per-window sub-interval fractions ``[batch, sub_steps]`` summing to 1.

    ``random`` draws ``sub_steps - 1`` sorted knots on ``(0, 1)`` and takes the
    spacings (the historical two-substep sampler is the ``sub_steps = 2`` case,
    where the first fraction is ``U(0, 1)``); ``equal`` returns ``1 / sub_steps``
    everywhere and is what evaluation uses so the metric stays deterministic.
    """
    sub_steps = int(sub_steps)
    if sub_steps < 1:
        raise ValueError(f"sub_steps must be >= 1, got {sub_steps}")
    if mode not in ("random", "equal"):
        raise ValueError(f"mode must be 'random' or 'equal', got {mode!r}")
    batch = int(batch)
    if sub_steps == 1:
        return torch.ones((batch, 1), device=device, dtype=dtype)
    if mode == "equal":
        return torch.full((batch, sub_steps), 1.0 / sub_steps, device=device, dtype=dtype)
    knots = torch.sort(
        torch.rand(batch, sub_steps - 1, device=device, dtype=dtype), dim=1
    ).values
    edges = torch.cat(
        [
            torch.zeros(batch, 1, device=device, dtype=dtype),
            knots,
            torch.ones(batch, 1, device=device, dtype=dtype),
        ],
        dim=1,
    )
    fractions = torch.diff(edges, dim=1)
    if min_fraction > 0.0:
        # Guard against degenerate near-zero sub-intervals (optional; off by
        # default so the historical two-substep distribution is unchanged).
        fractions = fractions.clamp_min(float(min_fraction))
        fractions = fractions / fractions.sum(dim=1, keepdim=True)
    else:
        # Sorted knots can coincide (duplicate spacings of exactly 0); a zero
        # sub-interval is not a no-op for the CfC cell, so keep them positive.
        fractions = fractions.clamp_min(torch.finfo(fractions.dtype).tiny)
    return fractions


def _as_fractions(fractions, reference: torch.Tensor) -> torch.Tensor:
    """Normalise ``fractions`` to a positive ``[S]`` or ``[B, S]`` tensor."""
    fracs = torch.as_tensor(fractions, dtype=reference.dtype, device=reference.device)
    if fracs.dim() == 0:
        fracs = fracs.reshape(1)
    if fracs.dim() > 2:
        raise ValueError(f"fractions must be 0-d/1-d/2-d, got shape {tuple(fracs.shape)}")
    if fracs.shape[-1] < 1:
        raise ValueError("fractions must contain at least one sub-interval")
    if not bool((fracs > 0).all()):
        raise ValueError("sub-interval fractions must be strictly positive")
    total = fracs.sum(dim=-1, keepdim=True)
    if not bool(torch.allclose(total, torch.ones_like(total), atol=1e-5)):
        raise ValueError(f"sub-interval fractions must sum to 1, got {total.reshape(-1).tolist()}")
    return fracs


def _expand_dt(dt, nodes: int, batch_index, dtype, device) -> torch.Tensor:
    """Per-node interval length ``[nodes]`` from a scalar or per-window tensor."""
    tensor = torch.as_tensor(dt, dtype=dtype, device=device)
    if tensor.numel() == 1:
        return tensor.reshape(1).expand(nodes)
    if tensor.numel() == nodes:
        return tensor.reshape(-1)
    if batch_index is not None and tensor.numel() == int(batch_index.max()) + 1:
        return tensor.reshape(-1)[batch_index]
    raise ValueError(
        f"cannot expand dt of shape {tuple(tensor.shape)} to {nodes} nodes"
    )


def _dt_feature(dt_nodes: torch.Tensor, dt_scale: float) -> torch.Tensor:
    """Bounded, monotone encoding of the interval length: ``log1p(dt)/log1p(scale)``.

    The recorded cadence spans 0.5..250 JOREK units (pitfalls #48), so a raw
    linear feature would be dominated by the coarse tail; this keeps the range
    ~0.1..1.4 and is exactly 1.0 at the nominal ``dt_scale``.
    """
    scaled = torch.log1p(dt_nodes.clamp_min(0.0)) / math.log1p(max(dt_scale, 1.0))
    return scaled.reshape(-1, 1)


def deterministic_fractions(sub_steps: int, alpha: float | None = None) -> list[float]:
    """Deterministic sub-interval split used by every evaluation path.

    ``sub_steps == 2`` keeps the historical ``[alpha, 1 - alpha]`` bracket (so
    recorded ``alpha = 0.5`` readings stay comparable).  Longer chains use equal
    sub-intervals; when ``alpha`` is given it becomes the weight of the *last*
    sub-interval, which lets the self-consistency gate vary the split without
    touching the total interval.
    """
    sub_steps = int(sub_steps)
    if sub_steps <= 1:
        return [1.0]
    if sub_steps == 2:
        a = 0.5 if alpha is None else float(alpha)
        return [a, 1.0 - a]
    if alpha is None:
        return [1.0 / sub_steps] * sub_steps
    share = (1.0 - float(alpha)) / (sub_steps - 1)
    return [share] * (sub_steps - 1) + [float(alpha)]


def model_kwargs_from_meta(meta: dict | None, **overrides) -> dict:
    """Constructor kwargs recovered from a checkpoint's stored ``meta['args']``.

    A checkpoint's architecture is part of the checkpoint.  Building the model
    from CLI defaults instead is a recurring source of silent mismatch: after
    the 2026-10-01 propagator rewrite every analysis script that hard-coded
    ``CfCInterpolator(...)`` would have constructed an LTC model and failed (or
    worse, silently loaded the wrong head shapes) on a pre-rewrite ncps
    checkpoint.  Anything the caller passes in ``overrides`` wins.
    """
    args = (meta or {}).get("args", {}) or {}
    kwargs = dict(
        latent_dim=int(args.get("latent", 32)),
        hidden_size=int(args.get("hidden", 64)),
        dt_scale=float(args.get("dt_scale", 50.0)),
        input_deriv=bool(
            args.get("input_deriv") or args.get("input_deriv2") or args.get("feedback_deriv")
        ),
        input_deriv2=bool(args.get("input_deriv2", False)),
        input_dt=bool(args.get("input_dt", False)),
        feedback_deriv=bool(args.get("feedback_deriv", False)),
        residual=bool(args.get("residual", False)),
        # checkpoints written before the rewrite carry no 'cell' key and were
        # all trained with the installed ncps cell
        cell=str(args.get("cell", "ncps")),
        backbone_units=int(args.get("backbone", 256)),
        backbone_layers=int(args.get("backbone_layers", 2)),
        state_dependent=bool(args.get("state_dependent", False)),
        target_activation=str(args.get("target_activation", "tanh")),
        init_decay=float(args.get("init_decay", 0.5)),
        predict_source=bool(args.get("predict_source", False)),
    )
    kwargs.update({key: value for key, value in overrides.items() if value is not None})
    return kwargs


class CfCInterpolator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_size: int = 64,
        dt_scale: float = 50.0,
        input_deriv: bool = False,
        input_deriv2: bool = False,
        input_dt: bool = False,
        feedback_deriv: bool = False,
        deriv_fn=None,
        residual: bool = False,
        cell: str = "ltc",
        backbone_units: int = 256,
        backbone_layers: int = 2,
        state_dependent: bool = False,
        target_activation: str = "tanh",
        init_decay: float = 0.5,
        predict_source: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt_scale = float(dt_scale)
        self.input_deriv = bool(input_deriv)
        # ``input_deriv2`` appends the graph-LSQ Laplacian of every field, i.e.
        # the second-order operator the PDEs need (verified against the recorded
        # Z_j/omega channels at corr 0.987-0.997; see pitfalls #21 revision).
        self.input_deriv2 = bool(input_deriv2)
        if self.input_deriv2:
            self.input_deriv = True
        # ``input_dt`` appends log1p(dt)/log1p(dt_scale) to the encoder input.
        # Without it the network only meets the interval length inside the cell
        # (``exp(-rate*ts)``), so the encoder cannot modulate its own features for
        # a 0.5-unit step vs a 250-unit one (pitfalls #48).  dt is a per-interval
        # constant, so this does not affect the semigroup property.
        self.input_dt = bool(input_dt)
        self.feedback_deriv = bool(feedback_deriv)
        self.deriv_fn = deriv_fn
        # ``cell="ltc"`` (default) is the closed-form LTC propagator: the cell
        # outputs (rate, target) and the state is advanced with the exact
        # solution of dh/dt = -rate*(h - target), so ts=0 is the identity and
        # splitting an interval is (to floating point) a no-op -- required
        # before a physics residual can be placed at an interpolated time t*.
        # ``cell="ncps"`` keeps the installed CfC cell for re-loading the
        # checkpoints trained before 2026-10-01 (pitfalls #43/#45).
        self.cell = str(cell)
        if self.cell not in ("ncps", "ltc"):
            raise ValueError(f"cell must be 'ncps' or 'ltc', got {cell!r}")
        self.backbone_units = int(backbone_units)
        self.state_dependent = bool(state_dependent)
        # Inside one recorded interval the *input* of the propagator is the
        # encoded state at the interval start; only the hidden state is chained
        # across sub-steps.  Feeding the decoded latent back in (what the legacy
        # ncps path does) makes the coefficients depend on the sub-step index
        # and destroys the split invariance, so it is only kept for the legacy
        # cell (it is part of how those checkpoints were trained).
        self.recycle_latent = self.cell == "ncps"
        # ``residual=True`` makes the decoder output the *increment* over the
        # interval instead of the field: ``u(t+dt) = u(t) + decode(...)``.  The
        # recorded fields change by only ~1-20% per frame, so an absolute
        # parameterisation spends nearly all of its loss budget on reproducing
        # the (already known) input field and almost none on the dynamics
        # (pitfalls #29).
        self.residual = bool(residual)
        # Value-only (9) or value + first-order local derivative features (9 + 18).
        input_dim = NUM_STATE_FIELDS * (1 + 2) if self.input_deriv else NUM_STATE_FIELDS
        if self.input_deriv2:
            input_dim += NUM_STATE_FIELDS
        if self.input_dt:
            input_dim += 1
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        processor_cls = JorekLTCProcessor if self.cell == "ltc" else JorekLNNProcessor
        if self.cell == "ltc":
            self.processor = processor_cls(
                latent_dim=latent_dim, hidden_size=hidden_size,
                backbone_units=backbone_units, backbone_layers=backbone_layers,
                state_dependent=state_dependent, target_activation=target_activation,
                init_decay=init_decay,
            )
        else:
            # legacy checkpoints carry the LSTM mixed-memory weights
            self.processor = processor_cls(
                latent_dim=latent_dim, hidden_size=hidden_size, mixed_memory=True
            )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, NUM_STATE_FIELDS),
        )
        # Optional augmented-state head: the field-independent projections
        # S = [dE_ion, rad_sink, Z_eff, Z_imp, N_imp] decoded from the latent.
        # Kept behind a flag because the current main path is pure-data field
        # training with *teacher* sources; see advance_with_source_integral.
        self.predict_source = bool(predict_source)
        self.source_head = (
            nn.Sequential(
                nn.Linear(latent_dim, latent_dim),
                nn.SiLU(),
                nn.Linear(latent_dim, NUM_MARKER_PROJECTION_FIELDS),
            )
            if self.predict_source
            else None
        )

    def encode(
        self,
        state: torch.Tensor,
        deriv: torch.Tensor | None = None,
        lap: torch.Tensor | None = None,
        dt_nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.input_deriv:
            if deriv is None:
                raise ValueError("input_deriv=True requires the derivative features")
            state = torch.cat([state, deriv], dim=-1)
        if self.input_deriv2:
            if lap is None:
                raise ValueError("input_deriv2=True requires the Laplacian features")
            state = torch.cat([state, lap], dim=-1)
        if self.input_dt:
            if dt_nodes is None:
                raise ValueError("input_dt=True requires the per-node interval dt")
            state = torch.cat([state, _dt_feature(dt_nodes, self.dt_scale)], dim=-1)
        return self.encoder(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def advance(
        self,
        latent: torch.Tensor,
        hidden: tuple | torch.Tensor | None,
        dt: torch.Tensor | float,
        controls: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple | torch.Tensor]:
        """One CfC cell step with a (possibly per-node) physical ``dt``."""
        timespan = torch.as_tensor(dt, dtype=latent.dtype, device=latent.device) / self.dt_scale
        return self.processor(
            latent, batch_index, hidden, time_steps=timespan, controls=controls
        )

    def step(
        self,
        state: torch.Tensor,
        hidden: tuple | torch.Tensor | None,
        dt: torch.Tensor | float,
        controls: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        deriv: torch.Tensor | None = None,
        lap: torch.Tensor | None = None,
        sub_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple | torch.Tensor]:
        """Encode ``state``, advance one interval and decode the next field.

        ``sub_steps`` splits the interval into that many equal cell calls (used
        by the burn-in so its granularity matches the supervised interval when
        ``--sub-steps`` is raised; pitfall #39b).
        """
        input_latent = self.encode(
            state, deriv, lap,
            _expand_dt(dt, state.shape[0], batch_index, state.dtype, state.device)
            if self.input_dt else None,
        )
        latent = input_latent
        steps = max(int(sub_steps), 1)
        for _ in range(steps):
            cell_input = latent if self.recycle_latent else input_latent
            latent, hidden = self.advance(
                cell_input, hidden, dt / steps, controls, batch_index
            )
        first = (latent, hidden)
        prediction = self.decode(first[0])
        if self.residual:
            prediction = state + prediction
        return prediction, first[0], first[1]

    def advance_with_source_integral(
        self,
        state: torch.Tensor,
        hidden,
        dt,
        controls: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        deriv: torch.Tensor | None = None,
        lap: torch.Tensor | None = None,
        num_substeps: int = 5,
        fractions=None,
    ) -> dict:
        """Propagate one interval and accumulate ``A = ∫ S dτ`` (trapezoid rule).

        JOREK records the projection as an *interval* quantity (an average over
        the fluid steps inside a frame), so the augmented state that has to be
        compared with the data is the integral of the decoded source over the
        interval - not its instantaneous value.  With a genuine propagator this
        can be accumulated at high frequency inside the interval:

            h_i = Phi(h_{i-1}, sub_dt);  S_i = source_head(proj(h_i))
            A  += (S_{i-1} + S_i) / 2 * sub_dt

        Returns the decoded field at the end of the interval, the accumulated
        integral ``A`` (``[nodes, 5]``), the per-substep sources and the updated
        hidden state.  ``source_interval_mean = A / dt`` is the quantity to
        compare with the recorded projection of the *target* frame.

        Requires ``predict_source=True`` (the head is off in the pure-data field
        training, which feeds recorded teacher sources as controls instead).
        """
        if self.source_head is None:
            raise ValueError(
                "advance_with_source_integral requires predict_source=True"
            )
        num_substeps = int(num_substeps)
        if num_substeps < 1:
            raise ValueError(f"num_substeps must be >= 1, got {num_substeps}")
        steps = (
            [1.0 / num_substeps] * num_substeps
            if fractions is None
            else list(_as_fractions(fractions, state).reshape(-1).tolist())
        )
        input_latent = self.encode(
            state, deriv, lap,
            _expand_dt(dt, state.shape[0], batch_index, state.dtype, state.device)
            if self.input_dt else None,
        )
        latent = input_latent
        if hidden is None:
            h_state = torch.zeros(
                (state.shape[0], int(self.processor.hidden_size)),
                device=state.device, dtype=state.dtype,
            )
        else:
            h_state = hidden[0] if isinstance(hidden, tuple) else hidden
        project = getattr(self.processor, "proj", None)
        if project is None:
            raise ValueError("the NCPS processor has no hidden->latent projection")

        def per_node(value) -> torch.Tensor:
            tensor = torch.as_tensor(value, dtype=state.dtype, device=state.device)
            if tensor.numel() == 1:
                return tensor.reshape(())
            if batch_index is not None and tensor.numel() == int(batch_index.max()) + 1:
                return tensor.reshape(-1)[batch_index].view(-1, 1)
            raise ValueError(
                "per-node dt requires a scalar or one entry per graph; got shape "
                f"{tuple(tensor.shape)}"
            )

        source_prev = self.source_head(project(h_state))
        total = torch.zeros_like(source_prev)
        sources = []
        for index, fraction in enumerate(steps):
            sub_dt = fraction * dt
            cell_input = latent if self.recycle_latent else input_latent
            latent, h_state = self.advance(cell_input, h_state, sub_dt, controls, batch_index)
            source = self.source_head(project(h_state))
            total = total + 0.5 * (source_prev + source) * per_node(sub_dt)
            sources.append(source)
            source_prev = source
        prediction = self.decode(latent)
        if self.residual:
            prediction = state + prediction
        return {
            "prediction": prediction,
            "source_integral": total,
            "source_rates": sources,
            "hidden": h_state,
            "end_latent": latent,
            "fractions": steps,
        }

    def multi_substep(
        self,
        state: torch.Tensor,
        hidden: tuple | torch.Tensor | None,
        fractions,
        dt: torch.Tensor | float,
        controls: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        deriv: torch.Tensor | None = None,
        lap: torch.Tensor | None = None,
        checkpoint: bool = False,
    ) -> dict:
        """Random ``S``-substep interpolation over one recorded interval.

        ``fractions`` is the per-window split of the interval: a scalar, a
        length-``S`` sequence, or a ``[B, S]`` tensor whose rows sum to 1.  Each
        sub-interval advances the *same* latent/hidden chain by ``f_i * dt`` and
        the decoder is applied once, at the end of the interval.

        Returns the decoded field at the end of the interval, the latent that
        enters the *last* sub-interval (``mid_latent``: the physics term and the
        feedback path both pair it with the final state), the end latent and the
        updated hidden state.

        ``checkpoint=True`` recomputes each cell call during the backward pass
        instead of storing it, which is what makes long chains (``S`` >> 2) fit
        on a 16 GB card: one retained cell call costs ~0.7 GB at
        ``batch * nodes = 132k`` rows, so 3 intervals x 8 sub-steps would need
        ~17 GB of activations alone.
        """
        fracs = _as_fractions(fractions, state)
        input_latent = self.encode(
            state, deriv, lap,
            _expand_dt(dt, state.shape[0], batch_index, state.dtype, state.device)
            if self.input_dt else None,
        )
        latent = input_latent
        steps = fracs.shape[-1]
        mid_latent = latent
        for index in range(steps):
            if index == steps - 1:
                mid_latent = latent
            step_dt = fracs[..., index] * _as_scalar(dt)
            cell_input = latent if self.recycle_latent else input_latent
            if checkpoint and torch.is_grad_enabled():
                latent, hidden = _checkpoint(
                    self.advance, cell_input, hidden, step_dt, controls, batch_index,
                    use_reentrant=False,
                )
            else:
                latent, hidden = self.advance(
                    cell_input, hidden, step_dt, controls, batch_index
                )
            if self.feedback_deriv and self.deriv_fn is not None and index < steps - 1:
                # Curriculum: probability of using the model's own derivative
                # features instead of the teacher's, annealed by the trainer.
                u_mid = self.decode(latent)
                if self.residual:
                    u_mid = state + u_mid
                probability = float(getattr(self, "feedback_prob", 1.0))
                use_model = probability >= 1.0 or (
                    probability > 0.0
                    and torch.rand((), device=u_mid.device).item() < probability
                )
                next_deriv = self.deriv_fn(u_mid) if use_model else deriv
                latent = self.encode(u_mid, next_deriv)
        prediction = self.decode(latent)
        if self.residual:
            prediction = state + prediction
        return {
            "prediction": prediction,
            "mid_latent": mid_latent,
            "end_latent": latent,
            "hidden": hidden,
        }

    def two_substep(
        self,
        state: torch.Tensor,
        hidden: tuple | torch.Tensor | None,
        alpha: float,
        dt: torch.Tensor | float,
        controls: torch.Tensor | None = None,
        batch_index: torch.Tensor | None = None,
        deriv: torch.Tensor | None = None,
        lap: torch.Tensor | None = None,
    ) -> dict:
        """Historical ``S = 2`` split ``[alpha, 1 - alpha]`` (ADR-0013)."""
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        return self.multi_substep(
            state, hidden, [float(alpha), 1.0 - float(alpha)], dt, controls, batch_index, deriv, lap
        )


def _as_scalar(dt: torch.Tensor | float) -> torch.Tensor | float:
    if isinstance(dt, torch.Tensor) and dt.numel() == 1:
        return float(dt.detach())
    return dt


__all__ = [
    "CfCInterpolator",
    "deterministic_fractions",
    "make_deriv_fn",
    "model_kwargs_from_meta",
    "sample_split_fractions",
]
