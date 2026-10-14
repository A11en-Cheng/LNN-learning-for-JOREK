# Native hybrid closure: primary local code evidence (2026-10-11)
Scope: read AGENTS.md and docs/adr/README.md; inspected local implementation only.
No web, papers, training, optimizer, GPU execution, or tests were used.
Line references describe the working-tree source inspected today, not a certified build.
ADR-0001–0017 are historical; no historical ADR is used as an acceptance gate here.

## Qualified answer
A field-only CfC candidate does not explicitly evolve the causal shard/marker state.
That limits an autonomous pellet/source closure claim; it does not establish that
CfC intrinsically cannot represent a driven system or that split coupling must fail.
The current coupled path supplies source features before stencil and temporal updates,
and its complete event driver evolves and saves physical source/shard state.
Causal numerical closure, continuous physical accuracy, and differentiability are
different properties; the code supports the first within its declared contracts,
does not prove the second, and deliberately limits the third.

## Field-only versus source-conditioned field core
- `src/models/native_torch_rollout.py:12`: field state contains U, raw deltas,
  memory, units, time, previous dt, scheme and configuration; no shard/marker cloud.
- `src/models/native_torch_rollout.py:55`: `advance_field` calls the field candidate
  and updates only that field state; its docstring explicitly excludes shard events.
- `src/models/native_coupling.py:825`: joint learning is the default coupled mode;
  frozen99 is a separately selected reference response, not the joint implementation.
- `src/models/native_coupling.py:847`: joint mode constructs `JointSourceEncoding`.
- `src/models/native_coupling.py:875`: trusted field-event context validates source
  time against the current interval; `:883` encodes feedback, projection history,
  independent shard density and the current/observed field; `:885` passes the encoding.
- `src/models/native_source_features.py:155`: source identity includes five feedback
  rows, five projection-delta rows and two shard-density rows, plus Gauss closure
  features and clock metadata; `:123` explicitly identifies pooling as lossy.
- `src/models/native_field_step.py:219`: encoding is added to the latent features;
  `:225` then runs stencil layers and `:229` then runs the temporal core.
- `src/models/native_temporal.py:37`: the CfC adapter receives those features,
  explicit memory and dt. This is source-conditioned CfC when that core is installed.
- `src/models/native_source_features.py:147`: joint response semantics are learned;
  these features are not an assembled physical increment or an exact MHD solve.

Thus “current implementation is field-only and never lets sources enter CfC” is false
for joint mode. Field-only remains a distinct, narrower execution path.

## Complete event order, clocks and saved augmented state
- `src/training/base_coupled.py:192`: `_advance` validates the control and clones
  runtime; `:198` computes independent density from the currently saved shards.
- `src/models/marker_state.py:787`: ordinary events first advance the particle/source
  window to the already committed fluid time using previous/current field endpoints.
- `src/models/native_coupling.py:1003`: births and source advance share a sampler
  constructed from current fields, saved raw fluid deltas and the completed dt.
- `src/training/base_coupled.py:216`: registered startup instead executes the first
  fluid/shard event with its registered zero-projection state before particle sampling.
- `src/models/marker_state.py:764`: solve consumes installed source and field history;
  `:771` updates shards in OLD fields; `:772` then commits new fields/deltas/memory.
- `src/models/marker_state.py:750`: startup uses marker sim-time as the fluid base;
  ordinary steps use committed fluid time. `:776` advances time by dt, records dt,
  increments fluid index/plan cursor and updates the available field endpoints.
- `src/training/base_coupled.py:224`: diagnostics refresh after commit;
  `src/models/native_coupling.py:1040` binds them to new field values and event time.
- `src/models/marker_state.py:563`: runtime carries both field endpoints, shards,
  marker state, clocks, step plan/cursor, raw fluid history and explicit field memory.
- `src/models/marker_state.py:133`: marker physical time is sim-time minus residual;
  its fixed particle dt is separate from the fluid timestep.
- `src/models/marker_state.py:400`: installs new feedback and new-minus-old history;
  `:402` advances marker sim-time, residual and window index after projection.
- `src/training/base_coupled.py:123`: Torch state retains this complete runtime plus
  field/history/memory tensors. `:323` saves detached complete session, model, control
  manifest, gradient scope and RNG; `:340` checks fingerprints and continuation.
- `src/models/marker_state.py:454`: inference saver likewise stores the entire
  augmented runtime, weights, configuration signature and RNG, not just field U.

This is an explicit causal hybrid transition, conditional on complete initial state,
configured inputs and valid controls. It does not need an unavailable future field
to generate the next source window; its lag and event order are part of the scheme.

## Gradient boundary and direct macrostep limitation
`src/training/base_coupled.py:208` uses live field tensors for the learned solve;
`:213` detaches its output into NumPy mirrors for physical runtime continuation.
`:226` retains live output tensors for subsequent field/history/memory computation.
Source/shard dependence on predicted fields is therefore present in forward values,
but absent from backpropagation across their NumPy recomputation (`:132–138`).
Torch source encoding still has differentiable learned weights and field-dependent
consumer features (`src/models/native_source_features.py:165`); this is not full
end-to-end differentiation of births, transport, ablation or event diagnostics.

One `CoupledNativeField.forward` call consumes supplied source/history; it does not
run particle windows, update shards or refresh diagnostics (`native_coupling.py:871`).
An arbitrarily large dt through that call, or `advance_field`, does not reproduce a
sequence of complete base events. `run_sequence` advances every declared control
(`src/training/base_coupled.py:281`); replacing them by one call requires separate
accuracy/convergence evidence. CfC dt support alone supplies no such guarantee.

## JOREK cross-check and version/production qualification
Fortran paths below are under `/home/ac_desktop/JOREK_code/jorek/` (read-only inspection).
`core/mod_jorek_timestepping.f90:393` solves; `:418` updates SPI shards before
`:424` commits node values, `:425` deltas and `:426` time. `:446` refreshes equilibrium,
`:448` energy and `:468` recorded time: solve → OLD-field shard → commit → diagnostics.
`models/pellet_module.f90:276` moves shards; `:307` erodes radius using saved ablation;
`:329` locates the advanced position and `:360` interpolates node fields before the
caller commits them. `:632` evaluates a Parks-model ablation branch using these inputs.
This confirms the inspected split order, not universal splitting stability/accuracy.

`src/models/native_coupling.py:930` declares small-shard-v1 reductions explicitly.
`:467–478` separately versions marker-source approximation, raw-delta sampling and budget.
`src/models/native_source_production.py:23` declares inspected-Ne mechanisms-v1,
NumPy stop-gradient and `production_build='not_certified'`; `:26` requires external ES.
Those implementation versions do not identify the executable that produced data.
Read-only `git -C /home/ac_desktop/JOREK_code/jorek rev-parse --show-toplevel HEAD`
verified root `/home/ac_desktop/JOREK_code/jorek` and HEAD
`54c675bcc39c33b082c0c35046e28122dc53b3aa`. No executable hash, build macros or
production provenance was verified. Source/build equivalence remains
unestablished, including macro-selected TiTe/impurity branches (`pellet_module.f90:358`).
An external causal input/provider can define valid driven dynamics; autonomy requires
that its origin and timing be accounted for. The complete hybrid interface is evidence
of numerical closure under those inputs, not certification of continuous physics.
