# Implementation progress log

## 2026-09-07 — Phase 0 closed and the single-sample overfit gate passed

- Data layer (#0001-#0005, #0002): EHL-2 2D fixtures and split manifest;
  protocol v2 with the five-field marker-projection channel plus the
  three-field background channel; machine-checked parameter-surface audit;
  the zero-dim parameter channel (h5 → namelist → JOREK defaults, with
  provenance).  The JOREK-side export ticket was superseded by the namelist
  channel (ADR-0007).
- Time dynamics (#0006): per-step physical dt (`step_dt_seconds`), multi-horizon
  skip sampling, stateful LNN with BPTT boundaries, per-step dt in the physics
  loss.
- Gate (#0007): real single-sample overfit passes — loss drop 99.993%, no NaN,
  reload difference 4.14e-7; 54 tests green.
- Planned (#0014): ADAS coronal closure prototype validated against recorded
  projections, feeding the three-mode forced rollout in #0012.

## 2026-09-07 (later) — conditioning wired, rollout gate still failing for a sharp reason

- #0011: condition encoder at the encoder input; the zero-dim channel grew to
  44 entries with scenario discriminators (n_spi, pellet_density, gamma_*,
  particlesource, ellip, bc_natural_open); ablation shows distinguishable
  responses (single-entry ~0.01-0.02, cross-run 0.04 mean |Δlatent|).
- #0009 retry (v2, 1000 steps, 6 runs, conditioned): still fails on both
  held-out runs (15-25% worse than persistence, drift 0.5).  One-step
  diagnostic: predicted |Δ| is ~56x the true single-step increment on a
  quiescent frame — the residual magnitude ignores the local state, pointing at
  under-learned state dependence and phase-skewed training windows.  Follow-up:
  revisit the delta convention (rate × dt) and rebalance window sampling.

## 2026-09-07 (end of day) — residual rate + dt_ref fix the magnitude bias, gate still open

- Residual contract changed to a rate with `x_{t+1} = x_t + rate*dt/dt_ref`;
  dt_ref is the mean training frame step stored in the checkpoint.  This fixes
  the vanishing-gradient problem caused by dt ~ 4e-6 s.
- v4 baseline (1000 steps, 6 runs, conditioned): short-horizon drift is back
  inside the registered threshold (0.0002 vs 0.5), but per-field RMSE is still
  3-39% above persistence with no improving horizon; a 100-step horizon
  re-accumulates drift (~9).  The overfit gate re-passes under the new contract
  (99.998% loss drop, reload 5.1e-7).
- Open decision: which lever to pull next for the rollout gate (longer
  training + schedule, horizon-1 training, per-field weights, longer temporal
  context, phase-balanced windows).

## 2026-09-05 — Model 502 schema and source-interface foundation

Scope authorised by the user: validate the existing project, define the
nine-field schema/protocol, and prepare the codebase for new Model 502 source
data.  No training run or external data download was performed.

### Completed

- Audited `redundant/jorek/models/model502/mod_model_settings.f90`: Model 502
  has `n_var=9` and one-based variables `psi,u,zj,w,rho,Ti,Vpar,rhoimp,Te`.
- Audited `redundant/jorek/models/model502/mod_sources.f90`: the first forcing
  contract is `particle_source`, `heat_i_source`, and `heat_e_source`.
- Added `src/data/model502_schema.py` as the canonical field order, HDF5
  reordering, transforms, source contract, condition names, and shape guards.
- Updated the HDF5 parser and sequence dataset to expose explicit source
  availability, reject partial source data, retain namelist availability masks,
  and optionally require sources for every frame.
- Changed encoder/decoder defaults to the mandatory nine-field Model 502
  interface. Decoder output is now a nine-field one-step residual.
- Added an optional, explicitly normalised three-control injection point to
  the LNN processor. Raw HDF5 sources are not automatically fed to the model.
- Corrected active loss/benchmark field indices from legacy five-field indices
  to the canonical Model 502 indices. The mass diagnostic now subtracts raw
  `S_rho` when it is present; the result remains a reduced diagnostic.
- Moved default configuration from incompatible BEST/Model 600 data to the
  local ITER Model 502 directory and disabled incompatible legacy checkpoints.
- Declared missing runtime/test dependencies in `requirements.txt`, removed the
  accidental syntax-invalid editor scratch file, and fixed `.gitignore` so its
  dataset rule no longer hides `src/data/` source code from version control.

### Validation status

- Static validation passed: `python -m compileall -q src tests` and
  `git diff --check` both pass.
- Created a clean local Python 3.14 validation environment and installed the
  declared dependencies. The checked-in `venv` remains unusable because it
  points to a missing Python 3.11 executable; it was not modified.
- `pytest` passed: 12 Model 502 schema/source/interface and migrated smoke
  tests. This includes real HDF5 parser loading, encoder/LNN/decoder forward
  and backward through the training smoke test, source-contract fixtures, and
  checkpoint reload equivalence.
- `tests/verify_data_integrity.py` passed on all 104 local HDF5 files: no read
  errors, NaN or Inf values, and no node-count warning.
- Real EHL-2 Model 502 frames parse as `[4125, 9]` with valid time and geometry.
  They correctly report `source_available=false` and are rejected when
  `require_source=true`.

### Required next verification

1. When source data arrives, use `require_source=True` and add a fixture proving
   all three source fields, units and time alignment.
2. Create a train/validation/test sequence manifest and compute state/source
   statistics only from the training split.
3. Run a fixed-seed single-sample overfit and short free-run rollout before
   enabling the source loss or any MoE work.

### Deferred deliberately

- Full Model 502 nine-equation PDE loss and charge-state-aware pressure closure.
- Ion/electron source energy residuals, SPI/MGI/pellet/radiation/ionisation
  extensions, and automatic source normalisation.
- MoE, gradient firewall, checkpointing, and 3D work.
