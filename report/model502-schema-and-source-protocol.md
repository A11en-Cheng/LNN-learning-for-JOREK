# Model 502 nine-field and source protocol

Status: active contract, created 2026-09-05.

This protocol is the acceptance contract for all new Model 502 data and model
checkpoints.  It is based on `redundant/jorek/models/model502/mod_model_settings.f90`
and `mod_sources.f90`.  It does not apply to the existing BEST/Model 600 data.

## 1. State contract

Every graph state is a float32 tensor shaped `[nodes, 9]` in this exact order:

| Index | Name | HDF5 `values` index | Transform |
|---:|---|---:|---|
| 0 | `psi` | 0 | linear, Z-score |
| 1 | `Phi` (`u`) | 1 | linear, Z-score |
| 2 | `Z_j` | 2 | linear, Z-score |
| 3 | `omega` | 3 | linear, Z-score |
| 4 | `rho` | 4 | `symlog = sign(x)·log1p(|x|)`, Z-score |
| 5 | `T_i` | 5 | `symlog = sign(x)·log1p(|x|)`, Z-score |
| 6 | `v_par` | 6 | linear, Z-score |
| 7 | `rho_imp` | 7 | `symlog = sign(x)·log1p(|x|)`, Z-score |
| 8 | `T_e` | 8 | `symlog = sign(x)·log1p(|x|)`, Z-score |

Statistics are computed only on the training split after the stated transform.

> Correction 2026-09-29: the HDF5 `values` channel order is the model502
> internal order (`mod_model_settings.f90`: `var_psi=1, var_u=2, var_zj=3, ...`)
> and `mod_export_restart.f90` copies it directly; the table above now uses the
> identity mapping.  The previous psi=2 / Z_j=0 rows came from a wrong
> permutation in `src/data/model502_schema.py`, which swapped psi and Z_j.
The checkpoint must record the schema version, field names, train-split ID,
mean and standard deviation.  Negative values in positive-only channels are a
data-quality event: the transform clamps them for numerical safety, but the
loader/ingestion report must count and explain them.

The decoder returns a normalized residual **rate** `rate_x [nodes, 9]`; update
a state only as `x_{t+1} = x_t + rate_x * dt`, where `dt` is the step in
**JOREK time units** (`time_b - time_a`).  The model's internal time is JOREK
time so the rate has O(1) gradients without any reference-step statistic;
physical seconds are `dt * t_norm` and are used only at the physics boundary
(projection alignment, PDE residuals, physical reports).  A five-field
checkpoint is incompatible and must not be silently loaded.

## 2. Conditions

The optional global condition vector is the canonical flattened vector built by
`src/data/parameter_channel.py`, with a same-sized boolean `condition_mask`
marking default-filled entries.  As of protocol v2 (ADR-0007) the
condition channel is built by `src/data/parameter_channel.py` from the HDF5
scalars, the run namelist (`jorek_namelist_spitest`, falling back to
`jorek_namelist`) and the JOREK defaults in `preset_parameters.f90`.  A
missing parameter is filled with its JOREK default and carries
`provenance=default`; consumers must read provenance rather than the raw value
to detect fill-ins.  Derived quantities (`time_evol_theta/zeta`) are computed
with the same logic as JOREK.  Conditioning is not yet wired into the network,
so data must retain the condition vector and provenance until its ablation is
implemented.

## 3. Source-control contract

Protocol v2 (ADR-0005, ADR-0007) defines two separate source channels, both
controls and never learning targets.  The network's learning targets remain
the nine native state fields; all source data is labelled a control.

> **2026-10-07 correction (ADR-0020 decision 1; ADR-0005/0007 read as （历史）)**:
> in a marker run (`use_marker=.t.`, the production setting) the recorded five
> projection channels **are** the actual field source for the energy / radiation /
> charge channels (`models/model502/mod_elt_matrix.f90:989-1003,1079,1129,1728`),
> while the analytic `total_imp_source` path only supplies the impurity density
> source (`:1219,1542,1717`).  The new architecture (roadmap_v2 Phase B1) therefore
> uses the marker feedback as its B1 target / shared S object; the "controls are
> never learning targets" sentence above applies to the v1 interpolation route
> only (history).  See `report/data-contract-v3.md` §1 and
> `docs/adr/0020-marker-source-contract.md`.

### 3.1 Marker projections (primary channel)

`marker_spi_source` (use_marker mode) writes one `projections<index>.h5` per
frame with five projected controls in this order:

| Index | Name | Meaning |
|---:|---|---|
| 0 | `dE_ion` | ionisation energy source |
| 1 | `rad_sink` | radiation energy sink |
| 2 | `Z_eff` | effective charge |
| 3 | `Z_imp` | impurity charge |
| 4 | `N_imp` | impurity density |

The parser loads the sibling `projections<index>.h5` matched by frame index and
exposes `Data.source [nodes, 5]`.  Alignment is checked on three axes: node
count, harmonic dimension (=1 for Phase 1 2D runs), and time.  `t_now` is
stored in **physical seconds** in projection files but in **JOREK units** in
state files; the conversion is `t_seconds = t_now * t_norm` using the state
frame's `t_norm` dataset (verified on real EHL-2 data).  A frame without a
matching projection file has `source_available=false`; it is a valid unforced
sample, not a zero-source sample.

Source controls are linearly Z-scored with the train-split statistics
(`source_stats` in the split manifest, computed only on training frames that
carry a projection) before being injected through the LNN control encoder.
Unforced frames pass `controls=None`; they are never zero-filled.

### 3.1a Output timing contract (ADR-0019, verified 2026-10-04)

Verified against JOREK source (`marker_spi_source.f90`,
`mod_project_particles.f90`, `mod_jorek_timestepping.f90`, `mod_particle_io.f90`),
the production log, and 620 matched frames per run:

- `jorek{n}.h5` = state `u(T_n)`; `t_now` in JOREK units.
- `projections{n}.h5` = time average of the source over `[T_{n−1}, T_n]`
  (particles advanced in the interpolated field `u_{n−1} → u_n`), stored with
  `t_now = T_n` in physical seconds; it is consumed by the solve that advances
  `u_n → u_{n+1}`.
- `part{n}.h5` = particle state at `T_{n−1}` (one JOREK step before the
  same-index field frame; `/time` in seconds), i.e. the start of the source
  sampling window.
- Correct pairing: `(u_k, p_{k−1}=part{k}, S_k=projections{k}) → u_{k+1}`.
- The legacy interpolation pipeline used `controls = source[:, k+1]`, i.e. a
  one-step-ahead source; those runs are not a valid baseline (ADR-0018/0019).
- Restart-checkpoint frames (e.g. 3550/4150/4950/5330/6600/7350, v400 5050)
  are offset by one step; production runs are **not regenerated**, so they are
  handled as *registered restart events*: the cross-boundary pair and the
  lagging `projections{n-1}` are skipped, the event frame `n` stays
  (ADR-0019 addendum / ADR-0020 decision 5; `data/restart_events.json`).
- New exports must be self-describing: per fluid step
  `(interval_start, interval_end, u_before, S_used, u_after, p_start, p_end)`.

### 3.2 Background sources (optional second channel)

Model 502 `mod_sources:sources` provides three node-local background controls:
`particle_source`, `heat_i_source`, `heat_e_source`.  Accepted layouts are one
vector dataset (`sources`/`source`/`model502_sources`/`controls`/`forcing`,
`[3, ..., nodes]`) or three nodal datasets with the documented aliases.  All
three or none: a partial record is an error.  The parser exposes
`Data.background_source [nodes, 3]`; the physics loss uses only
`particle_source` in the provisional mass-continuity residual, and neither
heat channel enters an energy residual until units and pressure convention are
verified.

This contract does not claim that pellet deposition, impurity injection,
radiation power or all `mod_elt_matrix.f90` source terms are exported; those
remain versioned extensions.

## 4. Data delivery checklist

For each new source-enabled sequence provide:

1. Model 502 HDF5 state frames with 9 `values` channels, mesh topology,
   boundary, and monotonic `t_now`.
2. The five projection fields for every forced frame, with shape and units
   recorded, plus the optional three background source fields (all-or-nothing).
3. The input namelist (`jorek_namelist_spitest` preferred), plus an explicit
   `model_id=502` provenance record.
4. A train/validation/test split manifest at sequence level (never random frame
   splits), data version/hash, and the ADR-0019 timing contract fields per
   export (`interval_start`, `interval_end`, `u_before`, `S_used`, `u_after`).
5. A small three-frame fixture that can be checked into a permissible test-data
   location or generated in CI.
6. Particle/marker snapshots with their own `time` dataset, aligned to the
   contract above (`part{n}` = window start `T_{n−1}`); restart-boundary frames
   flagged or excluded.

## 5. Current boundaries

- `rho_imp` is the Model 502 impurity-density channel (`var_rhoimp=8` in
  one-based Fortran indexing), not a generic neutral density label.
- The current momentum diagnostic uses a background two-temperature pressure
  approximation `rho * (T_i + T_e)`.  It is not the full charge-state-aware
  Model 502 closure.
- The baseline may train without sources.  A forced model run requires the
  source contract and source normalisation manifest first.
