# Model 502 nine-field and source protocol

Status: active contract, created 2026-09-05.

This protocol is the acceptance contract for all new Model 502 data and model
checkpoints.  It is based on `redundant/jorek/models/model502/mod_model_settings.f90`
and `mod_sources.f90`.  It does not apply to the existing BEST/Model 600 data.

## 1. State contract

Every graph state is a float32 tensor shaped `[nodes, 9]` in this exact order:

| Index | Name | HDF5 `values` index | Transform |
|---:|---|---:|---|
| 0 | `psi` | 2 | linear, Z-score |
| 1 | `Phi` (`u`) | 1 | linear, Z-score |
| 2 | `Z_j` | 0 | linear, Z-score |
| 3 | `omega` | 3 | linear, Z-score |
| 4 | `rho` | 4 | `log1p(clamp(x, 0))`, Z-score |
| 5 | `T_i` | 5 | `log1p(clamp(x, 0))`, Z-score |
| 6 | `v_par` | 6 | linear, Z-score |
| 7 | `rho_imp` | 7 | `log1p(clamp(x, 0))`, Z-score |
| 8 | `T_e` | 8 | `log1p(clamp(x, 0))`, Z-score |

Statistics are computed only on the training split after the stated transform.
The checkpoint must record the schema version, field names, train-split ID,
mean and standard deviation.  Negative values in positive-only channels are a
data-quality event: the transform clamps them for numerical safety, but the
loader/ingestion report must count and explain them.

The decoder returns a normalized residual **rate** `rate_x [nodes, 9]`; update
a state only as `x_{t+1} = x_t + rate_x * dt`, where `dt` is the physical
seconds between the two frames.  Physical `dt` is also passed separately
to the LNN and PDE diagnostics. A five-field checkpoint is incompatible
and must not be silently loaded.

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
   splits), data version/hash, and the source's temporal convention: source at
   `t`, `t + dt`, or interval average.
5. A small three-frame fixture that can be checked into a permissible test-data
   location or generated in CI.

## 5. Current boundaries

- `rho_imp` is the Model 502 impurity-density channel (`var_rhoimp=8` in
  one-based Fortran indexing), not a generic neutral density label.
- The current momentum diagnostic uses a background two-temperature pressure
  approximation `rho * (T_i + T_e)`.  It is not the full charge-state-aware
  Model 502 closure.
- The baseline may train without sources.  A forced model run requires the
  source contract and source normalisation manifest first.
