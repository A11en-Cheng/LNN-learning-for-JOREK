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

The decoder returns a normalized one-step residual `delta_x [nodes, 9]`; update
a state only as `x_{t+1} = x_t + delta_x`.  Physical `dt` is passed separately
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

Model 502 `mod_sources:sources` provides three node-local controls:

| Index | Name | Meaning in JOREK source routine |
|---:|---|---|
| 0 | `S_rho` | particle source in the background density equation |
| 1 | `S_T_i` | ion heat source |
| 2 | `S_T_e` | electron heat source |

For a forced frame, the HDF5 must contain all three fields, in JOREK internal
units and aligned to the state frame's nodes and `t_now`.  Accepted source
layouts are:

- one dataset named `sources`, `source`, `model502_sources`, `controls`, or
  `forcing`, shaped `[3, ..., nodes]`; or
- three nodal datasets, using `S_rho`/`particle_source`,
  `S_T_i`/`heat_i_source`, and `S_T_e`/`heat_e_source` (case-insensitive).

The parser exposes a raw physical `Data.source [nodes, 3]` only if all fields
are present.  A partial record is an error.  A source-free frame has
`source_available=false` and no `source` tensor; it is not a zero-source
training sample.  Set `JorekSequenceDataset(require_source=True)` for forced
training so that one incomplete frame fails the load.

Source controls must have separate training-split mean/std before being passed
to `JorekLNNProcessor(..., controls=source_normalized)`.  The physics loss uses
the raw `S_rho` only in the provisional mass-continuity residual.  `S_T_i` and
`S_T_e` are deliberately not inserted into an energy residual until their
units, pressure convention, and per-frame time alignment are verified.

This first source contract does not claim that pellet deposition, impurity
injection, radiation power, ionisation potential energy or all source terms in
`mod_elt_matrix.f90` have been exported.  Those require a versioned extension
after the new data identifies the exact JOREK output fields and units.

## 4. Data delivery checklist

For each new source-enabled sequence provide:

1. Model 502 HDF5 state frames with 9 `values` channels, mesh topology,
   boundary, and monotonic `t_now`.
2. The three source fields above for every frame, with shape and units recorded.
3. The input namelist or its four requested parameters, plus an explicit
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
