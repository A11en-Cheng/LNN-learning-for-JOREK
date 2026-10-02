# Roadmap review — 2026-09-29

Systematic review of the current roadmap ([new_version.md](../new_version.md),
ADR-0010…ADR-0014) against (a) the verified JOREK source behaviour and
(b) the historical evidence already produced in this repository
(#0012/#0014/#0015/#0016, the stage-0 physics round, the elliptic surrogate).

## 1. What still holds

- The gated structure (ADR-0002), the LOO + blocked-split protocol
  (ADR-0012), the "physics loss only at interpolated points" rule (ADR-0010)
  and the sub-interval integral residual (ADR-0013) remain sound.
- Phase 1 (window corpus, fragment state, dataset cache) is implemented and
  tested; the `t_ns` cut, burn-in and LOO folds are consistent with the data.
- The analytic JOREK backbone (ablation model 3, rocket model 6, erosion) is
  verified against recorded shard data (position/radius MAE small; ablation
  log-corr ~0.99 early, late-phase deficit from the nonlocal branch).
- The xtime integral supervision fixed the aggregate ablated-atom budget
  (-47% -> +7%) and halved the per-interval error.

## 2. The central finding: Phase 2.2 regressed against history

The historical split-head source surrogate (#0012/#0015/#0016) already solved
the marker -> projection mapping on the *same* held-out scenario family:

| model | val R2 (Z_eff / Z_imp / N_imp) | val R2 (dE_ion / rad_sink) |
|---|---|---|
| #0012 instantaneous node MLP (9 fields only) | 0.678 / 0.677 / 0.554 | 0.206 / 0.253 |
| #0015 GNN + 18-frame history | -0.291 / 0.266 / 0.216 | 0.574 / 0.520 |
| #0016 split heads (instantaneous + 3 past projections) | 0.705 / 0.714 / 0.607 | 0.808 / 0.864 |
| #0016 longer training (12万 samples, hidden 192) | 0.613 / 0.637 / 0.651 | 0.952 / 0.955 |
| **current Phase 2.2 head (2026-09-27, no history)** | **0.664 / 0.709 / 0.597** | **0.219 / 0.155** |

Reading:

- Charge/density channels are an **instantaneous local function of the nine
  fields**; the shard/analytic machinery adds little for them.
- Energy/radiation channels are **history-dominated**; the proven solution is
  to feed the past projections (or an equivalent memory) — #0016 reached
  0.81-0.96 R2, while the current head (no history) drops to 0.15-0.22.
- Therefore the current Phase-2.2 gate is architecturally inconsistent with
  the evidence: it cannot expect dE_ion/rad_sink to beat the analytic baseline
  without history.  This is a roadmap *regression*, not a modelling failure.

## 3. Verified JOREK behaviour that the roadmap does not yet encode

| fact | consequence for the roadmap |
|---|---|
| `projections*.h5` values are signed FEM basis coefficients; assembly is `HH(l,m)*size(l,m)*quantity/n_norm` | any source loss / moment extraction must use the same basis; naive sums are meaningless |
| the projection is the **time average of one fluid step** (`rhs = feedback_rhs/(n_steps*timesteps)`); a frame interval contains 50 fluid steps | the source-accumulator training window must be aligned with the gather semantics; currently unresolved |
| `part*.h5` stores the full particle state (`P_imp/x/v/weight/i_life/i_elm/st`) and the process is Markovian | the Eulerian-moment extraction (charge-state-resolved) is well-posed; the state is sufficient *given the field evolution* |
| field is static within one fluid step; `n_steps = fluid dt / 5e-9 s`; up to ~326 substeps | CfC must carry the sub-step physics or an equivalent reduced state; explicit ADAS evolution is the natural stiff integrator |
| intermediate fields between saved frames are not written | a particle replay needs field interpolation; full replay ~minutes per late interval, days for the dataset in NumPy |
| RNG state is not stored; `t_birth` is zero in the dumps | restarts are statistical; birth-time diagnostics are unavailable |
| nonlocal ablation tables (`nl_*`) are not exported | late-phase ablation remains the main physics gap; learned correction or a JOREK-side export |
| historical #0014: local coronal and node-local time-dependent charge-state closures both failed (R2 ~ 0, transport-dominated) | do not retry local closures; the memory must be carried (transport / past projections / latent state) |

## 4. Phase-by-phase verdict

| phase | verdict | main issue |
|---|---|---|
| Phase 1 data contract | mostly good, incomplete | missing: Eulerian-moment extraction, projection-window semantics, export wish list |
| Phase 2.1 fragment encoder | keep as component | fine; not the bottleneck |
| Phase 2.2 source head | **needs redesign before further training** | no history for the energy channels (historical regression); keep as an ablation |
| Phase 2.3 hybrid dynamics | partial, keep as component | radius/ablation gains; velocity hurt by the acceleration correction; analytic backbone is the workhorse |
| Phase 2.5 / ADR-0014 implicit CfC | promising, unproven | must beat the #0016-style explicit baseline; needs the moment set and accumulator semantics resolved first |
| Phase 3 field model | unchanged but untested | the existing 225k-parameter GNN+CfC+decoder has not been trained under the new contract; ADR-0014 merges it with the source/fragment path — decide whether it stays a separate stage or becomes the implicit decoder |
| Phase 4 physics loss + closed loop | sound in principle | physics residuals cannot repair a wrong source memory (historical #0014); the gate must include source-history checks |
| Phase 5 validation | protocol good | add layered reporting (injection/quench/quiescent) and re-run historical baselines under the new LOO protocol (old numbers used a 2-run val split) |

## 5. Recommended roadmap changes

1. **Add Phase 1.5 "state & source contract"**: (a) Eulerian moments extraction
   (charge-state-resolved nodal density + rate moments) from `part*.h5`;
   (b) projection-window semantics (per-fluid-step vs frame average) and the
   resulting training target; (c) JOREK-side export wish list
   (per-step/accumulated projections, `nl_*` tables, `t_birth`) with fallbacks.
2. **Restore the split-head + history baseline as the Phase-2 reference**
   (re-run under the LOO protocol) and set the gate against it; the current
   no-history head becomes an ablation.
3. **Define a baseline ladder for ADR-0014**: L0 persistence/analytic;
   L1 explicit split-head + history; L2 L1 + xtime integral; L3 implicit CfC
   (Eulerian moments + accumulator); L4 closed loop.  Each level must beat the
   previous one on the same LOO/blocked gates before the next is started.
4. **Resolve the moment-set sufficiency by experiment**: charge-state-resolved
   nodal density vs compressed moments; gate on the source R2 and the rate
   moments, not on the field MAE alone.
5. **Keep the physics/learning boundary explicit**: ADAS, FEM deposition and
   basis assembly are computed; transport, nonlocal ablation and model error
   are learned.  This is what preserves cross-device transferability.
6. **Re-run the historical baselines under the current protocol** before
   trusting any gate numbers; the old #0016 R2 used a different split and
   sampling.

## 6. Do-not-repeat list (historical failures)

- Physics loss as a regulariser on labelled single-step fits (ADR-0010).
- Local coronal / node-local charge-state closures ignoring marker transport
  (#0014: R2 ~ 0).
- Unbounded learned corrections (#0015 -> #0016; and the 2026-09-27 dynamics
  run that diverged at 1e9 before bounding).
- Dense attention over 1e6 markers; use Eulerian moments instead.
- Static frame-snapshot -> projection regression without history/memory.

## 7. Addendum (2026-09-29, later): verified assets and the JOREK work list

### 7.1 Assets that already exist (no JOREK change required)

| asset | content | cadence | use |
|---|---|---|---|
| `macroscopic_vars.dat` | times, energies, magnetic/kinetic energies, particlecontent, thermalenergy, heatingpower, particlesource, bnd_fluxes, energy_conservation, ... | per timestep, ~6546 blocks/run, run-specific | high-frequency history features; budget gates |
| `nonlocal_array.dat` | `psi_n`, surface-averaged `nl_avg_Te`, `nl_avg_ne` (1000 points) | single snapshot (overwritten each call) | validation target for a field-side surface-average reconstruction; if the reconstruction matches, the nonlocal ablation branch becomes computable |
| `part*.h5` | full particle state (`P_imp`, `x`, `v`, `weight`, `i_life`, `i_elm`, `st`) | per saved frame | Eulerian-moment extraction (charge-state-resolved) |
| `projections*.h5` | signed FEM coefficients of the five channels | per output step (frame cadence; some runs finer) | source target, with the window-semantics caveat |

### 7.2 Namelist facts that change the cost estimate

- `nout = 50` (50 fluid steps per output frame);
- `tstep_n` reaches 5 JOREK units; `tstep_particles = 5e-9 s`, so one fluid step
  can contain up to ~408 particle substeps (not ~40);
- one late frame therefore costs ~`50 x N_alive x ~408` particle substeps;
  a Python replay of a single late interval is minutes, the full dataset is
  days.

### 7.3 JOREK-side change list (priority order)

| priority | change | location | fallback |
|---|---|---|---|
| P0a | interval-averaged (or per-fluid-step) projection export | particle projection event / `mod_project_particles.f90` | trapezoidal aggregation of existing projections + recorded truncation error, cross-checked with `macroscopic_vars.dat` |
| P0b | **cancelled**: the nonlocal table is a one-time frozen snapshot (`int3d_new` is called once when `nl_avg_Te(1)==0`), so it is used offline as-is with the recorded shard psi/radius/species | no JOREK change | — |
| P1 | per-shard ablation history | `communication/mod_export_restart.f90` (change the `n_inj` dims to `n_spi_tot`) | per-frame `spi_abl_arr` + aggregate `xtime_*` (already in use) |
| P2 | `t_birth` / RNG state | particle writer | statistical restart is acceptable for now |

### 7.4 Additional gates added to the roadmap

- **L3 memory ablation (mandatory)**: pure implicit CfC memory vs CfC plus the
  explicit past-3-frame source features; fall back to the explicit history
  input if the implicit memory decays.
- **Moment reverse-assembly gate**: the extracted Eulerian moments must
  re-assemble the five projection channels to machine precision before any
  network training.
- **Projection-window timebox**: if the interval-averaged projection export is
  not available in time, use the trapezoidal fallback and record its truncation
  error in the evaluation.
- **L4 hard clamping**: all learned correction multipliers (radiation sink,
  nonlocal ablation, ...) are clamped to prior physical extrema in code, so a
  divergent closed loop cannot escape the analytic skeleton.
