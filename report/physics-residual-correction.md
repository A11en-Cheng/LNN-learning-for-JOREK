# Physics-residual correction — 2026-09-09

> Status 2026-09-24: the nine-field training workflow and its artifacts referenced
> below were retired and moved to `/tmp/repo_cleanup_20260924/` (temporary, not
> durable).  The physics conventions in this file remain the evidence record and
> are summarised canonically in `src/physics/definitions.md` §9.  Training-specific
> numbers (weights, run names, floors as regulariser signals) are historical only.

Evidence record for the physics-coupling loss rework.  Primary source:
`/mnt/d/Proj/jorek/models/model502/mod_elt_matrix_fft.f90` (equations 1-9,
lines ~1490-1935), cross-checked against the model500 equation set in
arXiv:2308.16124.

## What was broken (ranked)

1. **FD operators were mis-normalised.** `compute_gradient` divided by a
   scalar `sum(1/d)` instead of solving the 2x2 normal system.  On the
   4125-node Delaunay mesh, `grad(R)` measured `(0.47, 0)` instead of
   `(1, 0)`, and `lap(R^2)` had ~12.6 node-to-node noise (mean 2.2 instead of
   2).  Fixed with the inverse-distance-weighted least-squares normal
   equations in `src/physics/grad_ops.py`; `grad(R)` is now exact to 1e-5 and
   `lap(R^2)` has std 0.32.  The `psi -> Z_j` surrogate is insensitive to
   this change (corr 0.938 -> 0.937).

2. **The Poisson bracket carried a spurious `1/R`.** JOREK's axisymmetric
   bracket is `{a,b} = a_R b_Z - a_Z b_R` (no `1/R`); cylindrical weighting is
   carried by explicit powers of R in each term.  Replaced `_poisson` with
   `poisson_bracket`.

3. **The two elliptic operators were conflated.** `Z_j = Delta* psi =
   lap psi - psi_R/R` (eq 3) but `omega = lap u + u_R/R` (eq 4).  The old
   code used `lap u` for omega, missing `+u_R/R`.

4. **Vorticity equation structure was hand-derived with wrong signs and an
   arbitrary `/visco` scale.**  Correct node-collocated strong form (from the
   Fortran weak form, dropping tauIC/TG/zeta/continuity corrections):

   ```
   div(R^3 rho grad du/dt) = 1/2 {R^2|grad u|^2, R^2 rho}
                           + {R^4 rho omega, u}
                           + {psi, Z_j}
                           + 2 R d p/d Z            # see sign note below
                           + div(visco R grad omega)
   ```

5. **The pressure term carries the stored-frame sign.**  The Fortran weak
   form gives `-2R d p/d Z`, but with exact gradients the recorded frames
   satisfy `{psi, Z_j} = -[R^2, p]`, i.e. the stored `psi`/`Z_j` are the
   opposite sign of the Fortran internal variables.  The loss uses the
   stored-frame convention so the recorded trajectory minimises it.  Measured
   on frame 2750/2800:

   | convention | `|inertia - rhs|` rms |
   |---|---:|
   | data `{psi,Z_j} = -[R^2,p]` | **0.286** |
   | Fortran sign | 1.77 |

6. **The old LSQ `Delta* psi` was rank-deficient.**  It kept only one
   direction of each edge (~3.9 neighbours on this mesh) while fitting 5
   monomials.  Replaced by the MLS `div(grad(...))` form (`delta_star`).

7. **`current_definition` / `vorticity_definition` were implemented but never
   entered the loss** (`PHYSICS_WEIGHTS` had them at 0.0).  They are now
   enabled (weights 0.5 each).

8. **CfC collapsed per-graph dt to a mean.**  The installed ncps wrapper
   squeezes `timespans[:, t]` to `[B]`, which cannot broadcast against the
   cell's `[B, units]` time gate, so per-sample timespans were impossible
   through the public API.  `JorekLNNProcessor` now runs the single-step cell
   directly with a `[B*N, 1]` timespan.

## Measured floor at pred=target (frame 2750 -> 2800, val run)

Relative residuals (`|sum| / sum of term rms`), via the (archived) stage0
diagnostic `tests/diagnose_stage0_physics.py`:

| residual | relative | absolute rms | reading |
|---|---:|---:|---|
| induction | 0.90 | 4.6e-4 | all terms tiny (eta~1e-10), noisy relative form |
| vorticity | 0.16 | 0.286 | force balance holds (was 2.3e5, then 4.1e2) |
| current_definition | 0.13 | 0.065 | surrogate corr 0.937 |
| vorticity_definition | 0.07 | 9.4e-6 | `delta_plus` corr 0.99 |
| density | 0.96 | 1.8e-2 | reduced form omits sources/parallel advection |
| ion_energy | 1.00 | 2.2e5 | parallel conduction dominates, sources omitted |
| parallel_momentum | 0.99 | 0.22 | reduced form |
| impurity | 0.83 | 3.6e-5 | reduced form |
| electron_energy | 1.00 | 5.9e6 | parallel conduction dominates, sources omitted |

Consequences encoded in `PHYSICS_WEIGHTS`: ion/electron energy are
down-weighted to 0.2 because their floors are flat (omitted dominant terms
mean near-constant loss with little gradient); `vorticity_definition` is 0.05
because its relative normalisation against the tiny recorded `omega`
(|w| ~ 5e-3) gives it a Phi-gradient of ~3.6e3 (measured), which would
dominate the decoder gradient clip; vorticity and the two elliptic
consistency terms otherwise carry the informative signal.

Per-residual Phi-gradient at the trajectory (normalised pred units, from the
diagnostic `--grad` path): vorticity 0.26, vorticity_definition 3.6e3,
current_definition 0.012; psi-gradient: induction 9.5.  This table is the
first thing to consult when the corrected run still oscillates.

## Regulariser property

The vorticity residual must grow as the prediction moves away from the
recorded trajectory (it does: 0.286 -> 1.45 at a 10% field perturbation),
which is locked in by `test_vorticity_residual_grows_with_prediction_error`.
It is deliberately *not* asserted to be zero at the trajectory: the reduced
residual omits two-fluid, hyper-viscosity, zeta and continuity-consistency
terms, so a nonzero floor is expected and measured.

## Training-path float32 overflow (found in the first 400-step run)

The first corrected run (`stage0_phys_diag`, 400 steps) showed the physics
term pinned at ~50 on train splits while val sat at the ~18 floor.  Relative
residuals are bounded at 1 each, so a 4-graph step can never exceed
`sum(weights) * 4 = 23.8` without the `inf -> 10` training clamp firing.
Root cause: the model's out-of-range normalised predictions (`clamp(..., 20)`
in z-score space) de-standardise to ~1e11 densities via `expm1`, and products
of those values overflow float32 in the residual assembly on GPU (the CPU
reproduction stays just below the overflow threshold, which is why the
diagnostic at `pred=target` missed it).

Fix: `_clamp_physical` clips every de-normalised field to generous
regulariser bounds (orders of magnitude outside the recorded range) and the
residual assembly runs in float64, with the scalar loss cast back to float32
so gradients remain in the model dtype.  Per-residual weighted sums are now
logged to `physics_residuals.csv` alongside `channel_loss.csv`.

## Phi oscillation follow-up (second corrected run)

After the overflow fix the physics term is bounded (train 18.3-22.6 vs the
23.8 theoretical ceiling, no clamp), but `Phi` still oscillates between two
attractors: the data term (field weight 3) and the `vorticity_definition`
constraint.  Two changes address the mechanism:

* `--lambda-warmup`: the physics weight ramps linearly from 0 to
  `lambda_phys` over the warmup steps, so the data term fixes `Phi` before the
  constraint switches on.
* `OMEGA_REFERENCE`: `vorticity_definition` is normalised by a fixed constant
  (1e-2) instead of the frame-relative `|omega_hat|+|omega_target|`
  denominator.  (The old form was already a *global* relative norm, not
  per-node; the fixed scale's benefit is removing the denominator collapse on
  quasi-static frames, where both omega fields are ~0.)

Loss spikes were marked by the (archived) `tests/audit_frame_activity.py`,
which wrote `output/model502_baseline/frame_activity.json` (the most active
next-step frames per run - dominated by `Phi`/`omega` around injection events);
the training script recorded run/frame provenance in `spike_log.csv` whenever a
step exceeded `--spike-threshold`.

## Still open

- The energy residuals need the marker/background heat sources and the
  parallel projection to become informative; until then they are weak
  regularisers (weights 0.2).
- `src/physics/model502_physics.py` (the older `model502_residuals` layer)
  still uses the legacy half-edge LSQ and `1/R` bracket; it is superseded by
  `model502_full_loss.py` and should be removed or ported after the next run.
- The `psi -> Z_j` surrogate was trained with the old (biased) gradient
  features.  Its output was verified insensitive (0.938 -> 0.937), but a
  retrain on the corrected operators is the clean follow-up.
- The projected `Z_imp` source channel is zero in the val frames probed, so
  the two-temperature closure degenerates to unit charge
  (`n_e = rho`, `n_i = rho + rho_imp`).  Confirm the projection provenance
  before treating the energy residuals as physically faithful.
