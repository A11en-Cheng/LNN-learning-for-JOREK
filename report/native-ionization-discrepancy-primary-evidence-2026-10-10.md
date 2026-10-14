# Eion A/B discrepancy: primary evidence, 2026-10-10

Read `docs/adr/README.md` first; no ADR-0001–0017（历史）gate is used. Scope: static, read-only JOREK and current Python source audit, plus two existing saved-array norm calculations. This report was researched by the background agent requested by the research skill; it wrote only this report, with no implementation changes, training, JOREK execution/mutation, or new source windows. Main owns the paired-source multiwindow diagnostics at 64 cells per width; the older evidence below used 128 and independent streams.

## Finding and exact formula

Eion is **signed change of cumulative ionization energy**, not stored ionization energy, mean charge, kinetic energy, or a fresh random atomic transition. With incremental barriers I_j [eV], U_0=0 and U_z=Σ_{j=1}^z I_j,

`e_mk = w_m * EL_CHG * Σ_z (p_new[z]-p_old[z])*U_z` [J].

JOREK constructs U at `particles/examples/marker_spi_source.f90:274–278`, clips explicit charge-state updates at `543–588`, normalizes at `598`, computes the difference at `599–603`, then deposits at the OLD location at `616–642`. Channel 1 is `e_mk*t_norm/E_norm`, with **no second dt factor**; the window RHS is divided by actual gather at `673–676`. `dEion_dt` is thus a misleading alias for substep energy, not a per-second quantity at its calculation site. `E_lost_ion` accumulates that same signed energy (`604`). `ion_energy` instead holds kinetic energy (`609–611`) and must not be used as an independent Eion reference.

The exact current aliases are `p=old P_imp.copy()`, `q=charge_step(...)`, `e=(q-p)@r_[0,cumsum(potentials_ev)]*w*elementary_charge`, load channel 0, diagnostic `energy`, and feedback channel 0 (`src/models/marker_state.py:309–322,349`). `NativeMarkerSource.advance` supplies the actual `EL_CHG` explicitly (`src/models/native_coupling.py:545–549`). `NeonTables.ionization_potentials_ev` contains **incremental** barriers (parser `src/models/source_module.py:179–193`), despite its misleading “cumulative barrier” field comment (`171`); accumulating an already cumulative reference array would double-count. The coronal `_neon_closure` at `245–313` is a different quantity/operator and is not this time-dependent marker reference.

An independent formula check should use scalar charge-state sums and an independently constructed U, then compare with the production dot product before deposition. In the absence of clipping/renormalization, the identity is

`Σ_z Δp_z U_z = ne*dt*Σ_{z=0}^{Z-1} I_{z+1}*(p_z*S_z-p_{z+1}*A_{z+1})`.

After clipping, that raw-rate identity need not equal the actual source: use normalized p_new for the primary reference and record the clipping correction separately. The commented raw-rate expressions in JOREK (`560–586`) are not executed. For each marker of constant weight, summing e over its processed history telescopes to `w*EL_CHG*(p_final-p_initial)·U`; basis-weighted deposition does **not** telescope because position/basis changes. Exit truncates the processed history. A scalar, array-dot, and endpoint comparison can falsify arithmetic/alias errors without treating spatial deposition as an endpoint formula.

## Why a few births can matter

Neutral newborns have p_0=1 (`particles/mod_initialise_particles.f90:539–541`). Existing highly charged or near-equilibrium markers can carry large Z/Z2 but small Δp·U. Therefore a small birth count can contribute strongly to Eion while the established population dominates charge moments and radiation. “Existing cloud nearly charged” is a hypothesis about these imported states, not a measured charge histogram in this sidecar; being charged alone does not imply equilibrium or small Eion.

Read-only calculations on `reference_2917.npz`, keys `reference_2917_{whole,newborn}_weak`, shape `(5,16419)`, float64, give Euclidean unique-DOF `||newborn||/||whole||`:

| Anchor | Eion | Rad | Z2 | Z | N |
|---|---:|---:|---:|---:|---:|
| 4110-final | 0.3452972151 | 0.0197992652 | 0.0017321975 | 0.0060008988 | 0.0276655111 |
| 5680-final | 0.2838804048 | 0.0215172589 | 0.0001955247 | 0.0010022111 | 0.0079821536 |

These are norm ratios, not fractions of physical energy or additive percentages. Newborn loads were captured within the whole-cloud window, normalized by gather, then projected with the same operator (`tools/validate_native_birth_projection.py:255–272`). Linearity permits whole-minus-newborn decomposition under the fixed feedback/fields, but this calculation does not establish the old-cloud charge state or sign cancellation. The previous comparison reports 29/56 births versus 11,168/52,604 initially live markers and whole-cloud Gauss Eion A/B L2 ≈9.08%/7.01%, versus integral errors ≈0.327%/0.187% (`report/native-birth-projection-comparison-2026-10-10.md`, tables). These prior measurements motivate investigation; two seeds cannot identify bias or certify equivalence.

## Sensitivity, history, and falsifiable hypotheses

1. **Birth-dominated energy increment.** Prediction: newborn/old decompositions show a much larger newborn share for Eion than charge moments; old markers have smaller absolute Δp·U per represented atom. Falsify using initial charge histograms and cohort-specific signed/absolute energies. The saved norm ratios support the first prediction only.
2. **Localized Te/ne response.** The rates enter through ne*dt and log-table queries, so nearby birth points need not have similar Eion in steep temperature/density regions. Current ne uses old projected charge/number plus fluid impurity density; Te and ne floors and table endpoint clipping matter (`native_coupling.py:506–536`; `native_atomic_interpolation.py:20–36`). Prediction: differences concentrate near steep rate response or clipping regimes. Falsify with paired local Te/ne, rates and Δp·U, or fixed-point sensitivity perturbations holding p/history constant. Avoid assuming every regime is highly sensitive: saturated/equilibrated regimes may be insensitive.
3. **Timing/history and signed cancellation.** Ionization raises U; recombination lowers it. Large positive/negative substep or cohort terms may leave a small net denominator. Charge histories persist and the time-dependent field reader matters even with fixed field endpoints/rawD. Current sample time is `start+k*dt` (`marker_state.py:283,307`); primary JOREK samples at residual-clock times, while birth velocity initializes at sim time (see the prior primary report). Prediction: signed sums differ much less than absolute activity, or differences track source-time/charge-stage transitions. Falsify with separate positive/negative and endpoint identities, preserving actual gather and residual time. Constant-weight endpoint telescoping prevents attributing an integral discrepancy solely to intermediate sign oscillation when endpoints are identical.
4. **Few effective energy contributors.** Equal birth weights do not imply equal energy/basis-weighted contributions. For independent marker contributions a_m to a selected scalar observable, report `(Σ|a_m|)^2/Σa_m²` and cancellation `|Σa_m|/Σ|a_m|`; distinguish this concentration index from a confidence-interval sample size. Repeated substeps of one marker are correlated, not independent samples. Prediction: Eion is carried by fewer effective markers than N, even at the same nominal count. Falsify using per-marker history-integrated contributions. No effective sample count was computed here.
5. **Sampler or deterministic operator discrepancy.** Small probability-bin errors do not bound basis/rate-weighted source errors. At fixed sample count, stable signed spatial differences across paired windows/seeds or refinement dependence would support spatial-table bias; persistent identical-point charge discrepancies would support atomic/timing/precision errors. Test 64-cpw results with a separately controlled refinement and independent pointwise formula reference. Independent reference/reference noise in older reports is context, not proof that A/B is noise alone or that 64-cpw inherits 128-cpw accuracy.

Spatial and integral diagnostics answer different questions. Volume-weighted L2 sees relocation/local amplitude errors; an integral can cancel them. The native Gauss measure is `weights*jacobian*R*2pi`, with full Hermite evaluation (`validate_native_birth_projection.py:80–86`). Relative L2 uses the reference norm and integral error uses the signed reference integral (`34–50`); always retain absolute differences and denominators. Weak-load and flattened coefficient norms mix basis/derivative scaling and must not be called physical field errors. High Pearson and small integral error do not certify small local Eion error.

## Precision and units audit

JOREK particle `weight` and `P_imp` are real*8 (`particles/mod_particle_types.f90:59,78`), but newborn weight is explicitly rounded to real*4 then stored (`mod_initialise_particles.f90:539`). Do not equate storage precision with birth precision. Python saved weak loads here are float64; interpolation explicitly uses float64 queries. JOREK sums scalar charge terms; Python uses a dot product, so rounding order is not bit-identical, especially when Δp is small. No evidence here establishes that rounding explains percent-level A/B errors.

JOREK ADAS loader converts density log axis cm^-3→m^-3 by +6, temperature eV→K, coefficients cm³/s→m³/s by −6 in log space (`tools/mod_openadas.f90:103–130`), and recombination rows retain source charge 1..Z while SCD uses 0..Z−1 (`116–125`). Python instead queries log10(ne*1e−6), log10(Te_eV), converts SCD/ACD outputs by 1e−6, and indexes acd[:,z] as z+1→z (`native_coupling.py:590–596`; `marker_state.py:200–212`). These are equivalent unit/index conventions, not identical numeric interpolation certification; endpoint handling deserves identical-point checks.

JOREK `E_norm=1/(MU_ZERO*(GAMMA−1))` (`marker_spi_source.f90:261`); Python load formula hardcodes gamma=5/3 (`marker_state.py:316–319`). Equality requires that actual GAMMA. Density uses n_norm, energy uses t_norm/E_norm, then gather division produces the normalized rate; an extra ne, n_norm, dt or 2pi would be erroneous. JOREK `EL_CHG=1.602176565e−19` (`models/constants.f90:13`) equals current `marker_feedback.py:16`, imported at `native_coupling.py:11`. `advance_window` has a different default 1.602176634e−19 (`marker_state.py:245`), but the production caller overrides it. Its relative difference ≈4.31e−8 cannot explain these percent-level discrepancies. Audit an independent formula's imported constant rather than its function default. Uniform common energy-unit scaling also cancels in relative A/B Eion errors, though not absolute quantities.

## SHA256 snapshot

JOREK paths below are relative to `/home/ac_desktop/JOREK_code/jorek`; all other paths are relative to this repository. Exact line intervals above apply to these snapshots. Prior primary report: `report/native-birth-projection-primary-evidence-2026-10-10.md` (birth, timing, projection evidence); no new multiwindow result is claimed.

```text
4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c  JOREK/particles/examples/marker_spi_source.f90
269ce1cea5d5029c561f8ba93aacf4987f9c24163708f6df1a085cd6ba61f22d  JOREK/particles/mod_initialise_particles.f90
14fb42d89cf6e0ed55de5bfda4122aa230da4c67c1f4385abc92e1db9197b421  JOREK/particles/mod_particle_types.f90
105db98d6a0ac36f2546975e20423a8cdf2c374183131e8d28f55b3a9095342f  JOREK/tools/mod_openadas.f90
2597932cc2e84433b3f10d7a83cae0f0af6cefb81f5be2b688d826712ada6c3e  JOREK/models/constants.f90
e23328dda7512f920a8a773c93d4bb824aad6a3c8424c314748919b5cadf77f7  src/models/marker_state.py
f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b  src/models/native_coupling.py
23fc174293056877aeab0eb2ec1f5aee77263ae5b02f4d0e843c4b825df27618  src/models/native_atomic_interpolation.py
517aec5694de1baeacdc68c3158e59cb5da9ab0882f3f4360ec9b4e32dfa12e0  src/models/source_module.py
04ec4fe5ca228c8edf20518feb1c45ee8c59969bb467c9850abad525298b5c63  src/models/marker_feedback.py
2ec98026852e678043f97537bd074fdb66c5bbbcd2119df31d390f85ef83c25b  tools/validate_native_birth_projection.py
a2d1ec44b892b1ff7610785411d2254960e5f2c45458c8aa001b23eae7fbfdc0  output/native-birth-projection-2026-10-10/anchor4110-final/reference_2917.npz
b0290a711cba1d3cb58a1de2fbacbb2031227d3ae4d923f062ca7ed67df15f96  output/native-birth-projection-2026-10-10/anchor5680-final/reference_2917.npz
2c9a5f29d5fa4dc6bfbb58980f93011bec5df910d62a0b81bb425a6e32cb465c  report/native-birth-projection-comparison-2026-10-10.md
9edbae7dbceaa18c0a8aa83328c86c4d2cb1ad97359c3f87c49e6d6fa75c60d1  report/native-birth-projection-primary-evidence-2026-10-10.md
e314d4e2f2cd30c4d4f35ab15081e8a56bb17780d0dab4409013ad5c4ea060bf  docs/adr/README.md
```
