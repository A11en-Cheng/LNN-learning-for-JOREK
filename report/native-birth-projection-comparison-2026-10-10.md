# Native birth projection A/B, 2026-10-10

Both real phaseA anchors completed bounded fixed-field source windows with the full
one-million-slot initial cloud. No training, optimizer, or nine-field/network advance
ran. Full A/B seeds are 2917 and 2918 at both anchors. Independent reference/reference
full-window noise is available for both seeds at 4110 and seed 2918 at 5680.
Two seeds, and one inner noise pair, do **not** certify statistical equivalence or
identify systematic bias.

## Evidence and fixed state

Read `docs/adr/README.md`, the registered-restart/timing contracts in ADR-0019/0020,
`tools/validate_native_birth_entry.py`, the actual real builder, native birth,
`NativeMarkerSource.advance`, `advance_window`, and mesh deposit/project/integrate.
The compared source is marker feedback as specified by ADR-0020. No historical ADR
is used as a gate. These are registered poststartup imports, not invented ordinary
restart histories.

Each anchor builds once per invocation. Every arm starts from identical full cloud,
current/previous fields, saved rawD, all shards, marker clock, projection and its raw
history. Array SHA256 digests, source hashes and clock values are in `summary.json`
or `initial.json`. The instrumented runs assert these inputs remain unchanged.
The builder performs its real initial shard diagnostics. Subsequent windows hold
fields and old feedback fixed; shards do not advance. This is a source-window probe,
not a coupled field forecast or multiwindow feedback experiment.

Corrected native-mass rejection is the reference. New sampling is explicitly n0 CDF,
128 cells per width, 16,000,000 maximum cells. The harness passes
`birth_cells_per_width=128` and `birth_max_cells=16000000` to the builder and binds
its own comparison sampler explicitly. Generic sampler defaults are not modified.
Both use the same birth count, float32 weight, real locator and rawD/sim-time velocity
initialization. n0 retains the RZ marginal with uniform placeholder phi; it is not
production PCG32/MPI replay or the native joint phi law.

Seed once per case: reference uses base seed; reference-noise uses base + **1,000,003**;
new uses base + **2,000,003**. Thus actual streams are 2917/1002920/2002920 and
2918/1002921/2002921. These are independent-stream comparisons, not paired draws.
No substep reseeding occurs. MT19937 before/birth/after-window digests are recorded;
the command restores global NumPy state on exit. Same-seed reference replay matches
birth cloud, RNG state, saved loads/coefficients and final full-cloud digest exactly.

| Anchor | Initial slots / live | Births | Full substeps / gather SI | Builder time | Full window time | Peak process RSS |
|---|---|---:|---|---:|---:|---:|
| 4110 | 1,000,000 / 11,168 | 29 | 40 / 2.0428876129759165e-7 s | 8.60 s | 6.99–7.51 s | 1.46 GiB |
| 5680 | 1,000,000 / 52,604 | 56 | 81 / 4.085775225951833e-7 s | about 9 s | 43.24–46.48 s | 1.73 GiB |

New owned tables: 4110 has 2,763,724 cells, **88,519,016 bytes** (84.42 MiB),
3.34–3.95 s measured preparation; 5680 has 7,064,151 cells,
**226,180,560 bytes** (215.70 MiB), about 10.16–10.22 s preparation in
`anchor5680-final`. Birth times include table preparation, sampling, valid-mesh
rejection and velocity initialization. These are observed costs, not speedup claims.
Both tables use 64 angular integration points; inner has saturated total-source
nodes. Arrays are not source probabilities inferred from particle histograms.
BLAS/OpenMP/MKL and Torch use one thread per worker; source budget is 8 GiB.

## Measures and results

Five channels: Eion = ionization energy, Rad = radiation, Z2 = squared-charge moment,
Z = charge moment, N = number. Birth-only load is instantaneous neutral newborn
number, `weight/n_norm`; its other four channels are exactly zero. A full source
window subsequently develops all five channels through the public atomic/push chain.

During that **same full-cloud window**, deposition captures contributions belonging
to newly inserted slots, normalized by actual gather. The whole-cloud load includes
all original live markers as well. No separate newborn-feedback loop is introduced.
Old projection feedback remains identical and fixed throughout each window.

Relative L2 is `norm(candidate-reference)/norm(reference)`. Algebraic weak-load
norms use unique DOFs; coefficient norms flatten the recorded Hermite layout and
mix values with derivatives. They are **not physical field error norms**. Their
Pearson and integral diagnostics, including absolute errors, are saved per channel.
Weak integrals act on the constant test (value DOFs one, derivative DOFs zero);
original mesh coefficient integrals use its native convention without extra 2pi.

Physical field diagnostics evaluate saved coefficients at the full mesh's native
16 Gauss points per element, using `test_basis`, and weight by
`g.weights*g.jacobian*R*2pi`. L2 and Pearson use that volume measure; integrals
are the corresponding weighted field integrals. This postprocessing rebuilt only
the filtered mesh, without another source or field advance. It is the represented
filtered FEM field on this quadrature, not continuous truth or a source-bias proof.

Tables below are arithmetic **means of per-seed errors**, with A/B two seeds;
4110 noise uses two pairs and 5680 noise uses one pair (2918). Percent columns
multiply relative errors by 100. All individual values/ranges and undefined counts
are available in the compact JSON summaries.

Whole-cloud volume-weighted Gauss fields:

| Anchor | Channel | A/B L2 % | Reference-noise L2 % | A/B Pearson | A/B integral error % |
|---|---|---:|---:|---:|---:|
| 4110 | Eion | 9.08406 | 7.86373 | 0.995854566 | 0.326931 |
| 4110 | Rad | 0.491371 | 0.406668 | 0.999987972 | 0.0260947 |
| 4110 | Z2 | 0.0368576 | 0.0336544 | 0.999999932 | 0.000369613 |
| 4110 | Z | 0.134722 | 0.117892 | 0.999999085 | 0.00345045 |
| 4110 | N | 0.833809 | 0.681143 | 0.999964400 | 6.47134e-13 |
| 5680 | Eion | 7.01113 | 5.20719 | 0.997562614 | 0.186871 |
| 5680 | Rad | 0.509556 | 0.409829 | 0.999986658 | 0.00282051 |
| 5680 | Z2 | 0.00618799 | 0.00448823 | 0.999999998 | 3.18488e-05 |
| 5680 | Z | 0.0299039 | 0.0218945 | 0.999999949 | 0.000231285 |
| 5680 | N | 0.219113 | 0.207450 | 0.999997250 | 1.81062e-11 |

Newborn contribution alone, same volume-weighted Gauss measure:

| Anchor | Channel | A/B L2 % | Reference-noise L2 % | A/B Pearson | A/B integral error % |
|---|---|---:|---:|---:|---:|
| 4110 | Eion | 46.2214 | 39.6063 | 0.888536188 | 2.08514 |
| 4110 | Rad | 47.7260 | 39.1587 | 0.885069972 | 3.79136 |
| 4110 | Z2 | 45.2427 | 40.4133 | 0.890368464 | 0.838816 |
| 4110 | Z | 45.9109 | 39.6165 | 0.889251948 | 1.87162 |
| 4110 | N | 51.9534 | 42.9809 | 0.880046923 | 2.99076e-12 |
| 5680 | Eion | 29.2639 | 21.4941 | 0.968271104 | 1.66097 |
| 5680 | Rad | 28.1728 | 22.4414 | 0.968571888 | 0.623398 |
| 5680 | Z2 | 30.4752 | 21.7101 | 0.963875285 | 1.23538 |
| 5680 | Z | 29.6756 | 21.4309 | 0.967604312 | 1.91390 |
| 5680 | N | 28.5242 | 26.7571 | 0.965131177 | 1.80343e-08 |

Whole-cloud algebraic diagnostics; all entries relative L2 percent:

| Anchor | Channel | Weak A/B | Weak noise | Coefficient A/B | Coefficient noise |
|---|---|---:|---:|---:|---:|
| 4110 | Eion | 37.6749 | 34.7632 | 90.2290 | 95.9243 |
| 4110 | Rad | 2.16319 | 2.02700 | 7.15060 | 7.65052 |
| 4110 | Z2 | 0.177837 | 0.162767 | 0.669420 | 0.715454 |
| 4110 | Z | 0.631321 | 0.586011 | 2.41007 | 2.57554 |
| 4110 | N | 3.27149 | 3.29378 | 13.9827 | 15.6359 |
| 5680 | Eion | 15.0581 | 9.32098 | 34.5818 | 23.3467 |
| 5680 | Rad | 1.12808 | 0.751142 | 5.88768 | 3.95262 |
| 5680 | Z2 | 0.0108350 | 0.00633584 | 0.0965072 | 0.0610587 |
| 5680 | Z | 0.0542410 | 0.0328122 | 0.484626 | 0.318592 |
| 5680 | N | 0.430623 | 0.366399 | 3.48508 | 2.97725 |

One 4110 Eion coefficient A/B error is 107.34%, despite its Gauss-field error
being 9.06%. The norms answer different questions; the coefficient result must not
be called a physical field error. Whole-cloud dilution also must not hide the much
larger newborn-only differences from 29/56 finite samples.

Birth-only N weak L2 is 115.35–119.23% at 4110 and 42.68–64.90% at 5680;
coefficient L2 is 130.59–134.82% and 96.81–106.12%, respectively. N integral
agreement is near numerical precision. For Eion/Rad/Z2/Z, zero reference norm and
zero reference integral are explicitly **undefined**, represented by JSON null plus
status. Pearson for a constant channel is likewise undefined; no forced zero/one,
epsilon denominator or NaN is substituted.

Optional saved two-seed **ensemble-mean field** comparisons were also completed.
These compare mean fields first, not averages of per-seed errors. Whole-cloud A/B
Gauss L2 percentages in Eion/Rad/Z2/Z/N order are:
4110 `7.29079 / 0.402308 / 0.0287942 / 0.107537 / 0.694974`;
5680 `5.13258 / 0.368879 / 0.00460564 / 0.0219221 / 0.139607`.
Inner full-window noise ensemble is explicitly skipped because only one complete
noise pair exists. Two-seed averaging still does not isolate systematic error.

Finite Monte Carlo noise is directly demonstrated by the reference/reference arms.
A/B errors are of comparable order but often larger in these samples. This does not
prove either unbiasedness or a systematic defect. No significance test, approved
tolerance, resolution-convergence certificate, coarse-histogram inference, training
qualification or multiwindow stability claim is made.

## Artifacts, commands and unfinished scope

All artifacts are under `output/native-birth-projection-2026-10-10/`:

- `anchor4110-final/`: two full A/B and noise seeds, replay, table diagnostics,
  `summary.json`, `metrics-summary.json`, `gauss-metrics.json`,
  `gauss-metrics-summary.json`, `ensemble-gauss-metrics.json`, per-case JSON/NPZ.
- `anchor5680-final/`: full seed-2917 A/B + replay; two-seed birth diagnostics;
  other windows explicitly `not_requested`. Its Gauss and ensemble artifacts are saved.
- `anchor5680-seed2918/`: **all four full windows completed**, including independent
  reference-noise and exact replay. This supplies the second full A/B seed and
  the completed inner noise baseline. Both summary and Gauss compact JSON are saved.
- `anchor4110/`: preserved first successful two-seed probe.
- `anchor5680/`: preserved original seven **unfinished** 45-second-budget attempts,
  with birth deposition and birth-only Gauss diagnostics. These are not counted as
  completed windows, and partial gathered loads are not presented as full-window data.
- Corresponding `.log` files retain real builder stages, timings and failure evidence.

Window executions preceded the later summary/Gauss-only additions. Their recorded
validator hashes identify those measured versions; they are not relabelled with the
final file hash. Postprocessing records its own hash and input-summary hashes. No
particle-event algorithm was changed by these diagnostic additions. All completed
inner A/B inputs match exactly across the two preserved invocations.

Executed commands (all execution/postprocessing used the three thread environment
variables set to 1 and `.torch-cuda-env/bin/python -m tools.validate_native_birth_projection`):

```bash
# First probes, timeout 420s each, output directories anchor4110 and anchor5680:
--anchor 4110 --seeds 2917 2918 --cells-per-width 128 --birth-max-cells 16000000 --window-seconds 45 --output-dir output/native-birth-projection-2026-10-10/anchor4110
--anchor 5680 --seeds 2917 2918 --cells-per-width 128 --birth-max-cells 16000000 --window-seconds 45 --output-dir output/native-birth-projection-2026-10-10/anchor5680
# Instrumented 4110, timeout 420s:
--anchor 4110 --seeds 2917 2918 --cells-per-width 128 --birth-max-cells 16000000 --window-seconds 45 --output-dir output/native-birth-projection-2026-10-10/anchor4110-final
# Selected full inner windows, timeout 360s:
--anchor 5680 --seeds 2917 2918 --cells-per-width 128 --birth-max-cells 16000000 --window-seconds 120 --window-labels reference_2917 new_2917 reference_replay --output-dir output/native-birth-projection-2026-10-10/anchor5680-final
# Completed inner supplementation, timeout 300s:
--anchor 5680 --seed 2918 --cells-per-width 128 --birth-max-cells 16000000 --window-seconds 75 --output-dir output/native-birth-projection-2026-10-10/anchor5680-seed2918
# Gauss-only postprocessing: invoked once for each of these summaries:
--gauss-from-summary output/native-birth-projection-2026-10-10/anchor4110-final/summary.json
--gauss-from-summary output/native-birth-projection-2026-10-10/anchor5680/summary.json
--gauss-from-summary output/native-birth-projection-2026-10-10/anchor5680-final/summary.json
--gauss-from-summary output/native-birth-projection-2026-10-10/anchor5680-seed2918/summary.json
# Ensemble means, mesh only:
--gauss-from-summary output/native-birth-projection-2026-10-10/anchor4110-final/summary.json --ensemble-summaries
--gauss-from-summary output/native-birth-projection-2026-10-10/anchor5680-final/summary.json --ensemble-summaries output/native-birth-projection-2026-10-10/anchor5680-seed2918/summary.json
# Convenient summary only, no mesh build or source window:
--summary-only output/native-birth-projection-2026-10-10/anchor4110-final/summary.json
```

Seed/range/resource validation occurs before output-directory creation. New output
paths are required and JSON writes are exclusive. No older evidence is overwritten.
The first inner-final Gauss command exposed missing combined NPZ for deliberately
skipped windows; it failed without writing a result. Birth-only archive loading was
fixed, regression-tested, and that postprocessing then completed.

Local regression: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.torch-cuda-env/bin/python -m pytest -q tests/test_validate_native_birth_projection.py`
→ **11 passed**, 0.75 s. `git diff --check` on owned files passed. Tests cover
zero-reference undefined values, known metrics, weighted measures, Hermite Gauss
evaluation, RNG replay, seed rejection before mkdir, birth-only archive loading,
and mean-field versus mean-error ensemble semantics. Main-agent tests are separate.

Changed paths are only `tools/validate_native_birth_projection.py`,
`tests/test_validate_native_birth_projection.py`, this report, and new artifacts
under the specified output directory. No sampler/builder defaults, debug seed,
other reports, report README, data, training or optimizer files were edited.
