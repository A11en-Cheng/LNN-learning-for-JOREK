# Native birth physical-window statistics, 2026-10-10

Both anchors completed four consecutive physical source windows, eight seeds per window and three arms (refA corrected rejection, refB independent rejection, new64 table). This is **8 physical windows and 192 conditional arm cases**, not 192 independent windows or 32 independent Monte Carlo samples per run. Full native mesh and every one of the million original background slots are retained. No training, optimizer, NN update, field forecast or JOREK data change ran.

Read docs/adr/README.md first, ADR-0018/0019/0020, prior projection tool/reports, public NativeMarkerSource.advance and advance_window, birth/density code, and report/native-ionization-discrepancy-primary-evidence-2026-10-10.md. Historical ADRs are not gates.

## Timeline and shared conditional state

Available current j4111 → births → 40 particle substeps → gather/projection → install recorded j4112 → births → 4 substeps → gather → j4113 → 4 → gather → j4114 → 4 → gather. Inner: j5681 → 81 → gather → j5682 → 16 → gather → j5683 → 16 → gather → j5684 → 16 → gather. Recorded first tstep is 0.5 / 1 JU, subsequent tstep is 0.05 / 0.2 JU. Initial registered poststartup import retains actual startup history; ordinary windows do not invent initial history.

Each source window evaluates its available current record's fields, rawD, shards, t_now and tstep in recorded index order. field_static=True is the supported public source mode, with a saved-history time sampler and actual residual clock. Recorded spi_vol_arr is loaded at each window, including restoring the recorded first-window cache after builder diagnostic preparation. Independent shard density remains a separate kernel using explicit source-volume time/provenance. Ordinary source records fail on missing files, unknown units/index, restarts or a t_now/tstep gap.

The full common reference advance includes common births, captures their RHS, and returns the complete state used next window. Public MarkerCloud.insert reuses free slots, increments lifetimes, preserves native t_birth and never overwrites a live slot. Generated projection values/deltas and particle clocks persist. Next recorded fields and shards are installed after gather; future recorded marker clouds or feedback projections are not installed.

**Conditional limitation:** all arms share this deterministic common reference history. Alternative arms do not persist independently divergent clouds or feed their own feedback into the next window. Recorded fields and shards are prescribed, so this is source-window error, not a coupled NN/AR forecast.

**RNG limitation:** each window resets the diagnostic common reference birth stream to seed 2917, and each conditional arm is seeded once per case (refA 2917..2924; refB base+1,000,003; new64 base+2,000,003). No particle/substep reseeding occurs. This repeated per-window reset defines a diagnostic common trajectory; it is not formal production RNG restoration or production trajectory semantics. Seeds are independent within a window; nearby windows share history and reuse seed labels. They are not pooled as independent MC samples. All actual seed/RNG before, after birth and after window traces are saved.

## Executed windows

| Source j | Substeps | Births per arm | Full common advance s | Entire 24-arm window s | Peak process GiB |
|---|---:|---:|---:|---:|---:|
| 4111 | 40 | 29 | 6.89 | 38.99 | 1.423 |
| 4112 | 4 | 2 | 0.63 | 12.20 | 1.612 |
| 4113 | 4 | 2 | 0.53 | 12.29 | 1.633 |
| 4114 | 4 | 2 | 0.49 | 11.79 | 1.645 |
| 5681 | 81 | 56 | 47.33 | 115.68 | 1.449 |
| 5682 | 16 | 11 | 8.26 | 26.99 | 1.623 |
| 5683 | 16 | 17 | 7.87 | 26.76 | 1.628 |
| 5684 | 16 | 17 | 7.60 | 26.55 | 1.647 |

Both workers ran in parallel with BLAS/OMP/MKL/Torch threads 1. Full-advance budgets are 120 s and 8 GiB per process; chunk size 2048, birth proposal cap 1,000,000, cells per source width 64, maximum table cells 16,000,000. Observed full advances and entire ensemble windows stayed below 120 s; process peaks below 1.65 GiB. The first windows are reported separately because startup gathers are substantially heavier and have more births.

## Conditional decomposition and direct checks

Public source advance reads identical old feedback throughout the window. Marker charge/trajectory responses are independent given that fixed context, deposits add, and the same filtered projection is linear. Capture common newborn RHS during the full common advance; existing_rhs=common_whole_rhs−common_newborn_rhs. Each arm uses project(existing_rhs+arm_newborn_rhs). Compact newborns use the same old feedback and clocks, with no separate newborn feedback-update loop. The original background is never trimmed.

| Direct first-window new64, seed 2917 | Million slots | RHS relative discrepancy | Whole projection relative discrepancy |
|---|---:|---:|---:|
| j4111 | 1000000 | 2.484e-16 | 1.230e-13 |
| j5681 | 1000000 | 7.239e-17 | 1.408e-15 |

Both checks passed actual relative tolerance 1e-10, without epsilon normalization for zero norms. Additional direct first-window reruns compare every saved decomposition channel with the full million-slot advance; per-channel coefficient and physical Gauss errors are in each anchor's direct-channel-spotcheck.json. The initial complete fingerprint must match the original execution before that supplement proceeds.

Supplemental direct advances completed in 6.65 / 43.22 s, with matching initial fingerprints and all million slots. Physical volume-weighted relative discrepancies by channel:

| Channel | j4111 direct vs decomposition | j5681 direct vs decomposition |
|---|---:|---:|
| Eion | 1.238e-14 | 4.015e-16 |
| Radiation | 2.406e-14 | 3.636e-16 |
| Charge squared | 1.297e-14 | 2.133e-16 |
| Charge | 1.490e-14 | 2.193e-16 |
| Number | 1.594e-14 | 2.398e-16 |

Per-channel coefficient discrepancies are also saved: j4111 maximum 1.853e-13, j5681 maximum 2.353e-15. All five physical discrepancies passed the 1e-10 relative tolerance.

Independent small-marker fixture validates public advance/deposition/linear projection under fixed feedback, including positive R, nonzero velocity, full slot preservation and lifetime insertion. Saved-pair audit across all 128 arm/reference comparisons gives maximum all-channel coefficient background-identity residual relative to the arm difference 2.108e-10, and maximum physical Eion error-identity residual 2.850e-13. Coefficient ratios use a small difference denominator and can exceed the direct whole-field ratio; those residuals are not ignored.

## Physical Eion statistics

Full Hermite coefficients are evaluated at native Gauss points with physical volume weights weights*jacobian*R*2pi. Every channel has volume L2, weighted Pearson, signed reference/candidate integrals, absolute/relative integral error and explicit zero/constant-reference flags. Mean, median, p95 and max per-seed errors are separate from errors between ensemble-mean fields.

| Source j | Mean whole L2 new % | Mean whole L2 noise % | Mean newborn L2 new % | Mean whole integral relative error new % | L2 between whole ensemble means new % |
|---|---:|---:|---:|---:|---:|
| 4111 | 10.745 | 9.898 | 52.786 | 2.047 | 4.740 |
| 4112 | 7.329 | 7.618 | 162.099 | 1.666 | 4.147 |
| 4113 | 7.260 | 7.537 | 162.407 | 1.634 | 4.134 |
| 4114 | 7.205 | 7.494 | 162.295 | 1.608 | 4.111 |
| 5681 | 5.408 | 5.594 | 22.311 | 0.175 | 1.134 |
| 5682 | 2.764 | 2.959 | 54.134 | 0.399 | 0.552 |
| 5683 | 3.856 | 4.516 | 41.585 | 0.538 | 1.408 |
| 5684 | 5.642 | 5.702 | 48.025 | 0.692 | 1.679 |

The ordinary v400 windows have only two births. Newborn-relative Eion errors around 162% accompany reference/reference errors around 157%; whole-cloud errors are about 7.2–7.3%. Inner ordinary windows have 11/17/17 births and whole-cloud Eion errors around 2.8–5.6%. These are descriptive finite-sample results; they do not certify equivalence, significance, or an attribution solely to noise.

| Source j | Reference newborn/whole Eion norm range % | Ratio of mean whole new/noise error | Within-window bootstrap percentile range |
|---|---:|---:|---:|
| 4111 | 17.392–23.071 | 1.086 | 0.909–1.272 |
| 4112 | 1.136–10.895 | 0.962 | 0.736–1.140 |
| 4113 | 1.118–10.782 | 0.963 | 0.773–1.134 |
| 4114 | 1.110–10.721 | 0.961 | 0.757–1.126 |
| 5681 | 22.493–26.438 | 0.967 | 0.714–1.278 |
| 5682 | 4.044–6.149 | 0.934 | 0.720–1.158 |
| 5683 | 8.550–10.321 | 0.854 | 0.674–1.192 |
| 5684 | 10.823–13.006 | 0.990 | 0.785–1.289 |

Bootstrap resamples eight seed labels jointly within each window, 2000 resamples with a local Generator independent of event RNG. These are descriptive percentile ranges, not significance tests or equivalence bounds. No cross-window bootstrap or independent-sample pooling is performed.

Eion is signed cumulative ionization-energy change weight*(p_new−p_old)·U*EL_CHG, deposited at OLD positions and divided by gather after summation; there is no extra dt multiplier. The old cloud is not assumed fully ionized or at equilibrium (main's separate imported-state audit measured weighted mean Z≈4.33/6.09). Negative filtered FEM values can be ringing and are not evidence of recombination. Positive/negative field norms and cancellation diagnose the projected shape, not individual physical transition signs.

Per-pair Eion JSON retains newborn norm fraction, signed-integral fraction and denominator flag; old-background norm/integral; best-scale amplitude and shape residual; positive/negative norms and cancellation. Per-arm/common diagnostics retain summed processed per-marker cumulative-energy changes in joules. An independent endpoint telescoping audit was not executed; spatial basis-weighted deposition does not telescope.

## Five whole-cloud channels

Entries are mean per-seed physical L2 percentages, **new64 / refB noise**. Full newborn/whole distributions, all Pearson/integral metrics and ensemble-mean errors are in each j-window summary.

| Source j | Eion | Radiation | Charge squared | Charge | Number |
|---|---:|---:|---:|---:|---:|
| 4111 | 10.745 / 9.898 | 0.553 / 0.500 | 0.047 / 0.044 | 0.162 / 0.149 | 0.843 / 0.785 |
| 4112 | 7.329 / 7.618 | 0.131 / 0.135 | 0.004 / 0.004 | 0.020 / 0.021 | 0.310 / 0.309 |
| 4113 | 7.260 / 7.537 | 0.130 / 0.134 | 0.004 / 0.004 | 0.020 / 0.021 | 0.308 / 0.307 |
| 4114 | 7.205 / 7.494 | 0.129 / 0.133 | 0.004 / 0.004 | 0.020 / 0.020 | 0.306 / 0.305 |
| 5681 | 5.408 / 5.594 | 0.413 / 0.429 | 0.005 / 0.005 | 0.023 / 0.024 | 0.193 / 0.198 |
| 5682 | 2.764 / 2.959 | 0.476 / 0.481 | 0.010 / 0.011 | 0.030 / 0.033 | 0.095 / 0.107 |
| 5683 | 3.856 / 4.516 | 0.512 / 0.547 | 0.010 / 0.012 | 0.031 / 0.038 | 0.101 / 0.129 |
| 5684 | 5.642 / 5.702 | 0.565 / 0.544 | 0.012 / 0.012 | 0.036 / 0.037 | 0.113 / 0.128 |

The recorded-volume policy differs from earlier builder-refreshed 128-grid probes. Their two-seed differences cannot be attributed solely to 64-versus-128 source width. This experiment is the declared n0 table approximation, not production PCG32/MPI replay, a controlled refinement test, or continuous-source qualification.

## Commands, provenance, failures and ownership

Executed concurrently from repository root; all commands set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1:

```bash
.torch-cuda-env/bin/python -m tools.validate_native_birth_windows --anchor 4110 --output-dir output/native-birth-window-statistics-2026-10-10/anchor4110
.torch-cuda-env/bin/python -m tools.validate_native_birth_windows --anchor 5680 --output-dir output/native-birth-window-statistics-2026-10-10/anchor5680
.torch-cuda-env/bin/python -m tools.validate_native_birth_windows --spotcheck-saved output/native-birth-window-statistics-2026-10-10/anchor4110/summary.json
.torch-cuda-env/bin/python -m tools.validate_native_birth_windows --spotcheck-saved output/native-birth-window-statistics-2026-10-10/anchor5680/summary.json
.torch-cuda-env/bin/python -m pytest -q tests/test_validate_native_birth_windows.py
```

Execution snapshot.json and summary.json preserve commands and source SHA256 snapshots. Initial per-window fingerprints include all cloud arrays, old feedback/history, clocks, current/previous fields, rawD and shard arrays. conditional-identity-audit.json records timing/units, j-file and geometry hashes and every pair's residual. Supplemental direct results record their own validator hash because the per-channel supplement was added after the original execution. All 192 NPZs preserve newborn RHS and newborn/whole projected coefficients. Full million-slot states persist in memory; saved fingerprints/history digests/clocks document them without duplicating million-slot snapshots.

No initial physical execution failed. First fixture attempt: 1 failed/3 passed because zero R correctly triggered the cylindrical pusher guard; corrected the fixture to R>0. Focused final original run: 5 passed in 0.93 s; main independently ran these five plus two of its tests, 7 passed. Read-only report extraction attempts exceeded stdout limits and were retried with reduced output; they neither overwrote artifacts nor reran the ensembles.

After adding the supplemental direct-channel entry, the final focused run again passed **5 tests in 0.80 s**. Both supplemental physical checks completed successfully. Owned-file diff whitespace checks passed.

Changed paths owned by this task:

- tools/validate_native_birth_windows.py
- tests/test_validate_native_birth_windows.py
- report/native-birth-window-statistics-2026-10-10.md
- New output/native-birth-window-statistics-2026-10-10 artifacts: anchor snapshots/summaries, eight window summaries, 192 NPZs, conditional identity audit, supplemental direct-channel checks.

Builder/default changes, primary-source sidecars, main report and indices are owned by other agents and were not edited here.
