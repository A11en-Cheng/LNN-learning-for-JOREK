# Native autonomous pilot: data and cost audit (2026-10-09)

## Result and limits

Six two-event windows have actual complete marker initializations, original zero
restart feedback, full cubic nine-field records/raw deltas, and correctly timed
five-channel source labels. Their dynamic file/index support is disjoint within
each run. This is data feasibility, **not autonomous model/training qualification**.
No training, optimizer operation, JOREK/raw-data edits, or physical cloud cropping
was performed. Only the four user-assigned audit/test/config/report files changed.

Execution cost status is **required_measured_cost**, not a blocking verdict. Main reports about
447 seconds for v400's first 40-substep event. Scaling only by initial live count
times substeps gives inner's first event approximately 4,264 seconds (71 minutes),
and the six windows approximately 8.63 hours. These are rough workload proxies,
**not measured inner runtime**; births, field solve, nonlocal refresh, backprop and
different vectorization may substantially change them. These historical scalar
proxies do not establish whether the new `batched_v1` path fits a small pilot.
Main reports a historical real paired-source peak of approximately 3.12 GB;
the previous 1,536 MiB proposal was inadequate and is superseded by **8,192 MiB
(8,589,934,592 bytes)**. Resource budgets (900 seconds/window, 7,200 seconds total,
8,192 MiB process RSS) are **recommendations, not user-imposed hard limits**.
Main must measure autonomous full-cloud smoke wall time and peak RSS, then record
the adopted execution limits. Once adopted, limits remain explicit fail-without-
commit guards; never change physical step count/particle population to fit them.

## Actual selected data

Run base: `data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400`, with
`_injInner` for inner. Each selected marker snapshot has 1,000,000 slots;
each poststartup field has 846 shards. Marker arrays total 172,000,000 bytes
per cloud; two copies need 344,000,000 bytes before solver/Torch/scratch space.
All six passed the public `read_marker_snapshot` validation, including live
charge probability normalization and required marker fields. No empty cloud was
manufactured and no slots were discarded.

| Run / split | Imported anchor | Full dynamic support | Live | First + ordinary substeps | Initial-live × total-substeps proxy |
|---|---:|---|---:|---:|---:|
| v400 train | 4110 | 4110–4113 | 11,168 | 40 + 4 | 491,392 |
| v400 val | 5870 | 5870–5873 | 57,017 | 81 + 8 | 5,074,513 |
| v400 test | 6870 | 6870–6873 | 80,657 | 40 + 8 | 3,871,536 |
| inner train | 5680 | 5680–5683 | 52,604 | 81 + 16 | 5,102,588 |
| inner val | 6440 | 6440–6443 | 72,075 | 81 + 16 | 6,991,275 |
| inner test | 7220 | 7220–7223 | 107,040 | 81 + 8 | 9,526,560 |

Sum proxy is 31,057,864 particle steps. It holds initial live count fixed solely
for cost accounting, not physics; it is neither an exact work count nor a rigorous
lower bound when deaths occur. Smoke repeats and backward are not included.
Registered candidate scan found 8/9 v400 and 5/6 inner anchors with marker files;
earliest v400 2701 / inner 2703 lack `part02701.h5` / `part02703.h5` despite zero
projections. These are rejected initialization candidates. Nonselected anchors
were checked for existence/live counts, not certified by full reader validation.

Support includes imported field A, poststartup full field A+1, complete marker
A, original feedback A, and all future field/source labels needed for two events.
Shared namelist, geometry/constants and restart registry are allowed static
inputs; these are separate registered restarts, **not statistically independent
plasma trajectories**. Actual runner provenance must stay within this read set;
causal replay from another split or fit-on-all-data normalization invalidates the
split claim. Saved histories embedded in each initial record must not become a
cross-split training dataset.

## Clock and label contract

Both runs: `t_norm = 4.085775225951833e-7 s/JU`, nominal particle step `5e-9 s`.
Read imported `tstep` only to reconstruct the fixed particle step once:
`imported_dt / floor(imported_dt / nominal_dt)`. It is NOT the next field-control
plan. Imported .5 JU produces `5.107219032439791e-9 s`; imported 1 JU produces
`5.044166945619547e-9 s`. Subsequent windows retain residual particle time.
The tool uses the existing public `particle_window_schedule`, including its
existing near-integer handling; it does not change production scheduling.

Initial marker times (SI), verified against original projection timestamps:

| Anchor | Marker / original zero-projection time |
|---:|---:|
| 4110 | 0.003514379560602572 |
| 5870 | 0.003691497916647701 |
| 6870 | 0.0037936422972965887 |
| 5680 | 0.004113558497488448 |
| 6440 | 0.004231637401518401 |
| 7220 | 0.004357887856000258 |

Next controls are differences of consecutive field `t_now * t_norm`, not an
assumption from filenames, and are recorded-schedule metadata. Nominal JU controls
are .05, .1, .1, .2, .2, .1 respectively for the rows above; the JSON audit emits
the actual binary SI differences for each endpoint pair.

For event **4111→4112**, source label is **P4111** (timestamp matches field4111),
and field label is U4112. For **5681→5682**, source label is **P5681**, field label
U5682. The following events pair P4112/U4113 and P5682/U5683. In general source
label index is the field-event **start**, not its endpoint index. At restart this
is a nonzero physical particle gather despite equal imported/poststartup field
times. Audit output separately reports source logical endpoint, actual particle
start/end, integer substeps, gather duration and residual; it does not label the
zero-width duplicate field timestamp as a zero physical gather.

Initialization uses actual **zero P_A**, at original marker time, to rebuild
poststartup nonlocal cache. Nonzero P_A+1 belongs to the subsequent source event
and is a supervision label, NOT initialization aux. After initialization, only
predicted source projection feeds the next event. All nine next fields, including
psi and any ES computation based on it, must remain predicted: no teacher branch,
future recorded ES, or truth reset. Saved initial ES may initialize only once.

## Evidence and source scope

- `data/restart_events.json` registers event A+1 for all selected A.
- Raw HDF `jorek%06d`: `t_now`, `t_norm`, `tstep`, complete `values/deltas` shape
  `(9,4,1,4125)`, `n_spi_tot=846`; original `projections%06d` shape
  `(5,4,1,4125)` and SI `t_now`; `part%05d`: complete `groups/001`, SI `/time`.
- `src/data/marker_checkpoint.py`: public marker reader and restart-clock contract.
- Read-only JOREK source HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`,
  `particles/examples/marker_spi_source.f90:83–86`: imported fixed-step adjustment;
  `:350–355`: residual, integer substeps; `:423–443`: birth gather duration and
  projection gather time. This identifies inspected source, not a proven match
  to the production dirty executable.
- ADR-0020 is current authority for marker source/restart semantics; no v1 gate.

The audit reads future projections only for validation/labels. It does not execute
the learned chain. Data metadata include path/byte count, not an invented full-file
content digest. Runtime source/model fingerprints and all initial geometry/ES
validation remain the main runner's responsibility.

## Fixed proposal and actual tests

`config/native-autonomous-training-pilot.json`: seed 1, teacher false, two complete
events/window, two no-update smoke windows first, then at most one update/run
(two total), one val/test evaluation per run, no retries/hyperparameter search.
One million slots/live ceiling, chunk 2048 and at most 81 physical substeps/event
are resource guards, not reductions. An external deadline is still required;
callback-level elapsed-time checks cannot interrupt a slow callback.

Public audit seam tests followed TDD: missing module → first timing/label test
green; wrong-but-mutually-consistent marker/zero-aux time failed to reject →
restart-gap validation green; cross-split shared initialization failed → support
validation green. Additional negative cases cover missing marker, nonzero initial
feedback, zero live probability vector, future-misaligned source label, and
unregistered anchor. Used the codebase-design skill to keep this behind the two
public audit interfaces and TDD to verify file/time behavior without mocks.

Commands actually run:

```text
.torch-cuda-env/bin/python -m pytest -q tests/test_native_training_pilot_audit.py
8 passed in 0.12s
.torch-cuda-env/bin/python -m tools.audit_native_training_pilot --config config/native-autonomous-training-pilot.json
exit 0; data_feasible=true; full_support_disjoint=true; failures=[]; six windows
```

System `python` is unavailable; system `python3` lacks pytest. The repository's
`.torch-cuda-env/bin/python` was used for all reported tests and final audit.

Remaining work: main-owned full-cloud throughput optimization, two-run autonomous
smokes, intended-gradient checks, actual bounded training, checkpoint verification
and evaluation. No training result or autonomous capability is claimed here.

Batch entry points and updated execution status:
`NativeSourceMesh.points(e, st, values)` in `src/models/native_coupling.py:158`
already computes full cubic values/gradients for a bounded batch. At the initial
audit, the scalar source sampling loop at `:283–305` called `mesh.point` twice per
live particle; these line references describe that inspected revision. Main now
reports an optional `batched_v1` point/deposit/locate path with public source tests
passing and autonomous smoke about to run. This audit did not independently run
that smoke and makes no throughput claim for it. No source code was changed by
this audit. Measured full-cloud cost, not the scalar proxy, must inform the pilot.
