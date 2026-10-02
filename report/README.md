# Report index

Canonical reports, status as of 2026-09-24.  Domain vocabulary lives in
[CONTEXT.md](../CONTEXT.md); physics conventions live in
[src/physics/definitions.md](../src/physics/definitions.md).

| document | status | role |
|---|---|---|
| `model502-schema-and-source-protocol.md` | active | authoritative nine-field + source data contract |
| `model502-parameter-surface.md` (+ `.json`) | active | machine-checked zero-dimension parameter surface |
| `marker-projection-causality.md` | active | source-code evidence for marker-projection timing and causality |
| `physics-residual-correction.md` | active (physics evidence) | Fortran-derived operator/residual conventions; summary is canonical in `src/physics/definitions.md` §9 |
| `spi-analytic-baseline.md` | active (Phase-2 evidence) | analytic SPI model 3 vs recorded shard ablation; nonlocal-branch gap |
| `implementation-progress.md` | active (append-only) | dated log of the interpolation-centred route |
| `roadmap-review-2026-09-29.md` | active (review) | systematic roadmap review against JOREK source and historical evidence |
| `hyphy-glnn-audit-roadmap.md` | historical (2026-09-04) | audit of the pre-Model-502 repository and its now-superseded roadmap |

## Removed on 2026-09-24

Nine-field training outputs were deleted (`output/model502_baseline/**`: runs,
checkpoints, training plots, elliptic-surrogate checkpoint, audits).  The
nine-field training workflow scripts and the training-design records were moved
to `/tmp/repo_cleanup_20260924/` (temporary; not durable):

- workflow scripts: `train_stage0_segmented.py`, `train_baseline*.py`,
  `build_frame_cache.py`, `plot_stage0_metrics.py`, `diagnose_stage0_physics.py`,
  `eval_stage0_per_run.py`, `audit_*.py`, `scan_lambda_phys.py`,
  `run_source_surrogate_*.py`, `run_stage0_diag.sh`, ...
- records: `implementation-progress.md`, `current-work-summary.md`,
  `docs/adr/0008`, `docs/adr/0009`, the previous `report/README.md`.

Design documents: `README.md`, `CONTEXT.md` and `new_version.md` were rewritten
on 2026-09-24 for the interpolation-centred route
([ADR-0010](../docs/adr/0010-physics-loss-only-at-interpolated-times.md),
[ADR-0011](../docs/adr/0011-fragment-source-coupling-architecture.md));
`config.yaml` still describes the retired nine-field training pipeline and will
be regenerated together with the new implementation.
