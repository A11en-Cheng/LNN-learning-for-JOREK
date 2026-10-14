## Agent skills

### Issue tracker

Issues live in the GitHub repo `A11en-Cheng/LNN-learning-for-JOREK`; use the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: `CONTEXT.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.

### Roadmap & doc isolation (new architecture)

New architecture work follows `roadmap_v2.md` + `docs/adr/0018`/`0019`/`0020`. Read
`docs/adr/README.md` first: ADR-0001..0017 are v1 history — cite them as "(历史)", never as
gates or precedents. ADR-0020 is the authoritative overlay for roadmap_v2's conflicting
passages (marker source object, reduced marker chain, registered restarts, shared S
interface); its conformance map is
`report/conformance-roadmap-v2-2026-10-07.md`. Dirty-data entry points are audited in
`report/v2-doc-isolation-and-dirty-data-audit-2026-10-04.md`.
