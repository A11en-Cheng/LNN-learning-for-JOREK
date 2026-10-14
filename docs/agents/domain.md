# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`roadmap_v2.md`**: the current work route (Phase A–E, new architecture). `new_version.md`
  is v1 history; treat it as background only.
- **`docs/adr/README.md` first, then `docs/adr/`**: it separates current decisions
  (**ADR-0018/0019/0020**, new architecture: self-generated source + stencil-conv nine-field
  propagator + staggered closed loop; ADR-0020 = marker source object / reduced-order marker
  chain / augmented state / registered restarts / shared S interface) from v1 history
  (**0001–0017**, interpolation route).
  Cite v1 ADRs only as "(历史)" and never use their quantitative gates, teacher-source
  training semantics or evaluation protocols as prerequisites. If they conflict with
  0018/0019/0020, the latter win (0020 covers roadmap_v2's conflicting passages).
- **`CONTEXT.md`** at the repo root, or
- **`CONTEXT-MAP.md`** at the repo root if it exists: it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
  Sections marked `[v1]` are history; the current output-timing contract is ADR-0019.
  In multi-context repos, also check `src/<context>/docs/adr/` for context-scoped decisions.
- **`report/v2-doc-isolation-and-dirty-data-audit-2026-10-04.md`**: known dirty-data entry
  points (old manifests, checkpoints, teacher-source pairing, legacy code) and what is safe
  to reuse for new work.
- **`docs/jorek-normalization.md`**: code-verified unit/normalization reference (R/Z in
  meters, B in tesla, psi in T·m², Alfvén time `sqrt(mu0*rho0)`, transport-coefficient units
  and explicit-stability limits). Read before any time-scale/transport/stability work.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

Single-context repo (most repos):

```
/
├── CONTEXT.md
├── docs/adr/
│   ├── 0001-event-sourced-orders.md
│   └── 0002-postgres-for-write-model.md
└── src/
```

Multi-context repo (presence of `CONTEXT-MAP.md` at the root):

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← system-wide decisions
└── src/
    ├── ordering/
    │   ├── CONTEXT.md
    │   └── docs/adr/                  ← context-specific decisions
    └── billing/
        ├── CONTEXT.md
        └── docs/adr/
```

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
