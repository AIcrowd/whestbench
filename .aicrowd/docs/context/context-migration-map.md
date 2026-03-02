# Context Migration Map

Last updated: 2026-03-01

Goal: ensure critical context is available in `docs/context/` even if `CHALLENGE-CONTEXT.md` is removed.

## Coverage Table

| Legacy context area | Migrated to | Status |
|---|---|---|
| Challenge title, objective, motivation | `challenge-brief.md` | Covered |
| Stakeholders and team roles | `challenge-brief.md` | Covered |
| Problem formulation and circuit-generation intent | `challenge-brief.md` + `mvp-technical-snapshot.md` | Covered |
| Scoring philosophy and compute-budget framing | `challenge-brief.md` + `discussion-rationale-summary.md` + `mvp-technical-snapshot.md` | Covered |
| Runtime tolerance/penalty discussion | `discussion-rationale-summary.md` + `open-questions.md` | Covered |
| Baseline/performance-engineering concerns | `discussion-rationale-summary.md` + `open-questions.md` | Covered |
| AIcrowd integration model and starter-kit trajectory | `discussion-rationale-summary.md` + `aicrowd-starter-kit-patterns.md` | Covered |
| Private beta/test-solve strategy | `discussion-rationale-summary.md` | Covered |
| Timeline and phase strategy (including NeurIPS alignment discussion) | `challenge-brief.md` + `discussion-rationale-summary.md` | Covered |
| Participation scale expectations | `challenge-brief.md` + `discussion-rationale-summary.md` | Covered |
| Prize/legal/tie-handling concerns | `discussion-rationale-summary.md` + `open-questions.md` | Covered |
| Current MVP implementation details and limitations | `mvp-technical-snapshot.md` | Covered |
| AWS SageMaker sandbox/evaluator controls | `sagemaker-evaluator-sandbox-notes.md` | Covered |
| External source links used in research | `research-sources.md` | Covered |

## Post-Migration High-Priority Decisions

These decisions were added after legacy-context migration and are still mandatory context:

| Decision area | Captured in | Status |
|---|---|---|
| Agent-first starter-kit compatibility for participant workflows | `agent-first-starter-kit-requirements.md` + `challenge-brief.md` + `open-questions.md` | Covered |

## Intentional Non-Migration

The following were intentionally not copied as durable product context:

- conversational small-talk,
- meeting scheduling logistics,
- personal availability notes,
- duplicated prose where equivalent structured guidance now exists.

## Deletion Readiness Checklist for `CHALLENGE-CONTEXT.md`

Before deleting the legacy file, verify:

1. Future-agent onboarding references `docs/context/README.md` first.
2. No critical docs still state `CHALLENGE-CONTEXT.md` as required source-of-truth.
3. Any unresolved decisions from legacy context are represented in `open-questions.md`.
4. A maintainer has reviewed this mapping and confirmed no missing high-impact context.

## Maintenance Rule

If new legacy notes are added elsewhere, update this map and link each important topic to a durable doc destination inside `docs/context/`.
