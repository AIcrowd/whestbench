# Context Pack for Future Agents

Last updated: 2026-03-01

This folder is a high-signal handoff for future agents working on this repository.
It is intended to remain self-sufficient even if `CHALLENGE-CONTEXT.md` is deleted.

## Start Here

1. Read [`challenge-brief.md`](./challenge-brief.md) to understand intent, stakeholders, and evaluation philosophy.
2. Read [`mvp-technical-snapshot.md`](./mvp-technical-snapshot.md) to understand current code behavior and limitations.
3. Read [`agent-first-starter-kit-requirements.md`](./agent-first-starter-kit-requirements.md) for mandatory participant-agent compatibility constraints.
4. Read [`discussion-rationale-summary.md`](./discussion-rationale-summary.md) for product/ops rationale migrated from early discussions.
5. Read [`aicrowd-starter-kit-patterns.md`](./aicrowd-starter-kit-patterns.md) to understand likely participant-facing repository patterns.
6. Read [`sagemaker-evaluator-sandbox-notes.md`](./sagemaker-evaluator-sandbox-notes.md) before making infrastructure decisions.
7. Check [`open-questions.md`](./open-questions.md) before implementing new behavior.
8. Use [`context-migration-map.md`](./context-migration-map.md) to verify legacy-context coverage.

## Scope of This Pack

- Captures challenge intent, rationale, and implementation context needed for ongoing work.
- Adds external research links (primary sources only) for AIcrowd workflow and SageMaker sandboxing/security controls.
- Does not prescribe final product decisions where the source material is still ambiguous.

## Canonical Local Sources

- `docs/context/*.md` (this folder; primary context for future work)
- `README.md` (current toy contest mechanics)
- `src/circuit_estimation/*.py`, `main.py` (current implementation)

## Legacy Source Status

- `CHALLENGE-CONTEXT.md` is treated as a migration source, not a required dependency.
- Future agents should not rely on it being present.

## Update Protocol

When decisions are made, update these files in place and include:

- Decision date.
- Who made the decision.
- What changed.
- Which open question it resolves.
