# Open Questions and Decision Log

Last updated: 2026-03-01

This file lists unresolved decisions that block a production starter-kit + evaluator pipeline.

## Resolved Decisions

- 2026-03-01: Final starter kit must be explicitly compatible with agent-led exploration workflows (including Claude/Codex/Antigravity-style usage). See `agent-first-starter-kit-requirements.md` for implementation checklist.

## A. Benchmark and Scoring Spec

1. What is the final participant output contract?
   - Per-depth vector estimates?
   - Final-layer only?
   - Additional uncertainty outputs?

2. What is the final score formula?
   - Current toy code applies an adjusted MSE with runtime factor.
   - Need exact formal spec including tie-breaking and statistical confidence policy.

3. How should runtime tolerance be handled?
   - Keep current symmetric tolerance band (+/-10%)?
   - Use hard cap only?
   - Penalize overages continuously vs zero-out outputs?

4. What are official depth/width/budget distributions for public rounds?

## B. Submission Interface and Starter Kit

1. What is the final AIcrowd submission modality for this challenge?
   - code package, prediction artifact, or hybrid?

2. What exact files/entrypoints are required from participants?

3. Should the starter kit include multiple baseline tiers?
   - simple educational baseline,
   - optimized baseline,
   - optional native-accelerated baseline.

4. What approach classes are explicitly allowed/disallowed?
   - purely per-instance reasoning,
   - offline-trained predictors,
   - hybrid approaches with precomputed artifacts.

## C. Evaluator Runtime and Infrastructure

1. Should evaluator jobs use SageMaker Training Jobs or Processing Jobs?

2. What is the canonical per-submission resource envelope?
   - instance type(s), vCPU/memory limits,
   - maximum job runtime,
   - concurrency caps.

3. What metrics are exposed to participants after each submission?
   - per-budget runtime,
   - error components,
   - resource usage diagnostics.

4. What is the anti-gaming policy?
   - detection of hardware-specific overfitting,
   - randomization/holdout rotation policy,
   - repeated-submission probing limits.

5. What red-teaming protocol is required before public launch?
   - invite-only cohort size,
   - number of beta rounds,
   - promotion criteria from beta to public release.

## D. Governance, Legal, and Competition Design

1. Prize structure details still pending:
   - main leaderboard prizes,
   - algorithmic contribution prizes,
   - potential round/intermediate awards.

2. Rules for ties / statistical indistinguishability need explicit legal-safe wording.

3. Scope of organizer support/dev-rel during first weeks of launch needs concrete staffing commitment.

## Decision Logging Convention

When a question above is resolved, append:

- Date
- Decision owner
- Final decision
- Implementation impact (files/services/processes)
