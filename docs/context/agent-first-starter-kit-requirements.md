# Agent-First Starter Kit Requirements

Last updated: 2026-03-01
Decision owner: Challenge organizers
Decision status: Approved requirement

## Decision

The final starter kit must be intentionally compatible with agent-led exploration workflows, including popular coding-agent systems such as Claude, Codex, and Antigravity.

This is a design requirement for the participant experience, not an optional enhancement.

## Why This Matters

- Participants increasingly use agents to navigate unfamiliar codebases quickly.
- Clear machine-readable contracts and deterministic local workflows reduce unproductive agent loops.
- Better agent compatibility increases accessibility and accelerates exploration of the algorithmic solution space.

## Starter Kit Requirements

### 1. Single authoritative participant contract

Provide one canonical spec document that defines:

- expected submission inputs,
- required outputs/artifacts,
- runtime and resource constraints,
- failure semantics,
- scoring interpretation.

The same contract must be reflected consistently in code, docs, and local evaluation scripts.

### 2. Deterministic local evaluation path

Provide one-command local runs for:

- baseline execution,
- local scoring,
- tests/smoke checks.

Commands must have stable output shapes and explicit exit codes so agents can reason over results.

### 3. Machine-readable interfaces and schemas

Where possible, include explicit schemas for:

- config files,
- submission metadata,
- evaluator output/metrics payloads.

Avoid implicit conventions buried only in prose.

### 4. Agent-oriented repository affordances

Include and maintain:

- high-signal `README.md` with “start here” and minimal commands,
- `AGENTS.md` (or equivalent) with repo-specific guardrails and workflow hints,
- clear path-level ownership of key files (what to edit for what outcome),
- concise debugging and verification instructions.

### 5. Reproducibility and observability

Expose diagnostics that agents can consume during iterative development:

- per-budget metrics,
- runtime timings,
- structured failure reasons,
- deterministic seed policy (where applicable).

### 6. No hidden operational assumptions

Document all environment assumptions explicitly:

- Python/runtime/toolchain versions,
- required env vars,
- compute assumptions,
- what differs between local and hosted evaluation.

### 7. Safe extensibility for rapid exploration

Starter kit should support iterative experimentation without fragile rewiring:

- modular baseline code,
- clear extension points for custom estimators,
- stable evaluator call interface,
- examples showing how to plug in a new method.

## Acceptance Checklist for Release Readiness

Before declaring a starter-kit version “participant-ready,” verify:

1. A coding agent can run baseline + local evaluator from a clean clone using only documented commands.
2. Agent can identify where to implement a custom estimator without reading private/internal docs.
3. Agent can detect and interpret failure causes from structured logs/messages.
4. Agent can map local score outputs to hosted-score semantics from public docs.
5. A human reviewer can confirm no hidden steps are required beyond documented setup.

## Implementation Impact

This decision impacts:

- starter-kit file layout and docs,
- evaluator I/O design,
- baseline interface stability,
- local tooling/scripts and diagnostics,
- release checklist and QA process.

