# Challenge Brief

Last updated: 2026-03-01

## What This Challenge Is

The challenge is a theoretical computer science benchmark centered on mechanistic estimation for randomly generated Boolean circuits.

Participants are expected to estimate expected wire/output values under uniformly random inputs, under explicit runtime constraints, and outperform naive sampling baselines.

## Key People and Roles

- ARC (Jacob Hilton, Paul Christiano): benchmark/problem owners and baseline authors.
- AIcrowd (Sharada Mohanty + team): challenge operations, infra, participant support, evaluator hardening, and leaderboard ops.

## Proposal Metadata (Migrated)

- Proposal title: Mechanistic Estimation for Boolean Circuits
- Organization: Alignment Research Center (ARC)
- Initial proposal submission window: February 2026
- Reported prize pool target range in proposal discussions: USD 50,000 to USD 100,000

## Core Intent (From Discussions)

- The benchmark should reward algorithmic/mechanistic insight, not only brute-force sampling.
- Runtime/resource constraints are part of the objective, not just pass/fail gating.
- The starter kit should be highly accessible while still supporting serious optimization.
- A private beta / red-team phase is expected before public launch.

## Problem Formulation (Migrated)

- Inputs are randomly generated circuits with width/depth controls.
- Generator concept discussed:
  - start with `n` wires,
  - for each of `L` layers, build new wires from random wire pairs using random 2-input Boolean gates,
  - target is expectation/probability of output wires under uniform random inputs.
- Difficulty control is expected primarily through stratification knobs like depth (and potentially other generator variants).

## Example Parameter Regime Discussed

- Example width mentioned: around `n = 1024`.
- Example depth range mentioned: `L = 1..256`.
- Target error range discussed in early framing: roughly `1e-3` to `1e-6` MSE depending on regime/budget.

## Evaluation Philosophy (Current)

Recurring preferences from the thread/transcript:

- Score quality should be compared to sampling baselines across multiple compute budgets.
- Runtime should be integrated into scoring, likely with a tolerance band (discussion mentions +/-10%).
- Difficulty should be stratified (notably by depth), avoiding early benchmark saturation.
- Multiple rounds/phases are plausible, potentially including a later NeurIPS-affiliated phase.

## Runtime/Resource Philosophy (Migrated)

- Compute budgets should be explicit in evaluation.
- Methods may need to adapt behavior to budget inputs.
- Discussion considered both:
  - soft penalties for going out of budget, and
  - hard constraints at runtime.
- CPU/memory/resource accounting and participant feedback dashboards were considered important.

## Operational Expectations Mentioned

- Anticipated scale discussed: hundreds of participants and thousands of submissions (order of magnitude planning).
- AIcrowd intends to translate this local repo into a hardened evaluator stack and sandboxed execution path.
- Participants should be able to run local approximations of the evaluation loop.

## Timeline Signals (As Discussed)

Concrete dates mentioned in context docs:

- Proposal + initial emails: February 10-12, 2026.
- Meeting transcript date: February 16, 2026.
- ARC expected paper timing discussed: around mid-April 2026.
- Practical sequence discussed:
  - finalize benchmark spec + repository clarity,
  - run invite-only beta/test-solves,
  - then public launch,
  - optionally align or extend with NeurIPS timelines.

## Critical Product Requirements Implied by Context

- A single source-of-truth technical benchmark/evaluator spec document.
- A participant-facing starter kit that is easy to run locally.
- A production evaluator that can safely run potentially adversarial submissions.
- Clear, reproducible scoring behavior between local and hosted evaluation.
- The starter kit must be agent-first for participant workflows (Claude/Codex/Antigravity-style exploration support).

## Non-Goals at Current Stage

- No finalized prize/legal policy in this repo yet.
- No finalized public submission interface contract yet.
- No finalized decision between all candidate model domains (Boolean circuits vs potential later MLP/RNN variants) for future rounds.

## Where to Find Remaining Detail

- Discussion rationale and trade-offs: `discussion-rationale-summary.md`
- Concrete technical status of current code: `mvp-technical-snapshot.md`
- Agent-first starter-kit requirements: `agent-first-starter-kit-requirements.md`
- Open unresolved decisions: `open-questions.md`
