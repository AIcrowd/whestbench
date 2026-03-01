# Challenge Brief

Last updated: 2026-03-01

## What This Challenge Is

The challenge is a theoretical computer science benchmark centered on mechanistic estimation for randomly generated Boolean circuits.

Participants are expected to estimate expected wire/output values under uniformly random inputs, under explicit runtime constraints, and outperform naive sampling baselines.

Primary proposal reference: `CHALLENGE-CONTEXT.md` (proposal section and email thread).

## Key People and Roles

- ARC (Jacob Hilton, Paul Christiano): benchmark/problem owners and baseline authors.
- AIcrowd (Sharada Mohanty + team): challenge operations, infra, participant support, evaluator hardening, and leaderboard ops.

## Core Intent (From Discussions)

- The benchmark should reward algorithmic/mechanistic insight, not only brute-force sampling.
- Runtime/resource constraints are part of the objective, not just pass/fail gating.
- The starter kit should be highly accessible while still supporting serious optimization.
- A private beta / red-team phase is expected before public launch.

## Evaluation Philosophy (Current)

Recurring preferences from the thread/transcript:

- Score quality should be compared to sampling baselines across multiple compute budgets.
- Runtime should be integrated into scoring, likely with a tolerance band (discussion mentions +/-10%).
- Difficulty should be stratified (notably by depth), avoiding early benchmark saturation.
- Multiple rounds/phases are plausible, potentially including a later NeurIPS-affiliated phase.

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

## Non-Goals at Current Stage

- No finalized prize/legal policy in this repo yet.
- No finalized public submission interface contract yet.
- No finalized decision between all candidate model domains (Boolean circuits vs potential later MLP/RNN variants) for future rounds.

