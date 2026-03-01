# Core Runtime + README Documentation Uplift Design

Date: 2026-03-01  
Branch: `docs-onboarding-documentation`  
Status: Approved for planning

## 1. Problem

The repository should function as an educational onboarding resource for new contributors.  
Current docs explain mechanics, but core runtime files and README can still be hard to follow for mixed audiences.

## 2. Audience

Primary audience for this pass:

- Engineers comfortable with Python but new to this specific circuit-estimation problem.
- Research-oriented readers who want clear assumptions, estimator intuition, and evaluation semantics.

## 3. Scope

In scope:

- Root `README.md` rewrite as an onboarding-first introduction.
- Documentation quality pass for core Python runtime in `src/circuit_estimation/`.
- Rich, tutorial-style commentary in `estimators.py`.
- Small, behavior-preserving readability refactors when they improve explainability.

Out of scope:

- Functional feature changes.
- Performance tuning or algorithmic changes.
- Documentation overhaul for `tools/circuit-explorer`.

## 4. Documentation Architecture

The documentation pass will use three layers:

1. README narrative layer:
   - Explain the problem and scoring model conceptually.
   - Provide newcomer reading path through modules.
   - Document quickstart and extension points.

2. Module docstring layer:
   - Strengthen top-level module docstrings to explain role, boundaries, and interaction with the scoring pipeline.

3. API and inline guidance layer:
   - Add clear docstrings to classes/functions, including assumptions and shape contracts.
   - Add moderate inline comments only around non-obvious logic.
   - In `estimators.py`, add tutorial-style walkthrough comments for core mathematical flow.

## 5. README Structure

The new README should be organized as:

1. What this repository teaches.
2. Conceptual problem overview.
3. End-to-end evaluation flow.
4. Codebase map with suggested reading order.
5. Quickstart and CLI modes.
6. Estimator extension contract and pitfalls.
7. Verification commands.

## 6. Style Rules

- Prioritize precise docstrings over heavy inline commentary.
- Keep inline comments concise and high-signal.
- For estimator internals, prefer stepwise narrative comments that connect equations to code blocks.
- Keep behavior unchanged unless a tiny refactor improves readability without affecting semantics.

## 7. Risks and Mitigations

Risk: Comments become noisy or redundant.  
Mitigation: Only annotate non-trivial logic and keep explanatory text close to intent.

Risk: Documentation drifts from actual behavior.  
Mitigation: Tie docstrings to explicit contracts already enforced in tests and runtime validation.

Risk: Readability refactors accidentally alter behavior.  
Mitigation: Limit refactors to local naming/structure improvements and run full relevant tests afterward.

## 8. Acceptance Criteria

- README can be read start-to-finish as an onboarding intro.
- Core runtime modules have complete, consistent, and educational docstrings.
- `estimators.py` contains tutorial-style walkthrough commentary.
- Existing tests pass after documentation and readability edits.

## 9. Handoff

Next step is to produce a concrete implementation plan using the `writing-plans` workflow, then execute it in this worktree branch.
