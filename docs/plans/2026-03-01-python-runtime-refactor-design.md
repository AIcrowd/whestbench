# Python Runtime Refactor Design

Date: 2026-03-01  
Status: Approved for planning  
Scope: Python runtime + tests + README; exclude `CHALLENGE-CONTEXT.md`

## 1. Problem and Goals

Refactor the current Python implementation into a release-ready starter-kit baseline with:

- clear, layered documentation for mixed audiences (quick orientation + deep technical detail),
- cleaner architecture and explicit interfaces,
- strict quality gates (lint, format-check, type-check, tests),
- targeted behavior fixes where current behavior is ambiguous or inconsistent,
- richer and more explanatory tests.

Non-goal:

- Do not update `CHALLENGE-CONTEXT.md`.

## 2. Chosen Approach

Chosen approach: **Full package-style restructure**.

Rationale:

- Improves long-term maintainability and participant discoverability.
- Creates explicit boundaries needed for future RPC exposure.
- Supports agent-first starter-kit requirements with clearer extension points and contracts.

## 3. Architecture

### Proposed structure

- `src/circuit_estimation/domain.py`
  - Core entities (`Layer`, `Circuit`) and invariants.
- `src/circuit_estimation/generation.py`
  - Random gate/circuit generation and seed policy.
- `src/circuit_estimation/simulation.py`
  - Batched execution and empirical statistics.
- `src/circuit_estimation/estimators.py`
  - Mean propagation, covariance propagation, combined estimator.
- `src/circuit_estimation/scoring.py`
  - Contest params, profiling utilities, baseline timing, scoring.
- `src/circuit_estimation/cli.py`
  - Thin local entrypoint and error boundary.
- `main.py`
  - Compatibility wrapper calling package CLI.
- `tests/`
  - Mirrors module structure with enriched comments and coverage.

### Documentation style in every module

Each module should include:

1. What this module does.
2. Problem framing.
3. Constraints and gotchas.
4. Extension points.

## 4. Public Interfaces and Data Flow

### Participant-facing estimator interface

- `EstimatorFn = Callable[[Circuit, int], Iterator[NDArray[np.float32]]]`

### Evaluator entrypoint

- `score_estimator(estimator: EstimatorFn, n_circuits: int, n_samples: int, contest_params: ContestParams) -> float`

### Data flow

1. `generation.py` creates circuits from explicit RNG/seed policy.
2. `simulation.py` executes circuits and computes empirical means.
3. `estimators.py` emits per-layer estimates under budget.
4. `scoring.py` profiles runtime against baseline and computes adjusted MSE.
5. `cli.py` wires configuration and output for local workflows.

### Behavioral-fix policy

Targeted behavior changes are allowed when they improve correctness or remove ambiguity.  
Every change must be justified with tests that document intended behavior.

## 5. Future RPC Compatibility (Design-Only)

No RPC implementation now, but design should preserve an easy migration path by keeping core logic transport-agnostic.

Planned RPC-readiness constraints:

1. Versioned, serializable contracts for core entities.
2. A dedicated schema module (e.g., `protocol.py`) for request/response types.
3. Deterministic seed and error semantics documented for stable remote behavior.
4. IO/process concerns confined to `cli.py` (and future adapters), not core modules.

## 6. Validation, Errors, and Constraints

### Validation

- `Layer`/`Circuit` shape and consistency checks.
- Index bounds checks and generator invariants.
- `ContestParams` range/shape checks.
- Estimator output shape/depth checks in scoring.

### Error handling

- Raise clear typed errors (`ValueError`/`TypeError`) with stable messages.
- Keep numerical kernels pure and side-effect free.
- Provide structured failure boundary in CLI for future RPC mapping.

### Constraint docs

Document numerical and scoring gotchas explicitly (dtype behavior, clipping semantics, timeout/floor mechanics, baseline comparability assumptions).

## 7. Testing Strategy

### Test modules

- `tests/test_domain.py`
  - Entity invariants, validation errors, serialization round-trips.
- `tests/test_generation.py`
  - Reproducibility, index constraints, binary-output gate properties.
- `tests/test_simulation.py`
  - Layer execution correctness and controlled-circuit checks.
- `tests/test_estimators.py`
  - Exactness for linear cases, covariance identities, clipping invariants, budget mode switch.
- `tests/test_scoring.py`
  - Timeout zeroing, minimum runtime floor, shape mismatch failures, deterministic scoring.
- `tests/test_cli.py`
  - Entry-point smoke behavior and structured failure semantics.

### Test writing style

- Add concise comments explaining intent and challenge relevance.
- Emphasize regression tests for ambiguous areas and behavior fixes.
- Keep deterministic defaults; isolate stochastic checks.

## 8. Tooling and Release Gates

Add strict project checks and document one-command workflows:

1. `ruff check .`
2. `ruff format --check .`
3. `pyright`
4. `pytest` (quick + exhaustive paths)

`README.md` must include:

- start-here commands,
- contract and invariants,
- failure semantics,
- extension points for custom estimators,
- release verification steps.

## 9. Scope Boundaries

In-scope:

- Python runtime package refactor,
- tests refactor and enrichment,
- README update,
- design/decision notes in `docs/`.

Out-of-scope:

- `CHALLENGE-CONTEXT.md` updates,
- production RPC server implementation.

## 10. Accepted Decisions

1. Full package-style restructure selected over conservative or balanced alternatives.
2. `domain.py` retained (instead of `core.py`) for entity + invariant boundary naming.
3. Mixed-audience documentation standard adopted (layered explanations).
4. Targeted behavior fixes permitted when backed by tests and rationale.
5. Strict release gates required (lint + format-check + type-check + tests).
6. Refactor should remain compatible with future RPC exposure via transport-agnostic core modules.
