# Python Runtime Refactor Decisions

Last updated: 2026-03-01  
Decision owner: Starter-kit maintainers

## Decision Log

### 2026-03-01 - Package-style runtime restructure

- **Decision:** Move runtime logic into `src/circuit_estimation/*` modules and keep root files as compatibility wrappers.
- **Why:** Clearer extension points, better agent/human discoverability, and future RPC portability.
- **Implementation impact:**
  - Added `domain`, `generation`, `simulation`, `estimators`, `scoring`, `protocol`, `cli` modules.
  - Kept `circuit.py`, `estimators.py`, and `evaluate.py` import surfaces stable.

### 2026-03-01 - Strict local quality gates

- **Decision:** Require lint + format-check + type-check + tests before release claims.
- **Why:** Reduce silent regressions and make local/hosted behavior easier to trust.
- **Implementation impact:**
  - Added `ruff` + `pyright` dev dependencies and project config.
  - Added release command checklist to `README.md`.

### 2026-03-01 - Optional profiling diagnostics in scorer

- **Decision:** Add profiler hook with `wall_time_s`, `cpu_time_s`, `rss_bytes`, and `peak_rss_bytes`.
- **Why:** Align with challenge context calling for runtime/resource diagnostics and performance-vs-budget visibility.
- **Implementation impact:**
  - `score_estimator` accepts optional profiler callback.
  - `main.py --profile` emits structured diagnostics.
  - Profiling remains opt-in to avoid affecting default baseline behavior.

### 2026-03-01 - Explicit scoring failure semantics

- **Decision:** Make estimator shape/depth contract failures explicit (`ValueError`) rather than silently tolerating malformed outputs.
- **Why:** Faster debugging for participants and cleaner evaluator boundaries for future RPC/hosted execution.
- **Implementation impact:**
  - Added scoring validation for output width and expected depth.
  - Added regression tests for short-output failure mode.
