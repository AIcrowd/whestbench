# Python Runtime Refactor Decisions

Last updated: 2026-03-01  
Decision owner: Starter-kit maintainers

## Decision Log

### 2026-03-01 - Package-style runtime restructure

- **Decision:** Move runtime logic into `src/circuit_estimation/*` modules.
- **Why:** Clearer extension points, better agent/human discoverability, and future RPC portability.
- **Implementation impact:**
  - Added `domain`, `generation`, `simulation`, `estimators`, `scoring`, `protocol`, `cli` modules.

### 2026-03-01 - Remove legacy root module wrappers

- **Decision:** Delete `circuit.py`, `estimators.py`, and `evaluate.py`.
- **Why:** Ensure only the package implementation is authoritative and avoid dual import paths.
- **Implementation impact:**
  - Migrated tests to `circuit_estimation.*` imports.
  - Updated static analysis includes to remove deleted modules.
  - Added regression test to ensure legacy root modules are not reintroduced.

### 2026-03-01 - Strict local quality gates

- **Decision:** Require lint + format-check + type-check + tests before release claims.
- **Why:** Reduce silent regressions and make local/hosted behavior easier to trust.
- **Implementation impact:**
  - Added `ruff` + `pyright` dev dependencies and project config.
  - Added release command checklist to `README.md`.

### 2026-03-01 - Optional black-box call-level profiling diagnostics

- **Decision:** Add profiler hook with `wall_time_s`, `cpu_time_s`, `rss_bytes`, and `peak_rss_bytes`.
- **Why:** Align with challenge context calling for runtime/resource diagnostics and performance-vs-budget visibility.
- **Implementation impact:**
  - Profiling is collected at estimator call boundaries (one `(circuit, budget)` invocation), not per-depth internals.
  - `score_estimator_report(..., profile=True)` includes `profile_calls`, and `detail full` includes `profile_summary`.
  - `main.py --profile` surfaces structured diagnostics in both agent and human reporting modes.
  - Profiling remains opt-in to avoid affecting default baseline behavior.

### 2026-03-01 - Strict estimator output tensor contract

- **Decision:** Require estimator output to be a single-pass `ndarray` of shape `(max_depth, width)`.
- **Why:** Better RPC compatibility, simpler evaluator contracts, and explicit black-box boundaries for untrusted submissions.
- **Implementation impact:**
  - Added scoring validation for `ndarray` type, rank, width, and depth.
  - Updated in-repo reference estimators to return full-depth tensors in one call.
  - Added regression tests for non-ndarray and wrong-shape failures.

### 2026-03-01 - Dual-mode report rendering contract

- **Decision:** Standardize CLI/report output into:
  - default human dashboard: Rich multi-section terminal report with trend plots.
  - `--agent-mode`: pretty JSON only for machine consumers.
- **Why:** Support both machine consumers (future UI/agents) and local human debugging without changing scorer internals.
- **Implementation impact:**
  - Added `src/circuit_estimation/reporting.py` renderers.
  - Added CLI flags: `--agent-mode`, `--detail`, `--profile`.
  - Added `detail full` aggregate sections for downstream consumers.

### 2026-03-01 - Future-agent black-box policy note

- **Decision:** Treat participant estimator implementations as potentially adversarial/malicious black boxes.
- **Why:** Hosted evaluation cannot rely on in-estimator instrumentation or cooperative internal event emission.
- **Implementation impact:**
  - Evaluator diagnostics are external only (call-level timing/resource observations).
  - No assumed per-layer internal event API for participant estimators.
  - In-repo estimators documented as examples, not trusted integration contracts.

### 2026-03-01 - Remove synthetic per-layer runtime diagnostics from report schema

- **Decision:** Drop per-layer runtime-derived report fields (`time_ratio_by_layer`, `effective_time_s_by_layer`, `adjusted_mse_by_layer`, and related aggregates).
- **Why:** Estimator implementations are black boxes with call-level timing only; per-layer runtime attribution is not observable or trustworthy for adversarial submissions.
- **Implementation impact:**
  - `results.by_budget_raw` now carries call-level scalar runtime metrics:
    - `call_time_ratio_mean`
    - `call_effective_time_s_mean`
    - `timeout_rate`
    - `time_floor_rate`
  - `adjusted_mse` is budget-level scalar only (no per-layer adjusted fields).
  - `results.by_layer_overall` / `results.by_budget_layer_matrix` now include only MSE-derived layer aggregates.
  - Human reporting layer diagnostics are MSE-only; runtime visuals are budget-level and profile-call-level only.
