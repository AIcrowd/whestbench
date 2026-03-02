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
  - Default human CLI mode enables profiling to populate the Textual `Performance` tab.
  - Programmatic callers can still choose opt-in profiling via scorer API parameters.

### 2026-03-01 - Streaming estimator output contract

- **Decision:** Require estimator output to be a streamed iterator of depth rows from `predict(circuit, budget)`.
- **Why:** Matches reference runtime semantics, supports depth-wise runtime enforcement, and keeps participant API minimal.
- **Implementation impact:**
  - Added scoring validation for iterable output, row shape `(width,)`, finite values, and exact depth row count.
  - Updated in-repo reference estimators to `yield` one row per depth.
  - Added regression tests for non-iterable output, wrong row counts, and wrong row shape.

### 2026-03-01 - Dual-mode report rendering contract

- **Decision:** Standardize CLI/report output into:
  - default human dashboard: Textual multi-tab terminal UI with summary-first layout.
  - `--agent-mode`: pretty JSON only for machine consumers.
- **Why:** Support both machine consumers (future UI/agents) and local human debugging without changing scorer internals.
- **Implementation impact:**
  - Added `src/circuit_estimation/textual_dashboard/*` UI package for default human mode.
  - `src/circuit_estimation/reporting.py` remains for agent JSON and static fallback rendering.
  - CLI surface simplified to `--agent-mode` only (removed `--detail`, `--profile`, `--show-diagnostic-plots`).

### 2026-03-01 - Future-agent black-box policy note

- **Decision:** Treat participant estimator implementations as potentially adversarial/malicious black boxes.
- **Why:** Hosted evaluation cannot rely on in-estimator instrumentation or cooperative internal event emission.
- **Implementation impact:**
  - Evaluator diagnostics are external only (depth-yield timing boundaries + call-level resource observations).
  - No assumed estimator-internal instrumentation API for participant code.
  - In-repo estimators documented as examples, not trusted integration contracts.

### 2026-03-01 - Budget-by-depth runtime report semantics

- **Decision:** Keep runtime metrics as observable depth-boundary and budget-level values only.
- **Why:** Participant estimators are black boxes, but streamed `yield` boundaries allow trustworthy depth timing without introspecting estimator internals.
- **Implementation impact:**
  - `results.by_budget_raw` includes depth vectors:
    - `time_budget_by_depth_s`
    - `time_ratio_by_depth_mean`
    - `effective_time_s_by_depth_mean`
    - `timeout_rate_by_depth`
    - `time_floor_rate_by_depth`
  - `results.by_budget_raw` also keeps budget-level scalar runtime metrics:
    - `call_time_ratio_mean`
    - `call_effective_time_s_mean`
    - `timeout_rate`
    - `time_floor_rate`
  - `adjusted_mse` remains budget-level scalar.
  - `results.by_layer_overall` / `results.by_budget_layer_matrix` remain MSE-only aggregates.

### 2026-03-01 - Restore original budget semantics as budget-by-depth envelopes

- **Decision:** Define `budget` exactly as in the original MVP: a sampling trial count that induces a depth-wise runtime envelope.
- **Why:** This aligns participant tuning with the challenge reference implementation and removes ambiguity around budget units.
- **Implementation impact:**
  - Scoring computes `time_budget_by_depth_s` via `sampling_baseline_time(budget, width, depth)`.
  - Timeout/floor is applied per depth index against that envelope with `time_tolerance`.
  - Runtime-adjusted loss uses depth-wise ratios and then aggregates to budget-level `adjusted_mse`.
  - Documentation and terminology standardize on **budget-by-depth** (instead of budget-by-layer wording).
