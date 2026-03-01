# Agent/Human Reporting + Black-Box Estimator Design

Date: 2026-03-01  
Status: Approved for planning

## 1. Goal

Improve the starter-kit run UX so:

- default output is machine-friendly and agent-first,
- human mode provides rich educational reporting,
- evaluator assumptions match future participant reality (untrusted/adversarial estimators),
- estimator API is strict and efficient (`np.ndarray` in one pass).

## 2. Core Decisions

1. Two output modes:
   - `agent` mode (default): pretty JSON only.
   - `human` mode: rich terminal multi-section report with plots.
2. Profiling is evaluator-observed from outside estimator calls, not estimator-internal events.
3. Estimator is treated as a black box.
4. Estimator output contract is strict:
   - return `np.ndarray`,
   - shape `(max_depth, width)`,
   - one call per `(circuit, budget)`.
5. `--detail full` adds derived/computed metrics; default agent output prioritizes raw data.

## 3. Why This Changes the Previous Direction

Earlier local plan used iterator/yield boundaries for per-depth profiling events.
This is not suitable for the real contest model where participant estimators are untrusted and potentially adversarial.

The revised design moves to:

- single call-level enforcement/profiling,
- black-box execution semantics,
- strict full-output tensor contract.

## 4. CLI Contract

### Flags

- `--mode agent|human` (default: `agent`)
- `--detail raw|full` (default: `raw`)
- `--profile` (optional, call-level resource diagnostics)

### Output behavior

- `agent` mode:
  - stdout must contain only pretty JSON.
  - no extra prose/log lines.
- `human` mode:
  - rich report with sections and terminal-friendly plots.
  - includes meaningful interpretation text.

## 5. JSON Report Schema (Agent Mode)

Top-level fields:

- `schema_version`
- `mode`
- `detail`
- `run_meta`
  - `run_started_at_utc`
  - `run_finished_at_utc`
  - `run_duration_s`
- `environment`
  - `python_version`
  - `platform`
  - `os_name`
  - `os_release`
  - `os_version`
  - `hostname`
  - `cpu_count_logical`
  - `cpu_count_physical` (when available)
  - `total_memory_bytes` (when available)
- `run_config`
  - `n_circuits`
  - `n_samples`
  - `width`
  - `max_depth`
  - `layer_count`
  - `budgets`
  - `time_tolerance`
  - `profile_enabled`
- `circuits`
  - per-circuit metadata:
    - `circuit_index`
    - `wire_count`
    - `layer_count`
- `results`
  - `final_score`
  - `score_direction` (`lower_is_better`)
  - `by_budget_raw`:
    - `budget`
    - `mse_by_layer`
    - `time_ratio_by_layer`
    - `adjusted_mse_by_layer`
    - `timeout_flag_by_layer`
    - `time_floor_flag_by_layer`
    - `baseline_time_s_by_layer`
    - `effective_time_s_by_layer`
- `notes`

If `--profile` is set, include evaluator-observed call diagnostics:

- `profile_calls` (per `(budget, circuit_index)`):
  - `budget`
  - `circuit_index`
  - `wire_count`
  - `layer_count`
  - `wall_time_s`
  - `cpu_time_s`
  - `rss_bytes`
  - `peak_rss_bytes`

If `--detail full` is set, add:

- `results.by_budget_summary`
- `results.by_layer_overall`
- `results.by_budget_layer_matrix`
- `profile_summary` (if profiling enabled)

## 6. Human Report Content

Human mode should render:

1. Run Context
   - config, run time, machine/OS metadata.
2. Score Summary
   - final score and lower-is-better interpretation.
3. Budget Breakdown
   - table + plot for budget contributions and runtime behavior.
4. Layer/Depth View
   - depth curves and budget-by-depth summary.
5. Profiling Section (when `--profile`)
   - call-level runtime/CPU/memory tables and compact plots.

## 7. Evaluator Enforcement Model

For each `(circuit, budget)`:

1. Start external timing/resource measurement.
2. Call estimator once.
3. Stop measurement after return.
4. Validate output:
   - type is `np.ndarray`,
   - shape exactly `(max_depth, width)`.
5. If invalid, fail with explicit error.

No estimator-internal events are trusted or required.

## 8. Future-Agent Safety Note (Mandatory)

Docs must explicitly state:

- In-repo estimator implementations are educational examples.
- Real evaluation uses participant-submitted estimators that may be adversarial or malicious.
- Evaluator logic must treat estimators as untrusted black boxes.

## 9. Testing Impact

Add/adjust tests for:

- strict ndarray output validation,
- rejection of list/non-ndarray estimator outputs,
- call-level profiling payload generation,
- agent mode JSON-only output,
- human mode rich output sections,
- `--detail full` computed metrics expansion.
