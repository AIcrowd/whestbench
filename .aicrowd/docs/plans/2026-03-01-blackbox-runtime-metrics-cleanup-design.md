# Black-Box Runtime Metrics Cleanup Design

Date: 2026-03-01  
Status: Approved for implementation planning

## 1. Goal

Align scoring/reporting contracts with strict black-box evaluator semantics:

- participant estimators are untrusted black boxes,
- runtime/resource observations are only valid at call boundaries,
- no invalid per-layer runtime-derived fields remain in repo outputs.

## 2. Final Decisions

1. Breaking cleanup is allowed now (pre-release).
2. Keep `schema_version` unchanged (`1.0`).
3. Remove invalid per-layer runtime-derived fields directly (no deprecation window).
4. Keep runtime-aware scoring, but only as scalar budget-level terms.
5. `adjusted_mse_by_layer` is removed; only scalar `adjusted_mse` is kept.

## 3. Black-Box Observability Boundary

Allowed as directly observed:

- call-level wall/cpu/rss/peak_rss telemetry,
- call-level timeout/floor enforcement,
- prediction tensors returned by estimator.

Not allowed as direct measurement:

- per-layer estimator runtime,
- per-layer estimator CPU/memory.

If runtime is incorporated into score, it must be through call-level values only.

## 4. Contract Changes

### 4.1 Remove from `results.by_budget_raw`

- `time_ratio_by_layer`
- `adjusted_mse_by_layer`
- `baseline_time_s_by_layer`
- `effective_time_s_by_layer`
- per-layer timeout/floor runtime vectors

### 4.2 Keep/add in `results.by_budget_raw`

- `budget`
- `mse_by_layer` (prediction-derived, valid)
- `mse_mean` (scalar)
- `adjusted_mse` (scalar)
- `call_time_ratio_mean` (scalar)
- `call_effective_time_s_mean` (scalar)
- `timeout_rate` (scalar)
- `time_floor_rate` (scalar)

### 4.3 Final score

- Final score remains lower-is-better.
- Final score is mean of budget-level scalar `adjusted_mse`.

## 5. Detail Full Cleanup

`detail=full` must remove runtime-by-layer outputs and keep only valid prediction-derived layer views.

- Keep: prediction-centric layer summaries (for example `mse_mean_by_layer`).
- Remove: runtime-by-layer aggregates/matrices.

## 6. Estimator Execution Modes

Architecture remains compatible with both participant paths:

- single-circuit estimator call,
- batch-of-circuits estimator call.

For either mode, measurement/enforcement boundary is the call itself.
No internal or per-layer estimator event assumptions are allowed.

## 7. Reporting/UI Implications

- Layer diagnostics must be prediction-focused.
- Runtime summaries must be call/budget-level only.
- Any plots/tables implying measured per-layer runtime are removed or renamed to avoid semantic ambiguity.

## 8. Testing and Audit Requirements

1. Add forbidden-field tests for `raw` and `full` detail payloads.
2. Add scoring tests for scalar budget runtime fields and scalar adjusted score.
3. Update reporting tests to remove dependencies on runtime-by-layer fields.
4. Run repo-wide string audit for removed field names in `src/`, `tests/`, `README.md`, `.aicrowd/docs/context/` and update docs accordingly.

## 9. Documentation Updates

Update:

- `README.md` scoring contract and examples,
- `.aicrowd/docs/context/python-runtime-refactor-decisions.md` decision log,

to explicitly state:

- call-level black-box observability,
- removed invalid per-layer runtime-derived fields,
- scalar budget-level runtime-aware scoring.
