# Example Estimators Full Migration Design

Date: 2026-03-01
Status: Approved

## 1. Goal

Fully migrate starter estimator implementations to the new class-based estimator API and remove function-based estimator compatibility paths.

Primary user-facing outcome:

- participants learn from file-based class examples in `examples/estimators/`
- participants run and package estimators through the new CLI/harness surface
- core runtime no longer presents function-style estimator APIs as primary integration points

## 2. Scope and Non-Goals

In scope:

- move starter estimator examples to `examples/estimators/`
- represent examples as `Estimator(BaseEstimator)` classes
- remove old function-based estimator API from participant path (no wrappers)
- switch default local run flow to file-entrypoint runner path
- migrate estimator tests/docs to class/file-based expectations

Out of scope:

- adding new estimator algorithms
- changing scoring math semantics
- changing submission artifact schema beyond entrypoint usage already approved

## 3. Architecture

### 3.1 Example estimator location

Chosen option: `examples/estimators/`.

Planned files:

- `examples/estimators/mean_propagation.py`
- `examples/estimators/covariance_propagation.py`
- `examples/estimators/combined_estimator.py`

Each file exports `Estimator(BaseEstimator)` with `predict(circuit, budget)`.

### 3.2 Core runtime relationship

- core runner/loader paths remain under `src/circuit_estimation/*`
- example estimators are treated as participant-style files loaded via `EstimatorEntrypoint`
- default local evaluation path should use example entrypoint loading instead of direct function callable imports

## 4. Data Flow

1. CLI resolves estimator path (defaulting to combined example file for legacy smoke entrypoint behavior).
2. Loader imports estimator module and resolves class (`Estimator` by default).
3. Runner executes `setup` once and `predict` per call.
4. Scorer consumes `PredictOutcome` and computes standard metrics.
5. Report rendering remains unchanged by estimator form.

## 5. Error Handling

- missing/invalid example estimator files -> structured validation/load errors
- class resolution ambiguity -> explicit loader error with override guidance
- bad output shape/type from examples -> protocol/runtime error rates in score report
- all failures surfaced in JSON-only agent mode and concise human mode with optional traceback via `--debug`

## 6. Testing Strategy

Estimator migration tests will move from function imports to file/class-based behavior checks:

- loader-based class resolution tests for each example file
- numerical correctness tests that call class `predict` outputs
- budget-switch behavior tests for combined example estimator class
- CLI and default-run tests that confirm file-entrypoint runner execution
- removal checks ensuring old function estimator API is not used by current integration paths

## 7. Docs Strategy

- README and context docs point participants to `examples/estimators/` as starter references
- docs explicitly state full class-based estimator contract (`BaseEstimator`, `predict`, optional `setup/predict_batch`)
- runtime decision log records complete removal of function-based estimator path from participant workflow

## 8. Migration Sequence

1. add class-based example estimators under `examples/estimators/`
2. update defaults and runtime call sites to use entrypoint loading (runner path)
3. rewrite estimator tests to class/file-based checks
4. remove/retire old function-based estimator module usage
5. update docs and finalize verification gates

## 9. Acceptance Criteria

- `examples/estimators/` contains class-based starter estimators
- no participant-facing path depends on function-based estimator API
- default local run can execute combined example via entrypoint
- estimator/CLI/scoring tests pass with class-based examples
- docs consistently describe class-based file-entrypoint workflow
