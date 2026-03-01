# MVP Technical Snapshot

Last updated: 2026-03-01

## Repository Layout

- `src/circuit_estimation/domain.py`: `Layer`/`Circuit` entities and validation.
- `src/circuit_estimation/generation.py`: random gate/circuit sampling (seedable RNG path).
- `src/circuit_estimation/simulation.py`: batched execution + empirical mean helpers.
- `src/circuit_estimation/estimators.py`: mean/covariance propagation + combined estimator.
- `src/circuit_estimation/scoring.py`: contest params, baseline timing, scoring loop, optional profiler hook.
- `src/circuit_estimation/protocol.py`: serializable request/response DTOs for future RPC integration.
- `src/circuit_estimation/cli.py`: local run entrypoint used by `main.py`.
- `circuit.py`, `estimators.py`, `evaluate.py`: compatibility wrappers that preserve legacy imports.
- `main.py`: local smoke run with `--mode agent|human`, `--detail raw|full`, and optional `--profile`.

## Current Mathematical Representation

- Wires are represented as values in `{-1, +1}` internally, not `{0, 1}`.
- Gates are represented as affine/bilinear forms using coefficients:
  - `const`, `first_coeff`, `second_coeff`, `product_coeff`.

This representation is compact for propagation-style estimators. Participant-facing contracts now explicitly document estimator output shape/depth expectations in `README.md`.

## Circuit Generation and Execution

From `generation.py` + `simulation.py`:

- `random_gates(n)` samples gate coefficients and wire connections.
- `random_circuit(n, d)` builds `d` layers.
- `run_batched(circuit, inputs)` yields layer outputs for a batch.
- `empirical_mean(circuit, trials)` estimates per-layer mean vectors by Monte Carlo.
- validation failures now raise explicit `ValueError` with stable messages.

## Estimators in Repo

From `estimators.py`:

- `mean_propagation(circuit)`: tracks only means, ignores higher-order dependencies.
- `covariance_propagation(circuit)`: tracks means + full covariance matrix per layer.
- `combined_estimator(circuit, budget)`:
  - uses covariance propagation when `budget >= 30 * circuit.n`;
  - otherwise uses mean propagation.

The in-repo estimator implementations are examples for local development only. Future hosted evaluation must treat participant estimators as black box, potentially adversarial/malicious implementations.

## Current Scoring Logic

From `scoring.py` / `README.md`:

- Contest params currently include:
  - `width`
  - `max_depth`
  - `budgets`
  - `time_tolerance`
- For each budget:
  - baseline per-depth runtime is measured by batched sampling (`sampling_baseline_time`).
  - estimator is run once per `(circuit, budget)` and must return one tensor for all layers.
  - if estimator runtime exceeds `(1 + tolerance) * baseline_total_time`, output is zeroed for the full tensor.
  - if estimator runtime is below `(1 - tolerance) * baseline_total_time`, runtime is floored at that bound.
  - MSE is computed vs empirical means; then multiplied by time ratio.

Final score = average adjusted MSE across depths and budgets.

Additional scorer behavior:

- malformed estimator outputs (wrong width, wrong depth, or non-ndarray) raise explicit errors,
- optional profiler callback can emit call-level diagnostics:
  - `wall_time_s`,
  - `cpu_time_s`,
  - `rss_bytes`,
  - `peak_rss_bytes`.
- report path (`score_estimator_report`) can return:
  - raw mode payloads for machine use (`mode agent`),
  - full mode payloads with computed aggregates (`detail full`),
  - run metadata including host/machine/os details.

## Local Smoke Result (Observed)

Command run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run main.py
```

Observed output format:

- default (`mode agent`) emits pretty JSON with `results.final_score` and raw per-budget/per-layer metrics.
- human mode (`--mode human`) emits a Rich multi-section terminal report.

This is a local sanity surface, not a stable benchmark number.

## Important Gaps for Next Iterations

1. Hosted evaluator resource envelope and enforcement policy are still unresolved (`open-questions.md`).
2. No production containerized/sandboxed execution path in this repo yet.
3. Deterministic seeded evaluation flow exists locally but final public seed policy is not frozen.
4. Profiling metrics are local-process diagnostics, not yet equivalent to hosted infra accounting.
5. Participant output contract is clearer locally (strict ndarray + full-depth single-call return) but still needs final public challenge-spec freeze.

## Implication for Future Agents

Treat current code as a strong local starter baseline with explicit interfaces and verification gates, but not yet a production hosted evaluator. Next phase should prioritize hosted isolation semantics and finalized public benchmark policy.
