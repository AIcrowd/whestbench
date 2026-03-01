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
- `main.py`: local smoke run with default Textual human dashboard output and `--agent-mode` JSON mode.

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

From `src/circuit_estimation/estimators.py`:

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
  - `budget` is interpreted as sampling trial count.
  - sampling baseline defines a budget-by-depth runtime envelope:
    - `time_budget_by_depth_s[i]`: cumulative sampling runtime to depth `i`.
  - estimator is run once per `(circuit, budget)` and must return one tensor for all layers.
  - runtime enforcement is applied depth-wise against the envelope:
    - depth `i` is zeroed when wall time exceeds `(1 + tolerance) * time_budget_by_depth_s[i]`.
    - depth `i` runtime is floored to `(1 - tolerance) * time_budget_by_depth_s[i]` when faster than that bound.
    - because estimator API returns one tensor per call, the same observed call wall time is compared at each depth index.
  - MSE is computed vs empirical means (`mse_by_layer` + `mse_mean`).
  - Runtime-adjusted score uses depth-wise runtime ratios, with call-level scalar summaries exposed:
    - `time_budget_by_depth_s`
    - `time_ratio_by_depth_mean`
    - `effective_time_s_by_depth_mean`
    - `call_time_ratio_mean`
    - `call_effective_time_s_mean`
    - `adjusted_mse = mean(mse_by_depth * time_ratio_by_depth_mean)`

Final score = average budget-level `adjusted_mse` across budgets.

Additional scorer behavior:

- malformed estimator outputs (wrong width, wrong depth, or non-ndarray) raise explicit errors,
- optional profiler callback can emit call-level diagnostics:
  - `wall_time_s`,
  - `cpu_time_s`,
  - `rss_bytes`,
  - `peak_rss_bytes`.
- report path (`score_estimator_report`) can return:
  - raw mode payloads for machine use (`--agent-mode`),
  - full mode payloads with computed aggregates for human dashboard tabs, where layer aggregates are MSE-only,
  - run metadata including host/machine/os details.

## Local Smoke Result (Observed)

Command run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run main.py
```

Observed output format:

- default emits a Textual multi-tab terminal dashboard (or static fallback when Textual is unavailable).
- `--agent-mode` emits pretty JSON with `results.final_score` and raw per-budget/per-layer metrics.
  - `by_budget_raw` includes `mse_by_layer`, budget-by-depth runtime arrays, and call-level scalar runtime summaries.

This is a local sanity surface, not a stable benchmark number.

## Important Gaps for Next Iterations

1. Hosted evaluator resource envelope and enforcement policy are still unresolved (`open-questions.md`).
2. No production containerized/sandboxed execution path in this repo yet.
3. Deterministic seeded evaluation flow exists locally but final public seed policy is not frozen.
4. Profiling metrics are local-process diagnostics, not yet equivalent to hosted infra accounting.
5. Participant output contract is clearer locally (strict ndarray + full-depth single-call return) but still needs final public challenge-spec freeze.

## Implication for Future Agents

Treat current code as a strong local starter baseline with explicit interfaces and verification gates, but not yet a production hosted evaluator. Next phase should prioritize hosted isolation semantics and finalized public benchmark policy.
