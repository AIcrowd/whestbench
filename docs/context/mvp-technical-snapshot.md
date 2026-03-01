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
- `main.py`: small local smoke run and `--profile` diagnostics mode.

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

## Current Scoring Logic

From `scoring.py` / `README.md`:

- Contest params currently include:
  - `width`
  - `max_depth`
  - `budgets`
  - `time_tolerance`
- For each budget:
  - baseline per-depth runtime is measured by batched sampling (`sampling_baseline_time`).
  - estimator is run on each held-out circuit.
  - if estimator runtime at a depth exceeds `(1 + tolerance) * baseline_time`, output is zeroed for that depth.
  - if estimator runtime is below `(1 - tolerance) * baseline_time`, runtime is floored to `(1 - tolerance) * baseline_time`.
  - MSE is computed vs empirical means; then multiplied by time ratio.

Final score = average adjusted MSE across depths and budgets.

Additional scorer behavior:

- malformed estimator outputs (wrong width or wrong number of layers) now raise explicit errors,
- optional profiler callback can emit per-layer diagnostics:
  - `wall_time_s`,
  - `cpu_time_s`,
  - `rss_bytes`,
  - `peak_rss_bytes`.

## Local Smoke Result (Observed)

Command run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run main.py
```

Observed output score:

- `0.0071351177599899744`

This is just a toy sanity run, not a stable benchmark number.

## Important Gaps for Next Iterations

1. Hosted evaluator resource envelope and enforcement policy are still unresolved (`open-questions.md`).
2. No production containerized/sandboxed execution path in this repo yet.
3. Deterministic seeded evaluation flow exists locally but final public seed policy is not frozen.
4. Profiling metrics are local-process diagnostics, not yet equivalent to hosted infra accounting.
5. Participant output contract is clearer locally but still needs final public challenge-spec freeze.

## Implication for Future Agents

Treat current code as a strong local starter baseline with explicit interfaces and verification gates, but not yet a production hosted evaluator. Next phase should prioritize hosted isolation semantics and finalized public benchmark policy.
