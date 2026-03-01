# MVP Technical Snapshot

Last updated: 2026-03-01

## Repository Layout

- `circuit.py`: random circuit generation + batched forward simulation + empirical means.
- `estimators.py`: mean propagation, covariance propagation, and a budget-switched combined estimator.
- `evaluate.py`: scoring loop, baseline timing, contest params.
- `main.py`: small local smoke run.

## Current Mathematical Representation

- Wires are represented as values in `{-1, +1}` internally, not `{0, 1}`.
- Gates are represented as affine/bilinear forms using coefficients:
  - `const`, `first_coeff`, `second_coeff`, `product_coeff`.

This representation is compact for propagation-style estimators, but the participant-facing API still needs to define whether submission contracts use Boolean or signed-wire semantics.

## Circuit Generation and Execution

From `circuit.py`:

- `Layer` stores per-wire parent indices and gate coefficients.
- `random_gates(n)` samples gate coefficients and wire connections.
- `random_circuit(n, d)` builds `d` layers.
- `run_batched(circuit, inputs)` yields layer outputs for a batch.
- `empirical_mean(circuit, trials)` estimates per-layer mean vectors by Monte Carlo.

## Estimators in Repo

From `estimators.py`:

- `mean_propagation(circuit)`: tracks only means, ignores higher-order dependencies.
- `covariance_propagation(circuit)`: tracks means + full covariance matrix per layer.
- `combined_estimator(circuit, budget)`:
  - uses covariance propagation when `budget >= 30 * circuit.n`;
  - otherwise uses mean propagation.

## Current Scoring Logic

From `evaluate.py` / `README.md`:

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

## Local Smoke Result (Observed)

Command run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run main.py
```

Observed output score:

- `0.0071351177599899744`

This is just a toy sanity run, not a stable benchmark number.

## Important Gaps for Next Iterations

1. No formal submission interface contract yet.
2. No containerized/sandboxed execution path yet.
3. No deterministic seeding policy for reproducible leaderboard recomputation.
4. No unit/integration test suite for scoring invariants.
5. Timing uses local wall-clock behavior; no production-grade resource accounting yet.
6. Some score pipeline internals appear provisional (for example `baseline_performance` is computed but not used).

## Implication for Future Agents

Treat current code as a toy reference for algorithm/evaluator shape, not as production evaluator code. The next phase should prioritize explicit interfaces, reproducibility, and secure isolated execution semantics.

