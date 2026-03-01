# Circuit Estimation MVP

This repository is a local starter-kit style sandbox for a circuit-estimation challenge:
given a randomly generated layered circuit, predict the mean value of every wire at every layer
while respecting a runtime budget model.

## What This Repository Teaches

- How random Boolean-like layered circuits can be sampled and simulated.
- How estimator outputs are scored against empirical targets.
- How runtime is incorporated into error via adjusted MSE.
- How to build participant-style estimators that satisfy explicit output contracts.

## Conceptual Problem Overview

Each circuit has:

- `width`: number of wires per layer.
- `max_depth`: number of layers to predict.
- Layer rules: each output wire reads two input wires and applies an affine-bilinear update.

For a fixed `(circuit, budget)`, an estimator must return a tensor of shape
`(max_depth, width)` where each row predicts per-wire means for that layer.

The evaluator compares those predictions to empirical means from Monte Carlo simulation and
combines accuracy with runtime efficiency.

## How Evaluation Works (End-to-End)

For each scoring run:

1. Sample `n_circuits` random circuits (or use provided circuits).
2. For each circuit, estimate target means via simulation with `n_samples`.
3. For each budget:
4. Measure per-layer baseline sampling runtime.
5. Call estimator once per circuit with that budget.
6. Validate estimator output type/rank/shape.
7. Apply runtime semantics:
8. If call wall-time exceeds `(1 + time_tolerance) * baseline_total_time`, output is zeroed.
9. If call wall-time is below `(1 - time_tolerance) * baseline_total_time`, effective runtime is floored.
10. Compute per-layer MSE, multiply by time ratio, and average to budget score.
11. Average budget scores to final score (`lower_is_better`).

## Codebase Map (Suggested Reading Order)

1. `src/circuit_estimation/domain.py`  
   Data model (`Layer`, `Circuit`) and invariants.
2. `src/circuit_estimation/generation.py`  
   Random gate/circuit sampling.
3. `src/circuit_estimation/simulation.py`  
   Batched forward simulation and empirical means.
4. `src/circuit_estimation/estimators.py`  
   Reference estimators and approximation logic.
5. `src/circuit_estimation/scoring.py`  
   Evaluator loop, runtime accounting, report payload assembly.
6. `src/circuit_estimation/reporting.py`  
   Human dashboard and machine JSON rendering.
7. `src/circuit_estimation/cli.py` and `main.py`  
   Local entrypoints and CLI flags.

## Quickstart

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Run default local scoring

```bash
uv run main.py
```

### Output modes

- Default: Rich human dashboard in terminal.
- `--agent-mode`: pretty JSON only (machine-oriented output).
- `--detail raw` (default): raw per-budget/per-layer series.
- `--detail full`: includes derived summaries and matrices.
- `--profile`: adds per-call profiling metrics (`wall_time_s`, `cpu_time_s`, `rss_bytes`, `peak_rss_bytes`).

Examples:

```bash
uv run main.py
uv run main.py --agent-mode
uv run main.py --profile
uv run main.py --detail full --profile
```

## Extending the Estimator

Participant-facing contract in this repository:

- Signature: `Callable[[Circuit, int], NDArray[np.float32]]`
- Input: one `Circuit` plus one `budget`.
- Output: rank-2 array with exact shape `(max_depth, width)`.

Common failure cases:

- Wrong return type (not `np.ndarray`) -> `ValueError`.
- Wrong tensor rank -> `ValueError`.
- Depth mismatch (`shape[0] != max_depth`) -> `ValueError`.
- Width mismatch (`shape[1] != width`) -> `ValueError`.

Start with reference estimators in `src/circuit_estimation/estimators.py`, then evaluate via:

- `score_estimator_report(...)` for structured diagnostics.
- `score_estimator(...)` for final-score-only compatibility path.

## Verification Commands

Run before merging:

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pyright
uv run --group dev pytest -m "not exhaustive"
uv run --group dev pytest -m exhaustive
```

## Test Harness Shortcuts

```bash
./scripts/run-test-harness.sh quick
./scripts/run-test-harness.sh full
./scripts/run-test-harness.sh exhaustive
```

## Context for Future Agents

If you are iterating on evaluator/starter-kit design, begin with:

- `internal context docs`

That folder contains durable challenge context, technical notes, research links, and open questions.
