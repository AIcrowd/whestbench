<img src="assets/logo/logo.png" alt="logo" style="max-height: 100px;">

# ARC - Circuit Estimation Challenge

This repository is a starter-kit style implementation of the circuit-estimation problem used for local development and evaluator design iteration.

## What This Repository Teaches

- How to implement a participant estimator with the streaming `BaseEstimator` contract.
- How evaluation computes accuracy/runtime-adjusted score across budgets.
- How to use local tooling (`cestim`) to validate, run, and package submissions.
- How to iterate from simple baseline estimators to stronger budget-aware approaches.

## Conceptual Problem Overview

We generate random circuits made of layered gates. For each circuit, the evaluator estimates per-layer wire means via Monte Carlo simulation (ground truth).
Participants provide an estimator that receives:

- one `Circuit`
- one `budget`

and must stream exactly `max_depth` vectors via `yield`, where each emitted vector has shape `(width,)`.

Important security/architecture note: in-repo estimator implementations are examples only. Hosted evaluation should assume participant estimators may be adversarial/malicious and must be treated as black boxes.

## How Evaluation Works (End-to-End)

Given `n_circuits`, `n_samples`, and contest params (`width`, `max_depth`, `budgets`, `time_tolerance`):

1. Sample or accept circuits.
2. Compute empirical layer-wise target means for each circuit via batched simulation.
3. For each budget:
   - Measure baseline runtime by depth (`time_budget_by_depth_s`) using sampling.
   - Call estimator once per `(circuit, budget)` invocation and consume streamed depth rows.
   - At each emitted depth row `i`:
     - if cumulative wall time > `(1 + time_tolerance) * time_budget_by_depth_s[i]`, zero that row;
     - if cumulative wall time < `(1 - time_tolerance) * time_budget_by_depth_s[i]`, floor effective time to that lower bound.
   - Compute per-depth MSE and aggregate:
     - `mse_mean`
     - `call_time_ratio_mean`
     - `call_effective_time_s_mean`
     - `adjusted_mse = mean(mse_by_depth * time_ratio_by_depth_mean)`
4. Final score = mean `adjusted_mse` across budgets (lower is better).

Report payload:

- `results.by_budget_raw` contains raw per-budget metrics (including `mse_by_layer` and depth runtime vectors such as `time_budget_by_depth_s`).
- Human mode computes the richer payload used by dashboard tabs (summary, budgets, layers, performance, data).
- `score_estimator_report(..., profile=True, detail="full")` remains available as a programmatic API for advanced callers.

## Codebase Map (Suggested Reading Order)

1. `src/circuit_estimation/domain.py`: core `Layer` and `Circuit` entities + validation.
2. `src/circuit_estimation/generation.py`: random gate/circuit sampling.
3. `src/circuit_estimation/simulation.py`: batched execution and empirical means.
4. `src/circuit_estimation/estimators.py`: reference estimators and budget switch logic.
5. `src/circuit_estimation/scoring.py`: scoring loop, runtime enforcement, profiling hook.
6. `src/circuit_estimation/reporting.py`: static fallback rendering and agent JSON output.
7. `src/circuit_estimation/cli.py` and `main.py`: local CLI entrypoints.
8. `src/circuit_estimation/protocol.py`: DTOs for future RPC-style integration.

## Quickstart

### Prerequisite

Install [uv](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install CLI (recommended)

From repository root:

```bash
uv tool install -e .
```

This installs the `cestim` command globally from your local checkout.
If you work with multiple worktrees, read [Worktrees and CLI](docs/development/worktrees-and-cli.md).

### Run default local report

```bash
cestim
```

Default output is the Rich human dashboard. For machine consumers:

```bash
cestim --agent-mode
```

To run from the current checkout without relying on global `cestim`:

```bash
uv run main.py
```

To run `cestim` from the current checkout explicitly:

```bash
uv run --with-editable . cestim --agent-mode
```

## Extending the Estimator

Canonical participant interface:

- subclass `BaseEstimator`
- implement `predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]`

Contract:

- input: one circuit + one budget
- output: streamed depth rows via `yield`
- each emitted row: `np.ndarray` with shape `(width,)`
- required row count: exactly `max_depth` yields

Scoring API compatibility:

- `score_estimator(...)` accepts a callable with signature:
  - `Callable[[Circuit, int], Iterator[NDArray[np.float32]]]`
- for class-based estimators, pass `Estimator().predict`.

See the starter tutorial guide:

- `docs/guides/participant-streaming-estimator-guide.md`
- `examples/estimators/random_estimator.py` (start here: full interface walkthrough)
- `examples/estimators/mean_propagation.py` (first real baseline)
- `examples/estimators/covariance_propagation.py` (second-order moments)
- `examples/estimators/combined_estimator.py` (budget-aware routing)

Budget tuning intuition:

- `budget` is not seconds; it is the sampling trial count used to derive a reference time envelope by depth.
- Higher budgets increase the `time_budget_by_depth_s` envelope, which allows more estimator compute before depth outputs are zeroed.
- Runtime below the lower tolerance bound is floored, so use that slack to improve accuracy while staying safely under timeout.

Recommended extension path:

- start from `examples/estimators/random_estimator.py` to learn the interface,
- then copy one of the stronger starters from `examples/estimators/` into your own `estimator.py`,
- keep estimator-specific helper methods inside your estimator class,
- evaluate locally with `score_estimator(...)` or `score_estimator_report(...)`,
- compare via `cestim` (or `uv run main.py`).

## Verification Commands

Run before claiming release-ready status:

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pyright
uv run --group dev pytest -m "not exhaustive"
uv run --group dev pytest -m exhaustive
```

## Tools

### Circuit Explorer (Interactive UI)

```bash
cd tools/circuit-explorer && npm install && npm run dev
```

See `tools/circuit-explorer/README.md` for details.


# Authors
- Paul Christiano
- Jacob Hilton
- Sharada Mohanty 
