This repository is a starter-kit style implementation of the circuit-estimation problem used for local development and evaluator design iteration.

## What This Repository Teaches

- How randomly generated boolean circuits are represented and simulated.
- How estimators predict wire-mean trajectories across all layers in one call.
- How scoring combines prediction quality and runtime constraints into a single objective.
- How to inspect outcomes in either a Textual human dashboard (default) or machine JSON (`--agent-mode`).

## Conceptual Problem Overview

We generate random circuits made of layered gates. For each circuit, the evaluator estimates per-layer wire means via Monte Carlo simulation (ground truth).
Participants provide an estimator that receives:

- one `Circuit`
- one `budget`

and must return one `np.ndarray` with shape `(max_depth, width)` containing predictions for all layers in a single pass.

Important security/architecture note: in-repo estimator implementations are examples only. Hosted evaluation should assume participant estimators may be adversarial/malicious and must be treated as black boxes.

## How Evaluation Works (End-to-End)

Given `n_circuits`, `n_samples`, and contest params (`width`, `max_depth`, `budgets`, `time_tolerance`):

1. Sample or accept circuits.
2. Compute empirical layer-wise target means for each circuit via batched simulation.
3. For each budget:
   - Treat `budget` as a sampling trial count.
   - Measure the sampling runtime curve by depth:
     - `time_budget_by_depth_s[i] = cumulative wall time to reach depth i with budget samples`.
   - Call estimator once per circuit (`estimator(circuit, budget)`), expecting full-depth predictions.
   - Apply runtime enforcement against the budget-by-depth curve:
     - for each depth `i`, if call wall time > `(1 + time_tolerance) * time_budget_by_depth_s[i]`, zero depth `i`;
     - for each depth `i`, if call wall time < `(1 - time_tolerance) * time_budget_by_depth_s[i]`, floor depth runtime to that lower bound.
     - estimator API is single-call tensor output, so this comparison uses the observed call wall time at every depth index.
   - Compute per-depth MSE and aggregate:
     - `mse_mean`
     - `call_time_ratio_mean`
     - `call_effective_time_s_mean`
     - `adjusted_mse = mean(mse_by_depth * time_ratio_by_depth_mean)`
4. Final score = mean `adjusted_mse` across budgets (lower is better).

Report payload:

- `results.by_budget_raw` contains raw per-budget metrics (including `mse_by_layer`, `time_budget_by_depth_s`, and scalar call-level runtime summaries).
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

### Run default local report

```bash
uv run main.py
```

Default output is the full-screen Textual human dashboard (with automatic static fallback when unsupported). For machine consumers:

```bash
uv run main.py --agent-mode
```

Default `uv run main.py` behavior:

- launches full-screen Textual dashboard in supported interactive terminals,
- falls back to static terminal report when Textual is unavailable (for example non-TTY or limited terminal capabilities).

## Extending the Estimator

Implement a callable with signature:

- `Callable[[Circuit, int], NDArray[np.float32]]`

Contract:

- input: one circuit + one budget
- output: rank-2 ndarray of shape `(max_depth, width)`
- each row: predicted wire means for one layer depth

Budget tuning intuition:

- `budget` is not seconds; it is the sampling trial count used to derive a reference time envelope by depth.
- Higher budgets increase the `time_budget_by_depth_s` envelope, which allows more estimator compute before depth outputs are zeroed.
- Runtime below the lower tolerance bound is floored, so use that slack to improve accuracy while staying safely under timeout.

Recommended extension path:

- add new estimators under `src/circuit_estimation/estimators.py` (or new module),
- evaluate locally with `score_estimator(...)` or `score_estimator_report(...)`,
- compare via `uv run main.py`.

## Verification Commands

Run before claiming release-ready status:

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pyright
uv run --group dev pytest -m "not exhaustive"
uv run --group dev pytest -m exhaustive
```

## Planning Context for Future Agents

If you are iterating on starter-kit/evaluator design, start from:

- `internal context docs`

That folder contains durable challenge context, design decisions, and open questions. It is the primary source of truth for future changes.

## Tools

### Circuit Explorer (Interactive UI)

```bash
cd tools/circuit-explorer && npm install && npm run dev
```

See `tools/circuit-explorer/README.md` for details.
