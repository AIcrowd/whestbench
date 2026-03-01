This repository is a starter-kit style implementation of the circuit-estimation problem used for local development and evaluator design iteration.

## What This Repository Teaches

- How randomly generated boolean circuits are represented and simulated.
- How estimators predict wire-mean trajectories across all layers in one call.
- How scoring combines prediction quality and runtime constraints into a single objective.
- How to inspect outcomes in either a human dashboard (default) or machine JSON (`--agent-mode`).

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
   - Measure baseline total runtime for sampling (`baseline_total_time`).
   - Call estimator once per circuit (`estimator(circuit, budget)`), expecting full-depth predictions.
   - Apply runtime enforcement at call level:
     - if call wall time > `(1 + time_tolerance) * baseline_total_time`, zero the full returned tensor;
     - if call wall time < `(1 - time_tolerance) * baseline_total_time`, floor effective time to that lower bound.
   - Compute per-layer MSE and aggregate:
     - `mse_mean`
     - `call_time_ratio_mean`
     - `call_effective_time_s_mean`
     - `adjusted_mse = mse_mean * call_time_ratio_mean`
4. Final score = mean `adjusted_mse` across budgets (lower is better).

Report payload:

- `results.by_budget_raw` contains raw per-budget metrics (including `mse_by_layer` and scalar call-level runtime metrics).
- `detail=full` adds derived tables/matrices (`by_budget_summary`, `by_layer_overall`, `by_budget_layer_matrix`).
- `--profile` adds call-level profiling events (`wall_time_s`, `cpu_time_s`, `rss_bytes`, `peak_rss_bytes`).

## Codebase Map (Suggested Reading Order)

1. `src/circuit_estimation/domain.py`: core `Layer` and `Circuit` entities + validation.
2. `src/circuit_estimation/generation.py`: random gate/circuit sampling.
3. `src/circuit_estimation/simulation.py`: batched execution and empirical means.
4. `src/circuit_estimation/sdk.py`: participant-facing `BaseEstimator` and setup context.
5. `src/circuit_estimation/loader.py`: deterministic loading of `Estimator(BaseEstimator)` from file.
6. `src/circuit_estimation/runner.py`: in-process/subprocess runner contracts and outcomes.
7. `examples/estimators/`: class-based starter estimators (`mean`, `covariance`, `combined`).
8. `src/circuit_estimation/scoring.py`: scoring loop, runtime enforcement, profiling hook.
9. `src/circuit_estimation/reporting.py`: human dashboard and agent JSON rendering.
10. `src/circuit_estimation/cli.py` and `main.py`: local CLI entrypoints.
11. `src/circuit_estimation/protocol.py`: DTOs for future RPC-style integration.

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

Default output is a Rich human dashboard. For machine consumers:

```bash
uv run main.py --agent-mode
```

Useful flags:

```bash
# full derived aggregates
uv run main.py --detail full

# include call-level profiling metrics
uv run main.py --profile

# show optional diagnostic plots in human mode
uv run main.py --show-diagnostic-plots
```

## Extending the Estimator

Participant estimators now use a stateful class API:

- class base: `BaseEstimator`
- required: `Estimator.predict(circuit: Circuit, budget: int) -> np.ndarray`
- optional: `setup(context)` and `predict_batch(circuits: Iterable[Circuit], budget: int)`

Contract:

- `predict` returns rank-2 ndarray shape `(max_depth, width)`
- values must be finite
- setup is allowed and treated as a separate lifecycle phase

Starter examples for this API live under `examples/estimators/`.
The old function-style estimator API is no longer the participant path.

Local participant workflow (installable CLI):

```bash
# scaffold estimator.py + optional requirements.txt
cestim init my-estimator

# validate estimator contract (JSON-only with --agent-mode)
cestim validate --estimator my-estimator/estimator.py --agent-mode

# run local scoring (human report by default; JSON-only in --agent-mode)
cestim run --estimator my-estimator/estimator.py --runner subprocess

# build submission artifact with generated manifest.json
cestim package --estimator my-estimator/estimator.py --output submission.tar.gz
```

Submission artifact contract:

- required machine contract: `manifest.json` (generated)
- participant code: `estimator.py`
- optional participant files: `requirements.txt`, `submission.yaml`, `APPROACH.md`

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
