This repository is a starter-kit style implementation of the circuit-estimation problem used for local development and evaluator design iteration.

## What This Repository Teaches

- How randomly generated boolean circuits are represented and simulated.
- How estimators stream wire-mean trajectories one depth row at a time.
- How scoring combines prediction quality and runtime constraints into a single objective.
- How to inspect outcomes in either a human dashboard (default) or machine JSON (`--agent-mode`).

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
   - Call estimator once per circuit (`estimator(circuit, budget)`) and consume streamed depth rows.
   - At each emitted depth row `i`:
     - if cumulative wall time > `(1 + time_tolerance) * time_budget_by_depth_s[i]`, zero that row;
     - if cumulative wall time < `(1 - time_tolerance) * time_budget_by_depth_s[i]`, floor effective time to that lower bound.
   - Compute per-depth MSE and aggregate:
     - `mse_mean`
     - `call_time_ratio_mean`
     - `call_effective_time_s_mean`
     - `adjusted_mse = mse_mean * call_time_ratio_mean`
4. Final score = mean `adjusted_mse` across budgets (lower is better).

Report payload:

- `results.by_budget_raw` contains raw per-budget metrics (including `mse_by_layer` and depth runtime vectors such as `time_budget_by_depth_s`).
- `detail=full` adds derived tables/matrices (`by_budget_summary`, `by_layer_overall`, `by_budget_layer_matrix`).
- `--profile` adds call-level profiling events (`wall_time_s`, `cpu_time_s`, `rss_bytes`, `peak_rss_bytes`).

## Codebase Map (Suggested Reading Order)

1. `src/circuit_estimation/domain.py`: core `Layer` and `Circuit` entities + validation.
2. `src/circuit_estimation/generation.py`: random gate/circuit sampling.
3. `src/circuit_estimation/simulation.py`: batched execution and empirical means.
4. `src/circuit_estimation/estimators.py`: reference estimators and budget switch logic.
5. `src/circuit_estimation/scoring.py`: scoring loop, runtime enforcement, profiling hook.
6. `src/circuit_estimation/reporting.py`: human dashboard and agent JSON rendering.
7. `src/circuit_estimation/cli.py` and `main.py`: CLI entrypoint implementation and fallback launcher.
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

### Run default local report

```bash
cestim
```

Default output is a Rich human dashboard. For machine consumers:

```bash
cestim --agent-mode
```

Useful flags:

```bash
# full derived aggregates
cestim --detail full

# include call-level profiling metrics
cestim --profile

# show optional diagnostic plots in human mode
cestim --show-diagnostic-plots
```

Without installing globally, you can still run the CLI via:

```bash
uv run --with-editable . cestim --agent-mode
```

## Extending the Estimator

Implement a callable with signature:

- `Callable[[Circuit, int], Iterator[NDArray[np.float32]]]`

Contract:

- input: one circuit + one budget
- output: streamed depth rows via `yield`
- each emitted row: `np.ndarray` with shape `(width,)`
- required row count: exactly `max_depth` yields

See the starter tutorial guide:

- `docs/context/participant-streaming-estimator-guide.md`

Recommended extension path:

- add new estimators under `src/circuit_estimation/estimators.py` (or new module),
- evaluate locally with `score_estimator(...)` or `score_estimator_report(...)`,
- compare via `cestim --detail full --profile` (or `uv run cestim --detail full --profile`).

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

- `docs/context/README.md`

That folder contains durable challenge context, design decisions, and open questions. It is the primary source of truth for future changes.

## Tools

### Circuit Explorer (Interactive UI)

```bash
cd tools/circuit-explorer && npm install && npm run dev
```

See `tools/circuit-explorer/README.md` for details.
