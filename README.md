This repository is intended to illustrate the basic mechanics of the competition.

## Planning Context for Future Agents

If you are iterating on starter-kit/evaluator design, begin with:

- `docs/context/README.md`

This folder contains the durable challenge context, MVP technical notes, infrastructure research, and open questions.
Future agents should treat it as the primary source even if `CHALLENGE-CONTEXT.md` is removed.
The folder also includes mandatory `agent-first` starter-kit requirements for participant agent workflows.

## Getting Started

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Run

```bash
uv run main.py
```

That's it. `uv` reads `pyproject.toml`, auto-creates a venv, installs dependencies, and runs the script with the **human dashboard** output by default.

### CLI Output Modes

- default: Rich human dashboard with tables and terminal plots.
- `--agent-mode`: pretty JSON only, machine-parseable (recommended for automated agents/UIs).
- `detail raw` (default): raw per-budget/per-layer data only.
- `detail full`: includes derived aggregates (`by_budget_summary`, `by_layer_overall`, `by_budget_layer_matrix`) and `profile_summary` when profiling is enabled.

### Human dashboard structure

The default human report is a tri-objective dashboard with:

- top row: `Run Context` + `Readiness Scorecard`,
- second row: `Hardware & Runtime`,
- core lanes: `Budget Intelligence` and `Layer Intelligence`,
- profile lane (only with `--profile`): `Profile Summary`, runtime plot, memory plot.

Layout is adaptive:

- wide/medium terminals (`>=110` columns): two-pane top row with hardware stacked below,
- narrow terminals (`<110`): stacked pane layout in the same narrative order.

Examples:

```bash
# default: human dashboard + detail raw
uv run main.py

# machine-parseable JSON for agents
uv run main.py --agent-mode

# include call-level profiling metrics
uv run main.py --profile

# rich report + full computed aggregates + profiling
uv run main.py --detail full --profile
```

Profiling is call-level (per estimator invocation on one `(circuit, budget)` pair) and records `wall_time_s`, `cpu_time_s`, `rss_bytes`, and `peak_rss_bytes`.

## Test Harness

This repository now includes a `pytest`-based test harness with:
- Core unit/integration checks for `circuit.py`, `estimators.py`, and `evaluate.py`
- Exhaustive validation checks that compare estimator behavior against exact enumeration on small circuits

### Run the harness

```bash
# Run only fast checks (default for local iteration)
./scripts/run-test-harness.sh quick

# Run fast checks + exhaustive checks (recommended before PR/merge)
./scripts/run-test-harness.sh full

# Run only exhaustive checks
./scripts/run-test-harness.sh exhaustive
```

The script uses `uv run --group dev pytest ...` and installs `pytest` from the `dev` dependency group automatically when needed.

## Release Quality Gates

Run all checks before release:

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pyright
uv run --group dev pytest -m "not exhaustive"
uv run --group dev pytest -m exhaustive
```

---

## Tools

### Circuit Explorer (Interactive UI)

Visual tool for exploring small circuits — see how gate operations, signal statistics, and estimator accuracy evolve with depth.

```bash
cd tools/circuit-explorer && npm install && npm run dev
```

See [`tools/circuit-explorer/README.md`](tools/circuit-explorer/README.md) for details.

---

## Parameters

The contest definition depends on: max_depth, width, budgets, time_tolerance

These are defined in ContestParams.

## Default values

Here are some values, to be adjusted based on early feedback:
- max_depth = 300
- width = 1000
- budgets = [1e2, 1e3, 1e4, 1e5, 1e6]
- time_tolerance = 0.1

These are defined in default_contest_params.

## Circuit sampling

The sampling procedure is: each gate has 2 random distinct inputs and implements a random boolean function.

This is implemented in random_circuit.

## Score function

Scoring is defined in terms of n_circuits and n_samples. We can adaptively increase these parameters to increase precision of the estimates and hopefully ensure we have a statistically significant difference between top competitors. This should be relatively easy as long as we don't make the circuits too deep.

Scoring procedure:

- We draw n_circuits circuits at random. For each we draw n_samples inputs and take the average.
- We then compute the target means for each wire in each circuit.
- For each budget in budgets:
  - We compute baseline sampling runtime per layer from a batched forward pass.
  - We run your estimator once per circuit (single call) and expect predictions for all layers at once.
  - Runtime enforcement is at call level:
    - if call wall time is above `(1 + time_tolerance) * baseline_total_time`, the whole output tensor is zeroed.
    - if call wall time is below `(1 - time_tolerance) * baseline_total_time`, effective runtime is floored to that lower bound.
- For each budget and depth:
  - MSE is computed against empirical means.
  - Adjusted MSE = `MSE * (effective_time / baseline_time)`.
- Final score is the mean adjusted MSE across budgets and layers.

(This is implemented in score_estimator)

## Participant Contract (Current Local API)

- Estimator signature:
  - `Callable[[Circuit, int], NDArray[np.float32]]`
- Input:
  - a generated `Circuit`,
  - one `budget` value from evaluator configuration.
- Required output:
  - one rank-2 `np.ndarray` with shape `(max_depth, width)`,
  - each row is the predicted wire means for one layer depth.

Violating output shape/depth requirements raises explicit errors in scoring.

Important: in-repo estimators are reference examples only. Future hosted evaluation should assume participant-submitted estimators may be adversarial or malicious and must be treated as black box implementations.

## Extension Points

- Implement custom estimators in `src/circuit_estimation/estimators.py` or a new module with the same callable signature.
- Evaluate locally via:
  - `score_estimator(...)` in `src/circuit_estimation/scoring.py`,
  - `score_estimator_report(...)` in `src/circuit_estimation/scoring.py` for structured raw/full report payloads,
  - `uv run main.py` for default baseline run.
- Optional diagnostics:
  - use `uv run main.py --profile` to emit call-level runtime/resource events.

## Failure Semantics

Scoring applies the following runtime rules per estimator call (single circuit + single budget):

- If elapsed time exceeds `(1 + time_tolerance) * baseline_total_time`, estimator output tensor is zeroed.
- If elapsed time is below `(1 - time_tolerance) * baseline_total_time`, effective runtime is floored to `(1 - time_tolerance) * baseline_total_time`.
- If estimator output width mismatches `width`, scoring raises `ValueError`.
- If estimator output depth mismatches `max_depth`, scoring raises `ValueError`.

## Deterministic Seed Policy

- Circuit generation supports explicit seeded RNG objects (`np.random.default_rng(seed)`).
- For reproducible local experiments, pass seeded RNGs when constructing circuits.
- Default CLI flow remains stochastic unless a seeded circuit workflow is explicitly used.
