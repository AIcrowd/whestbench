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

That's it. `uv` reads `pyproject.toml`, auto-creates a venv, installs dependencies, and runs the script — no manual setup needed.

### Optional profiling run

```bash
uv run main.py --profile
```

This prints the score plus per-layer diagnostics (`wall_time_s`, `cpu_time_s`, `rss_bytes`, `peak_rss_bytes`).

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
- We also compute the average variance of a random wire in a random circuit at depth d for each d up to max_depth.
- For each budget in budgets:
  - We compute the time required to run a forward pass with budget samples (for each layer), which becoms the time_budget.
  - We run your algorithm on each circuit.
  - The algorithm outputs predictions for each layer one at a time, and we record how long it took to get to each layer.
  - If your program goes over (1 + time_tolerance) * time_budget for a given layer, we zero out its predictions on that layer.
  - If your program goes under (1 - time_tolerance) * time_budget, we set it's time to (1 - time_tolerance) * time_budget.
- Your loss at a given depth and budget is defined as:
  - Base loss = (Your mean squared error) / (sampling mean squared error)
  - Final loss = (base loss) * (your average time) / (sampling average time).
Your total loss is the average loss over all depths and time limits.

(This is implemented in score_estimator)
