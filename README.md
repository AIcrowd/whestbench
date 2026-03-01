This repository is intended to illustrate the basic mechanics of the competition.

## Planning Context for Future Agents

If you are iterating on starter-kit/evaluator design, begin with:

- `internal context docs`

This folder contains distilled challenge context, MVP technical notes, infrastructure research, and open questions.

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
