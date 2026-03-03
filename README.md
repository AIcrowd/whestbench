<img src="assets/logo/logo.png" alt="Circuit Estimation Challenge logo" style="height: 120px;">

# Circuit Estimation Challenge Starter Kit

Build, test, and iterate budget-aware estimators for random layered circuits.

## 60-Second Overview

You are given:

- one random `Circuit`
- one integer `budget`

Your estimator must stream exactly one vector per layer, each with shape `(width,)`, estimating expected wire values after that layer.

The evaluator compares your streamed predictions to Monte Carlo ground truth and applies runtime-aware adjustments by depth.

Lower score is better.

### 🧠 Why this challenge matters

This benchmark targets a core research question: when can we estimate model behavior by using structure, instead of relying only on brute-force sampling?

ARC's recent work frames "matching or beating sampling" as an important and difficult milestone for mechanistic estimation:

- [Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/)
- [AlgZoo: uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

This challenge instantiates that question in random Boolean circuits, where evaluation is explicit, reproducible, and compute-aware.

Practical goal for participants: improve error under fixed runtime budgets, not only via oversampling with effectively unlimited compute.

## 5-Minute Quickstart

Install [`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the CLI from this repository:

```bash
uv tool install -e .
```

Sanity-check CLI wiring:

```bash
cestim smoke-test
```

Run your first full loop:

```bash
cestim init ./my-estimator
cestim validate --estimator ./my-estimator/estimator.py
cestim run --estimator ./my-estimator/estimator.py --runner subprocess
cestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

For local editable invocation without global install:

```bash
uv run --with-editable . cestim smoke-test
```

## Documentation

Start at: [Documentation Index](docs/index.md)

### Getting Started

- [Install and CLI Quickstart](docs/getting-started/install-and-cli-quickstart.md)
- [First Local Run](docs/getting-started/first-local-run.md)

### Concepts

- [Problem Setup](docs/concepts/problem-setup.md)
- [Scoring Model](docs/concepts/scoring-model.md)

### How-To

- [Write an Estimator](docs/how-to/write-an-estimator.md)
- [Inspect and Traverse Circuit Structure](docs/how-to/inspect-circuit-structure.md)
- [Validate, Run, and Package](docs/how-to/validate-run-package.md)
- [Use Circuit Explorer](docs/how-to/use-circuit-explorer.md)

### Reference

- [Estimator Contract](docs/reference/estimator-contract.md)
- [CLI Reference](docs/reference/cli-reference.md)
- [Score Report Fields](docs/reference/score-report-fields.md)

### Troubleshooting

- [Common Participant Errors](docs/troubleshooting/common-participant-errors.md)

## Example Estimators

Starter estimators are in `examples/estimators/`:

- `random_estimator.py`: interface walkthrough, intentionally low quality
- `mean_propagation.py`: first-order baseline
- `covariance_propagation.py`: second-order baseline
- `combined_estimator.py`: budget-aware baseline

Recommended reading order:

1. `random_estimator.py`
2. `mean_propagation.py`
3. `covariance_propagation.py`
4. `combined_estimator.py`

## Current Platform Status

This starter kit supports local development, validation, scoring, and packaging.

Hosted submission/upload instructions are not part of this repository yet; until then, use local `cestim package` artifacts for iteration.

## Verification Commands

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pyright
uv run --group dev pytest -m "not exhaustive"
uv run --group dev pytest -m exhaustive
```

## Authors

- Paul Christiano
- Jacob Hilton
- Sharada Mohanty
