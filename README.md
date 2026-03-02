<img src="assets/logo/logo.png" alt="logo" style="height: 120px;">

# Circuit Estimation Challenge Starter Kit

Build, test, and iterate budget-aware estimators for random layered circuits.

## 60-Second Overview

You are given:

- one random `Circuit`
- one integer `budget`

Your estimator must stream exactly one vector per layer, each with shape `(width,)`, estimating expected wire values after that layer.

The evaluator compares your streamed predictions to Monte Carlo ground truth and applies runtime-aware adjustments by depth.

Lower score is better.

## 5-Minute Quickstart

Install [`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install the CLI from this repository:

```bash
uv tool install -e .
```

Initialize a starter estimator:

```bash
cestim init ./my-estimator
```

Validate the estimator contract:

```bash
cestim validate --estimator ./my-estimator/estimator.py
```

Run a local score report:

```bash
cestim run --estimator ./my-estimator/estimator.py --runner subprocess
```

Package a submission artifact:

```bash
cestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

For local editable invocation without global install:

```bash
uv run --with-editable . cestim --json
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
