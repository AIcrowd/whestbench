<div align="center">
  <img src="assets/logo/logo.png" alt="Circuit Estimation Challenge logo" style="height: 120px;">
  <br>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-green?logo=python&logoColor=white" alt="Python 3.10+"></a>
</div>

# Circuit Estimation Challenge

Can you predict a circuit's behavior by analyzing its structure, instead of just running it thousands of times?

This challenge asks you to build **mechanistic estimators** — algorithms that exploit the wiring and gate rules of random layered circuits to estimate expected wire values, rather than relying solely on brute-force Monte Carlo sampling. The question is both practical and foundational: when can structure-aware estimation compete with or beat pure sampling under the same compute budget?

## ⚡ 60-Second Overview

You are given:

- one random `Circuit`
- one integer `budget`

Your estimator must stream exactly one vector per layer, each with shape `(width,)`, estimating expected wire values after that layer.

Your score combines prediction accuracy with compute efficiency: can you match sampling's accuracy while using less time? See [Scoring Model](docs/concepts/scoring-model.md) for details. Lower score is better.

### 🧠 Why this challenge matters

<div align="center">
  <img src="assets/circuit-explorer-visualization.svg" alt="Circuit Explorer Visualization" width="100%">
</div>

The natural way to estimate a circuit's expected output is brute force: sample many random inputs, propagate them, average the results. Sampling is the ground truth — with enough samples it converges to the exact answer. But it's inefficient: the error only shrinks as 1/√k with k samples, and it learns nothing from the circuit's structure.

**Mechanistic estimation** asks: can we beat sampling at this task? Instead of brute-force evaluation, analyze the circuit's wiring and gate rules to estimate expected wire values directly. Because sampling scales so poorly, there is room for structure-aware methods to reach the same accuracy in far less compute. ARC's research suggests this is both possible and hard — simple structural methods (like mean propagation) work at shallow depth, but break down as correlations accumulate through layers.

- [Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/)
- [AlgZoo: uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

This challenge instantiates that question in random Boolean circuits, where evaluation is explicit, reproducible, and compute-aware.

> **Your practical goal:** beat sampling. Build an estimator that reaches the same accuracy as brute-force sampling but in less compute time. Your score directly measures how efficiently you estimate relative to the sampling baseline.

## 🚀 5-Minute Quickstart

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
cestim run --estimator ./my-estimator/estimator.py
cestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

`cestim run` uses `--runner subprocess` by default.

For faster repeated evaluations, pre-create a dataset and reuse it:

```bash
cestim create-dataset -o my_dataset.npz
cestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
```

Quick debug sequence when `run` fails:

```bash
cestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
cestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz --debug
cestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz --runner inprocess --debug
```

For local editable invocation without global install:

```bash
uv run --with-editable . cestim smoke-test
```

## 📚 Documentation

Start at: [Documentation Index](docs/index.md)

### 🏁 Getting Started

- [Install and CLI Quickstart](docs/getting-started/install-and-cli-quickstart.md)
- [First Local Run](docs/getting-started/first-local-run.md)

### 💡 Concepts

- [Problem Setup](docs/concepts/problem-setup.md)
- [Scoring Model](docs/concepts/scoring-model.md)

### 🛠 How-To

- [Write an Estimator](docs/how-to/write-an-estimator.md)
- [Inspect and Traverse Circuit Structure](docs/how-to/inspect-circuit-structure.md)
- [Validate, Run, and Package](docs/how-to/validate-run-package.md)
- [Use Evaluation Datasets](docs/how-to/use-evaluation-datasets.md)
- [Use Circuit Explorer](docs/how-to/use-circuit-explorer.md)

### 📖 Reference

- [Estimator Contract](docs/reference/estimator-contract.md)
- [CLI Reference](docs/reference/cli-reference.md)
- [Score Report Fields](docs/reference/score-report-fields.md)

### 🔧 Troubleshooting

- [Common Participant Errors](docs/troubleshooting/common-participant-errors.md)

## 🧪 Example Estimators

Starter estimators are in [`examples/estimators/`](examples/estimators/):

- [`random_estimator.py`](examples/estimators/random_estimator.py): interface walkthrough, intentionally low quality
- [`mean_propagation.py`](examples/estimators/mean_propagation.py): first-order baseline
- [`covariance_propagation.py`](examples/estimators/covariance_propagation.py): second-order baseline
- [`combined_estimator.py`](examples/estimators/combined_estimator.py): budget-aware baseline

Recommended reading order:

1. [`random_estimator.py`](examples/estimators/random_estimator.py)
2. [`mean_propagation.py`](examples/estimators/mean_propagation.py)
3. [`covariance_propagation.py`](examples/estimators/covariance_propagation.py)
4. [`combined_estimator.py`](examples/estimators/combined_estimator.py)

Try them out (adjust `--n-circuits` and `--n-samples` to control evaluation size):

```bash
# Quick smoke run (10 circuits, 500 samples — fast)
cestim run --estimator examples/estimators/mean_propagation.py --n-circuits 10 --n-samples 500

# Full evaluation against the combined estimator
cestim run --estimator examples/estimators/combined_estimator.py --n-circuits 100 --n-samples 10000

# Compare estimators on the same dataset for fair scoring
cestim create-dataset --n-circuits 100 --n-samples 10000 -o eval.npz
cestim run --estimator examples/estimators/mean_propagation.py --dataset eval.npz
cestim run --estimator examples/estimators/covariance_propagation.py --dataset eval.npz
cestim run --estimator examples/estimators/combined_estimator.py --dataset eval.npz
```

## 📡 Current Platform Status

This starter kit supports local development, validation, scoring, and packaging.

Hosted submission/upload instructions are not part of this repository yet; until then, use local `cestim package` artifacts for iteration.

## ✅ Verification Commands

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pyright
uv run --group dev pytest -m "not exhaustive"
uv run --group dev pytest -m exhaustive
```

## 👥 Authors

- Paul Christiano
- Jacob Hilton
- Sharada Mohanty
