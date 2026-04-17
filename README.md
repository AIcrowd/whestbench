<div align="center">
  <img src="assets/logo/logo.png" alt="ARC Whitebox Estimation Challenge logo" style="height: 120px;">
  <br>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-green?logo=python&logoColor=white" alt="Python 3.10+"></a>
</div>

# ARC Whitebox Estimation Challenge

Can you predict a network's behavior by analyzing its structure, instead of just running it thousands of times?

This challenge asks you to build **mechanistic estimators** — algorithms that exploit the topology and layer rules of random MLPs to estimate expected neuron values, rather than relying solely on brute-force Monte Carlo sampling. The question is both practical and foundational: when can structure-aware estimation compete with or beat pure sampling under the same compute budget?

## ⚡ 60-Second Overview

You are given:

- one random `MLP`
- one integer `budget`

Your estimator must return one vector per layer, each with shape `(width,)`, estimating expected neuron values after that layer.

Your score is pure MSE under a FLOP budget constraint: predictions that exceed the FLOP cap are zeroed, and you are ranked by MSE otherwise. See [Scoring Model](docs/concepts/scoring-model.md) for details. Lower score is better.

### 🧠 Why this challenge matters

<div align="center">
  <img src="assets/whestbench-explorer-visualization.svg" alt="WhestBench Explorer Visualization" width="100%">
</div>

The natural way to estimate a network's expected output is brute force: sample many random inputs, propagate them, average the results. Sampling is the ground truth — with enough samples it converges to the exact answer. But it's inefficient: the error only shrinks as 1/√k with k samples, and it learns nothing from the network's structure.

**Mechanistic estimation** asks: can we beat sampling at this task? Instead of brute-force evaluation, analyze the network's topology and layer rules to estimate expected neuron values directly. Because sampling scales so poorly, there is room for structure-aware methods to reach the same accuracy in far less compute. ARC's research suggests this is both possible and hard — simple structural methods (like mean propagation) work at shallow depth, but break down as correlations accumulate through layers.

- [Competing with sampling](https://www.alignment.org/blog/competing-with-sampling/)
- [AlgZoo: uninterpreted models with fewer than 1,500 parameters](https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/)

This challenge instantiates that question in random MLPs, where evaluation is explicit, reproducible, and compute-aware.

> **Your practical goal:** beat sampling. Build an estimator that reaches the same accuracy as brute-force sampling but within a fixed FLOP budget. Your score is the MSE of your predictions under that budget — lower is better.

## 🚀 5-Minute Quickstart

Install [`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install the CLI:

```bash
git clone git@github.com:AIcrowd/whestbench.git
cd whestbench
uv tool install -e .
```

Sanity-check CLI wiring:

```bash
whest smoke-test
```

Explore networks visually (requires Node.js):

```bash
whest visualizer
```

Run your first full loop:

```bash
whest init ./my-estimator
whest validate --estimator ./my-estimator/estimator.py
whest run --estimator ./my-estimator/estimator.py
whest package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

`whest run --estimator ...` uses local in-process execution by default for faster iteration.

For faster repeated evaluations, pre-create a dataset and reuse it:

```bash
whest create-dataset -o my_dataset.npz
whest run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
```

Quick debug sequence when `run` fails:

```bash
whest run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
whest run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz --debug
whest run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz --runner local --debug
```

For local editable invocation without global install:

```bash
uv run --with-editable . whest smoke-test
```

## 📚 Documentation

Start at: [Documentation Index](docs/index.md)

### 🏁 Getting Started

- [Install and CLI Quickstart](docs/getting-started/install-and-cli-quickstart.md)
- [First Local Run](docs/getting-started/first-local-run.md)
- [From Problem to Code](docs/getting-started/from-problem-to-code.md) — complete walkthrough to a scored estimator

### 💡 Concepts

- [Problem Setup](docs/concepts/problem-setup.md)
- [Scoring Model](docs/concepts/scoring-model.md)
- [How Ground Truth Is Generated](docs/concepts/ground-truth.md)

### 🛠 How-To

- [Write an Estimator](docs/how-to/write-an-estimator.md)
- [Algorithm Ideas](docs/how-to/algorithm-ideas.md) — survey of estimation strategies
- [Manage Your FLOP Budget](docs/how-to/manage-flop-budget.md)
- [Inspect and Traverse MLP Structure](docs/how-to/inspect-mlp-structure.md)
- [Validate, Run, and Package](docs/how-to/validate-run-package.md)
- [Use Evaluation Datasets](docs/how-to/use-evaluation-datasets.md)
- [Use WhestBench Explorer](docs/how-to/use-whestbench-explorer.md)
- [Profile Simulation Backends](docs/how-to/profile-simulation-backends.md) — profile whest FLOP usage and analytical correctness
- [Performance Tips](docs/how-to/performance-tips.md)
- [Debugging Checklist](docs/how-to/debugging-checklist.md)

### 📖 Reference

- [Estimator Contract](docs/reference/estimator-contract.md)
- [CLI Reference](docs/reference/cli-reference.md)
- [Score Report Fields](docs/reference/score-report-fields.md)
- [Code Patterns](docs/reference/code-patterns.md)

### 🔧 Troubleshooting

- [Common Participant Errors](docs/troubleshooting/common-participant-errors.md)
- [FAQ](docs/troubleshooting/faq.md)

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

Try them out (adjust `--n-mlps` to control evaluation size):

```bash
# Quick iteration (fewer MLPs = faster evaluation)
whest run --estimator examples/estimators/mean_propagation.py --n-mlps 3

# Full evaluation (default settings)
whest run --estimator examples/estimators/mean_propagation.py --n-mlps 10

# Compare estimators on the same dataset for fair scoring
whest create-dataset --n-mlps 100 -o eval.npz
whest run --estimator examples/estimators/mean_propagation.py --dataset eval.npz
whest run --estimator examples/estimators/covariance_propagation.py --dataset eval.npz
whest run --estimator examples/estimators/combined_estimator.py --dataset eval.npz
```

## 📡 Current Platform Status

This starter kit supports local development, validation, scoring, and packaging.

Hosted submission/upload instructions are not part of this repository yet; until then, use local `whest package` artifacts for iteration.

## ✅ Verification Commands

```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pyright
uv run --group dev pytest -m "not exhaustive"
uv run --group dev pytest -m exhaustive
```

## 🪝 Local Push Guardrail

Enable the repository pre-push hook so pushes are blocked until local lint/tests pass:

```bash
git config core.hooksPath .githooks
```

Run a check manually at any time:

```bash
.githooks/pre-push
```

Current checks include:

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pytest -m "not exhaustive" -x -q`

This setup requires `uv` to be installed and on `PATH`.

## 👥 Authors

- Paul Christiano
- Jacob Hilton
- Sharada Mohanty
