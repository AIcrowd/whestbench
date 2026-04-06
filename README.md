<div align="center">
  <img src="assets/logo/logo.png" alt="Network Estimation Challenge logo" style="height: 120px;">
  <br>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-green?logo=python&logoColor=white" alt="Python 3.10+"></a>
</div>

# Network Estimation Challenge

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
  <img src="assets/network-explorer-visualization.svg" alt="Network Explorer Visualization" width="100%">
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
git clone git@github.com:AIcrowd/network-estimation-challenge-internal.git
cd network-estimation-challenge-internal
uv tool install -e .
```

Sanity-check CLI wiring:

```bash
nestim smoke-test
```

Explore networks visually (requires Node.js):

```bash
nestim visualizer
```

Run your first full loop:

```bash
nestim init ./my-estimator
nestim validate --estimator ./my-estimator/estimator.py
nestim run --estimator ./my-estimator/estimator.py
nestim package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz
```

`nestim run` uses `--runner server` by default.

For faster repeated evaluations, pre-create a dataset and reuse it:

```bash
nestim create-dataset -o my_dataset.npz
nestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
```

Quick debug sequence when `run` fails:

```bash
nestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz
nestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz --debug
nestim run --estimator ./my-estimator/estimator.py --dataset my_dataset.npz --runner local --debug
```

For local editable invocation without global install:

```bash
uv run --with-editable . nestim smoke-test
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
- [Inspect and Traverse MLP Structure](docs/how-to/inspect-mlp-structure.md)
- [Validate, Run, and Package](docs/how-to/validate-run-package.md)
- [Use Evaluation Datasets](docs/how-to/use-evaluation-datasets.md)
- [Use Network Explorer](docs/how-to/use-network-explorer.md)
- [Profile Simulation Backends](docs/how-to/profile-simulation-backends.md) — profile mechestim FLOP usage and analytical correctness

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

Try them out (adjust `--n-mlps` to control evaluation size):

```bash
# Quick iteration (fewer MLPs = faster evaluation)
nestim run --estimator examples/estimators/mean_propagation.py --n-mlps 3

# Full evaluation (default settings)
nestim run --estimator examples/estimators/mean_propagation.py --n-mlps 10

# Compare estimators on the same dataset for fair scoring
nestim create-dataset --n-mlps 100 -o eval.npz
nestim run --estimator examples/estimators/mean_propagation.py --dataset eval.npz
nestim run --estimator examples/estimators/covariance_propagation.py --dataset eval.npz
nestim run --estimator examples/estimators/combined_estimator.py --dataset eval.npz
```

## 📡 Current Platform Status

This starter kit supports local development, validation, scoring, and packaging.

Hosted submission/upload instructions are not part of this repository yet; until then, use local `nestim package` artifacts for iteration.

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
