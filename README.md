<div align="center">
  <img src="assets/logo/logo.png" alt="ARC Whitebox Estimation Challenge logo" style="height: 120px;">
  <br>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-green?logo=python&logoColor=white" alt="Python 3.10+"></a>
</div>

# ARC Whitebox Estimation Challenge — `whestbench`

Library + CLI for the ARC Whitebox Estimation Challenge. Generates random ReLU MLPs, runs FLOP-budgeted estimators against Monte Carlo ground truth, and produces score reports.

## For participants

**👉 Start at the [whest-starterkit](https://github.com/AIcrowd/whest-starterkit).** That repo is the on-ramp: a working `estimator.py`, four worked examples, stage-by-stage walkthroughs from "just iterate locally" to "package a submission".

For an interactive visualization of small random MLPs and estimator behavior, see the **[WhestBench Explorer](https://aicrowd.github.io/whestbench-explorer/)** — an in-browser companion that's optional but useful for building intuition.

This repo is the underlying engine. You don't need to clone it directly.

## For library / CLI users

```python
from whestbench import BaseEstimator, MLP, sample_mlp
import flopscope as flops
import flopscope.numpy as fnp


class MyEstimator(BaseEstimator):
    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        return fnp.zeros((mlp.depth, mlp.width))
```

CLI entry point (registered as both `whest` and `whestbench`):

```bash
whest validate --estimator path/to/estimator.py
whest run --estimator path/to/estimator.py --runner local
whest doctor
```

See `docs/reference/cli-reference.md` for the full command surface.

### Using evaluation datasets (schema 3.0)

Datasets are HuggingFace Hub repositories — Parquet files + metadata sidecar. Use
the published evaluation dataset directly, or bake your own:

```python
# Use the published evaluation dataset
from datasets import load_dataset
import whestbench

ds = load_dataset(
    "aicrowd/arc-whestbench-2026",
    revision="v1",
    split="public",
)

# Or bake your own (schema 3.0 directory output)
# $ whest dataset bake --n-mlps 8 --n-samples 1_000_000 --width 256 --depth 8 \
#       --seed 42 --output ./my-eval

# Iterate as MLP instances (same interface as on-the-fly sampling)
for mlp in whestbench.iter_mlps(ds):
    y_pred = my_estimator.predict(mlp)
```

Run evaluation against a published or local dataset:

```bash
# HF Hub (pinned revision required)
whest run --estimator ./estimator.py \
    --dataset hf://aicrowd/arc-whestbench-2026@v1

# Local directory
whest run --estimator ./estimator.py --dataset ./my-eval
```

See `docs/reference/dataset-format.md` for the schema 3.0 specification.

### Optional GPU backend

For baking large ground-truth datasets (`n_samples ≥ 10⁸`), install the torch
backend extra:

```bash
pip install whestbench[gpu]
```

Then use `whest dataset bake --torch --device auto ...`. See
[GPU Dataset Generation](docs/reference/gpu-dataset-generation.md) for details.
For parallel baking across multiple GPUs, see
[Parallel bake](docs/how-to/parallel-bake.md).

## Repository layout

```
src/whestbench/
├── __init__.py            ← public API surface
├── cli.py                 ← `whest`/`whestbench` entry point
├── concurrency.py         ← parallel execution helpers
├── dataset.py             ← evaluation dataset I/O (schema 3.0 bake + load)
├── dataset_io.py          ← Parquet+sidecar on-disk I/O, merge
├── dataset_torch.py       ← GPU/torch backend for dataset baking
├── doctor.py              ← `whest doctor` environment checks
├── domain.py              ← MLP, SetupContext, scoring spec
├── estimators.py          ← BaseEstimator + reference impls (mean/cov/combined)
├── generation.py          ← sample_mlp
├── hardware.py            ← hardware probing
├── hub.py                 ← publish_dataset (HF Hub upload)
├── loader.py              ← estimator module loading
├── packaging.py           ← submission packaging
├── presentation/          ← Rich rendering helpers
├── profiler.py            ← FLOP profiler integration
├── protocol.py            ← Server runner JSON protocol
├── reporting.py           ← Rich score report + smoke panels
├── runner.py              ← local/server runner orchestration
├── scoring.py             ← evaluate_estimator, ContestSpec
├── sdk.py                 ← Python SDK surface
├── simulation.py          ← Monte Carlo ground truth via flopscope
├── subprocess_worker.py   ← isolated estimator subprocess
└── templates/             ← `whest init` + dataset card Jinja2 templates
docs/
├── index.md               ← Library/CLI reference index
├── how-to/                ← Task walkthroughs (publish-to-hf-hub, parallel-bake)
└── reference/             ← cli-reference, dataset-format, estimator-contract, ...
```

## Releases

Tagged via `release-please`. See `docs/RELEASING.md`.

Underlying FLOP accounting library: [`AIcrowd/flopscope`](https://github.com/AIcrowd/flopscope) (replaced the deprecated `whest`).

## License

See [LICENSE](LICENSE).
