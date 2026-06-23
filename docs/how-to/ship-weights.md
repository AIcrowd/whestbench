# Ship weights and helper modules with your submission

How to bundle pre-trained weights and multi-file code alongside `estimator.py`,
and load them efficiently at predict time.

## When to use this

- Your estimator reads offline-trained weights (numpy arrays, a custom model file, etc.)
- Your estimator spans more than one Python file (helper modules, utilities)
- You want to split a large estimator into composable pieces

## 1. Split code across modules

Put helper modules next to `estimator.py`. They ship automatically — no manifest
entries, no special registration.

```
my-submission/
  estimator.py    ← entrypoint; the file you pass to --estimator
  layers.py       ← helper module
  utils.py        ← another helper
  weights.npz     ← pre-trained weights
```

Import helpers with a plain relative import — they're on `sys.path` when the
submission is extracted:

```python
# estimator.py
from layers import build_model   # layers.py is in the same directory
from utils import normalize      # utils.py is in the same directory
```

If a `.py` file is not imported from `estimator.py`, `whest package` warns about
it (it's likely dead code). Add it to `.whestignore` if it shouldn't ship.

## 2. Author weights offline (plain numpy is fine)

Compute and save weights in a separate training script before packaging:

```python
# train.py  (not shipped — add to .whestignore)
import numpy as np

W1 = ...   # your trained weights
W2 = ...

np.savez("weights.npz", W1=W1, W2=W2)
```

`np.savez` writes a standard `.npz` archive (zip of `.npy` files, no pickle).
On the grader, load it with `fnp.load` (the flopscope wrapper) — plain `numpy`
is not available in the grader sandbox, so use `fnp.load`, not `np.load`.

## 3. Bundle: files are picked up automatically

Run `whest package` from any directory:

```bash
whest package --estimator ./my-submission/estimator.py
```

`whest package` bundles every file in the `estimator.py` directory except the
built-in ignore set (`.git`, `__pycache__`, `*.pyc`, `*.tar.gz`, etc.) and any
patterns in `.gitignore` / `.whestignore`.

The preview lists every file that will be included with its size and total count:

```
Packaging ./my-submission/estimator.py → submission-*.tar.gz
Files to bundle (4 files, 1.2 MB):
  estimator.py  (3.1 KB)
  layers.py     (1.8 KB)
  utils.py      (900 B)
  weights.npz   (1.2 MB)
Package these 4 files (1.2 MB)? [y/N]
```

Press `y` to confirm, or pass `--yes` / `-y` to skip the prompt (for CI):

```bash
whest package --estimator ./my-submission/estimator.py --yes
```

### Caps

Submissions are capped at **50 MB** total and **50 files**. If you exceed either
cap, the command aborts with a message naming the largest files. Exclude files
you don't need via `.whestignore`.

### Excluding scratch files with `.whestignore`

Add patterns to `.whestignore` (same glob syntax as `.gitignore`) to prevent
files from being bundled:

```
# .whestignore
train.py
*.log
scratch/
checkpoints/
```

`whest init` creates a starter `.whestignore` when scaffolding a new project.

## 4. Load weights at predict time

Load weights in `setup()`, not `predict()`. `setup()` runs once before FLOP
tracking starts, so the load costs **0 FLOPs**.

```python
from pathlib import Path
from typing import Optional

import flopscope as flops
import flopscope.numpy as fnp

from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP


class WeightedEstimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        if ctx.submission_dir is not None:
            # fnp.load is pickle-free (reads .npz zip archives). 0 FLOPs.
            data = fnp.load(Path(ctx.submission_dir) / "weights.npz")
            self.W1 = data["W1"]
            self.W2 = data["W2"]
        else:
            # Fallback for bare `whest run --estimator estimator.py`
            # (no packaged submission; ctx.submission_dir is None)
            self.W1 = fnp.zeros((4, 4))
            self.W2 = fnp.zeros((4, 4))

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        # Use self.W1 / self.W2 here — already loaded, no FLOPs charged
        ...
```

`ctx.submission_dir` is `None` outside a packaged submission (e.g. when you
run `whest run --estimator estimator.py` directly). Guard with
`if ctx.submission_dir is not None` or provide a sensible default.

### Using a `flops.Module` subclass

If your model is defined as a `flopscope.Module`, use `from_file` for the same
0-FLOPs load pattern:

```python
class MyModel(flops.Module):
    ...
    @classmethod
    def from_file(cls, path: Path) -> "MyModel":
        data = fnp.load(path)
        model = cls()
        model.weights = data["W"]
        return model


class MyEstimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        if ctx.submission_dir is not None:
            self.model = MyModel.from_file(Path(ctx.submission_dir) / "model.npz")
```

## End-to-end example

```
my-submission/
  estimator.py
  layers.py
  weights.npz
  .whestignore     (contains: train.py)
  train.py         (excluded — in .whestignore)
```

```bash
# 1. Train offline (not shipped)
uv run python train.py

# 2. Validate the estimator loads correctly
whest validate --estimator ./my-submission/estimator.py

# 3. Preview and package
whest package --estimator ./my-submission/estimator.py

# 4. Dry-run to verify what would be submitted
whest submit --estimator ./my-submission/estimator.py --dry-run

# 5. Submit
whest submit --estimator ./my-submission/estimator.py
```

## Next step

- [Estimator contract](../reference/estimator-contract.md) — full `SetupContext` field reference
- [CLI reference — whest package](../reference/cli-reference.md#whest-package) — flags, ignore rules, size caps
- [CLI reference — whest submit](../reference/cli-reference.md#whest-submit) — `--dry-run` and submission options
