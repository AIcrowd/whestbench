# Estimator Contract

## When to use this page

Use this page when you need exact estimator I/O requirements.

## Required interface

`predict(self, mlp: MLP, budget: int) -> fnp.ndarray`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

### What `setup()` costs on the grader

`setup()` runs outside the per-MLP FLOP budget, and outside the
`residual_wall_time_s` penalty too — nothing it does is billed to any MLP.
That part is worth relying on.

What it is not is a once-per-submission hook. The grader runs `setup()` once
per worker process, and a submission is served by several workers (a worker
also restarts its participant process after certain failures), so in practice
`setup()` runs roughly 5-15 times per submission. Each run gets its own hard
**5 s** budget; blowing it fails the whole submission with `SETUP_TIMEOUT`, not
just one MLP. `predict()` has a separate hard per-MLP wall-clock cap — setup
does not borrow from it and cannot spend it.

So `setup()` is the place to *load* precomputed work, not to *do* it. Compute
offline, ship the artifact, and read it here.

The 5 s cap is the same locally and on the grader: `whest run` and
`whest validate` enforce it too, so a local `SETUP_TIMEOUT` means what a graded
one does. Budget accordingly — 5 s is enough to read an `.npz`, and is not
enough to build one.

### `SetupContext` fields

| Field | Type | Description |
|---|---|---|
| `width` | `int` | Neuron count for generated MLPs |
| `depth` | `int` | Number of layers per MLP |
| `flop_budget` | `int` | FLOP cap for the estimator |
| `api_version` | `str` | Contract version string |
| `scratch_dir` | `str \| None` | Optional writable directory for caching |
| `submission_dir` | `str \| None` | Read-only path to the extracted submission directory. All files shipped beside `estimator.py` are extracted here and importable as modules. Load shipped data files with `fnp.load(Path(ctx.submission_dir) / "weights.npz")` (0 FLOPs, pickle-free) or `MyModel.from_file(...)`. `None` outside the runner/grader (e.g. in plain `whest run` without a packaged submission). |
| `seed` | `int` | Per-run seed from `--seed` (default `0`). Use in `setup()` to reproduce one-time random initialisation. See [Setup-time reproducibility](#setup-time-reproducibility). |

### Multi-file submissions

Submissions are folder-based: `whest package` bundles every file in the `estimator.py` directory (minus the built-in ignore set, `.gitignore`, and `.whestignore`). Helper modules placed next to `estimator.py` are automatically included and importable at run time — no special registration needed.

```python
# myproject/
#   estimator.py       ← entrypoint
#   layers.py          ← helper module
#   weights.npz        ← pre-trained weights

# estimator.py
from pathlib import Path
import flopscope.numpy as fnp
from layers import build_model   # importable: layers.py is in the same directory

class MyEstimator(BaseEstimator):
    def setup(self, ctx: SetupContext) -> None:
        if ctx.submission_dir is not None:
            data = fnp.load(Path(ctx.submission_dir) / "weights.npz")
            self.model = build_model(data["W"])

    def predict(self, mlp, budget):
        ...
```

`ctx.submission_dir` points to the directory where the submission was extracted; it is `None` when running outside a packaged submission context (e.g. bare `whest run --estimator estimator.py` without a tarball). Guard with `if ctx.submission_dir is not None` or fall back to a default.

Loading shipped weights with `fnp.load` (or `np.load`) costs **0 FLOPs** — the load happens in `setup()`, which runs before any FLOP tracking starts. If you use a `flops.Module` subclass, call `MyModel.from_file(Path(ctx.submission_dir) / "model.npz")` in `setup()` for the same reason.

See [Ship weights and helper modules](https://github.com/AIcrowd/whestbench/blob/main/docs/how-to/ship-weights.md) for a step-by-step walkthrough.

## Input object quick reference

| Object | Field | Meaning |
|---|---|---|
| `MLP` | `width` | Number of neurons per layer |
| `MLP` | `depth` | Number of weight matrices (layers) |
| `MLP` | `weights` | Ordered weight matrices, each `(width, width)` |
| `MLP` | `seed` | Per-MLP grader-supplied seed; use this to seed estimator-internal randomness for reproducibility under regrade. |
| `MLP` | `name` | Human-readable slug like `"danielle-johnson"` derived deterministically from `seed`. Stable across runs and CPU/GPU backends at the WhestBench release's pinned `faker` version. Useful for log lines and error messages; safe to ignore. Empty string only when an `MLP` is constructed outside an evaluator bake path. |

For traversal examples, see [Inspect and Traverse MLP Structure](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/inspect-mlp-structure.md) (in the starter kit).

## Output requirements per `predict` call

| Requirement | Rule |
|---|---|
| Shape | Return a 2D array with shape `(mlp.depth, mlp.width)` |
| Numeric validity | Every value is finite |

## FLOP tracking

Your estimator must use flopscope primitives (`import flopscope as flops
import flopscope.numpy as fnp`) for all numerical computation. flopscope tracks FLOP usage analytically. If the total FLOPs across your entire `predict` call exceed `flop_budget`, all predictions for that MLP are replaced with zero vectors and your MSE for that MLP is computed against zeros.

### Ops not supported on the grader

Five NumPy callback ops — `apply_along_axis`, `apply_over_axes`, `fromfunction`, `fromiter`, `frompyfunc` — run in-process locally but raise `flopscope.errors.RemoteCallbackError` on the AIcrowd grader (they run a Python callback that can't cross the client/server boundary). Enumerate them at runtime with `flopscope.remote_unsupported_ops()`. Precompute the result or avoid these ops in submitted code.

## Failure semantics

When `predict()` cannot return a valid result — for any reason — the affected MLP is
scored as if the estimator had returned a zero array, and the multiplier in the
budget-adjusted score `s_m` is forced to `1.0` (no compute discount). Concretely:

- **FLOP budget exhausted** (`flopscope.BudgetExhaustedError`) → `Y_hat = 0`, `s_m = MSE(0, Y) * 1.0`
- **Wall-time / residual-time budget exhausted** → same
- **Combined-budget post-check** (`C_m = F_m + λ·R_m > B_m`) → same
- **`predict()` raised an exception** (any subclass of `Exception`, including `MemoryError`,
  `ValueError` from `validate_predictions`, custom estimator exceptions) → same
- **Invalid output shape** (not `(depth, width)`) → same
- **Non-finite values** (any `inf` or `NaN`) → same
- **Subprocess worker hard-killed** (OOM, segfault, timeout, non-zero exit) → same

The scoring loop continues across the remaining MLPs and produces a finite `adjusted_final_layer_score`.
Per-MLP diagnostic fields (`error`, `error_code`, `traceback`, `budget_exhausted`,
`time_exhausted`, `residual_wall_time_exhausted`, `combined_budget_exhausted`) are preserved
so failures remain debuggable.

The "no compute discount on failure" rule (multiplier forced to 1.0) ensures that a failed
run is strictly worse than a trivial-zero submission that succeeds (which receives the
0.1 multiplier floor — the minimum discount, a factor-of-ten cap).

## Memory limit

`ContestSpec.memory_limit_mb` (default `65_536`, i.e. 64 GB — matches the Phase 1 grader allocation) bounds the address space available to your estimator. Enforcement depends on the runner:

- **`--runner subprocess`** (used by the grader): the worker calls `resource.setrlimit(RLIMIT_AS, ...)` before importing your estimator module. Any allocation that would exceed the cap raises `MemoryError` inside `predict()`, which routes through the failure path described above (zero-prediction MSE × 1.0).
- **`--runner local`**: the limit is advisory only. WhestBench cannot safely call `setrlimit` on the CLI process itself. The runner emits a single warning at start (`"memory_limit_mb=… is advisory in --runner local: enforcement requires --runner subprocess (uses RLIMIT_AS) or external sandboxing (cgroups)."`) and continues without enforcement. Use `--runner subprocess` if you want the limit actually enforced locally.

Platforms without `RLIMIT_AS` (Windows, some BSDs) log a warning to the worker's stderr and continue without enforcement. The grader's evaluation environment is Linux, where enforcement is reliable.

## Wall-clock cap

`ContestSpec.wall_time_limit_s` (default `60.0` seconds — matches the Phase 1 grader cap) is an operational backstop on per-MLP `predict()` execution. If a single `predict()` call's elapsed wall-clock time exceeds the cap, the estimator's prediction is replaced with zeros and the MLP is scored through the failure path (zero-prediction MSE × 1.0, no compute discount). This is intentionally generous — the primary compute constraint is the effective FLOP budget `C_m = F_m + λ·R_m`; the wall-clock cap only catches stalled or runaway submissions.

The CLI flag `--wall-time-limit SECONDS` accepts a positive float. To disable the cap programmatically, construct your own `ContestSpec` with `wall_time_limit_s=None`.

## Reproducibility under the grader seed

### Predict-time reproducibility

If your estimator uses randomness — Monte Carlo sampling, randomized hashing,
random projections, etc. — seed it from `mlp.seed`. The grader supplies a fixed
per-MLP seed that is identical across all submissions for a given MLP, derived
deterministically from the suite seed. Submissions that use unseeded randomness
or their own seeds are NOT guaranteed to reproduce under regrade and may be
disqualified for prize eligibility.

Example:

```python
import flopscope.numpy as fnp

def predict(self, mlp, budget):
    rng = fnp.random.default_rng(mlp.seed)
    # ... use rng for any internal randomness
```

If your estimator is deterministic (no internal randomness), you can ignore `mlp.seed`.

### Setup-time reproducibility

If your estimator does randomized one-time setup (e.g., sampling a random
projection basis, jittering initial weights, choosing random hyperparameters),
seed it from `ctx.seed` inside `setup()`. When the grader passes `--seed`, the same value is forwarded to `ctx.seed` for every MLP in the run; participants running locally can pass `--seed` themselves to reproduce a given setup.

```python
import flopscope.numpy as fnp

def setup(self, ctx: SetupContext) -> None:
    self.setup_rng = fnp.random.default_rng(ctx.seed)
    # ... use self.setup_rng for any one-time random work
```

Do **not** call `fnp.random.seed(ctx.seed)` (or `np.random.seed(ctx.seed)`) —
that mutates the process-global RNG and breaks composability with other
libraries. Use `fnp.random.default_rng(ctx.seed)` to get an isolated `Generator`.

`ctx.seed` defaults to `0` when no `--seed` was passed; estimators that don't
read it are unaffected. The seed is recorded in the run output under
`run_config.seed` for audit-trail purposes — a reviewer can read it from a
participant's JSON output and re-run with `--seed N` to reproduce the
participant's setup state. See [score-report-fields.md](score-report-fields.md)
for the `run_config.seed` field.

`ctx.seed` and `mlp.seed` are independent: `mlp.seed` controls per-MLP
randomness inside `predict()`, `ctx.seed` controls one-time setup. With
`--dataset`, the dataset supplies `mlp.seed` values (baked at the dataset's
own seed) while `--seed` controls `ctx.seed` only. See
[cli-reference.md](cli-reference.md) for the `--seed` flag semantics.

## Next step

- [Write an Estimator](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/how-to/write-an-estimator.md) (in the starter kit)
- [Common Participant Errors](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/troubleshooting/common-participant-errors.md) (in the starter kit)
