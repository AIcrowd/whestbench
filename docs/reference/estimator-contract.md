# Estimator Contract

## When to use this page

Use this page when you need exact estimator I/O requirements.

## Required interface

`predict(self, mlp: MLP, budget: int) -> fnp.ndarray`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

### `SetupContext` fields

| Field | Type | Description |
|---|---|---|
| `width` | `int` | Neuron count for generated MLPs |
| `depth` | `int` | Number of layers per MLP |
| `flop_budget` | `int` | FLOP cap for the estimator |
| `api_version` | `str` | Contract version string |
| `scratch_dir` | `str \| None` | Optional writable directory for caching |

## Input object quick reference

| Object | Field | Meaning |
|---|---|---|
| `MLP` | `width` | Number of neurons per layer |
| `MLP` | `depth` | Number of weight matrices (layers) |
| `MLP` | `weights` | Ordered weight matrices, each `(width, width)` |
| `MLP` | `seed` | Per-MLP grader-supplied seed; use this to seed estimator-internal randomness for reproducibility under regrade. |

For traversal examples, see [Inspect and Traverse MLP Structure](../how-to/inspect-mlp-structure.md).

## Output requirements per `predict` call

| Requirement | Rule |
|---|---|
| Shape | Return a 2D array with shape `(mlp.depth, mlp.width)` |
| Numeric validity | Every value is finite |

## FLOP tracking

Your estimator must use flopscope primitives (`import flopscope as flops
import flopscope.numpy as fnp`) for all numerical computation. flopscope tracks FLOP usage analytically. If the total FLOPs across your entire `predict` call exceed `flop_budget`, all predictions for that MLP are replaced with zero vectors and your MSE for that MLP is computed against zeros.

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

The scoring loop continues across the remaining MLPs and produces a finite `adjusted_final_layer_mse`.
Per-MLP diagnostic fields (`error`, `error_code`, `traceback`, `budget_exhausted`,
`time_exhausted`, `residual_wall_time_exhausted`, `combined_budget_exhausted`) are preserved
so failures remain debuggable.

The "no compute discount on failure" rule (multiplier forced to 1.0) ensures that a failed
run is strictly worse than a trivial-zero submission that succeeds (which receives the
0.5 multiplier floor — the minimum discount).

## Memory limit

`ContestSpec.memory_limit_mb` (default `4096`) bounds the address space available to your estimator. Enforcement depends on the runner:

- **`--runner subprocess`** (used by the grader): the worker calls `resource.setrlimit(RLIMIT_AS, ...)` before importing your estimator module. Any allocation that would exceed the cap raises `MemoryError` inside `predict()`, which routes through the failure path described above (zero-prediction MSE × 1.0).
- **`--runner local`**: the limit is advisory only. WhestBench cannot safely call `setrlimit` on the CLI process itself. The runner emits a single warning at start (`"memory_limit_mb=… is advisory in --runner local: enforcement requires --runner subprocess (uses RLIMIT_AS) or external sandboxing (cgroups)."`) and continues without enforcement. Use `--runner subprocess` if you want the limit actually enforced locally.

Platforms without `RLIMIT_AS` (Windows, some BSDs) log a warning to the worker's stderr and continue without enforcement. The grader's evaluation environment is Linux, where enforcement is reliable.

## Reproducibility under the grader seed

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

## Next step

- [Write an Estimator](../how-to/write-an-estimator.md)
- [Common Participant Errors](../troubleshooting/common-participant-errors.md)
