# Flopscope Primer

Flopscope is a numpy-compatible array library that tracks FLOPs analytically rather than timing them on hardware. Every arithmetic operation on a `fnp.ndarray` increments a FLOP counter instead of (or in addition to) performing the computation. This is how WhestBench enforces fair FLOP budgets across different machines.

Source: [github.com/AIcrowd/flopscope](https://github.com/AIcrowd/flopscope)

## BudgetContext

All estimator predictions run inside a `BudgetContext`. When the budget is exhausted, a `BudgetExhaustedError` is raised and your predictions are zeroed out.

```python
import flopscope as flops
import flopscope.numpy as fnp

with flops.BudgetContext(flop_budget=1_000_000) as ctx:
    x = fnp.ones(100)
    y = x @ fnp.eye(100)  # matmul: 100 * 100 * 100 = 1M FLOPs
    # BudgetExhaustedError raised here if budget exceeded
```

You don't need to create `BudgetContext` yourself — the framework does it before calling your `predict()` method. The `budget` argument tells you how many FLOPs you have.

`BudgetContext` also supports `wall_time_limit_s` when you want a cooperative
wall-clock limit in addition to the FLOP cap:

```python
with flops.BudgetContext(flop_budget=1_000_000, wall_time_limit_s=2.0) as ctx:
    ...
```

The timer starts when the context is entered and is checked before and after
each counted flopscope/NumPy call. If it is exceeded, flopscope raises
`TimeExhaustedError`.

## Operation FLOP Costs

| Category | Operations | Cost |
|----------|-----------|------|
| **Free** (0 FLOPs) | `fnp.array`, `fnp.zeros`, `fnp.ones`, `fnp.eye`, `fnp.asarray`, `fnp.reshape`, `.T`, indexing, `fnp.stack`, `fnp.concatenate`, `.copy()`, `.astype()` | 0 |
| **Pointwise** (1 FLOP/element) | `+`, `-`, `*`, `/`, `fnp.exp`, `fnp.sqrt`, `fnp.abs`, `fnp.maximum`, `fnp.where`, `fnp.log`, comparisons | N elements |
| **Reductions** (input size) | `fnp.sum`, `fnp.mean`, `fnp.var`, `fnp.max`, `fnp.min`, `fnp.all`, `fnp.any` | N elements |
| **Matmul** | `@`, `fnp.matmul` | M * N * K for (M,N) @ (N,K) |

**Key insight:** Matmul dominates. A single `(100, 100) @ (100, 100)` costs 1M FLOPs. A pointwise `exp` on 100 elements costs 100 FLOPs.

## Array Creation

```python
import flopscope as flops
import flopscope.numpy as fnp

x = fnp.zeros(100)                          # 1D zeros
X = fnp.zeros((64, 100), dtype=fnp.float32)  # 2D zeros, explicit dtype
I = fnp.eye(100, dtype=fnp.float32)          # identity matrix
a = fnp.array([1.0, 2.0, 3.0])             # from list
b = fnp.asarray(numpy_array)                # convert from numpy (free)
```

All array creation is **free** (0 FLOPs).

## Random Number Generation

```python
import flopscope as flops
import flopscope.numpy as fnp

rng = fnp.random.default_rng(42)            # seeded RNG
x = rng.standard_normal((1000, 64))        # Gaussian samples
x = x.astype(fnp.float32)                   # cast to float32 (free)
```

Random generation itself is free. FLOPs are counted when you operate on the arrays.

## Budget Inspection

Use `budget.summary()` for the current explicit context and
`fnp.budget_summary()` for the accumulated session/global view:

```python
with flops.BudgetContext(flop_budget=10_000_000) as ctx:
    # ... your computations ...
    print(ctx.summary())        # current context only
    print(fnp.budget_summary())  # process/session-wide summary
    print(ctx.flops_used)       # integer FLOP count
```

Both summaries also include four timing fields that satisfy a strict
decomposition identity, `wall_time_s = flopscope_backend_time_s + flopscope_overhead_time_s + residual_wall_time_s`:

- `wall_time_s`: total elapsed time in the context
- `flopscope_backend_time_s`: time spent inside counted flopscope numpy kernels
- `flopscope_overhead_time_s`: time spent inside flopscope's own dispatch (wrapper preambles, FLOP bookkeeping, namespace push/pop)
- `residual_wall_time_s`: everything else - participant Python, GC, and any ops not attributed to a flopscope backend or callback bucket (see the 0.7.0 note below)

This decomposition lets you see whether time is going to numpy compute, framework dispatch, or your own Python.

> **flopscope 0.7.0 timing re-attribution.** As of flopscope 0.7.0, data-movement NumPy ops (concatenate, stack-family, tile, repeat, take, pad, …) are timed as `flopscope_backend_time_s`, not `residual_wall_time_s`; Python-callback ops bill their callback time to residual. The identity `wall_time_s = flopscope_backend_time_s + flopscope_overhead_time_s + residual_wall_time_s` still holds; FLOP counts are unchanged.

## WhestBench-specific limits

Flopscope's `BudgetContext` measures `wall_time_s`, `flopscope_backend_time_s`,
`flopscope_overhead_time_s`, and `residual_wall_time_s`. It also accepts
`wall_time_limit_s`, which it checks while counted flopscope operations run.

WhestBench exposes some of those concepts as run-level CLI knobs:

- `--wall-time-limit`: passed through to the estimator's `BudgetContext`
- `--residual-wall-time-limit`: enforced by WhestBench after `predict()` returns,
  using the reported `residual_wall_time_s`. Because `residual_wall_time_s` no longer
  includes flopscope's own dispatch time, this gate measures only your
  Python work — not the framework's bookkeeping tax.

So if you see `time_exhausted`, that came from Flopscope's `wall_time_limit_s`.
If you see `residual_wall_time_exhausted`, that came from WhestBench scoring
logic comparing Flopscope's measured `residual_wall_time_s` with the configured
`--residual-wall-time-limit`.

## Residual wall-time charging (lambda)

WhestBench's effective compute budget combines analytical FLOPs and residual wall time
via a conversion rate `λ` (`whestbench.budget.LAMBDA_FLOPS_PER_SECOND`, default `1e11`;
configurable per run via `whest run --lambda-flops-per-second`):

```
C_m = F_m + λ · R_m
```

- `F_m` = analytical FLOPs counted by flopscope (`flops_used`)
- `R_m` = residual wall time — the third bucket of the time decomposition. Specifically,
  `residual_wall_time_s` = `wall_time_s − flopscope_backend_time_s − flopscope_overhead_time_s`.
  This is participant Python (loops, control flow), GC pauses, and Python-callback ops.
  It explicitly **excludes** flopscope's own dispatch overhead (the second bucket) and,
  as of flopscope 0.7.0, data-movement numpy ops (those now bill to backend).
- `λ` = the configured residual-penalty rate λ (default 1e11 FLOPs/s; set per-run with `whest run --lambda-flops-per-second`).

The combined `C_m` is capped at `B_m = flop_budget`. If `C_m > B_m`, the MLP is marked
`combined_budget_exhausted` and the prediction is replaced with zeros.

Why charge non-flopscope time at all? It lets participants use any Python they like —
not just flopscope-instrumented operations — but holds them accountable for that work
in the compute budget. Pure-flopscope solutions get the entire budget for analytical
work; pure-Python solutions trade some FLOP headroom for residual time.

## Common Gotchas

**numpy arrays still count FLOPs.** Since `fnp.ndarray` is backed by numpy, a raw numpy array passed to flopscope operations will still be tracked. Convert explicitly with `fnp.array()` or `fnp.asarray()` — but those conversions are themselves subject to the numeric-dtype rule below, so `fnp.array([1.0, None])` and `fnp.array(['a', 'b'])` raise `UnsupportedDtypeError` rather than producing an `object` or string array. Build numeric data with `fnp.array(..., dtype=fnp.float64)` from values that are already numeric scalars, and keep ragged or mixed data in a Python list of numeric arrays instead of one array. Converting with plain numpy first is not a remedy: the grader sandbox ships no numpy.

**Pythonic operators are tracked.** `x @ w` counts the same FLOPs as `fnp.matmul(x, w)`. Use whichever reads better.

**dtype decides both cost and admission.** dtype scales FLOP cost: the charged cost is `flop_cost * dtype_rate * complex_factor * weight`, and `float64` carries a rate of `2.0`, so a `float64` operation costs twice the same operation in `float32`. dtype also decides admission — flopscope accepts only numeric dtypes (`dtype.kind in "biufc"`: bool, signed and unsigned integer, float, complex). Anything else raises `UnsupportedDtypeError` (importable from `flopscope.errors`) wherever it reaches a registered operation, whether as an operand, an explicit `dtype=`, a fill value or distribution parameter, or an `out=` destination. The one carve-out is a dtype NumPy materialises with zero itemsize, such as an empty structured spec (`'V0'`); `'U0'` and `'S0'` are not exempt, because NumPy promotes them to `'U1'`/`'S1'` on allocation.

## Testing

Use flopscope's testing utilities:

```python
import flopscope as flops
import flopscope.numpy as fnp

fnp.testing.assert_allclose(actual, expected, atol=1e-6)
fnp.testing.assert_array_equal(actual, expected)
```

These work like numpy's testing functions but on flopscope arrays.
