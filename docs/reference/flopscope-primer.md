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

Both summaries can also include timing data:

- `wall_time_s`: total elapsed time in the context
- `tracked_time_s`: time spent inside counted flopscope calls
- `untracked_time_s`: everything else in the context

This is useful during development to understand where both FLOPs and time go.

## WhestBench-specific limits

Flopscope itself only knows about `wall_time_limit_s` on `BudgetContext`.
WhestBench adds two run-level knobs on top:

- `--wall-time-limit`: passed through to the estimator's `BudgetContext`
- `--untracked-time-limit`: enforced by WhestBench after `predict()` returns,
  using the reported `untracked_time_s`

So if you see `untracked_time_exhausted` in a report, that came from
WhestBench scoring logic, not from a `BudgetContext` parameter.

## Common Gotchas

**numpy arrays still count FLOPs.** Since `fnp.ndarray` is backed by numpy, a raw numpy array passed to flopscope operations will still be tracked. Use `fnp.array()` or `fnp.asarray()` to convert explicitly.

**Pythonic operators are tracked.** `x @ w` counts the same FLOPs as `fnp.matmul(x, w)`. Use whichever reads better.

**dtype matters for precision, not FLOPs.** `float32` and `float64` operations cost the same FLOPs. Use `float32` for memory efficiency and `float64` for numerical stability where needed.

## Testing

Use flopscope's testing utilities:

```python
import flopscope as flops
import flopscope.numpy as fnp

fnp.testing.assert_allclose(actual, expected, atol=1e-6)
fnp.testing.assert_array_equal(actual, expected)
```

These work like numpy's testing functions but on flopscope arrays.
