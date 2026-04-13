# MechEstim Primer

MechEstim is a numpy-compatible array library that tracks FLOPs analytically rather than timing them on hardware. Every arithmetic operation on a `me.ndarray` increments a FLOP counter instead of (or in addition to) performing the computation. This is how WhestBench enforces fair FLOP budgets across different machines.

Source: [github.com/AIcrowd/mechestim](https://github.com/AIcrowd/mechestim)

## BudgetContext

All estimator predictions run inside a `BudgetContext`. When the budget is exhausted, a `BudgetExhaustedError` is raised and your predictions are zeroed out.

```python
import mechestim as me

with me.BudgetContext(flop_budget=1_000_000) as ctx:
    x = me.ones(100)
    y = x @ me.eye(100)  # matmul: 2 * 100 * 100 * 100 = 2M FLOPs
    # BudgetExhaustedError raised here if budget exceeded
```

You don't need to create `BudgetContext` yourself — the framework does it before calling your `predict()` method. The `budget` argument tells you how many FLOPs you have.

## Operation FLOP Costs

| Category | Operations | Cost |
|----------|-----------|------|
| **Free** (0 FLOPs) | `me.array`, `me.zeros`, `me.ones`, `me.eye`, `me.asarray`, `me.reshape`, `.T`, indexing, `me.stack`, `me.concatenate`, `.copy()`, `.astype()` | 0 |
| **Pointwise** (1 FLOP/element) | `+`, `-`, `*`, `/`, `me.exp`, `me.sqrt`, `me.abs`, `me.maximum`, `me.where`, `me.log`, comparisons | N elements |
| **Reductions** (input size) | `me.sum`, `me.mean`, `me.var`, `me.max`, `me.min`, `me.all`, `me.any` | N elements |
| **Matmul** | `@`, `me.matmul` | 2 * M * N * K for (M,N) @ (N,K) |

**Key insight:** Matmul dominates. A single `(100, 100) @ (100, 100)` costs 2M FLOPs. A pointwise `exp` on 100 elements costs 100 FLOPs.

## Array Creation

```python
import mechestim as me

x = me.zeros(100)                          # 1D zeros
X = me.zeros((64, 100), dtype=me.float32)  # 2D zeros, explicit dtype
I = me.eye(100, dtype=me.float32)          # identity matrix
a = me.array([1.0, 2.0, 3.0])             # from list
b = me.asarray(numpy_array)                # convert from numpy (free)
```

All array creation is **free** (0 FLOPs).

## Random Number Generation

```python
import mechestim as me

rng = me.random.default_rng(42)            # seeded RNG
x = rng.standard_normal((1000, 64))        # Gaussian samples
x = x.astype(me.float32)                   # cast to float32 (free)
```

Random generation itself is free. FLOPs are counted when you operate on the arrays.

## Budget Inspection

Use `me.budget_summary()` inside a `BudgetContext` to see current usage:

```python
with me.BudgetContext(flop_budget=10_000_000) as ctx:
    # ... your computations ...
    print(me.budget_summary())  # shows FLOPs used so far
    print(ctx.flops_used)       # integer FLOP count
```

This is useful during development to understand where your budget goes.

## Common Gotchas

**numpy arrays still count FLOPs.** Since `me.ndarray` is backed by numpy, a raw numpy array passed to mechestim operations will still be tracked. Use `me.array()` or `me.asarray()` to convert explicitly.

**Pythonic operators are tracked.** `x @ w` counts the same FLOPs as `me.matmul(x, w)`. Use whichever reads better.

**dtype matters for precision, not FLOPs.** `float32` and `float64` operations cost the same FLOPs. Use `float32` for memory efficiency and `float64` for numerical stability where needed.

## Testing

Use mechestim's testing utilities:

```python
import mechestim as me

me.testing.assert_allclose(actual, expected, atol=1e-6)
me.testing.assert_array_equal(actual, expected)
```

These work like numpy's testing functions but on mechestim arrays.
