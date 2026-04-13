# Code Patterns

Quick reference for mechestim operations. All examples assume `import mechestim as me`.

## Operators are tracked

Python arithmetic operators (`+`, `-`, `*`, `/`, `@`) on `me.ndarray` values are
FLOP-tracked — you do not need to use the verbose `me.add`, `me.multiply`, etc. forms.

```python
import mechestim as me

a = me.ones(4)
b = me.ones(4)

# These are all equivalent and all tracked:
c = a + b           # tracked: same as me.add(a, b)
d = a * b           # tracked: same as me.multiply(a, b)
e = a / b           # tracked: same as me.divide(a, b)

W = me.eye(4)
v = me.ones(4)
f = W @ v           # tracked: same as me.matmul(W, v)
g = W.T @ v         # tracked: transpose is free, matmul is tracked
h = W.T @ W @ v     # tracked: two matmuls, chained with @
```

Use operators whenever they improve readability. The verbose `me.*` forms are still
available but are no longer required for tracking purposes.

## Operation costs

| What you want | Code | FLOP cost | Notes |
|---|---|---|---|
| Create zeros | `me.zeros((n, n))` | 0 | Free |
| Create ones | `me.ones(n)` | 0 | Free |
| Identity matrix | `me.eye(n)` | 0 | Free |
| Wrap existing data | `me.array(data)` | 0 | Free |
| Matrix multiply | `me.matmul(A, B)` | O(m x n x k) | Dominates budgets |
| Element-wise add | `me.add(a, b)` | 1 per element | |
| Element-wise multiply | `me.multiply(a, b)` | 1 per element | |
| Element-wise divide | `me.divide(a, b)` | 1 per element | |
| ReLU | `me.maximum(x, 0.0)` | 1 per element | |
| Square root | `me.sqrt(x)` | 1 per element | |
| Exponential | `me.exp(x)` | 1 per element | |
| Logarithm | `me.log(x)` | 1 per element | |
| Transpose | `me.transpose(W)` | 0 | Free |
| Reshape | `me.reshape(x, shape)` | 0 | Free |
| Extract diagonal | `me.diag(M)` | 0 | Free |
| Set diagonal | `me.fill_diagonal(M, v)` | 0 | Free, in-place |
| Outer product | `me.outer(a, b)` | n x m | |
| Sum | `me.sum(x, axis=0)` | input size | |
| Mean | `me.mean(x, axis=0)` | input size | |
| Max | `me.max(x)` | input size | |
| Stack arrays | `me.stack(rows, axis=0)` | 0 | Free |
| Concatenate | `me.concatenate([a, b])` | 0 | Free |
| Index/slice | `x[0]`, `x[:, 3]` | 0 | Free |

## Common patterns

### Standard normal PDF (for ReLU expectation)

```python
import mechestim as me

def norm_pdf(x):
    """phi(x) = exp(-x^2/2) / sqrt(2*pi)"""
    return me.exp(-0.5 * x * x) / me.sqrt(2.0 * me.pi)
```

### Standard normal CDF

Pure mechestim implementation using the Abramowitz & Stegun approximation (accurate to <7.5e-8):

```python
import mechestim as me

_P = 0.2316419
_A1, _A2, _A3 = 0.319381530, -0.356563782, 1.781477937
_A4, _A5 = -1.821255978, 1.330274429

def norm_cdf(x):
    t = 1.0 / (1.0 + _P * me.abs(x))
    poly = ((((_A5 * t + _A4) * t + _A3) * t + _A2) * t + _A1) * t
    pdf = me.exp(-0.5 * x * x) / me.sqrt(2.0 * me.pi)
    cdf = 1.0 - pdf * poly
    return me.where(x >= 0, cdf, 1.0 - cdf)
```

Alternatively, if you add `scipy` to your `requirements.txt`:

```python
# Optional: requires scipy as a user-provided dependency
from scipy.special import ndtr

def norm_cdf(x):
    return me.array(ndtr(me.asarray(x, dtype=me.float64)).astype(me.float32))
```

### ReLU expectation (E[max(0, z)] where z ~ N(mu, sigma^2))

```python
import mechestim as me

alpha = mu_pre / sigma_pre
E_relu = mu_pre * norm_cdf(alpha) + sigma_pre * norm_pdf(alpha)
```

See [`examples/estimators/mean_propagation.py`](../../examples/estimators/mean_propagation.py) for a complete working estimator using these patterns.

### Per-neuron variance propagation (diagonal)

```python
import mechestim as me

# var_pre[i] = sum_j W[j,i]^2 * var[j]
var_pre = (w * w).T @ var
```

## Next step

- [Manage Your FLOP Budget](../how-to/manage-flop-budget.md)
- [Algorithm Ideas](../how-to/algorithm-ideas.md)
- [Estimator Contract](./estimator-contract.md)
