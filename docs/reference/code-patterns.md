# Code Patterns

Quick reference for mechestim operations. All examples assume `import mechestim as me`.

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
    return me.multiply(
        me.exp(me.multiply(-0.5, me.multiply(x, x))),
        1.0 / float(me.sqrt(2.0 * me.pi)),
    )
```

### Standard normal CDF

mechestim wraps numpy — you can use scipy directly:

```python
import mechestim as me
from scipy.special import ndtr

def norm_cdf(x):
    return me.array(ndtr(me.asarray(x, dtype=me.float64)).astype(me.float32))
```

### ReLU expectation (E[max(0, z)] where z ~ N(mu, sigma^2))

```python
import mechestim as me

alpha = me.divide(mu_pre, sigma_pre)
E_relu = me.add(
    me.multiply(mu_pre, norm_cdf(alpha)),
    me.multiply(sigma_pre, norm_pdf(alpha)),
)
```

### Per-neuron variance propagation (diagonal)

```python
import mechestim as me

# var_pre[i] = sum_j W[j,i]^2 * var[j]
var_pre = me.matmul(me.transpose(me.multiply(w, w)), var)
```

## Next step

- [Manage Your FLOP Budget](../how-to/manage-flop-budget.md)
- [Algorithm Ideas](../how-to/algorithm-ideas.md)
- [Estimator Contract](./estimator-contract.md)
