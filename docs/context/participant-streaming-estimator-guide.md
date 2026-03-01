# Participant Guide: Streaming Estimators

This guide explains the only contract you need to implement:

- write `predict(circuit, budget)`
- `yield` one `(width,)` prediction row per circuit depth
- yield exactly `circuit.d` rows

You do not need to edit scorer internals or runner logic.
You can start from the class-based templates in `examples/estimators/`.

## Minimal Skeleton

```python
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from circuit_estimation.domain import Circuit
from circuit_estimation.sdk import BaseEstimator


class Estimator(BaseEstimator):
    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        x_mean = np.zeros(circuit.n, dtype=np.float32)
        for layer in circuit.gates:
            first = x_mean[layer.first]
            second = x_mean[layer.second]
            x_mean = (
                layer.first_coeff * first
                + layer.second_coeff * second
                + layer.const
                + layer.product_coeff * first * second
            ).astype(np.float32)
            yield x_mean
```

## Common Mistakes

1. Returning one full `(depth, width)` tensor at the end instead of streaming rows.
2. Yielding vectors with wrong shape (must be `(circuit.n,)`).
3. Yielding too few or too many rows (must be exactly `circuit.d`).
4. Yielding non-finite values (`nan`, `inf`).

## Budget-by-Depth Tuning Intuition

Scoring compares your cumulative elapsed time at each depth against a baseline:

- baseline: `time_budget_by_depth_s[i]`
- timeout rule: if you are slower than `(1 + tolerance) * baseline` at depth `i`, that depth row is zeroed
- floor rule: if you are faster than `(1 - tolerance) * baseline`, effective time is floored

Practical approach:

1. Start with a cheap method that always emits valid rows.
2. Add a higher-quality branch for larger budgets.
3. Switch methods early in `predict` based on `budget` and `circuit.n`.
4. Measure scores across multiple budgets to tune your threshold.

## Local Debugging Checklist

1. Run `cestim --agent-mode` and inspect JSON for `time_budget_by_depth_s`, `timeout_rate_by_depth`, and score.
2. Run `cestim --detail full --profile` to inspect profile diagnostics.
3. If evaluation fails, fix row shape/count/finite issues first before tuning math quality.
