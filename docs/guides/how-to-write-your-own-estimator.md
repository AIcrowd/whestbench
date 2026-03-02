# How To Write Your Own Estimator

Your job is to implement one class: `Estimator`, subclassing `BaseEstimator`.

## Contract You Must Follow

Required method:

- `predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]`

Optional lifecycle hooks:

- `setup(self, context: SetupContext) -> None`
- `teardown(self) -> None`

Output requirements in `predict`:

- emit exactly `circuit.d` rows
- each row shape must be `(circuit.n,)`
- values must be finite
- stream rows with `yield`

Do not return a single `(depth, width)` array at the end.

## Minimal Skeleton

```python
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from circuit_estimation import BaseEstimator
from circuit_estimation.domain import Circuit


class Estimator(BaseEstimator):
    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        x_mean = np.zeros(circuit.n, dtype=np.float32)
        for _layer in circuit.gates:
            # Replace with your real update logic.
            yield x_mean
```

## Recommended Starting Point

Read and run in this order:

1. `examples/estimators/random_estimator.py`
2. `examples/estimators/mean_propagation.py`
3. `examples/estimators/covariance_propagation.py`
4. `examples/estimators/combined_estimator.py`

## Common Mistakes

1. Returning final tensor instead of streaming rows.
2. Wrong row shape.
3. Too few or too many rows.
4. Non-finite values.
5. Ignoring budget entirely in advanced estimators.

## Budget-Aware Tuning Intuition

`budget` is a sampling trial-count signal used to derive timing envelopes.
Higher budget usually means more allowed compute before timeout penalties apply.

Practical path:

1. Implement a cheap safe baseline.
2. Add a higher-quality path for larger budgets.
3. Switch by `budget` (and optionally by circuit size).
4. Measure with `cestim run --detail full --profile`.
