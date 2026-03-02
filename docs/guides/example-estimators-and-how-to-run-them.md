# Example Estimators And How To Run Them

The starter kit includes four example estimators.

## 1) Random Estimator

File: `examples/estimators/random_estimator.py`

- best first read for lifecycle and stream contract
- intentionally poor predictive quality

Run:

```bash
cestim run --estimator examples/estimators/random_estimator.py --runner subprocess
```

## 2) Mean Propagation

File: `examples/estimators/mean_propagation.py`

- first-moment baseline
- fast and simple

Run:

```bash
cestim run --estimator examples/estimators/mean_propagation.py --runner subprocess
```

## 3) Covariance Propagation

File: `examples/estimators/covariance_propagation.py`

- second-order approximation (tracks pairwise effects)
- usually better quality at higher compute

Run:

```bash
cestim run --estimator examples/estimators/covariance_propagation.py --runner subprocess
```

## 4) Combined Estimator

File: `examples/estimators/combined_estimator.py`

- budget-aware switch between cheaper and richer methods
- practical baseline for challenge-style tuning

Run:

```bash
cestim run --estimator examples/estimators/combined_estimator.py --runner subprocess
```

## Suggested Learning Path

1. Start with `random_estimator.py` to internalize the contract.
2. Move to `mean_propagation.py` for first useful baseline behavior.
3. Study `covariance_propagation.py` for richer approximation.
4. Use `combined_estimator.py` as a template for budget-aware routing.
