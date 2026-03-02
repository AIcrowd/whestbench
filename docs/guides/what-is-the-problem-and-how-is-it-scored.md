# What Is The Problem And How Is It Scored?

This challenge asks you to estimate layer-wise wire expectations in random nonlinear circuits under a compute budget.

## Problem Setup

Each circuit has:

- `n` wires (`width`)
- `d` layers (`max_depth`)
- each layer computes wire outputs with gate parameters and wire references

Participants implement an estimator that receives:

- one `Circuit`
- one integer `budget`

and must stream exactly `d` vectors of shape `(n,)`, one vector per depth.

Interpretation:

- row `i` is your estimate of expected wire values after layer `i`
- values are compared to empirical Monte Carlo means

## Ground Truth

Ground truth for each circuit is approximated by simulation with `n_samples` sampled inputs. The evaluator computes empirical means at each depth and each wire.

## Runtime-Aware Evaluation

For every budget `b`:

1. Baseline timing is measured by depth using sampling (`time_budget_by_depth_s`).
2. Your estimator emits one row per depth.
3. Cumulative elapsed runtime is checked at every depth against tolerance bounds.

Rules at depth `i`:

- If elapsed runtime exceeds upper bound, row `i` is zeroed.
- If elapsed runtime is below lower bound, effective runtime is floored.
- Otherwise runtime is used as measured.

This enforces both correctness of streaming and budget-aware behavior.

## Score Components

Per budget, reports include:

- `mse_by_layer`
- `mse_mean`
- `call_time_ratio_mean`
- `call_effective_time_s_mean`
- `adjusted_mse`

Intuitively, `adjusted_mse` is prediction error weighted by runtime behavior under the depth-wise envelope.

Final challenge score:

- `final_score = mean(adjusted_mse over budgets)`
- lower is better

## Practical Strategy

1. Always satisfy stream contract first (shape, count, finite values).
2. Build a cheap robust baseline that never times out.
3. Add higher-quality paths for larger budgets.
4. Tune budget switches empirically with local reports.

## Common Failure Modes

- Returning one `(depth, width)` tensor instead of streaming rows.
- Emitting wrong row shape (for example `(1, n)` or `(n, 1)`).
- Emitting too few or too many rows.
- Emitting `nan` or `inf` values.
- Ignoring runtime envelope and getting depth rows zeroed.
