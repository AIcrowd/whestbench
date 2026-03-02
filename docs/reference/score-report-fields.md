# Score Report Fields

## Top-Level Fields

Typical report sections include:

- `schema_version`
- `mode`
- `detail`
- `run_meta`
- `run_config`
- `results`

## Core Result Fields

Inside `results`:

- `final_score`: leaderboard metric (lower is better)
- `by_budget_raw`: per-budget diagnostics

Inside each `by_budget_raw` entry:

- `budget`
- `mse_by_layer`
- `mse_mean`
- `adjusted_mse`
- `time_budget_by_depth_s`
- `time_ratio_by_depth_mean`
- `effective_time_s_by_depth_mean`
- `call_time_ratio_mean`
- `call_effective_time_s_mean`
- timeout/floor/runtime error rates

## Interpretation

- `mse_mean` reflects prediction quality before runtime adjustment.
- `adjusted_mse` reflects quality under runtime-aware scoring.
- `final_score` is the average adjusted error across budgets.

## Next

- [Scoring Model](../concepts/scoring-model.md)
- [CLI Reference](./cli-reference.md)
