# Score Report Fields

## When to use this page

Use this page to interpret `cestim run` output fields.

## Top-level fields

Typical report sections include:

- `schema_version`
- `mode`
- `detail`
- `run_meta`
- `run_config`
- `results`

## Core result fields

Inside `results`:

- `final_score`: leaderboard metric (lower is better)
- `by_budget_raw`: per-budget diagnostics

Inside each `by_budget_raw` entry:

### Scalar aggregates

| Field | Description |
|---|---|
| `budget` | Sampling trial count for this evaluation level |
| `mse_mean` | Mean of `mse_by_layer` across all depths |
| `adjusted_mse` | MSE adjusted by relative compute usage — the per-budget score |
| `call_time_ratio_mean` | Your runtime / sampling baseline runtime (averaged across depths and circuits) |
| `call_effective_time_s_mean` | Your effective runtime in seconds (averaged, floored if faster than baseline) |
| `timeout_rate` | Average fraction of circuits timed out across depths |
| `time_floor_rate` | Average fraction of circuits floored across depths |

### Per-depth arrays (length `d`)

| Field | Description |
|---|---|
| `mse_by_layer` | Per-depth MSE array — your main diagnostic for where estimation breaks down |
| `time_budget_by_depth_s` | Sampling baseline runtime per depth |
| `time_ratio_by_depth_mean` | Per-depth time ratio (your cumulative runtime / sampling baseline at each depth), averaged across circuits |
| `effective_time_s_by_depth_mean` | Per-depth effective runtime (floored if faster than baseline), averaged across circuits |
| `timeout_rate_by_depth` | Per-depth fraction of circuits where your estimator exceeded the time limit |
| `time_floor_rate_by_depth` | Per-depth fraction of circuits where your estimator was faster than the baseline floor |

## ✅ Interpretation guide

- `mse_by_layer` is your most actionable diagnostic — look for depths where error spikes.
- `mse_mean` reflects prediction quality before runtime adjustment.
- `adjusted_mse` reflects quality under runtime-aware scoring.
- `final_score` is the average `adjusted_mse` across budgets.
- `time_ratio_by_depth_mean` reveals which depths are slow relative to sampling.
- `timeout_rate_by_depth` shows where your estimator is timing out per depth.

## ➡️ Next step

- [Scoring Model](../concepts/scoring-model.md)
- [CLI Reference](./cli-reference.md)
