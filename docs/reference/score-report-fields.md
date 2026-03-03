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

| Field | Description |
|---|---|
| `budget` | Sampling trial count for this evaluation level |
| `mse_by_layer` | **Per-depth** MSE array (length `d`) — your main diagnostic for where estimation breaks down |
| `mse_mean` | Mean of `mse_by_layer` across all depths |
| `adjusted_mse` | MSE adjusted by relative compute usage — the per-budget score |
| `call_time_ratio_mean` | Your runtime / sampling baseline runtime (averaged across circuits) |
| `call_effective_time_s_mean` | Your effective runtime in seconds (floored if faster than baseline) |
| `timeout_rate` | Fraction of circuits where your estimator exceeded the time limit |
| `time_floor_rate` | Fraction of circuits where your estimator was faster than the baseline floor |
| `runtime_error_rate` | Fraction of circuits where your estimator raised an exception |
| `protocol_error_rate` | Fraction of circuits where output shape/format was invalid |
| `oom_rate` | Fraction of circuits where your estimator ran out of memory |

## ✅ Interpretation guide

- `mse_by_layer` is your most actionable diagnostic — look for depths where error spikes.
- `mse_mean` reflects prediction quality before runtime adjustment.
- `adjusted_mse` reflects quality under runtime-aware scoring.
- `final_score` is the average `adjusted_mse` across budgets.

## ➡️ Next step

- [Scoring Model](../concepts/scoring-model.md)
- [CLI Reference](./cli-reference.md)
