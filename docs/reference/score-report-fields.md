# Score Report Fields

## When to use this page

Use this page to interpret `nestim run` output fields.

## Top-level fields

Typical report sections include:

- `schema_version`
- `mode`
- `detail`
- `run_meta`
- `run_config`
- `run_config.dataset` (present when `--dataset` is used)
- `results`

## Core result fields

Inside `results`:

- `adjusted_mse`: leaderboard metric (lower is better)
- `by_budget_raw`: per-budget diagnostics

Inside each `by_budget_raw` entry:

### Scalar aggregates

| Field | Description |
|---|---|
| `budget` | Sampling trial count for this evaluation level |
| `mse_mean` | Mean of `mse_by_layer` across all depths |
| `adjusted_mse` | MSE adjusted by relative compute usage — the per-budget score |
| `call_time_ratio_mean` | Your runtime / sampling baseline runtime (averaged across depths and MLPs) |
| `call_effective_time_s_mean` | Your effective runtime in seconds (averaged, floored if faster than baseline) |
| `timeout_rate` | Average fraction of MLPs timed out across depths |
| `time_floor_rate` | Average fraction of MLPs floored across depths |

### Per-depth arrays (length `d`)

| Field | Description |
|---|---|
| `mse_by_layer` | Per-depth MSE array — your main diagnostic for where estimation breaks down |
| `time_budget_by_depth_s` | Sampling baseline runtime per depth |
| `time_ratio_by_depth_mean` | Per-depth time ratio (your cumulative runtime / sampling baseline at each depth), averaged across MLPs |
| `effective_time_s_by_depth_mean` | Per-depth effective runtime (floored if faster than baseline), averaged across MLPs |
| `timeout_rate_by_depth` | Per-depth fraction of MLPs where your estimator exceeded the time limit |
| `time_floor_rate_by_depth` | Per-depth fraction of MLPs where your estimator was faster than the baseline floor |

## ✅ Interpretation guide

- `mse_by_layer` is your most actionable diagnostic — look for depths where error spikes.
- `mse_mean` reflects prediction quality before runtime adjustment.
- `adjusted_mse` reflects quality under runtime-aware scoring.
- `adjusted_mse` is the average `adjusted_mse` across budgets.
- `time_ratio_by_depth_mean` reveals which depths are slow relative to sampling.
- `timeout_rate_by_depth` shows where your estimator is timing out per depth.

## Dataset traceability fields

When using `nestim run --dataset`, the report includes `run_config.dataset`:

| Field | Description |
|---|---|
| `path` | Absolute path to the dataset file |
| `sha256` | SHA-256 hash of the file for integrity |
| `seed` | RNG seed used to generate the dataset |
| `n_mlps` | Number of MLPs in the dataset |
| `n_samples` | Samples per MLP used for ground truth |
| `baselines_recomputed` | `true` if baselines were recomputed due to hardware mismatch |

See [Use Evaluation Datasets](../how-to/use-evaluation-datasets.md) for usage.

## ➡️ Next step

- [Scoring Model](../concepts/scoring-model.md)
- [CLI Reference](./cli-reference.md)
