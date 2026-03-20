# Score Report Fields

## When to use this page

Use this page to interpret `nestim run` output fields.

## Top-level fields

Typical report sections include:

- `schema_version`
- `mode`
- `run_meta`
- `run_config`
- `run_config.dataset` (present when `--dataset` is used)
- `results`

## Core result fields

Inside `results`:

| Field | Description |
|---|---|
| `primary_score` | Leaderboard metric — final-layer MSE normalized by sampling MSE, averaged across MLPs. Lower is better. |
| `secondary_score` | All-layer MSE normalized by sampling MSE, averaged across MLPs. Lower is better. |
| `per_mlp` | Array of per-MLP detail records (see below) |

### Per-MLP fields

Each entry in `per_mlp`:

| Field | Type | Description |
|---|---|---|
| `mlp_index` | `int` | Index of the MLP in the evaluation set |
| `time_budget_s` | `float` | Sampling baseline wall time for this MLP (seconds) |
| `time_spent_s` | `float` | Your estimator's wall time for this MLP (seconds) |
| `fraction_spent` | `float` | `max(time_spent / time_budget, 0.5)` — clamped time ratio |
| `final_mse` | `float` | MSE of your final-layer predictions vs ground truth |
| `all_layer_mse` | `float` | MSE of your all-layer predictions vs ground truth |
| `primary_score` | `float` | `final_mse / sampling_mse` for this MLP |
| `secondary_score` | `float` | `all_layer_mse / sampling_mse` for this MLP |

If the estimator raised an error, the entry also includes:

| Field | Type | Description |
|---|---|---|
| `error` | `str` | Error message from the failed prediction |

## Interpretation guide

- `final_mse` is your most actionable diagnostic — it directly drives `primary_score`.
- `fraction_spent` reveals compute behavior: values near 0.5 mean you're much faster than sampling; values > 1.0 mean timeout (predictions zeroed).
- `primary_score` reflects accuracy under time-aware scoring. Compare it with `final_mse` to diagnose whether runtime or accuracy is the bottleneck.

## Dataset traceability fields

When using `nestim run --dataset`, the report includes `run_config.dataset`:

| Field | Description |
|---|---|
| `path` | Absolute path to the dataset file |
| `sha256` | SHA-256 hash of the file for integrity |
| `seed` | RNG seed used to generate the dataset |
| `n_mlps` | Number of MLPs in the dataset |

## Next step

- [Scoring Model](../concepts/scoring-model.md)
- [CLI Reference](./cli-reference.md)
