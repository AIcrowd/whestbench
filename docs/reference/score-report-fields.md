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
| `primary_score` | Leaderboard metric — final-layer MSE averaged across MLPs. Lower is better. |
| `secondary_score` | All-layer MSE averaged across MLPs. Lower is better. |
| `per_mlp` | Array of per-MLP detail records (see below) |

### Per-MLP fields

Each entry in `per_mlp`:

| Field | Type | Description |
|---|---|---|
| `mlp_index` | `int` | Index of the MLP in the evaluation set |
| `flops_used` | `int` | Total FLOPs used by your estimator for this MLP |
| `budget_exhausted` | `bool` | Whether the estimator exceeded the FLOP budget (predictions zeroed if true) |
| `final_mse` | `float` | MSE of your final-layer predictions vs ground truth |
| `all_layer_mse` | `float` | MSE of your all-layer predictions vs ground truth |

If the estimator raised an error, the entry also includes:

| Field | Type | Description |
|---|---|---|
| `error` | `str` | Error message from the failed prediction |

## Interpretation guide

- `final_mse` is your most actionable diagnostic — it directly drives `primary_score`.
- `budget_exhausted` is the first thing to check if your score is unexpectedly high — exceeded budget means your predictions were zeroed.
- `flops_used` vs `flop_budget` shows how much headroom you have. If you are consistently near the cap, consider lighter methods.
- `primary_score` is raw MSE — compare across runs to see whether estimator changes are helping.

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
