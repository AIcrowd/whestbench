# Score Report Fields

## When to use this page

Use this page to interpret `whest run` output fields.

## Top-level fields

Typical report sections include:

- `schema_version`
- `mode`
- `run_meta`
- `run_config`
- `run_config.seed` (present when `whest run --seed` is provided)
- `run_config.dataset` (present when `--dataset` is used)
- `results`

## Host metadata

`run_meta.host` is always an object. If you set `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1`, WhestBench still records cheap host fields and any values available through `psutil`, but fallback-backed fields such as `cpu_count_physical` and `ram_total_bytes` may be `null`.

## Core result fields

Inside `results`:

| Field | Description |
|---|---|
| `primary_score` | Leaderboard metric — final-layer MSE averaged across MLPs. Lower is better. |
| `secondary_score` | All-layer MSE averaged across MLPs. Lower is better. |
| `breakdowns` | Aggregate FLOP/time breakdowns keyed by section name. Includes `sampling` and `estimator`. |
| `per_mlp` | Array of per-MLP detail records (see below) |

### Per-MLP fields

Each entry in `per_mlp`:

| Field | Type | Description |
|---|---|---|
| `mlp_index` | `int` | Index of the MLP in the evaluation set |
| `flops_used` | `int` | Total FLOPs used by your estimator for this MLP |
| `budget_exhausted` | `bool` | Whether the estimator exceeded the FLOP budget (predictions zeroed if true) |
| `time_exhausted` | `bool` | Whether the estimator exceeded the wall-clock limit for this MLP (predictions zeroed if true) |
| `untracked_time_exhausted` | `bool` | Whether WhestBench judged non-flopscope time to exceed `untracked_time_limit_s` (predictions zeroed if true) |
| `wall_time_s` | `float` | Total elapsed wall-clock time measured for this MLP's estimator context |
| `tracked_time_s` | `float` | Wall time inside counted flopscope numpy kernels — the participant's actual numpy compute |
| `flopscope_overhead_time_s` | `float` | Wall time inside flopscope's own dispatch code (wrapper preambles, FLOP bookkeeping, namespace push/pop). Framework cost, not participant cost. |
| `untracked_time_s` | `float` | Wall time inside the predict context that is neither tracked numpy nor flopscope dispatch — i.e. participant Python (loops, control flow), GC, uninstrumented numpy |
| `final_mse` | `float` | MSE of your final-layer predictions vs ground truth |
| `all_layer_mse` | `float` | MSE of your all-layer predictions vs ground truth |
| `breakdowns` | `dict \| null` | Per-MLP breakdown container. Currently includes estimator-only data under `estimator`. Sampling is aggregate-only. |

If the estimator raised an error, the entry also includes:

| Field | Type | Description |
|---|---|---|
| `error` | `str` \| `dict` | Legacy string message, or structured object: `{"message": str, "details": object}` |
| `error_code` | `str` | Stable identifier: `PREDICT_ERROR` for a `RunnerError`, or the Python exception class name otherwise |
| `traceback` | `str \| null` | Formatted traceback string for the failure. Forwarded from the subprocess worker when `--runner subprocess`/`server` is used; captured locally otherwise. |

For structured `error` objects, `error.details` includes:

- `expected_shape`: `List[int]` with expected `(depth, width)`.
- `got_shape`: `List[int]` observed from estimator output.
- `cause_hints`: `List[str]` with user-facing hints.
- `hint`: short summary hint.

## Time decomposition

Every `predict()` call satisfies a strict three-bucket identity:

```
wall_time_s ≈ tracked_time_s + flopscope_overhead_time_s + untracked_time_s
```

- `tracked_time_s` — numpy kernels actually crunching numbers via `flopscope.numpy.*`.
- `flopscope_overhead_time_s` — flopscope's own dispatch (wrapper preambles, FLOP bookkeeping, namespace push/pop).
- `untracked_time_s` — everything else inside the wall window: participant Python, GC, uninstrumented numpy.

The decomposition holds at every level: per-MLP, aggregated across MLPs, and per namespace inside `breakdowns`.

## Breakdown containers

When namespace-aware flopscope data is available, WhestBench adds breakdown containers in
these places:

- `results.breakdowns.estimator` - aggregated estimator breakdown across all evaluated MLPs
- `results.breakdowns.sampling` - aggregated sampling breakdown across all evaluated MLPs
- `results.per_mlp[].breakdowns.estimator` - one normalized estimator breakdown per MLP

Namespace normalization rules:

- sampling work is namespaced under `sampling.*`
- unlabeled estimator work becomes `estimator.estimator-client`
- explicit estimator namespace `phase` becomes `estimator.phase`
- nested estimator namespace `phase.subphase` becomes `estimator.phase.subphase`

Each breakdown summary also includes timing totals:

- `tracked_time_s` - accumulated time inside counted flopscope operations
- `flopscope_overhead_time_s` - accumulated time inside flopscope's own dispatch
- `untracked_time_s` - everything else (participant Python, GC, uninstrumented numpy)

For `results.breakdowns.*`, those values are aggregated across all evaluated
MLPs.

## Interpretation guide

- `final_mse` is your most actionable diagnostic — it directly drives `primary_score`.
- `budget_exhausted` is the first thing to check if your score is unexpectedly high — exceeded budget means your predictions were zeroed.
- `time_exhausted` means the estimator crossed the wall-clock limit configured through `wall_time_limit_s` / `--wall-time-limit`.
- `untracked_time_exhausted` means the non-flopscope portion of execution crossed WhestBench's `untracked_time_limit_s` / `--untracked-time-limit`.
- `flops_used` vs `flop_budget` shows how much headroom you have. If you are consistently near the cap, consider lighter methods.
- High `tracked_time_s` relative to wall: numpy compute is the dominant cost. Healthy for a numpy-heavy estimator.
- High `flopscope_overhead_time_s` relative to wall: many small ops are paying the per-call dispatch tax. Consider batching with larger numpy primitives.
- High `untracked_time_s` relative to wall: participant Python is the bottleneck (tight loops, per-element attribute access, calls into uninstrumented libraries). This is the bucket future versions of WhestBench will penalise on.
- `primary_score` is raw MSE — compare across runs to see whether estimator changes are helping.

## Dataset traceability fields

When using `whest run --dataset`, the report includes `run_config.dataset`:

| Field | Description |
|---|---|
| `path` | Absolute path to the dataset file |
| `sha256` | SHA-256 hash of the file for integrity |
| `seed` | RNG seed used to generate the dataset |
| `n_mlps` | Number of MLPs in the dataset |

## Next step

- [Scoring Model](../concepts/scoring-model.md)
- [CLI Reference](./cli-reference.md)
