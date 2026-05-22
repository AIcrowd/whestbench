# Score Report Fields

## When to use this page

Use this page to interpret `whest run` output fields.

## Top-level fields

Typical report sections include:

- `schema_version`
- `mode`
- `run_meta`
- `run_config`
- `run_config.seed` (always present; `null` when `--seed` is not provided)
- `run_config.dataset` (present when `--dataset` is used)
- `results`

## Run configuration fields

`run_config` records the parameters that governed the run:

| Field | Description |
|---|---|
| `seed` | The `--seed` value passed at the CLI, or `null` when `--seed` is omitted. When set, it determines both MLP generation (without `--dataset`) and `SetupContext.seed` for the participant's `setup()` call. When `null`, `ctx.seed` defaults to `0`. See [estimator-contract.md](estimator-contract.md) for the reproducibility contract and [cli-reference.md](cli-reference.md) for `--seed` flag semantics. |
| `dataset` | Present when `--dataset` is used. See [Dataset traceability fields](#dataset-traceability-fields) below. |

## Host metadata

`run_meta.host` is always an object. If you set `WHEST_SKIP_HARDWARE_FALLBACK_PROBES=1`, WhestBench still records cheap host fields and any values available through `psutil`, but fallback-backed fields such as `cpu_count_physical` and `ram_total_bytes` may be `null`.

## Core result fields

Inside `results`:

| Field | Description |
|---|---|
| `adjusted_final_layer_score` | Budget-adjusted leaderboard metric — suite mean of per-MLP `adjusted_final_layer_score = final_layer_mse × max(0.1, C_m/B_m)`; failure → × 1.0. Lower is better. |
| `all_layers_mse` | Raw all-layers MSE averaged across MLPs (no budget multiplier). Diagnostic — reveals where approximation error accumulates. |
| `final_layer_mse` | Raw final-layer MSE averaged across MLPs (no multiplier). |
| `per_layer_mse` | Per-layer MSE averaged across MLPs. `list[float]` of length `depth`. The last element equals `final_layer_mse` and the list mean equals `all_layers_mse`. Diagnostic only, no budget multiplier. |
| `best_mlp_adjusted_final_layer_score` | Minimum per-MLP `adjusted_final_layer_score`. |
| `worst_mlp_adjusted_final_layer_score` | Maximum per-MLP `adjusted_final_layer_score`. |
| `mean_score_multiplier` | Mean of per-MLP `max(0.1, C_m/B_m)` (1.0 on failure). Bounded [0.1, 1.0]. |
| `mean_compute_utilization` | Mean of per-MLP `C_m/B_m`, unclamped — can exceed 1.0 when an MLP busted the cap. |
| `n_failed_mlps` | Count of MLPs with any failure flag or `error_code` set. |
| `mean_effective_compute` | Mean of per-MLP `effective_compute`. |
| `failure_breakdown` | Dict with independent counts per failure flag: `budget_exhausted`, `time_exhausted`, `residual_wall_time_exhausted`, `combined_budget_exhausted`, `error`. Sums can exceed `n_failed_mlps` because one MLP can carry multiple flags. |
| `breakdowns` | Aggregate FLOP/time breakdowns keyed by section name. Includes `sampling` and `estimator`. |
| `per_mlp` | Array of per-MLP detail records (see below) |

### Per-MLP fields

Each entry in `per_mlp`:

| Field | Type | Description |
|---|---|---|
| `mlp_index` | `int` | Index of the MLP in the evaluation set |
| `mlp_name` | `str` | Human-readable slug for this MLP (e.g. `"danielle-johnson"`). Same value as `mlps[i].name` on the corresponding `MLP`; derived deterministically from `mlp_index`'s per-MLP seed. Use it as a stable label in your own logs and dashboards. |
| `flops_used` | `int` | Total FLOPs used by your estimator for this MLP |
| `effective_compute` | `float` | C_m = F_m + λ·R_m. Combined FLOP-equivalent compute used by the estimator. |
| `adjusted_final_layer_score` | `float` | s_m. The per-MLP budget-adjusted score that flows into the suite mean. |
| `combined_budget_exhausted` | `bool` | Whether the post-hoc check `C_m > B_m` fired (predictions zeroed if true). |
| `budget_exhausted` | `bool` | Whether the estimator exceeded the FLOP budget (predictions zeroed if true) |
| `time_exhausted` | `bool` | Whether the estimator exceeded the wall-clock limit for this MLP (predictions zeroed if true) |
| `residual_wall_time_exhausted` | `bool` | Whether WhestBench judged non-flopscope time to exceed `residual_wall_time_limit_s` (predictions zeroed if true) |
| `wall_time_s` | `float` | Total elapsed wall-clock time measured for this MLP's estimator context |
| `flopscope_backend_time_s` | `float` | Wall time inside counted flopscope numpy kernels - the participant's actual numpy compute |
| `flopscope_overhead_time_s` | `float` | Wall time inside flopscope's own dispatch code (wrapper preambles, FLOP bookkeeping, namespace push/pop). Framework cost, not participant cost. |
| `residual_wall_time_s` | `float` | Wall time inside the predict context that is neither flopscope backend execution nor flopscope dispatch - i.e. participant Python (loops, control flow), GC, uninstrumented numpy |
| `final_layer_mse` | `float` | MSE of your final-layer predictions vs ground truth |
| `all_layers_mse` | `float` | MSE of your all-layer predictions vs ground truth |
| `per_layer_mse` | `list[float]` | Per-layer MSE for this MLP. Length equals `depth`. `per_layer_mse[-1] == final_layer_mse` and `mean(per_layer_mse) == all_layers_mse` within float precision. |
| `breakdowns` | `dict \| null` | Per-MLP breakdown container. Currently includes estimator-only data under `estimator`. Sampling is aggregate-only. |
| `traceback` | `str \| null` | Non-null when this MLP's run did not produce real predictions — captures the Python traceback for either an estimator exception or a budget/time exhaustion. `null` on clean runs. For subprocess/server runners, the traceback is forwarded from the worker. |

When the estimator raised an unhandled exception (not budget/time exhaustion), the entry also includes:

| Field | Type | Description |
|---|---|---|
| `error` | `str` \| `dict` | Legacy string message, or structured object: `{"message": str, "details": object}` |
| `error_code` | `str` | Stable identifier: `PREDICT_ERROR` for a `RunnerError`, or the Python exception class name otherwise |

For structured `error` objects, `error.details` includes:

- `expected_shape`: `List[int]` with expected `(depth, width)`.
- `got_shape`: `List[int]` observed from estimator output.
- `cause_hints`: `List[str]` with user-facing hints.
- `hint`: short summary hint.

## Time decomposition

Every `predict()` call satisfies a strict three-bucket identity:

```
wall_time_s = flopscope_backend_time_s + flopscope_overhead_time_s + residual_wall_time_s
```

- `flopscope_backend_time_s` - numpy kernels actually crunching numbers via `flopscope.numpy.*`.
- `flopscope_overhead_time_s` - flopscope's own dispatch (wrapper preambles, FLOP bookkeeping, namespace push/pop).
- `residual_wall_time_s` - everything else inside the wall window: participant Python, GC, uninstrumented numpy.

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

- `flopscope_backend_time_s` - accumulated time inside counted flopscope operations
- `flopscope_overhead_time_s` - accumulated time inside flopscope's own dispatch
- `residual_wall_time_s` - everything else (participant Python, GC, uninstrumented numpy)

For `results.breakdowns.*`, those values are aggregated across all evaluated
MLPs.

## Budget-adjusted scoring

The leaderboard ranks submissions by `adjusted_final_layer_score`, the suite mean of the
budget-adjusted per-MLP score:

```
adjusted_final_layer_score = final_layer_mse × max(0.1, C_m / B_m)   for valid runs
adjusted_final_layer_score = final_layer_mse × 1.0                    for failures (no compute discount)

C_m = F_m + λ · R_m                      (effective compute, FLOPs and FLOP-equivalents)
λ = 1e11 FLOPs/second                    (conversion rate; see flopscope-primer.md)
```

Where `F_m` is the analytical FLOPs counted by flopscope (`flops_used`), `R_m` is the
residual wall-time bucket (`residual_wall_time_s` — neither flopscope-backend nor
flopscope-overhead), and `B_m` is `flop_budget`. The `max(0.1, ...)` floor caps the
discount at 10× so an arbitrarily cheap-but-wrong submission cannot dominate the ranking.

> **Why "score" not "MSE"?** Once `final_layer_mse` is multiplied by the budget
> factor `max(0.1, C_m/B_m)`, the result is no longer a mean-squared-error between
> predictions and targets — it is a derived ranking score (denoted `s_m`). The
> `_score` suffix in `adjusted_final_layer_score` reflects this; the raw
> diagnostics `final_layer_mse` and `all_layers_mse` keep the `_mse` suffix because
> they remain genuine MSEs.

## Interpretation guide

- `final_layer_mse` is your most actionable diagnostic — it directly drives `adjusted_final_layer_score`.
- `budget_exhausted` is the first thing to check if your score is unexpectedly high — exceeded budget means your predictions were zeroed.
- `time_exhausted` means the estimator crossed the wall-clock limit configured through `wall_time_limit_s` / `--wall-time-limit`.
- `residual_wall_time_exhausted` means the non-flopscope portion of execution crossed WhestBench's `residual_wall_time_limit_s` / `--residual-wall-time-limit`.
- `flops_used` vs `flop_budget` shows how much headroom you have. If you are consistently near the cap, consider lighter methods.
- High `flopscope_backend_time_s` relative to wall: numpy compute is the dominant cost. Healthy for a numpy-heavy estimator.
- High `flopscope_overhead_time_s` relative to wall: many small ops are paying the per-call dispatch tax. Consider batching with larger numpy primitives.
- High `residual_wall_time_s` relative to wall: participant Python is the bottleneck (tight loops, per-element attribute access, calls into uninstrumented libraries). This is the bucket future versions of WhestBench will penalise on.
- `adjusted_final_layer_score` is the budget-adjusted leaderboard metric and is always ≤ the raw `final_layer_mse`
  mean (the multiplier is at most 1.0 — it equals 1.0 at full budget use or on failures
  and drops to 0.1 at the discount floor — a factor-of-ten cap). A value close to raw `final_layer_mse`
  means you used near-full budget; a value close to one-tenth of raw `final_layer_mse`
  means you used ≤10% of the budget and got the maximum discount.
- `all_layers_mse` is a diagnostic aggregate with no budget multiplier. Use it to understand where approximation error accumulates across all layers, not just the final layer.
- `per_layer_mse` decomposes `all_layers_mse` layer-by-layer (length = `depth`). Useful for spotting which layers your estimator struggles on — e.g. early layers vs. final layer. By construction `per_layer_mse[-1] == final_layer_mse` and `mean(per_layer_mse) == all_layers_mse` (within float precision).

## Dataset traceability fields

When using `whest run --dataset`, the report includes `run_config.dataset`:

| Field | Description |
|---|---|
| `path` | Absolute path to the dataset file |
| `sha256` | SHA-256 hash of the file for integrity |
| `seed` | RNG seed used to generate the dataset |
| `n_mlps` | Number of MLPs in the dataset |
| `seed_protocol` | Object with `name` and `version`. WhestBench currently requires `version = "2.0"`. |

### Dataset format compatibility

The `.npz` files produced by `whest create-dataset` carry a `seed_protocol.version` in their embedded metadata. WhestBench refuses to load datasets at any other version: loading a v1.0 dataset raises `ValueError("Incompatible dataset seed_protocol version: file has '1.0', this whestbench requires '2.0'. Re-bake the dataset with `whest create-dataset`.")`.

The v2.0 format adds a per-MLP `seed` (stored as the `mlp_seeds` array in the `.npz`) that is exposed to estimators via `mlp.seed` — see [estimator-contract.md](./estimator-contract.md) for how to consume it. Auto-migration is intentionally not implemented because the v1.0 spawn protocol (2 streams per MLP) cannot produce a deterministic third stream; re-baking from the original spec seed is the only correct path.

Schema 2.4 added the per-MLP `name` slug (stored as the `mlp_names` array in the `.npz`). It is a pure function of `mlp_seeds` at the WhestBench release's pinned `faker` version, so loading a 2.3 file under 2.4 code transparently synthesizes the same names a fresh 2.4 bake would produce — no re-bake required. See [estimator-contract.md](./estimator-contract.md) for the `mlp.name` field exposed to estimators.

## Next step

- [CLI Reference](./cli-reference.md)
- [Scoring Model](https://github.com/AIcrowd/whest-starterkit/blob/main/docs/concepts/scoring-model.md) (in the starter kit)
