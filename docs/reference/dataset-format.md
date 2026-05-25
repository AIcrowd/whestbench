# WhestBench dataset format (schema 3.0)

WhestBench schema 3.0 stores evaluation datasets as a directory of Parquet files
plus two JSON/Markdown sidecars. This layout is native to the `datasets` library
(`datasets.load_dataset(...)` works directly on the directory), works with HuggingFace
Hub as a first-class dataset repository, and supports parallel distributed baking
with bit-exact merging.

The earlier `.npz` format (schemas 2.x) is no longer produced or loaded. Re-bake
with `whest dataset bake` to migrate.

## On-disk layout

```
<dataset_root>/
├── data/
│   └── <split>-NNNNN-of-MMMMM.parquet   # one row per MLP
├── metadata.json                          # whestbench provenance sidecar
└── README.md                              # HuggingFace dataset card
```

- `<split>` is the split name: `public` for the evaluation split, `holdout` for the
  private grader split. Controlled by `whest dataset bake --split`.
- `NNNNN-of-MMMMM` is the standard HF shard numbering; single-host bakes produce
  `00000-of-00001`.
- `metadata.json` is a flat JSON object with provenance, reproducibility, and hardware
  fields (see below).
- `README.md` is a rendered Jinja2 template with a YAML front-matter block that
  HuggingFace Hub uses to display the dataset card.

## Parquet schema (one row per MLP)

Eight columns per row. The `depth` and `width` dimensions are fixed for a given
dataset and captured in `metadata.json`.

This table mirrors the schema section in the published dataset card. They are
maintained in lockstep — any update here must also land in
`src/whestbench/templates/dataset_card.md.j2`.

| Column | Type / shape | What this is |
|---|---|---|
| `mlp_id` | `int32` | 0-based index of this MLP within the dataset (the absolute index across all parallel-bake slices). |
| `mlp_name` | `string` | Stable, deterministic human-readable slug like `"danielle-johnson"`, derived from `mlp_seed` via the `faker` library. Useful for log lines; carries no information beyond `mlp_seed`. |
| `mlp_seed` | `int64` | Seed an estimator should consume if it uses randomness (e.g. Monte Carlo). Per-MLP, derived from the contest seed; passed to `predict(mlp: MLP, budget: int)` as `mlp.seed`. |
| `weights` | `float32[depth, width, width]` | The MLP's layer weight matrices. The network has no biases and uses ReLU activations. Layer `l` computes `h_l(x) = max(0, W_l @ h_{l-1}(x))`. Weights are drawn i.i.d. from `N(0, 2/width)` (He initialization) at bake time. |
| `all_layer_means` | `float32[depth, width]` | **Ground truth.** Entry `[l, j]` is the empirical mean of neuron `j`'s post-ReLU output at layer `l`, averaged over many independent Gaussian inputs: `E_{x ~ N(0, I)}[ h_l(x)_j ] ≈ (1/N) Σ_i h_l(x_i)_j`, where `N = n_samples`. Computed by direct Monte Carlo. This is what an estimator predicts. |
| `final_means` | `float32[width]` | The last row of `all_layer_means` — i.e. `E[h_{depth}(x)_j]` for each output neuron `j`. Materialised as its own column because the primary scoring metric (`final_layer_mse`) only looks at this row. |
| `avg_variance` | `float64` | The mean across the final-layer neurons of the per-neuron output variance: `(1/width) Σ_j Var[h_{depth}(x)_j]`. A single scalar per MLP. Used as a normaliser in budget-adjusted scoring so that networks with naturally low output variance don't dominate the MSE rankings. |
| `sampling_budget_breakdown` | `string` (JSON) | FLOP accounting for the bake that produced the ground truth for **this** row — useful as provenance. Not related to the estimator's FLOP budget at evaluation time. Decode with `json.loads(...)`. |

### Notes on individual columns

**`mlp_id`** — matches the MLP's position in the logical dataset. Partial bakes
(from `--slice`/`--mlp-range`) have `mlp_id` values starting from their slice
offset; after `whest dataset merge`, `mlp_id` is monotonically increasing from 0.

**`mlp_name`** — the name is derived deterministically from `mlp_seed` using the
`faker` library at a pinned version. The same `--seed` and `--n-mlps` always
produce the same name list, on any hardware. Bumping the `faker` version pin
requires a deliberate re-bake.

**`weights`** — stored as `float32`. The weight matrices for each layer are
`weights[i]` of shape `(width, width)`. The forward pass uses no biases and ReLU
between layers; inputs are standard Gaussian, sampled fresh per Monte-Carlo draw
when ground truth is computed.

**`sampling_budget_breakdown`** — a JSON string with the per-namespace FLOP
counts and wall time consumed by the ground-truth Monte Carlo, accounted via
[`flopscope`](https://github.com/AIcrowd/flopscope). Parse with
`json.loads(row["sampling_budget_breakdown"])`. This is provenance metadata
about the bake itself, **not** the estimator's FLOP budget at evaluation time
(which is set at runtime via `whest run --flop-budget N`).

## metadata.json schema

`metadata.json` is a flat JSON object with the following fields.

### Base fields (all bakes)

| Field | Type | Description |
|---|---|---|
| `schema_version` | `string` | Always `"3.0"` for this format |
| `format` | `string` | Always `"hf-datasets-parquet"` |
| `backend` | `string` | `"flopscope"` (CPU path) or `"torch"` (GPU path) |
| `seed_protocol.name` | `string` | Always `"whestbench_seedsequence_hierarchy"` |
| `seed_protocol.version` | `string` | Always `"2.0"` |
| `seed` | `integer` or `null` | Root seed passed to `--seed`. `null` if auto-generated |
| `n_mlps` | `integer` | Number of MLPs in this dataset (or partial) |
| `n_samples` | `integer` | Ground-truth samples per MLP |
| `width` | `integer` | Neuron count per layer |
| `depth` | `integer` | Number of weight matrices |
| `created_at_utc` | `string` | ISO-8601 UTC timestamp of bake completion |
| `hardware` | `object` | Hardware fingerprint from the baking host |

### Torch-specific fields (when `backend == "torch"`)

| Field | Type | Description |
|---|---|---|
| `device` | `string` | `"cuda"`, `"mps"`, or `"cpu"` |
| `torch_version` | `string` | PyTorch version string, e.g. `"2.3.0"` |
| `cuda_device_name` | `string` | GPU name (CUDA only), e.g. `"NVIDIA L40S"` |
| `cuda_device_capability` | `[int, int]` | CUDA compute capability (CUDA only), e.g. `[8, 9]` |
| `mps_device_name` | `string` | Processor name (MPS only) |

### Partial-bake fields (when `--slice` or `--mlp-range` was used)

| Field | Type | Description |
|---|---|---|
| `is_partial` | `boolean` | Always `true` for partial bakes |
| `mlp_range` | `[int, int]` | `[start, end)` range of MLPs in this partial |
| `total_n_mlps` | `integer` | Logical total MLP count across all partials |

A dataset with `is_partial=true` is refused by `whestbench.load_dataset` — run
`whest dataset merge` first to assemble a complete dataset.

### Merged dataset fields (produced by `whest dataset merge`)

| Field | Type | Description |
|---|---|---|
| `merged_at_utc` | `string` | ISO-8601 UTC timestamp of the merge |
| `hardware_fingerprints` | `array` | List of per-partial hardware objects, each including `mlp_range` |

`is_partial`, `mlp_range`, and `total_n_mlps` are removed by the merge step.
`n_mlps` is set to the total count.

### Example metadata.json (CPU bake)

```json
{
  "schema_version": "3.0",
  "format": "hf-datasets-parquet",
  "backend": "flopscope",
  "seed_protocol": {
    "name": "whestbench_seedsequence_hierarchy",
    "version": "2.0"
  },
  "seed": 42,
  "n_mlps": 10,
  "n_samples": 10000000,
  "width": 256,
  "depth": 8,
  "created_at_utc": "2026-05-25T12:00:00+00:00",
  "hardware": {
    "cpu_brand": "Intel Xeon Platinum 8480+",
    "cpu_count": 64,
    "ram_gb": 512.0
  }
}
```

## README.md (HF dataset card)

`README.md` is rendered from a Jinja2 template at bake time. It contains:

- A YAML front-matter block with `license`, `tags`, `task_categories`, and HF
  dataset card metadata required for correct Hub display.
- A quick-start code snippet.
- A dataset summary table (split, MLPs, width, depth, samples, schema version, seed protocol).
- The full Parquet column schema.
- Reproducibility information including the exact `whest dataset bake` command to re-bake.
- Hardware provenance (for merged datasets, lists each host's GPU and mlp_range).

When `whest dataset push` uploads a local directory, it re-renders `README.md` with
the actual `repo_id` and `revision` (tag) so the published card has real values rather
than placeholders.

## Loading

### Bare `datasets.load_dataset`

Use this when you only need the raw data and don't need schema validation or the
metadata sidecar:

```python
from datasets import load_dataset

# Local directory
ds = load_dataset("./my-eval", split="public")

# HF Hub
ds = load_dataset(
    "aicrowd/arc-whestbench-2026",
    revision="v1",
    split="public",
)
print(ds)  # Dataset({features: [...], num_rows: 10})
print(ds[0]["mlp_name"])  # "danielle-johnson"
```

### `whestbench.load_dataset` wrapper

Use this for the recommended workflow. It validates `metadata.json`, refuses partial
datasets (suggesting the merge step), and attaches metadata to the returned `Dataset`
object for later retrieval via `whestbench.metadata(ds)`:

```python
import whestbench

# Local
ds = whestbench.load_dataset("./my-eval")

# HF Hub (pin a revision — bare repo without revision is rejected by whest run)
ds = whestbench.load_dataset(
    "aicrowd/arc-whestbench-2026",
    revision="v1",
    split="public",
)

# Access metadata sidecar
md = whestbench.metadata(ds)
print(md["seed"], md["n_mlps"], md["backend"])

# Iterate as MLP instances
for mlp in whestbench.iter_mlps(ds):
    print(mlp.name, mlp.weights[0].shape)

# Random access
mlp_0 = whestbench.mlp_at(ds, 0)
```

### `iter_mlps` / `mlp_at`

Both functions return `whestbench.MLP` objects constructed via `MLP.from_row(row)`.
The `MLP` object exposes the same interface as MLPs produced on-the-fly by
`whestbench.sample_mlp`: `mlp.weights`, `mlp.width`, `mlp.depth`, `mlp.name`,
`mlp.seed`.

## Schema version policy

| Version | Format | Notes |
|---|---|---|
| 3.0 | Parquet + sidecar directory | Current. Required by this release. |
| 2.4 | `.npz` with `mlp_names` field | Legacy. Rejected by `load_dataset` with a re-bake hint. |
| 2.3 | `.npz` | Legacy. |
| 2.2 | `.npz` | Legacy. |

`schema_version` tracks the storage format (2.x = npz, 3.0 = Parquet).
`seed_protocol.version` (always `"2.0"` in schema 3.0) tracks the RNG hierarchy
that produces per-MLP seeds. These two version numbers are independent — the
seed protocol could be bumped without changing the storage format, and vice versa.

HuggingFace git tags (e.g. `v1`, `v2`) are content versions for a specific published
dataset. They are independent of the schema version — a dataset at tag `v2` is still
schema 3.0.

## Partial datasets and merging

### Baking partials

`--slice K/N` divides a logical dataset of `n_mlps` into N equal slices and bakes
slice K (0-indexed). The output metadata is marked `is_partial=true` and includes
`mlp_range=[start, end)` and `total_n_mlps`.

```bash
# 4 workers each bake 250 of 1000 MLPs
whest dataset bake --slice 0/4 --n-mlps 1000 --seed 42 ... --output ./p0
whest dataset bake --slice 1/4 --n-mlps 1000 --seed 42 ... --output ./p1
whest dataset bake --slice 2/4 --n-mlps 1000 --seed 42 ... --output ./p2
whest dataset bake --slice 3/4 --n-mlps 1000 --seed 42 ... --output ./p3
```

`--mlp-range START-END` is the lower-level alternative. Both endpoints are inclusive
on the CLI, but the Python API uses half-open `[start, end)` intervals internally.
`--slice 0/4` with `n_mlps=1000` is equivalent to `--mlp-range 0-249`.

### Merging

`whest dataset merge` validates all partials, checks for gap-free coverage of
`[0, total_n_mlps)`, concatenates the Parquet files in order, and writes a new
complete dataset directory:

```bash
whest dataset merge ./p0 ./p1 ./p2 ./p3 --output ./final
```

### Bit-equivalence property

The bit-equivalence guarantee means a worker baking `--slice K/N` produces rows
that are bitwise identical to the corresponding rows of a single-host bake with the
same `--seed` and `--n-mlps`. This holds because:

1. The seed hierarchy derives each MLP's per-layer seeds from a global
   `numpy.random.SeedSequence(root_seed)` that is expanded over all `n_mlps` logical
   slots before any slicing occurs. A worker baking slot `i` uses the same derived
   seed regardless of which slice it's in.
2. MLP names are derived from all `n_mlps` logical seeds so that `slice_names[K]`
   equals `full_names[K]`.

Note: bit-equivalence is per-backend. The `flopscope` (CPU) and `torch` backends
use different RNG algorithms and produce statistically equivalent (not bitwise
identical) results at the same seed.
