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

- `<split>` is the split name. Controlled by `whest dataset bake --split`.
  Dataset authors can separately declare the HF config with `--config`; the
  default is `default`.
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
| `mlp_name` | `string` | Stable, deterministic human-readable slug like `"danielle-johnson"`, derived from `mlp_seed`. Useful for log lines; carries no information beyond `mlp_seed`. |
| `mlp_seed` | `int64` | Per-MLP seed. Under seed_protocol 3.0 (new bakes), this is the **input** seed — the canonical value stored in the parquet. `mlp.seed` (participant-facing) is derived locally from this value via `SeedSequence(mlp_seed).spawn(3)[2]`. Under legacy seed_protocol 2.0, this column stored the already-derived estimator seed. |
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
| `seed_protocol.name` | `string` | `"whestbench_explicit_per_mlp_seeds"` (3.0, new bakes) or `"whestbench_seedsequence_hierarchy"` (2.0, legacy). |
| `seed_protocol.version` | `string` | `"3.0"` (new bakes) or `"2.0"` (legacy). |
| `seed` | `integer` or `null` | **Present under seed_protocol 2.0 only.** Root seed passed to `--seed`. `null` if auto-generated. Absent in 3.0 datasets. |
| `split` | `string` | Split name for a single-split bake. New bakes populate this; legacy metadata may omit it. |
| `config` | `string` | HF dataset config for a single-split bake. Defaults to `"default"`; legacy metadata may omit it. |
| `n_mlps` | `integer` | Number of MLPs in this dataset (or partial) |
| `n_samples` | `integer` | Ground-truth samples per MLP |
| `width` | `integer` | Neuron count per layer |
| `depth` | `integer` | Number of weight matrices |
| `created_at_utc` | `string` | ISO-8601 UTC timestamp of bake completion |
| `hardware` | `object` | Hardware fingerprint from the baking host |

### Provenance fields

These pin the exact code + runtime state that produced a dataset, so a reader
can reproduce a bake without guessing which whestbench/flopscope/torch versions
or determinism flags were in effect. See
[Parallel bake → Bit-equivalence requirements](../how-to/parallel-bake.md#bit-equivalence-requirements)
for the operational consequences.

| Field | Type | Description |
|---|---|---|
| `whestbench_version` | `string` | Installed whestbench package version (e.g. `"0.3.0"`). `"unknown"` if `importlib.metadata` couldn't resolve it. |
| `flopscope_version` | `string` | Installed flopscope package version. Weight init uses `flopscope.numpy` so this matters for bit-exact weights. |

`validate_metadata` treats these as informational and does not require them
(absence doesn't fail validation), but `whest dataset bake` always populates
them.

### Torch-specific fields (when `backend == "torch"`)

| Field | Type | Description |
|---|---|---|
| `device` | `string` | `"cuda"`, `"mps"`, or `"cpu"` |
| `torch_version` | `string` | PyTorch version string, e.g. `"2.3.0"` |
| `cuda_device_name` | `string` | GPU name (CUDA only), e.g. `"NVIDIA L40S"` |
| `cuda_device_capability` | `[int, int]` | CUDA compute capability (CUDA only), e.g. `[8, 9]` |
| `cuda_driver_version` | `string` | NVIDIA driver version (CUDA only, best-effort via `nvidia-smi`). Absent if `nvidia-smi` is unavailable. |
| `mps_device_name` | `string` | Processor name (MPS only) |
| `mlps_per_batch` | `integer` | Number of MLPs the bake processed per device-side batch. |
| `chunk_size` | `integer` | Number of MC samples per device-side chunk. **Pinning this to a fixed value across workers + reference re-bakes is required for cross-host bit-exact verification** (see [parallel-bake.md](../how-to/parallel-bake.md)). |
| `bake_config` | `object` | Determinism flag state at bake time. See below. |

#### `bake_config` object (torch path only)

Captures the state of torch's determinism levers + the cuBLAS workspace env var
at bake time. Two bakes that should produce bit-identical numeric columns must
have matching `bake_config` values (and matching `chunk_size`).

| Field | Type | Description |
|---|---|---|
| `cudnn_deterministic` | `boolean` | Value of `torch.backends.cudnn.deterministic` at bake time. |
| `cudnn_benchmark` | `boolean` | Value of `torch.backends.cudnn.benchmark` at bake time. |
| `cublas_workspace_config` | `string` or `null` | Value of the `CUBLAS_WORKSPACE_CONFIG` env var at bake time, or `null` if unset. Recommended value for deterministic cuBLAS: `":4096:8"`. |
| `torch_use_deterministic_algorithms` | `boolean` | Value of `torch.are_deterministic_algorithms_enabled()` at bake time. |

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

### Example metadata.json (CPU bake, seed_protocol 3.0)

```json
{
  "schema_version": "3.0",
  "format": "hf-datasets-parquet",
  "backend": "flopscope",
  "seed_protocol": {
    "name": "whestbench_explicit_per_mlp_seeds",
    "version": "3.0"
  },
  "n_mlps": 10,
  "n_samples": 10000000,
  "width": 256,
  "depth": 8,
  "created_at_utc": "2026-05-25T12:00:00+00:00",
  "hardware": {
    "cpu_brand": "Intel Xeon Platinum 8480+",
    "cpu_count": 64,
    "ram_gb": 512.0
  },
  "whestbench_version": "0.3.0",
  "flopscope_version": "0.3.0"
}
```

### Example metadata.json (torch CUDA bake, seed_protocol 3.0)

```json
{
  "schema_version": "3.0",
  "format": "hf-datasets-parquet",
  "backend": "torch",
  "seed_protocol": {
    "name": "whestbench_explicit_per_mlp_seeds",
    "version": "3.0"
  },
  "n_mlps": 50,
  "n_samples": 1000000000,
  "width": 256,
  "depth": 8,
  "created_at_utc": "2026-05-26T03:45:00+00:00",
  "hardware": { "...": "..." },
  "whestbench_version": "0.3.0",
  "flopscope_version": "0.3.0",
  "torch_version": "2.3.0+cu121",
  "device": "cuda",
  "cuda_device_name": "NVIDIA L40S",
  "cuda_device_capability": [8, 9],
  "cuda_driver_version": "535.183.01",
  "mlps_per_batch": 16,
  "chunk_size": 524288,
  "bake_config": {
    "cudnn_deterministic": true,
    "cudnn_benchmark": false,
    "cublas_workspace_config": ":4096:8",
    "torch_use_deterministic_algorithms": false
  }
}
```

Under seed_protocol 3.0 there is no top-level `seed` field. Each MLP's input seed is
stored in the parquet `mlp_seed` column.

### Example metadata.json (legacy seed_protocol 2.0)

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

Legacy datasets (e.g. `aicrowd/arc-whestbench-2026-smoke-test`) use seed_protocol 2.0
and continue to load correctly. New bakes always write seed_protocol 3.0.

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
`seed_protocol.version` tracks the RNG algorithm that produces per-MLP seeds.
These two version numbers are independent — the seed protocol can be bumped without
changing the storage format, and vice versa.

## Seed protocols

### `whestbench_seedsequence_hierarchy` version `2.0` (legacy, read-only)

The original seeding scheme. A single **root seed** (`--seed N`) is expanded via
`numpy.random.SeedSequence(root_seed)` into `n_mlps` child sequences. Each child
spawns three streams: weights, samples, and estimator. The parquet `mlp_seed` column
stored the already-derived **estimator** seed (stream index 2), not the input seed.
New bakes can no longer write seed_protocol 2.0; `--seed N` on the CLI now rejects
with a migration hint.

### `whestbench_explicit_per_mlp_seeds` version `3.0` (new, default)

Each MLP receives an **independent input seed** (64-bit integer). Seeds are either
auto-generated via `secrets.randbits(63)` or supplied explicitly via
`--mlp-seeds FILE` (JSON array of N ints). The parquet `mlp_seed` column stores
the **input** seed — the canonical, portable value.

Within each MLP, the three RNG streams are still derived locally:
`SeedSequence(mlp_seed).spawn(3)` → `[weight_seq, sample_seq, estimator_seq]`.
`mlp.seed` (participant-facing) equals `int(estimator_seq.generate_state(1)[0])`,
unchanged from 2.0 from the participant's perspective.

#### Building a 3.0 dataset

```bash
# Auto-generated seeds (recommended for production bakes):
whest dataset bake --n-mlps 10 --n-samples 1e7 --width 256 --depth 8 \
    --output ./my-eval

# Explicit seeds (for reproducible small datasets or tests):
echo '[1001,2002,3003,4004]' > my-seeds.json
whest dataset bake --n-mlps 4 --n-samples 100 --width 4 --depth 2 \
    --mlp-seeds my-seeds.json --output ./tiny-eval

# Explicit HF config coordinate for authoring config-per-split repos:
whest dataset bake --n-mlps 100 --n-samples 1e9 --width 256 --depth 8 \
    --split full --config full --output ./full
```

In Python:

```python
from whestbench.dataset import create_dataset

# Auto-generated:
create_dataset(n_mlps=10, n_samples=1_000_000, width=256, depth=8,
               output_path="./my-eval")

# Explicit:
create_dataset(n_mlps=4, n_samples=100, width=4, depth=2,
               mlp_seeds=[1001, 2002, 3003, 4004],
               output_path="./tiny-eval")

# Explicit config coordinate:
create_dataset(n_mlps=100, n_samples=1_000_000_000, width=256, depth=8,
               split="full", config="full", output_path="./full")
```

#### Extracting seeds from a published dataset

```python
import whestbench

ds = whestbench.load_dataset("aicrowd/arc-whestbench-2026", revision="v1", split="public")
md = whestbench.metadata(ds)
if md["seed_protocol"]["version"] == "3.0":
    seeds = ds["mlp_seed"]   # list of input seeds
    print(seeds)
```

#### `--slice` + seed_protocol 3.0

Under 3.0, all workers baking a given split must receive the **same** `--mlp-seeds`
JSON file. Each worker uses `--slice K/N` to select its subset of rows; it draws
the corresponding seeds from that shared file. Seeds for different splits must use
different JSON files to preserve cross-split independence.

HuggingFace git tags (e.g. `v1`, `v2`) are content versions for a specific published
dataset. They are independent of the schema version — a dataset at tag `v2` is still
schema 3.0.

## Partial datasets and merging

### Baking partials

`--slice K/N` divides a logical dataset of `n_mlps` into N equal slices and bakes
slice K (0-indexed). The output metadata is marked `is_partial=true` and includes
`mlp_range=[start, end)` and `total_n_mlps`.

```bash
# Generate once, share the same file with all workers.
whest dataset generate-seeds --n-mlps 1000 > seeds.json

# 4 workers each bake 250 of 1000 MLPs
whest dataset bake --slice 0/4 --n-mlps 1000 --mlp-seeds seeds.json ... --output ./p0
whest dataset bake --slice 1/4 --n-mlps 1000 --mlp-seeds seeds.json ... --output ./p1
whest dataset bake --slice 2/4 --n-mlps 1000 --mlp-seeds seeds.json ... --output ./p2
whest dataset bake --slice 3/4 --n-mlps 1000 --mlp-seeds seeds.json ... --output ./p3
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
same `--mlp-seeds` file and `--n-mlps`. This holds because:

1. Under seed_protocol 3.0, each slot's input seed comes directly from the shared
   `--mlp-seeds` JSON file. A worker baking slot `i` reads `seeds[i]` from that file
   regardless of which slice it's assigned, so the derived weight/sample/estimator
   streams are identical to a single-host bake.
2. MLP names are derived from the same per-MLP input seeds so that `slice_names[K]`
   equals `full_names[K]`.

Note: bit-equivalence is per-backend. The `flopscope` (CPU) and `torch` backends
use different RNG algorithms and produce statistically equivalent (not bitwise
identical) results at the same seed.

## Multi-split datasets

A dataset directory can contain multiple splits as sibling parquet files in `data/`, with a single `metadata.json` describing all of them via an optional `splits:` sub-dict.

### On-disk layout

```
my-eval/
├── data/
│   ├── public-00000-of-00001.parquet
│   └── holdout-00000-of-00001.parquet
├── metadata.json
└── README.md
```

### metadata.json shape

```json
{
  "schema_version": "3.0",
  "format": "hf-datasets-parquet",
  "backend": "torch",
  "seed_protocol": {"name": "whestbench_explicit_per_mlp_seeds", "version": "3.0"},
  "n_samples": 1000000000,
  "width": 256,
  "depth": 8,
  "created_at_utc": "...",
  "hardware": {...},
  "splits": {
    "public":  {"config": "default", "n_mlps": 50, "created_at_utc": "...", "hardware_fingerprints": [...]},
    "holdout": {"config": "holdout", "n_mlps": 50, "created_at_utc": "...", "hardware_fingerprints": [...]}
  },
  "default_split": "public"
}
```

Under seed_protocol 3.0 there is no per-split `seed` field; seeds are stored in
the parquet `mlp_seed` column for each split.

### Field placement

| Field | Single-split | Multi-split |
|---|---|---|
| `schema_version`, `format`, `seed_protocol` | top-level | top-level |
| `backend`, `width`, `depth`, `n_samples` | top-level | top-level — must match across all splits (validated at combine time) |
| `split`, `config` | top-level optional coordinate for new bakes | per-split (`splits.<name>.config`) |
| `n_mlps`, `seed` | top-level | per-split (`splits.<name>.{n_mlps,seed}`) |
| `created_at_utc` | top-level | top-level (= earliest of splits) + optional per-split |
| `hardware` | top-level (bake host) | top-level (combine host) + per-split `hardware_fingerprints` for provenance |
| `splits` | absent | present |
| `is_partial`, `mlp_range`, `total_n_mlps` | present iff partial | not allowed (multi-split + partial is invalid) |

The discriminator is the presence of the `splits` field. No `schema_version` bump — the multi-split shape is a purely additive extension of schema 3.0.

### Loading

```python
from whestbench import load_dataset, metadata, iter_mlps

dsd = load_dataset("./my-eval")             # → DatasetDict
ds  = load_dataset("./my-eval", split="public")   # → Dataset

print(metadata(dsd)["splits"].keys())        # full multi-split metadata
print(metadata(dsd, split="public")["seed"]) # single-split-shaped projection

for mlp in iter_mlps(dsd["public"]):
    mlp.validate()
```

### Building a multi-split dataset

Bake each split as a complete single-split dataset, then combine. Under seed_protocol
3.0, each split uses its own seeds JSON file:

```bash
# Generate independent seed files for each split.
whest dataset generate-seeds --n-mlps 50 > public-seeds.json
whest dataset generate-seeds --n-mlps 50 > holdout-seeds.json

whest dataset bake --n-mlps 50 --n-samples 1e9 --width 256 --depth 8 --split public  --config default --mlp-seeds public-seeds.json  --output ./pub
whest dataset bake --n-mlps 50 --n-samples 1e9 --width 256 --depth 8 --split holdout --config holdout --mlp-seeds holdout-seeds.json --output ./hold
whest dataset combine-splits ./pub ./hold --output ./eval-r1
whest dataset push ./eval-r1 --repo aicrowd/arc-whestbench-2026-evals --tag round-1 --private
```

`combine-splits` preserves the baked config coordinate. If exactly one input
declares `config="default"`, the combined metadata records that split as
`default_split`, so `whest run --dataset ...` can keep a split-oriented UX.

### The `public` / `holdout` naming convention

The contest's evaluation dataset uses split names `public` (visible-during-contest scores) and `holdout` (private/final-leaderboard scores). The dataset-card template special-cases these names with leaderboard-specific wording. Other names render generically. Tooling itself accepts any HF-Hub-compatible split name (regex `[a-z][a-z0-9]*(-[a-z0-9]+)*`).
