# Publishing a dataset to HuggingFace Hub

Step-by-step walkthrough for baking a WhestBench evaluation dataset locally and
publishing it to HuggingFace Hub. Once published, participants (and other machines)
can load it directly with `datasets.load_dataset` or `whestbench.load_dataset`.

**Prerequisites:**
- `pip install whestbench` (or `whestbench[gpu]` for GPU bakes)
- A HuggingFace account with write access to the target repo
- `HF_TOKEN` with write scope (see step 1)

## 1. Set up authentication

```bash
# Option A — interactive login (stores a token in ~/.cache/huggingface/token)
huggingface-cli login

# Option B — environment variable (preferred in CI)
export HF_TOKEN=hf_your_write_token_here
```

The `HF_TOKEN` environment variable is read automatically by `whest dataset push`
and `whestbench.publish_dataset`. You can also pass it explicitly via `--token`.

## 2. Bake locally

Bake a dataset to a local directory. Choose `--n-mlps` and `--n-samples` appropriate
to your use case. For bit-exact reproducibility, pass an explicit
`--mlp-seeds` JSON file.

```bash
whest dataset bake \
    --n-mlps 10 \
    --n-samples 10_000_000 \
    --width 256 \
    --depth 8 \
    --output ./my-bake
```

For larger bakes, see [Parallel bake across multiple GPUs](./parallel-bake.md) and
[GPU Dataset Generation](../reference/gpu-dataset-generation.md).

## 3. Inspect before publishing

Verify the bake parameters before uploading. This is cheap and catches any
misconfiguration before it goes out:

```bash
whest dataset inspect ./my-bake
```

Expected output:

```
WhestBench dataset
  schema_version: 3.0
  format: hf-datasets-parquet
  backend: flopscope
  seed: 42
  n_mlps: 10
  n_samples: 10000000
  width: 256
  depth: 8
  created_at_utc: 2026-05-25T12:00:00+00:00
```

You can also verify the dataset loads correctly before pushing:

```python
import whestbench

ds = whestbench.load_dataset("./my-bake")
print(len(ds), "MLPs loaded")
for mlp in whestbench.iter_mlps(ds):
    print(mlp.name, mlp.weights[0].shape)
    break
```

## 4. Publish

Push the local directory to HF Hub. Use `--tag` to create a versioned git tag —
this is strongly recommended so participants can pin a specific version.

```bash
whest dataset push ./my-bake \
    --repo aicrowd/arc-whestbench-2026 \
    --tag v1 \
    --message "Bake: 10 MLPs, seed=42, 10M samples"
```

Expected output:

```
Uploaded to aicrowd/arc-whestbench-2026; commit abc1234def; tag v1
```

For a private repo (e.g. holdout sets), add `--private`:

```bash
whest dataset push ./my-bake \
    --repo aicrowd/arc-whestbench-2026-holdout \
    --tag v1 \
    --private \
    --message "Holdout bake: seed=99"
```

### What gets uploaded

- `data/<split>-00000-of-00001.parquet` — the MLP data
- `metadata.json` — provenance sidecar
- `README.md` — rendered dataset card (re-rendered with the actual `repo_id` and `tag` before upload, including any declared HF config layout)

## 5. Verify on HF Hub

Visit the dataset page to confirm the upload succeeded:

```
https://huggingface.co/datasets/aicrowd/arc-whestbench-2026/tree/v1
```

You should see the three files (`data/`, `metadata.json`, `README.md`) and the
dataset card rendered from the README.

You can also inspect from the CLI without downloading:

```bash
whest dataset inspect aicrowd/arc-whestbench-2026 --revision v1
```

## 6. Pull on another machine

On any other machine with `whestbench` installed:

```bash
whest dataset pull aicrowd/arc-whestbench-2026 \
    --revision v1 \
    --output ./local-copy
```

For a private repo, pass `--token` or set `HF_TOKEN` first.

## 7. Load in a participant script

### Using `datasets.load_dataset` directly

```python
from datasets import load_dataset

ds = load_dataset(
    "aicrowd/arc-whestbench-2026",
    revision="v1",
    split="public",
)
print(ds)  # Dataset({features: ['mlp_id', 'mlp_name', ...], num_rows: 10})
print(ds[0]["mlp_name"])  # "danielle-johnson"
```

### Using `whestbench.load_dataset` (recommended)

The wrapper validates the schema and attaches metadata for later retrieval:

```python
import whestbench

ds = whestbench.load_dataset(
    "aicrowd/arc-whestbench-2026",
    revision="v1",
    split="public",
)

# Iterate as MLP instances
for mlp in whestbench.iter_mlps(ds):
    y_pred = my_estimator.predict(mlp)

# Access metadata
md = whestbench.metadata(ds)
print(md["seed"], md["n_mlps"])
```

### Running evaluation against the published dataset

```bash
whest run --estimator ./estimator.py \
    --dataset hf://aicrowd/arc-whestbench-2026@v1

# Or equivalently:
whest run --estimator ./estimator.py \
    --dataset aicrowd/arc-whestbench-2026 \
    --revision v1
```

Note: bare `aicrowd/arc-whestbench-2026` without `--revision` is rejected by
`whest run` — always pin a revision.

## Troubleshooting

**`401 Unauthorized`** — Your `HF_TOKEN` doesn't have write access to the target
repo, or it has expired. Generate a new token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) with
`write` scope.

**`404 Repository not found`** — The repo doesn't exist yet. `whest dataset push`
creates it automatically; ensure you have permission to create repos under the
target org (e.g. `aicrowd/`).

**`FileExistsError: output already exists`** — `whest dataset bake` refuses to
overwrite an existing directory. Delete or rename the old output first, or choose
a new `--output` path.

**Dataset rejected with "partial dataset" error** — You pushed a slice bake without
merging first. Run `whest dataset merge` on all slices, then push the merged result.
See [Parallel bake](./parallel-bake.md).

### Multi-split datasets

`whest dataset push` handles multi-split datasets natively. The local directory must contain one parquet per split in `data/` and a `metadata.json` with a `splits:` dict; this is the shape produced by `whest dataset combine-splits`. If the input bakes declared `--config`, the push preserves that config-per-split layout in the published dataset card. The push uploads all parquets in one commit; tag with `--tag round-N` for per-round eval datasets.

For private repos (e.g. the evaluation dataset), pass `--private` on first push to create the repo as private. Subsequent pushes preserve the privacy setting.
