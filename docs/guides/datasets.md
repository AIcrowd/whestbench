# Datasets — a complete guide

WhestBench uses [HuggingFace Datasets](https://huggingface.co/docs/datasets) as
its dataset format and HF Hub as the distribution channel. This guide walks you
through every dataset-related verb in `whest`, in the order you'd typically
encounter them.

> If you only have 5 minutes, read the **Quick start** below. The rest of the
> guide builds on it.

## Quick start

You have a working estimator at `./estimator.py`. Bake a tiny evaluation
dataset locally, then score against it:

```bash
# 1. Generate 10 MLPs with ground-truth statistics → ./my-eval/
whest dataset bake --n-mlps 10 --n-samples 1000 --width 64 --depth 4 \
                   --output ./my-eval

# 2. Inspect what got written
whest dataset info ./my-eval

# 3. Score your estimator against the same MLPs every run
whest run --estimator estimator.py --dataset ./my-eval
```

Why this matters: without `--dataset`, `whest run` regenerates MLPs and ground
truth on every invocation. Baking a dataset once and reusing it makes your runs
deterministic and ~10× faster.

[Continue to: lifecycle ↓](#the-dataset-lifecycle)

## The dataset lifecycle

```
+--------+        +-----------+        +----------+        +--------+
| local  | upload |    HF     | down-  |  local   |  run   | scores |
|  bake  | -----> | Hub repo  | load → |  cache   | -----> | report |
| (./out)|        | (org/...) |        |  (~/.hf) |        |        |
+--------+        +-----------+        +----------+        +--------+
   ^                                                            |
   |____________________________________________________________|
                   iterate on estimator code
```

- **Local-only workflow:** `bake → run`. Best when you're iterating fast and
  don't care about sharing the dataset. See
  [Working locally](#working-locally).
- **Team workflow:** `bake → upload → … later … → download → run`. The HF
  repo's tag pins which exact dataset everyone scores against. See
  [Publishing to HuggingFace Hub](#publishing-to-huggingface-hub) and
  [Downloading from HF Hub and the local cache](#downloading-from-hf-hub-and-the-local-cache).
- **CI / leaderboard workflow:** `bake → upload --tag v1-warmup`. Participants
  pull by tag. Streaming (`whest run --streaming`) is the natural fit for
  per-PR CI gates — see [Streaming mode](#streaming-mode).

Each verb is detailed below.

## Working locally

You want to iterate fast: bake a small dataset to disk, inspect it, and reuse
it across `whest run` invocations. No network, no HF account needed.

### `whest dataset bake` — create a dataset

You're starting a new evaluation. Bake 100 MLPs of moderate size with their
ground-truth statistics to `./my-eval/`:

```bash
whest dataset bake \
    --n-mlps 100 \
    --n-samples 10000 \
    --width 256 --depth 8 \
    --output ./my-eval
```

Representative output:

```
→ Baking 100 MLPs (width=256, depth=8, n_samples=10000) to ./my-eval
  ✓ Generated weights         100/100
  ✓ Computed ground truth     100/100   31.7s
✓ Wrote ./my-eval (2.0 GB)
```

The result on disk:

```
my-eval/
├── data/public-00000-of-00001.parquet   # weights + ground-truth stats
├── metadata.json                         # schema_version, seed_protocol, …
└── README.md                             # dataset card
```

**Key flags:**

- `--mlp-seeds <file.json>` — pin per-MLP seeds explicitly. JSON array of N
  distinct int63 values. Required for bit-exact reproducibility with another
  bake.
- `--mlp-range START-END` or `--slice K/N` — bake a slice of a larger logical
  dataset. The slice is bit-equivalent to the corresponding portion of a
  single-host bake at the same seeds.
- `--torch` — use the GPU backend (requires `whestbench[gpu]`).
- `--split <name>` — assign a split name (default `public`). See
  [Multi-split datasets](#multi-split-datasets).

> If it broke, see the [Troubleshooting](#troubleshooting) section — bake errors
> usually trace back to seed shape, an existing output directory, or running
> out of RAM on large `--n-samples`.

### `whest dataset info` — what's in a dataset

You've baked or downloaded a dataset and want a one-screen summary before
running against it:

```bash
whest dataset info ./my-eval
```

Reports `schema_version`, `seed_protocol`, `n_mlps`, `n_samples`,
hardware fingerprint, and per-split row counts. `info` also works against HF
Hub directly:

```bash
whest dataset info aicrowd/arc-whestbench-public-2026 --revision v1-warmup
```

No download required — `info` only fetches `metadata.json`.

### `whest dataset merge` — assemble parallel bakes

You have a multi-host cluster and want to bake a 1,000-MLP dataset in two
slices, then concatenate. Both workers must share the same `--mlp-seeds` file
so the result is bit-equivalent to a single-host bake:

```bash
# Two workers each bake a slice…
whest dataset bake --n-mlps 1000 --slice 0/2 --output ./partial-a \
                   --mlp-seeds seeds.json
whest dataset bake --n-mlps 1000 --slice 1/2 --output ./partial-b \
                   --mlp-seeds seeds.json

# … then merge.
whest dataset merge ./partial-a ./partial-b --output ./full
```

The merged dataset is bit-equivalent to a single-host bake of the same size at
the same seeds. See also: [parallel-bake how-to](../how-to/parallel-bake.md).

### `whest run --dataset <local-dir>` — score against a baked dataset

You're iterating on `estimator.py`. Score it against the first 50 MLPs of
your baked dataset (fast feedback loop):

```bash
whest run --estimator estimator.py --dataset ./my-eval --n-mlps 50
```

`--n-mlps K` clamps the run to the first K MLPs of the dataset (useful for
quick iteration). Pass `--split <name>` if the dataset is
[multi-split](#multi-split-datasets).

Once you're happy with local results, [publish the dataset](#publishing-to-huggingface-hub)
so teammates can score against the same MLPs.

## Publishing to HuggingFace Hub

You've baked a dataset locally and want to share it with the team — or pin a
specific revision so a CI gate scores everyone against the same MLPs. Upload
to a HuggingFace Hub dataset repo.

### Authenticate once

```bash
hf auth login   # opens a browser; or pass --token <hf_xxx>
```

Tokens with `write` scope are required to push. You can also set the token
without the interactive flow:

```bash
export HF_TOKEN=hf_xxx
```

`whest dataset upload` reads `HF_TOKEN` as a fallback when `--token` isn't
passed. See also: the
[publish-to-hf-hub how-to](../how-to/publish-to-hf-hub.md) for an
end-to-end walkthrough.

### `whest dataset upload`

You have `./my-eval` from the previous section. Push it as a private repo and
pin the resulting commit with a tag:

```bash
whest dataset upload ./my-eval \
    --repo aicrowd/my-eval \
    --tag v1 \
    --private   # omit for public datasets
```

Representative output:

```
→ Uploading ./my-eval to aicrowd/my-eval (private)
  ✓ Repo exists / created
  ✓ Uploaded 2.0 GB                 ████████████████████ 100%   34.1s
  ✓ Tag v1 created at d2f9a1c
✓ Done: https://huggingface.co/datasets/aicrowd/my-eval/tree/v1
```

The repo is created if it doesn't exist. The tag is created at the resulting
commit so

```bash
whest run --dataset hf://aicrowd/my-eval@v1
```

pins to this exact revision.

**Repo naming.** Use `<org>/<dataset-name>`. Keep names short and
hyphen-separated (e.g. `aicrowd/arc-whestbench-public-2026`).

**Tag conventions.** HF doesn't enforce semver; the de-facto pattern is
`v<MAJOR>.<MINOR>` (e.g. `v1.0`, `v1.1`) or descriptive (`v1-warmup`,
`v1-holdout`). See
[HF's revision docs](https://huggingface.co/docs/huggingface_hub/guides/cli).

### What gets published

The dataset card (`README.md`) is auto-generated from `metadata.json` at bake
time. It includes splits, hardware fingerprint, seed protocol, and a runnable
quick-start snippet. Edit `README.md` after `bake` and before `upload` to
add custom content.

The card's YAML front-matter is what HuggingFace Hub renders on the dataset
page (tags, license, language, etc.). Don't strip it.

> `whest dataset push` continues to work as a deprecated alias for `upload`
> through v0.6. v0.7 will remove it. Same applies to `pull` → `download` and
> `inspect` → `info`.

> If it broke (401, 403, repo already exists, network errors), jump to
> [Troubleshooting](#troubleshooting).
