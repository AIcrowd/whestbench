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

## Downloading from HF Hub and the local cache

You want to score against a dataset published by your team or the contest
organisers. There are two paths.

### `whest dataset download` — explicit fetch

Use when you want a real on-disk copy you can inspect, ship to another
machine, or commit to a separate artifact store:

```bash
whest dataset download aicrowd/arc-whestbench-public-2026 \
    --revision v1-warmup \
    --output ./eval
```

Representative output:

```
→ Downloading aicrowd/arc-whestbench-public-2026@v1-warmup → ./eval
  Preflight: 1 parquet shard, 2.0 GB, 1,000 MLPs
  ✓ Downloaded 2.0 GB              ████████████████████ 100%   28.9s
✓ Wrote ./eval (cache: ~/.cache/huggingface/hub/datasets--aicrowd--arc-whestbench-public-2026)
```

With `--output` set, files are materialised under the named directory; the HF
cache also picks them up.

### Auto-fetch via `whest run`

You can skip the explicit download — `whest run` does it lazily on first use:

```bash
whest run --estimator estimator.py \
          --dataset hf://aicrowd/arc-whestbench-public-2026@v1-warmup
```

This downloads on first invocation (showing a progress bar) and caches.
Subsequent runs are ~10× faster (the cache hit prints `Loaded from cache`).

### HF cache layout

After a fetch, the HF cache lives at three places:

| Path | What's there |
|---|---|
| `~/.cache/huggingface/hub/datasets--<org>--<name>/` | Raw blobs (Git LFS / Xet objects) + the revision snapshot symlinks |
| `~/.cache/huggingface/datasets/<org>___<name>/` | The `datasets` library's regenerated Arrow tables (memory-mapped) |
| `~/.cache/huggingface/xet/{chunk_cache,shard_cache,staging}/` | Xet chunk-level dedup cache (since `hf_xet ≥ 1.0`) |

Total disk usage is roughly `2× download size` (the parquet blob + Arrow
rebuild). The hub cache uses content-addressed dedup, so the same blob is
shared across revisions and even repos.

### Cleaning up

Defer to HF's own cache CLI — it understands the layout above and will not
accidentally orphan blobs that are still referenced from another revision:

```bash
hf cache ls                  # show what's there
hf cache prune               # drop unreferenced revisions
hf cache rm <selector>       # remove a specific repo or revision
hf cache verify              # check integrity
```

Full reference: [HF cache management](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).

### Cache location overrides

| Env var | What it sets | Default |
|---|---|---|
| `HF_HOME` | Root of all HF state | `~/.cache/huggingface` |
| `HF_HUB_CACHE` | Hub-only cache (blobs/snapshots) | `$HF_HOME/hub` |
| `HF_DATASETS_CACHE` | datasets-library Arrow cache | `$HF_HOME/datasets` |
| `HF_XET_CACHE` | Xet chunk staging | `$HF_HOME/xet` |

When running on NFS, point `HF_XET_CACHE=/local/ssd` to avoid roundtrips.
See [Performance tuning](#performance-tuning) for more knobs.

> `whest dataset pull` continues to work as a deprecated alias for `download`
> through v0.6.

> If it broke (long pause, disk full, gated dataset, `cas-bridge.xethub.hf.co`
> URLs you don't recognise), jump to [Troubleshooting](#troubleshooting).

## Streaming mode

You want to score against a small slice of a remote dataset without paying
the cost of a full download. `whest run --streaming` consumes the dataset
row-group-by-row-group over HTTP instead of downloading it first.

### When to use

- You're iterating on estimator code with `--n-mlps 5` (or some small K).
  Streaming fetches only the first ⌈K/47⌉ row groups (~95 MB each for the
  warmup dataset) instead of the full 2 GB.
- You're on a constrained-disk environment (CI runner, container).
- You want a fast first-row response time more than total throughput.

### When NOT to use

- Repeated full evaluations of the same dataset. Streaming does NOT populate
  the cache — every run re-fetches. Use the
  [default materialise path](#downloading-from-hf-hub-and-the-local-cache)
  instead.
- Anything that needs random access. `IterableDataset` is iteration-only;
  `len(ds)`, `ds[i]`, and `ds.shuffle(seed=…)` don't work as expected.

### Trade-off table

| Property | Materialise (default) | `--streaming` |
|---|---|---|
| First-row latency, cold cache | ~30 s (full download) | ~5 s |
| First-row latency, warm cache | ~2 s | ~5 s (re-fetch) |
| Disk usage | ~4 GB (blob + Arrow) | 0 |
| Subsequent runs | ~2 s (cache hit) | ~5 s (re-fetch every time) |
| Random access | Yes | No |

### Authentication and streaming

Unauthenticated requests to HF are rate-limited and noticeably slower. Run
`hf auth login` once to set a token; streaming throughput typically improves
30–50% authenticated.

### Example

```bash
whest run --estimator estimator.py \
          --dataset hf://aicrowd/arc-whestbench-public-2026@v1-warmup \
          --streaming \
          --n-mlps 5
```

You'll see a `⚠ Streaming from HF` warning at startup, then a progress
indicator while the first row group is fetched, then scoring begins.

> Streaming is incompatible with `--json` output (it would corrupt JSON
> ordering) and `len(ds)` raises on a streaming dataset. Both are documented
> under [Troubleshooting](#troubleshooting).

## Multi-split datasets

A dataset can contain multiple disjoint groups of MLPs — typically `public`
(open to participants for tuning) and `holdout` (used only by the leaderboard
grader). One repo, two splits.

### When and why

- **Leaderboard datasets:** participants score against `public` locally,
  the leaderboard grader scores against `holdout`. Same parquet schema, same
  hardware fingerprint, different seeds.
- **Train/validation flow:** split a dataset into `train`/`val`/`test` for
  meta-learning experiments on top of WhestBench.

### Baking a split

Each split is baked separately. Make sure to use distinct `--mlp-seeds` files
so the splits don't overlap:

```bash
whest dataset bake --n-mlps 500 --split public  --output ./eval-public
whest dataset bake --n-mlps 500 --split holdout --output ./eval-holdout
```

### Combining splits into one multi-split directory

```bash
whest dataset combine-splits ./eval-public ./eval-holdout --output ./eval-full
```

The result is a single dataset directory with both splits in `data/`,
suitable for `whest dataset upload` to a single HF repo.

### Selecting a split when running

```bash
whest run --estimator estimator.py \
          --dataset hf://aicrowd/eval-full@v1 \
          --split public
```

Without `--split`, multi-split datasets are rejected by `whest run` (the
scoring path scores against exactly one split at a time, by design).

### Inspecting splits

```bash
whest dataset info ./eval-full
# Reports each split's n_mlps and seed.
```

> If `combine-splits` complains about overlapping `mlp_seed`s or mismatched
> hardware fingerprints, see [Troubleshooting](#troubleshooting).

## Performance tuning

These are power-user knobs. The defaults are fine for almost everyone.

### Xet high-performance mode

If you have ≥64 GB RAM and a fat uplink:

```bash
export HF_XET_HIGH_PERFORMANCE=1
```

Saturates both bandwidth and CPU cores. Helpful when downloading
many-GB datasets to a workstation. Reference:
[HF Xet storage docs](https://huggingface.co/docs/hub/xet/using-xet-storage).

### Local SSD for the Xet cache

If your HF cache is on NFS or a slow disk:

```bash
export HF_XET_CACHE=/local/ssd/hf-xet
```

Keeps the chunk staging cache on fast local storage. The main hub cache
(`HF_HUB_CACHE`) can stay on NFS — only the per-chunk Xet metadata is
roundtrip-sensitive.

### Disabling Xet entirely

```bash
export HF_HUB_DISABLE_XET=1
```

Falls back to plain LFS transport. Rarely useful; only reach for it if you've
confirmed a Xet-specific bug.

### Disabling progress bars (CI)

```bash
export HF_HUB_DISABLE_PROGRESS_BARS=1
```

Whestbench's `say.*` lines still emit; only the progress bars are suppressed.
For complete silence add `--quiet` to the `whest` invocation.

## Troubleshooting

**"I see a long pause and no output."**
Cache miss on a cold HF cache. Watch the progress bar — for the warmup
dataset it's ~30 s on a 70 MB/s link. To avoid silent re-downloads, run
[`whest dataset download`](#whest-dataset-download--explicit-fetch) ahead
of time, or `ls ~/.cache/huggingface/hub/` to confirm progress.

**"Downloads feel slow."**
You're probably unauthenticated; HF rate-limits anonymous traffic.
Run `hf auth login` once and re-run. See also
[Authentication and streaming](#authentication-and-streaming).

**"Disk filled up."**
HF stores blobs in both `~/.cache/huggingface/hub/` (raw download) and
`~/.cache/huggingface/datasets/` (regenerated Arrow). Use
`hf cache prune` to drop unreferenced revisions, then `hf cache ls` to
verify reclaimed space. See [Cleaning up](#cleaning-up).

**"401/403 on upload."**
Your token doesn't have `write` scope. Re-login with
`hf auth login --token <new-token>` from a token created with write access.
For org-owned repos, your account also needs membership in the org.

**"Cannot use `--streaming` with `--json` output."**
Known limitation — streaming progress events would corrupt JSON ordering.
Drop `--json`, or drop `--streaming`.

**"`len(ds)` raises on a streaming dataset."**
Expected per HF docs. Use `whestbench.metadata(ds)["n_mlps"]` instead — it
reflects the upstream `metadata.json`, not the local materialised count.

**"I see `cas-bridge.xethub.hf.co` URLs but the file is LFS."**
That's HF's Xet bridge transparently serving legacy LFS content via the Xet
CDN edge. No action required. If you need to force plain-LFS transport for
debugging, set `HF_HUB_DISABLE_XET=1` (see
[Disabling Xet entirely](#disabling-xet-entirely)).

**"Dataset is gated."**
Request access on the dataset page (HF will email you a link from
`https://huggingface.co/datasets/<repo>`), then re-run. Make sure you're
authenticated with the same account that was granted access.
