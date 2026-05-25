# Parallel bake across multiple GPUs / hosts

Bake one large dataset across N workers, then merge the partials into a single
canonical artifact that is bit-equivalent to what a single-host bake would have
produced.

## When to use this

At the default sampling rate (`n_samples=1_000_000_000`), a single L40S GPU takes
roughly 4 hours for 100 MLPs (measured; see
[GPU Dataset Generation](../reference/gpu-dataset-generation.md) for the full timing
table). Splitting the work across multiple workers reduces wall time proportionally:

- 1 L40S × 100 MLPs × 10⁹ samples ≈ **~4 h**
- 4 L40S workers × 25 MLPs each × 10⁹ samples ≈ **~1 h**
- 8 L40S workers × 12–13 MLPs each × 10⁹ samples ≈ **~30 min**

Parallel baking is also useful for fault tolerance — if one worker fails, you only
need to re-bake its slice.

## 1. Bake each slice

Use `--slice K/N` to assign each worker a disjoint range of MLPs. All workers must
use the **same** `--seed`, `--n-mlps`, `--n-samples`, `--width`, and `--depth` — the
merge step enforces this.

The following example bakes 1000 MLPs across 4 workers. Run each command on its own
host (or in a separate job):

**Worker 0** (MLPs 0–249):
```bash
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --seed 42 \
    --slice 0/4 \
    --torch --device auto \
    --output ./partial-0
```

**Worker 1** (MLPs 250–499):
```bash
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --seed 42 \
    --slice 1/4 \
    --torch --device auto \
    --output ./partial-1
```

**Worker 2** (MLPs 500–749):
```bash
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --seed 42 \
    --slice 2/4 \
    --torch --device auto \
    --output ./partial-2
```

**Worker 3** (MLPs 750–999):
```bash
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --seed 42 \
    --slice 3/4 \
    --torch --device auto \
    --output ./partial-3
```

Each worker writes a directory marked `is_partial=true` in `metadata.json`. The
loader refuses to load partial datasets directly — you must merge them first.

## 2. Fetch partials locally

Once all workers finish, collect the partial directories on a single machine.

```bash
# scp example (adjust hostnames and paths)
scp -r worker-0:/data/partial-0 ./partial-0
scp -r worker-1:/data/partial-1 ./partial-1
scp -r worker-2:/data/partial-2 ./partial-2
scp -r worker-3:/data/partial-3 ./partial-3

# Or rsync (preserves timestamps, supports resumption)
rsync -avz worker-0:/data/partial-0/ ./partial-0/
rsync -avz worker-1:/data/partial-1/ ./partial-1/
rsync -avz worker-2:/data/partial-2/ ./partial-2/
rsync -avz worker-3:/data/partial-3/ ./partial-3/
```

## 3. Merge

`whest dataset merge` validates all partials, checks that their `mlp_range` values
cover `[0, 1000)` exactly once (no gaps, no overlaps), concatenates the Parquet
shards in MLP-index order, and writes a complete dataset directory:

```bash
whest dataset merge \
    ./partial-0 ./partial-1 ./partial-2 ./partial-3 \
    --output ./final-eval
```

Expected output:

```
Merged 4 partials to ./final-eval
```

The merge fails loudly on any of:
- Partials disagree on `seed`, `n_samples`, `width`, `depth`, `backend`, or
  `total_n_mlps` (`MergeIncompatibleError`)
- Ranges have gaps — e.g. `[0,250)` and `[500,750)` with nothing in between
  (`MergeIncompleteError`)
- Ranges overlap (`MergeOverlapError`)
- A partial's actual row `mlp_id` values don't match its declared `mlp_range`
  (`MergeCorruptError`)

## 4. Verify bit-equivalence (optional)

To confirm the parallel bake matches a serial bake on the same seed, bake a small
reference dataset on a single host and compare `all_layer_means`:

```python
import numpy as np
from datasets import load_dataset

# Load the merged result
merged = load_dataset("./final-eval", split="public")

# Bake a tiny reference (e.g. first 4 MLPs) on one host for verification
# whest dataset bake --n-mlps 4 --n-samples 1000000 \
#     --width 256 --depth 8 --seed 42 --output ./reference-4

reference = load_dataset("./reference-4", split="public")

# Compare means for the overlapping MLPs
for i in range(len(reference)):
    merged_means = np.array(merged[i]["all_layer_means"])
    ref_means = np.array(reference[i]["all_layer_means"])
    max_diff = np.abs(merged_means - ref_means).max()
    print(f"MLP {i}: max |Δmean| = {max_diff:.2e}")
    assert max_diff == 0.0, f"MLP {i}: not bit-exact!"

print("Bit-equivalence verified for first 4 MLPs.")
```

Expected output (for the CPU backend):
```
MLP 0: max |Δmean| = 0.00e+00
MLP 1: max |Δmean| = 0.00e+00
MLP 2: max |Δmean| = 0.00e+00
MLP 3: max |Δmean| = 0.00e+00
Bit-equivalence verified for first 4 MLPs.
```

For the torch backend, bit-equivalence holds within each backend (flopscope or torch)
but not across backends — they use different RNG algorithms.

## 5. Inspect and publish

Inspect the merged dataset, then push to HuggingFace Hub as a single artifact.
See [Publishing a dataset to HuggingFace Hub](./publish-to-hf-hub.md) for the full
publish walkthrough.

```bash
# Inspect
whest dataset inspect ./final-eval

# Publish
whest dataset push ./final-eval \
    --repo aicrowd/arc-whestbench-2026-eval \
    --tag v1 \
    --message "Parallel bake: 1000 MLPs, seed=42, 4 workers"
```

## Slicing model

### `--slice K/N`

Divides the logical dataset of `--n-mlps` into N equal slices and assigns slice K
(0-indexed). For `n_mlps=1000` and `N=4`:

| `--slice` | `mlp_range` |
|---|---|
| `0/4` | `[0, 250)` |
| `1/4` | `[250, 500)` |
| `2/4` | `[500, 750)` |
| `3/4` | `[750, 1000)` |

If `n_mlps` is not evenly divisible by N, the last slice gets the remainder.

### `--mlp-range START-END`

The lower-level alternative to `--slice`. Both endpoints are **inclusive** on the
CLI (e.g. `--mlp-range 0-249` covers MLPs 0 through 249 inclusive). The Python API
uses half-open `[start, end)` intervals internally.

Use `--mlp-range` for irregular splits or when you need to re-run only a specific
MLP range after a failure.

```bash
# Re-run just MLPs 250–499 after a worker failure
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 --seed 42 \
    --mlp-range 250-499 \
    --torch --device auto \
    --output ./partial-1-retry
```

### Bit-equivalence requirements

The merge step produces a dataset bit-equivalent to a single-host bake only when:

1. All workers use the **same** `--seed` and **same** `--n-mlps`. The seed hierarchy
   expands over all `n_mlps` logical slots before slicing, so each slot gets the same
   derived seed regardless of which worker processes it.
2. All workers use the **same backend** (`flopscope` vs `torch`). The two backends
   use different RNG algorithms and produce statistically equivalent but not bitwise
   identical results at the same seed.
3. For the `torch` backend on CUDA, bitwise reproducibility additionally requires
   the same torch version (CUDA kernel implementations may differ between versions).
