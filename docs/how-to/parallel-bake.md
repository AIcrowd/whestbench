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
use the **same** `--mlp-seeds` file, `--n-mlps`, `--n-samples`, `--width`, and
`--depth` — the merge step enforces this.

Generate the seeds file once before launching workers:

```bash
whest dataset generate-seeds --n-mlps 1000 > seeds.json
```

The following example bakes 1000 MLPs across 4 workers. Run each command on its own
host (or in a separate job):

**Worker 0** (MLPs 0–249):
```bash
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --mlp-seeds seeds.json \
    --slice 0/4 \
    --torch --device auto \
    --output ./partial-0
```

**Worker 1** (MLPs 250–499):
```bash
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --mlp-seeds seeds.json \
    --slice 1/4 \
    --torch --device auto \
    --output ./partial-1
```

**Worker 2** (MLPs 500–749):
```bash
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --mlp-seeds seeds.json \
    --slice 2/4 \
    --torch --device auto \
    --output ./partial-2
```

**Worker 3** (MLPs 750–999):
```bash
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 \
    --mlp-seeds seeds.json \
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
- Partials disagree on `n_samples`, `width`, `depth`, `backend`, or
  `total_n_mlps` (`MergeIncompatibleError`)
- Ranges have gaps — e.g. `[0,250)` and `[500,750)` with nothing in between
  (`MergeIncompleteError`)
- Ranges overlap (`MergeOverlapError`)
- A partial's actual row `mlp_id` values don't match its declared `mlp_range`
  (`MergeCorruptError`)

## 4. Verify bit-equivalence (optional)

To confirm the parallel bake matches a serial bake on the same seeds file, bake a
small reference dataset on a single host and compare `all_layer_means`:

```python
import numpy as np
from datasets import load_dataset

# Load the merged result
merged = load_dataset("./final-eval", split="public")

# Bake a tiny reference (e.g. first 4 MLPs) on one host for verification.
# Pass the SAME --chunk-size as the parallel workers — otherwise the auto-tuned
# chunk_size differs (workers: B=mlps_per_slice; reference: B=4) and reductions
# accumulate in different orders, producing ~5e-4 spurious diffs on CUDA.
# echo '[<seed0>,<seed1>,<seed2>,<seed3>]' > ref-seeds.json  # use seeds[0:4] from seeds.json
# whest dataset bake --n-mlps 4 --n-samples 1000000 \
#     --width 256 --depth 8 --mlp-seeds ref-seeds.json \
#     --chunk-size 524288 --output ./reference-4

reference = load_dataset("./reference-4", split="public")

# Compare means for the overlapping MLPs
for i in range(len(reference)):
    merged_means = np.array(merged[i]["all_layer_means"])
    ref_means = np.array(reference[i]["all_layer_means"])
    max_diff = np.abs(merged_means - ref_means).max()
    print(f"MLP {i}: max |Δmean| = {max_diff:.2e}")
    assert max_diff == 0.0, f"MLP {i}: not bit-exact!"

    # avg_variance loses ~1 float64 ULP from the (sum_sq/n - mean²) subtraction,
    # so compare with np.isclose rather than strict equality. rtol=1e-12 covers
    # ULP noise that scales with the variance magnitude; atol=1e-15 guards near
    # zero. Observed noise on N=1e9 bakes is ~1e-17, so this is ~100× headroom.
    merged_var = float(merged[i]["avg_variance"])
    ref_var = float(reference[i]["avg_variance"])
    assert np.isclose(merged_var, ref_var, rtol=1e-12, atol=1e-15), (
        f"MLP {i}: variance not within ULP tol "
        f"(merged={merged_var}, ref={ref_var})"
    )

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
    --repo aicrowd/arc-whestbench-2026 \
    --tag v1 \
    --message "Parallel bake: 1000 MLPs, 4 workers"
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
# Re-run just MLPs 250–499 after a worker failure (use the same seeds.json as the original bake)
whest dataset bake \
    --n-mlps 1000 --n-samples 1_000_000_000 \
    --width 256 --depth 8 --mlp-seeds seeds.json \
    --mlp-range 250-499 \
    --torch --device auto \
    --output ./partial-1-retry
```

### Bit-equivalence requirements

The merge step produces a dataset bit-equivalent to a single-host bake only when:

1. All workers use the **same** `--mlp-seeds` file and **same** `--n-mlps`. Under
   seed_protocol 3.0, each slot reads its input seed directly from that shared file,
   so the derived weight/sample/estimator streams are identical regardless of which
   worker processes the slot.
2. All workers use the **same backend** (`flopscope` vs `torch`). The two backends
   use different RNG algorithms and produce statistically equivalent but not bitwise
   identical results at the same seeds.
3. For the `torch` backend on CUDA, bitwise reproducibility additionally requires
   the same torch version (CUDA kernel implementations may differ between versions).
4. For the `torch` backend on CUDA, **all workers and any reference re-bake must use
   the same `--chunk-size`**. The default is auto-tuned per call from
   `mlps_per_batch` (which derives from `--n-mlps` minus slicing) and the device's
   free memory — so a worker baking a 1-MLP slice (`mlps_per_batch=1`, auto chunk
   ≈ 1048576) and a 4-MLP reference bake (`mlps_per_batch=4`, auto chunk ≈ 524288)
   will pick different chunk sizes, accumulate float reductions in different
   orders, and disagree by ~5e-4 absolute on `all_layer_means`, `final_means`, and
   `avg_variance`. Pinning `--chunk-size` to a fixed value across every bake
   (workers AND any reference bake) eliminates this. For `width=256`,
   `--chunk-size 524288` is a safe choice across all batch sizes from 1 to 16.

   Cross-host CUDA non-determinism beyond chunk-size has been ruled out in practice
   when the standard PyTorch determinism flags are set (`cudnn.deterministic=True`,
   `cudnn.benchmark=False`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). With those + a
   pinned `--chunk-size`, parallel-vs-serial bakes match bit-exactly on
   `weights`, `all_layer_means`, and `final_means`. `avg_variance` differs by
   ~1 float64 ULP (~1e-17 on N=1e9 bakes) due to the `(sum_sq/n − mean²)`
   subtraction; compare it with `np.isclose(rtol=1e-12, atol=1e-15)` rather
   than strict equality. `rtol` covers ULP noise that scales with variance
   magnitude; `atol` guards near zero.

## Multi-split datasets

For datasets with multiple splits (e.g. the evaluation dataset with `public` and
`holdout`), bake each split independently — each split has its own seed file and the
seeds must be uncorrelated — then combine.

Under seed_protocol 3.0, each split has its own JSON file of per-MLP seeds. All
workers baking a given split must receive the SAME JSON file (they internally
slice it by `--slice K/N`); seeds for different splits MUST be different files
to preserve cross-split independence.

(The orchestrator in `whest-evaluation-utils/gpu-dataset-bake/` automates this.)

```bash
# Generate independent seed files for each split (once, before launching workers).
whest dataset generate-seeds --n-mlps 50 > public-seeds.json
whest dataset generate-seeds --n-mlps 50 > holdout-seeds.json

# Parallel-bake the public split (4 workers, same seeds file).
for K in 0 1 2 3; do
  whest dataset bake --n-mlps 50 --n-samples 1e9 --width 256 --depth 8 \
    --split public --mlp-seeds public-seeds.json --slice $K/4 \
    --torch --device cuda --output ./pub-p$K &
done
wait
whest dataset merge ./pub-p* --output ./pub-complete

# Parallel-bake the holdout split (4 workers, different seeds file).
for K in 0 1 2 3; do
  whest dataset bake --n-mlps 50 --n-samples 1e9 --width 256 --depth 8 \
    --split holdout --mlp-seeds holdout-seeds.json --slice $K/4 \
    --torch --device cuda --output ./hold-p$K &
done
wait
whest dataset merge ./hold-p* --output ./hold-complete

# Combine into one multi-split directory.
whest dataset combine-splits ./pub-complete ./hold-complete --output ./eval

# Inspect, push.
whest dataset inspect ./eval
whest dataset push ./eval --repo aicrowd/arc-whestbench-2026-evals --tag round-1 --private
```

Each per-split bake is independent — workers in different splits don't share any seed
state. The combine step validates that all splits agree on the invariants (`width`,
`depth`, `n_samples`, `backend`) but allows different per-split `n_mlps`.
