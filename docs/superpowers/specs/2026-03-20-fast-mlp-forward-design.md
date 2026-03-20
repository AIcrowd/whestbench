# Fast MLP Forward Pass — Design Spec

## Problem

The current MLP forward pass in `simulation.py` uses pure NumPy and stores all
layer activations simultaneously. At the default scoring spec (width=256,
depth=16, ground_truth_budget=16.7M samples), `run_mlp_all_layers` requires
~272 GB of memory — making it impossible to run. Even at CLI defaults (width=100,
2.56M samples) it pushes ~16 GB. The matmul+ReLU loop also leaves performance
on the table by not leveraging fused CPU kernels.

## Solution

Add `simulation_fast.py` — a PyTorch CPU backend with chunked streaming — while
keeping `simulation.py` as the readable, trusted reference oracle.

## Architecture

### File layout

```
src/network_estimation/
    simulation.py          # Reference (untouched)
    simulation_fast.py     # Optimized PyTorch path (new)
tests/
    test_simulation.py     # Reference tests (untouched)
    test_simulation_fast.py # Correctness tests (new)
```

### API contract

`simulation_fast.py` exposes the same four functions with identical signatures
and return types:

```python
def relu(x: NDArray[np.float32]) -> NDArray[np.float32]
def run_mlp(mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]
def run_mlp_all_layers(mlp: MLP, inputs: NDArray[np.float32]) -> List[NDArray[np.float32]]
def output_stats(mlp: MLP, n_samples: int) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]
```

### Graceful fallback

```python
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if not _HAS_TORCH:
    from .simulation import relu, run_mlp, run_mlp_all_layers, output_stats
```

When PyTorch is absent, the module re-exports the reference implementations
unchanged. Zero behavior difference.

### Module structure

The file uses an early-return pattern: the fallback `from .simulation import
...` block is followed by a module-level return-equivalent (the rest of the
file is inside `if _HAS_TORCH:` or, more practically, the four function defs
are only reached when `_HAS_TORCH` is True). All `torch.randn` calls
explicitly pass `dtype=torch.float32` to avoid sensitivity to
`torch.set_default_dtype()`.

## Core forward pass

`run_mlp` and `run_mlp_all_layers` convert inputs and weights to
`torch.Tensor`, run matmul+ReLU under `torch.no_grad()`, and convert back to
NumPy.

### Weight caching

MLP is a frozen dataclass but contains a `list` field (unhashable), so it
cannot be used as a `WeakKeyDictionary` key. Instead, torch weight tensors
are cached in a plain dict keyed on `id(mlp)`, with a `weakref.ref`
destructor callback registered on each MLP to evict the cache entry when
the object is garbage-collected. This avoids repeated NumPy-to-Torch
conversion when the same MLP is reused across `baseline_time` and
`estimator.predict`.

### Thread control

At module load, `torch.set_num_threads(N)` is set from `os.cpu_count()`
(capped at 4 to match the target 4-vCPU AWS instance). This ensures
consistent timing across environments.

## Chunked `output_stats` — the main optimization

Instead of materializing all samples and all layers, process in cache-friendly
chunks and accumulate statistics online.

### Algorithm

```
accumulators: layer_sums (depth, width), final_sum_sq (width,)

for each chunk of size C from n_samples:
    x = torch.randn(C, width, dtype=torch.float32)
    for each layer:
        x = relu(x @ w)
        layer_sums[layer] += x.sum(dim=0)
    final_sum_sq += (x**2).sum(dim=0)

layer_means = layer_sums / n_samples
final_mean = layer_means[-1]
avg_variance = mean(final_sum_sq / n_samples - final_mean**2)
```

### Memory

Working set is O(chunk_size * width) — about 2-8 MB regardless of n_samples.
Compared to O(n_samples * width * depth) for the reference path.

### Chunk size selection

`_pick_chunk_size(width)` targets a working set that fits comfortably in L2/L3
cache: `max(1024, min(16384, 2**20 // width))`. For typical widths (100-256)
this yields 4096-10240 rows and a ~2-8 MB working set. For very small widths
the chunk may be smaller; the formula is a heuristic, not a guarantee.
Configurable via optional parameter for benchmarking.

### Numerical stability note

The variance formula `E[X^2] - E[X]^2` is mathematically exact but can suffer
from catastrophic cancellation when values are large. For ReLU-activated MLPs
with He initialization, activations stay bounded and this is not a practical
concern at the sample sizes used here (thousands to millions). If precision
ever becomes an issue, Welford's online algorithm is a drop-in replacement.

### RNG note

Chunked `torch.randn` produces different random draws than a single
`np.random.randn`. This is fine — `output_stats` is Monte Carlo estimation
and any i.i.d. Gaussian source produces statistically equivalent results.

## Integration

### Callers that switch to `simulation_fast`

- `scoring.py` — imports `run_mlp` and `output_stats`
- `dataset.py` — imports `output_stats`

### `__init__.py` — deliberately stays on `simulation`

`__init__.py` re-exports `relu`, `run_mlp`, `run_mlp_all_layers`, and
`output_stats` from `simulation`. This is the participant-facing public API
and intentionally remains on the reference implementation. Participants who do
`from network_estimation import run_mlp` get the readable NumPy version.
Internal callers (`scoring.py`, `dataset.py`) import `simulation_fast`
directly for the optimized path.

### `baseline_time` behavioral note

`scoring.py:baseline_time` uses `run_mlp` to measure wall-clock time that
becomes the estimator's time budget. Switching to the fast `run_mlp` means
baseline times will be shorter, giving estimators less time. This is the
correct behavior: the baseline should reflect the platform's actual forward
pass speed. A faster baseline is a harder (fairer) benchmark — estimators must
beat the optimized sampling, not the unoptimized one.

### What stays on `simulation`

- The reference module itself (untouched, importable by participants)
- `__init__.py` public API (participant-facing)
- Participant estimators
- Existing test suite

### Dependency

PyTorch added as an optional dependency under `[dependency-groups]` to match
the existing project convention:

```toml
[dependency-groups]
dev = [...]
fast = ["torch>=2.0"]
```

### No CLI changes

The speedup is transparent. No flags, no configuration.

## Testing

### `tests/test_simulation_fast.py`

1. **Exact match** (deterministic, small MLP): `run_mlp` and
   `run_mlp_all_layers` produce identical outputs to the reference when given
   the same MLP and inputs.

2. **Statistical equivalence**: `output_stats` on both paths with large
   n_samples (50,000) and small MLP (width=8, depth=4). Assert means within
   `atol=0.05`.

3. **Fallback**: Mock `_HAS_TORCH = False`, verify re-export of reference
   functions.

4. **Chunk boundary**: Run with `n_samples` not a multiple of chunk_size
   (e.g. 10007). No off-by-one errors.

## Expected impact

- **Speed:** 2-5x faster forward passes from MKL/oneDNN BLAS + fused
  matmul+ReLU + cache-friendly chunking.
- **Memory:** O(MB) instead of O(GB). The scoring.py defaults become actually
  runnable.
- **Portability:** Works on Mac (Accelerate) and Linux (MKL) with same code.
  Falls back to NumPy reference if PyTorch is absent.
