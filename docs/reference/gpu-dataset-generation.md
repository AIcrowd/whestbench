# Generating Large Datasets on GPU

For ground-truth bakes with `n_samples ≥ 10⁸`, the default CPU path is slow. The
optional torch backend runs the same computation on GPU (or torch CPU for dev).
A `n_samples=10⁹` bake at default config (10 MLPs) takes ~30 hours on CPU but
~15–30 min on a single GPU. Larger n_mlps scales linearly — see
[Performance expectations](#performance-expectations) below for measured numbers.

## Install

```bash
pip install whestbench[gpu]
```

This pulls in `torch` as an optional dependency. The standard `pip install
whestbench` does not include torch.

## Quick start

```bash
# Auto-detect best available device (cuda > mps > cpu)
# WARNING: this takes ~4 hours on L40S, ~14 h on M3 Max. Calibrate first
# (see "Calibration recipe" below) before committing to a multi-hour bake.
whest dataset bake \
    --torch --device auto \
    --n-mlps 100 --n-samples 1_000_000_000 \
    --width 256 --depth 8 --seed 42 \
    --output ./ground-truth

# Smaller production-realistic example (10 MLPs × 10⁹ ≈ 25 min on L40S)
whest dataset bake \
    --torch --device cuda \
    --n-mlps 10 --n-samples 1_000_000_000 \
    --width 256 --depth 8 --seed 42 \
    --output ./data

# Develop on laptop using torch CPU (works without GPU)
whest dataset bake \
    --torch --device cpu \
    --n-mlps 5 --n-samples 100_000 \
    --width 256 --depth 8 \
    --output ./dev
```

Output is a **directory** (schema 3.0 layout), not a single `.npz` file:

```
./ground-truth/
├── data/public-00000-of-00001.parquet
├── metadata.json
└── README.md
```

Load it with `whestbench.load_dataset` or push to HF Hub with
`whest dataset push`. The array schema is identical to a CPU bake — the same
8 Parquet columns, same `mlp_name` values at the same seed.

> The seed → name mapping is stable across machines as long as the installed
> `faker` version matches the pin in `pyproject.toml`. Bumping `faker` is a
> deliberate operation; the lock-down test in `tests/test_naming.py` trips when
> faker's wordlists change, and reference datasets must be re-baked alongside
> the version bump.

## Parallel bakes with `--slice`

For very large bakes, use `--slice K/N` to distribute across multiple GPU workers.
Each worker produces a partial directory; run `whest dataset merge` afterwards.

```bash
# 4 workers
whest dataset bake --slice 0/4 --torch --device cuda \
    --n-mlps 400 --n-samples 1_000_000_000 \
    --width 256 --depth 8 --seed 42 --output ./p0
# ... (repeat for slices 1/4, 2/4, 3/4)

# Merge
whest dataset merge ./p0 ./p1 ./p2 ./p3 --output ./final
```

See [Parallel bake across multiple GPUs](../how-to/parallel-bake.md) for the full
walkthrough.

## Publishing

After baking (and optionally merging), push to HF Hub:

```bash
whest dataset push ./ground-truth \
    --repo aicrowd/arc-whestbench-2026-eval \
    --tag v1
```

See [Publishing a dataset to HuggingFace Hub](../how-to/publish-to-hf-hub.md).

## Device selection

| `--device` | Behavior |
|---|---|
| omitted | Use the default flopscope CPU path (no torch needed). |
| `auto` | Resolves `cuda > mps > cpu` at runtime. |
| `cuda` | Explicit CUDA. Errors if `torch.cuda.is_available()` is False. |
| `mps` | Apple Silicon GPU. Errors if MPS is unavailable. |
| `cpu` | Torch on CPU. First-class dev option, not a silent fallback. |

There is **no automatic fallback to CPU** if a GPU device is requested but
unavailable. Explicit device choices are honored or rejected loudly.

`--max-threads` cannot be combined with `--torch`; torch manages threading
internally.

## Performance expectations

**Key finding from L40S benchmarking**: at `width=256`, effective throughput
is bottlenecked at **~7–10 TFLOP/s on modern GPUs regardless of peak fp32 spec**.
The matmul is too small to saturate tensor cores, and TF32/fp16 give negligible
speedup at this size (measured: ~2% on L40S). **Don't extrapolate from peak
fp32 ratings; they overestimate by 5–10× for this workload.**

### Measured (NVIDIA L40S, AWS g6e.xlarge)

| n_mlps | n_samples | wall time | effective throughput |
|---|---|---|---|
| 10 | 10⁶ | 1.41 s | ~7.5 TF |
| 100 | 10⁶ | 13.78 s | ~7.5 TF |
| 10 | 10⁹ | ~23 min (linear projection) | — |
| 100 | 10⁹ | ~3.9 h (linear projection)<sup>†</sup> | — |

<sup>†</sup> *TODO: confirm with full-bake measurement — calibration anchored
on N=10⁶ predicts 3.9 hours; a 100 MLPs × 10⁹ bake is in progress as of
this writing.*

Scaling on L40S is **fully linear** in `n_mlps` and `n_samples`. Quadratic
in `width`. The `mlps_per_batch` knob has near-zero impact at L40S scale
(measured ≤ 0.4% spread across B ∈ {4, 8, 16, 32}).

### Extrapolations to other GPUs

Anchor: 7.5 TFLOP/s effective on L40S. For modern Ampere+/Ada/Hopper at
`width=256`, expect 5–10 TFLOP/s in practice — variation between cards is
small because the small-matmul ceiling binds before peak compute matters.

| Hardware | 10 MLPs × 10⁹ (est.) | 100 MLPs × 10⁹ (est.) |
|---|---|---|
| L40S (g6e.xlarge) | **~23 min (measured)** | **~3.9 h (measured)** |
| H100 PCIe | ~15–25 min | ~2.5–4 h |
| RTX 4090 | ~20–35 min | ~3.5–6 h |
| A100 80GB | ~25–40 min | ~4–6.5 h |
| RTX 3090 | ~30–50 min | ~5–8 h |
| Apple M3 Max (mps) | ~2.3 h (measured) | ~14 h (measured) |
| CPU (flopscope) | ~30 h | ~12 days |

**Strong recommendation**: run a 60-second calibration on your actual GPU
before committing to a multi-hour bake — see [Calibration recipe](#calibration-recipe)
below.

## Calibration recipe

A 60-second `N=10⁶` run on any GPU gives a precise wall-time projection for
your actual `N=10⁹` bake. Run this once when you spin up the instance:

```python
import time
from pathlib import Path
import torch
from whestbench.dataset_torch import create_dataset_torch

# Warmup (kernel compilation, ~0.2s on cuda)
create_dataset_torch(
    n_mlps=2, n_samples=10_000, width=256, depth=8,
    seed=0, output_path=Path('/tmp/warmup'), device='cuda')

# Calibration anchored on n_mlps=10 to match the production setup
t0 = time.perf_counter()
create_dataset_torch(
    n_mlps=10, n_samples=1_000_000, width=256, depth=8,
    seed=42, output_path=Path('/tmp/cal'), device='cuda')
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f'{elapsed:.2f}s at N=10⁶ → projected {elapsed*1000/60:.1f} min at N=10⁹')
```

`torch.cuda.synchronize()` is critical — CUDA ops are async; without it
you'd measure dispatch time, not compute time.

If the projection looks reasonable, proceed with the full bake. If it's
2× higher than expected, check `torch.backends.cuda.matmul.allow_tf32`
(default `False` in recent torch) — but expect only marginal speedup
since matmuls are small.

## Verifying the output

Datasets baked with `--torch` have identical Parquet column layout to default
(flopscope) datasets. Provenance is in `metadata.json`:

```python
import whestbench

ds = whestbench.load_dataset("./ground-truth")
md = whestbench.metadata(ds)
backend = md.get("backend", "flopscope")   # "torch" or "flopscope"
device = md.get("device")                  # "cuda" | "mps" | "cpu" if torch
torch_version = md.get("torch_version")
```

## Reproducibility

Datasets are deterministic per `(seed, device, torch_version)`. The seed
hierarchy is identical to the flopscope path; only the leaf RNG that produces
input samples changes (numpy PCG64 → torch Philox/MT).

**Important:** the same `seed` on the CPU (flopscope) and torch paths will not
produce bitwise-identical datasets — different RNG algorithms. They are
statistically equivalent: per-neuron means agree within Monte Carlo noise
(~3×10⁻⁵ at N=10⁹).

## Precision strategy

The torch backend uses fp32 matmul + fp64 reduction accumulators on CUDA and
CPU, matching the flopscope path's numerical semantics. On MPS (which does not
support fp64) the accumulators fall back to fp32 — this is acceptable for dev
workflows where N ≤ 10⁵, since fp32 accumulation error is comparable to Monte
Carlo noise at those scales. For production N=10⁹ bakes, use `--device cuda`.

## Python API for tuning

For power-user tuning beyond what the CLI exposes:

```python
from whestbench.dataset_torch import create_dataset_torch

create_dataset_torch(
    n_mlps=100, n_samples=10**9,
    width=256, depth=8,
    seed=42, output_path="ground_truth",
    device="cuda",
    mlps_per_batch=32,   # default: min(n_mlps, 16). Larger uses more GPU memory.
    chunk_size=1 << 20,  # default: memory-aware on cuda, 65536 on mps/cpu.
)
```

See the docstring for full parameter semantics. The CLI exposes `--device`,
`--mlps-per-batch`, and `--chunk-size`; these are also available as Python-API
knobs for benchmarking.

## Troubleshooting

**`ImportError: create_dataset_torch requires torch`** — Install the gpu
extra: `pip install whestbench[gpu]`.

**`RuntimeError: CUDA requested but torch.cuda.is_available() is False`** —
Either CUDA isn't installed at the system level, or torch was installed
without CUDA support. Check `python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"`.
For dev without a GPU, use `--device cpu`.

**Out of memory on GPU** — Lower the Python-API knobs:
- `mlps_per_batch`: fewer MLPs in parallel.
- `chunk_size`: smaller chunks of samples per step.

The auto-tuned defaults target ~25% of free GPU memory; on very full GPUs you
may need to override.

**Dataset looks slightly different from a CPU bake at the same seed** —
Expected (see Reproducibility above). To verify equivalence, compare means
within `~5/sqrt(n_samples)` tolerance.

**Progress bar shows fewer chunks than expected** — On GPU the chunk size is
much larger than on CPU (~64K–1M vs 4K), so there are 16–256× fewer chunks per
MLP. Total work units `n_mlps * chunks_per_mlp` still reflects the same total
samples processed.

**Wall time is much longer than peak-fp32 math suggests** — Expected. Peak
fp32 specs assume tensor cores can saturate, which requires large matmul
dimensions. At `width=256` the matmuls are too small; effective throughput
plateaus at ~7–10 TFLOP/s on most modern GPUs regardless of whether the
card is rated for 30 TF (L40S fp32) or 100 TF (H100 fp32). Tools like
`nvidia-smi` will correctly show 100% GPU utilization despite the low
effective TFLOP/s — the card is fully busy, the kernels are just shape-bound.
TF32 / fp16 give only ~2% speedup at this matmul size (measured), so don't
rely on them to close the gap. See [Performance expectations](#performance-expectations).

**`mlps_per_batch` doesn't seem to do anything** — Correct. On CUDA at
`width=256`, varying `mlps_per_batch` between 4 and 32 has < 1% effect on
wall time (measured on L40S). The bottleneck is the per-chunk matmul shape,
not the batching layer. Don't waste time tuning it.
