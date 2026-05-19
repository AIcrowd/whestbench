# Generating Large Datasets on GPU

For ground-truth bakes with `n_samples ≥ 10⁸`, the default CPU path is slow. The
optional torch backend runs the same computation on GPU (or torch CPU for dev).
A `n_samples=10⁹` bake at default config takes ~12 days on CPU but ~5–20 min on
a single GPU.

## Install

```bash
pip install whestbench[gpu]
```

This pulls in `torch` as an optional dependency. The standard `pip install
whestbench` does not include torch.

## Quick start

```bash
# Auto-detect best available device (cuda > mps > cpu)
whest create-dataset --device auto \
    --n-mlps 100 --n-samples 1000000000 -o ground_truth.npz

# Explicit cuda
whest create-dataset --device cuda --seed 42 -o data.npz

# Develop on laptop using torch CPU (works without GPU)
whest create-dataset --device cpu --n-mlps 5 --n-samples 100000 -o dev.npz
```

The output `.npz` is read by `whestbench.dataset.load_dataset()` unchanged —
the array schema is identical to a default (flopscope) dataset.

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

`--max-threads` cannot be combined with `--device`; torch manages threading
internally.

## Performance expectations

At default config (`width=256, depth=8, n_mlps=10, n_samples=10⁹`):

| Hardware | Realistic throughput | Wall time |
|---|---|---|
| H100 PCIe | ~30 TF | ~5–7 min |
| RTX 4090 | ~40 TF | ~4–5 min |
| A100 80GB | ~10 TF | ~15–20 min |
| RTX 3090 | ~18 TF | ~10–12 min |
| Apple M3 Max (mps) | ~3 TF | ~1 hour |
| CPU (flopscope) | — | ~12 days |

Wall times scale roughly linearly in `n_mlps` and `n_samples`, quadratically in `width`.

## Verifying the output

Datasets baked with `--device` have identical array layout to default
(flopscope) datasets. Provenance is in metadata:

```python
from whestbench.dataset import load_dataset

bundle = load_dataset("ground_truth.npz")
backend = bundle.metadata.get("backend", "flopscope")   # "torch" or "flopscope"
device = bundle.metadata.get("device")                  # "cuda" | "mps" | "cpu" if torch
torch_version = bundle.metadata.get("torch_version")
```

Files written by either path have `schema_version="2.3"`. Pre-existing files
baked before this feature have `"2.2"` and no `backend` key (default to
`"flopscope"`). All three are loaded identically — the loader doesn't care
which path produced them.

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
    width=256, depth=8, flop_budget=17_000_000_000,
    seed=42, output_path="ground_truth.npz",
    device="cuda",
    mlps_per_batch=32,   # default: min(n_mlps, 16). Larger uses more GPU memory.
    chunk_size=1 << 20,  # default: memory-aware on cuda, 65536 on mps/cpu.
)
```

See the docstring for full parameter semantics. The CLI exposes `--device`
only; `mlps_per_batch` and `chunk_size` are Python-API knobs for benchmarking.

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
