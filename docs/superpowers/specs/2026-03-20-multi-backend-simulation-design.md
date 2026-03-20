# Multi-Backend Simulation Profiling System

## Problem

The current `simulation_fast.py` uses PyTorch as the sole optimized backend, but benchmarks show it can be slower than NumPy on certain hardware (e.g., Apple Silicon under Rosetta 2). The production target is AWS x86 Linux with varying vCPU counts, and we cannot predict which backend will perform best without measuring on that hardware.

The goal is to provide challenge participants with a highly optimized MLP forward pass so they can focus on estimation strategies rather than simulation performance.

## Solution

Create multiple simulation backends behind a common ABC interface, with a `nestim profile-simulation` CLI command to benchmark them head-to-head on any hardware. This enables data-driven backend selection on the target AWS instances.

## Architecture

### Backend Interface (ABC)

File: `src/network_estimation/simulation_backend.py`

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
from numpy.typing import NDArray
from .domain import MLP

class SimulationBackend(ABC):
    """Abstract interface for MLP forward pass backends."""

    name: str  # e.g. "numpy", "pytorch", "numba", "scipy", "jax"

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this backend's dependencies are installed."""
        ...

    @classmethod
    def install_hint(cls) -> str:
        """Pip install command to enable this backend."""
        return ""

    @abstractmethod
    def relu(self, x: NDArray[np.float32]) -> NDArray[np.float32]: ...

    @abstractmethod
    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]: ...

    @abstractmethod
    def run_mlp_all_layers(self, mlp: MLP, inputs: NDArray[np.float32]) -> List[NDArray[np.float32]]: ...

    @abstractmethod
    def output_stats(self, mlp: MLP, n_samples: int) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]: ...
```

All backends accept and return NumPy arrays. Internal conversions (to torch tensors, jax arrays, etc.) are hidden. `is_available()` is a classmethod so the registry can check without instantiation.

### Backend Registry

File: `src/network_estimation/simulation_backends.py`

```python
BACKENDS: Dict[str, Type[SimulationBackend]] = {
    "numpy": NumPyBackend,
    "pytorch": PyTorchBackend,
    "numba": NumbaBackend,
    "scipy": SciPyBackend,
    "jax": JAXBackend,
}
```

Functions:
- `get_available_backends()` — Returns only backends whose dependencies are installed.
- `get_backend(name=None)` — Returns a backend instance. Reads `NESTIM_BACKEND` env var if name not provided, defaults to `"numpy"`.

If a requested backend is unavailable, raises `RuntimeError` with the install hint.

### Backend Implementations

Each backend lives in its own file implementing `SimulationBackend`:

| File | Backend | Key Technique | Dependency |
|------|---------|---------------|------------|
| `simulation_numpy.py` | NumPyBackend | Reference implementation, `np.maximum` + `@` operator | numpy (always available) |
| `simulation_pytorch.py` | PyTorchBackend | Weight tensor caching with weakref, `torch.relu`, thread control (cap 4) | torch>=2.0 |
| `simulation_numba.py` | NumbaBackend | `@njit` JIT-compiled matmul+ReLU loop | numba>=0.58 |
| `simulation_scipy.py` | SciPyBackend | `scipy.linalg.blas.sgemm` direct BLAS calls | scipy (already a runtime dep) |
| `simulation_jax.py` | JAXBackend | `@jax.jit` fused operations, `jax.random` for RNG | jax[cpu]>=0.4 |

All backends implement chunked `output_stats` with online accumulation to cap memory at O(chunk_size * width). Chunk sizing uses the `_pick_chunk_size(width)` heuristic targeting 2-8 MB working set.

The existing `simulation.py` stays untouched as the readable reference oracle. `simulation_fast.py` is removed — its logic moves into `simulation_pytorch.py`.

### Integration — scoring.py, dataset.py, __init__.py

`scoring.py` and `dataset.py` change from:
```python
from .simulation_fast import output_stats, run_mlp
```

To:
```python
from .simulation_backends import get_backend
```

The backend is instantiated at call sites:
```python
backend = get_backend()
backend.run_mlp(mlp, inputs)
backend.output_stats(mlp, n_samples)
```

`__init__.py` continues to export from `simulation.py` (the reference NumPy implementation) for the participant-facing API. Participants always get the readable reference. Backend selection only affects internal scoring/dataset computation.

### Profiler CLI

New subcommand: `nestim profile-simulation`

```
nestim profile-simulation [--preset {quick,standard,exhaustive}] [--backends numpy,pytorch,...] [--output results.json]
```

**Flags:**
- `--preset` (default: `standard`) — Controls the parameter sweep grid.
- `--backends` — Comma-separated list to restrict which backends to profile. Default: all available.
- `--output` — Path to save JSON results. If omitted, only prints terminal table.

**Presets:**

| Preset | Widths | Depths | n_samples | Est. time |
|--------|--------|--------|-----------|-----------|
| `quick` | 256 | 4 | 10K, 100K | ~30s |
| `standard` | 64, 256 | 2, 4 | 10K, 100K, 1M | ~2min |
| `exhaustive` | 64, 128, 256 | 2, 4, 8 | 10K, 100K, 500K, 1M, 16.7M | ~10min |

**Flow:**

1. Discover available backends (or filter by `--backends`).
2. Print skipped backends with install hints.
3. **Pre-flight correctness check** — For each available backend, run `run_mlp` and `output_stats` on a small fixed MLP (width=8, depth=4, 1000 samples) and compare against NumPy reference. Exact match for `run_mlp` (rtol=1e-5), statistical match for `output_stats` (atol=0.05). Failed backends marked "FAIL" and excluded from timing.
4. **Timing sweep** — For each (backend x operation x parameter combo), run 3 iterations, report median wall time.
5. **Terminal output** — Rich table with columns: Backend, Operation, Width, Depth, N_Samples, Median Time, Speedup vs NumPy, Status.
6. **JSON output** (if `--output`) — Full results including hardware info (platform, cpu, cpu_count), backend versions, per-run timings, and correctness status.

### Dependencies & Packaging

`pyproject.toml` changes — the `fast` dependency group is replaced with per-backend groups:

```toml
[dependency-groups]
dev = [
    "pytest>=8.2.0",
    "pyright>=1.1.0",
    "ruff>=0.6.0",
]
pytorch = ["torch>=2.0"]
numba = ["numba>=0.58"]
jax = ["jax[cpu]>=0.4"]
all-backends = ["torch>=2.0", "numba>=0.58", "jax[cpu]>=0.4"]
```

SciPy is already a runtime dependency. NumPy is always available.

Install: `uv sync --group all-backends` or `uv sync --group pytorch` etc.

### Testing Strategy

**Correctness tests** (`tests/test_simulation_backends.py`):
- Parametrized across all backends via `@pytest.mark.parametrize`.
- Each backend tested against the NumPy reference (`simulation.py`):
  - `run_mlp` exact match (rtol=1e-5, atol=1e-6).
  - `run_mlp_all_layers` exact match per layer.
  - `output_stats` statistical equivalence (means atol=0.05, variance within 10%).
- Backends that aren't installed skip cleanly via `pytest.importorskip`.
- Small fixed MLPs (width=8, depth=4) for exact tests, larger (width=64, depth=4, 50K samples) for statistical tests.

**Backend contract tests:**
- `is_available()` matches whether instantiation succeeds.
- `install_hint()` returns a non-empty string for optional backends.
- All methods return correct dtypes (`np.float32`) and shapes.

**Profiler tests** (`tests/test_profiler.py`):
- `profile-simulation` CLI runs without error on `--preset quick`.
- JSON output structure validated when `--output` is provided.
- Unavailable backends show "SKIPPED" status.

**Removed:** `test_simulation_fast.py` — its coverage moves into parametrized backend tests.

## File Changes Summary

**New files:**
- `src/network_estimation/simulation_backend.py` — ABC
- `src/network_estimation/simulation_backends.py` — Registry
- `src/network_estimation/simulation_numpy.py` — NumPy backend
- `src/network_estimation/simulation_pytorch.py` — PyTorch backend
- `src/network_estimation/simulation_numba.py` — Numba backend
- `src/network_estimation/simulation_scipy.py` — SciPy BLAS backend
- `src/network_estimation/simulation_jax.py` — JAX backend
- `tests/test_simulation_backends.py` — Parametrized correctness tests
- `tests/test_profiler.py` — Profiler CLI tests

**Modified files:**
- `src/network_estimation/scoring.py` — Import from registry
- `src/network_estimation/dataset.py` — Import from registry
- `src/network_estimation/cli.py` — Add `profile-simulation` subcommand
- `pyproject.toml` — Replace `fast` group with per-backend groups

**Removed files:**
- `src/network_estimation/simulation_fast.py` — Logic moves to `simulation_pytorch.py`
- `tests/test_simulation_fast.py` — Coverage moves to `test_simulation_backends.py`
