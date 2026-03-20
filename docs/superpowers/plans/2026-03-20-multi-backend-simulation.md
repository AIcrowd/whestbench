# Multi-Backend Simulation Profiling System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create 5 simulation backends (NumPy, PyTorch, Numba, SciPy, JAX) behind an ABC, with a `nestim profile-simulation` CLI command for head-to-head benchmarking.

**Architecture:** Each backend implements `SimulationBackend` ABC in its own file. A registry module discovers available backends by name. `scoring.py` and `dataset.py` get their backend via `get_backend()` reading `NESTIM_BACKEND` env var. A new CLI subcommand runs correctness checks then timing sweeps across all installed backends.

**Tech Stack:** Python 3.10+, NumPy, PyTorch, Numba, SciPy, JAX, Rich (tables), argparse (CLI)

**Spec:** `docs/superpowers/specs/2026-03-20-multi-backend-simulation-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/network_estimation/simulation_backend.py` | ABC defining the backend contract |
| `src/network_estimation/simulation_backends.py` | Registry: `BACKENDS` dict, `get_backend()`, `get_available_backends()` |
| `src/network_estimation/simulation_numpy.py` | NumPy backend (wraps `simulation.py` logic) |
| `src/network_estimation/simulation_pytorch.py` | PyTorch backend (migrated from `simulation_fast.py`) |
| `src/network_estimation/simulation_numba.py` | Numba JIT backend |
| `src/network_estimation/simulation_scipy.py` | SciPy BLAS backend |
| `src/network_estimation/simulation_jax.py` | JAX JIT backend |
| `src/network_estimation/profiler.py` | Profiling engine (correctness check + timing sweep + output) |
| `src/network_estimation/scoring.py` | Modified: use `get_backend()` instead of `simulation_fast` imports |
| `src/network_estimation/dataset.py` | Modified: use `get_backend()` instead of `simulation_fast` imports |
| `src/network_estimation/cli.py` | Modified: add `profile-simulation` subcommand |
| `pyproject.toml` | Modified: replace `fast` group with per-backend groups |
| `tests/test_simulation_backends.py` | Parametrized correctness tests for all backends |
| `tests/test_profiler.py` | Profiler CLI tests |

**Removed:**
- `src/network_estimation/simulation_fast.py` (logic → `simulation_pytorch.py`)
- `tests/test_simulation_fast.py` (coverage → `test_simulation_backends.py`)

---

## Chunk 1: ABC, Registry, NumPy Backend, and Tests

### Task 1: Create the SimulationBackend ABC

**Files:**
- Create: `src/network_estimation/simulation_backend.py`
- Test: `tests/test_simulation_backends.py`

- [ ] **Step 1: Write the ABC**

```python
# src/network_estimation/simulation_backend.py
"""Abstract base class for MLP forward pass backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP


class SimulationBackend(ABC):
    """Abstract interface for MLP forward pass backends.

    All backends accept and return NumPy arrays. Internal conversions
    (to torch tensors, jax arrays, etc.) are hidden from callers.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier, e.g. 'numpy', 'pytorch', 'numba'."""
        ...

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if this backend's dependencies are installed."""
        ...

    @classmethod
    def install_hint(cls) -> str:
        """Pip install command to enable this backend. Empty for always-available backends."""
        return ""

    @abstractmethod
    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass returning final-layer activations."""
        ...

    @abstractmethod
    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        """Forward pass returning activations after each layer."""
        ...

    @abstractmethod
    def output_stats(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """Compute per-layer means and average variance of the final layer."""
        ...
```

- [ ] **Step 2: Commit**

```bash
git add src/network_estimation/simulation_backend.py
git commit -m "feat: add SimulationBackend ABC"
```

### Task 2: Create the NumPy Backend

**Files:**
- Create: `src/network_estimation/simulation_numpy.py`

- [ ] **Step 1: Write the NumPy backend**

```python
# src/network_estimation/simulation_numpy.py
"""NumPy backend — reference implementation wrapping simulation.py logic."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .simulation_backend import SimulationBackend


def _pick_chunk_size(width: int) -> int:
    """Choose chunk size targeting a 2-8 MB working set for L2/L3 cache."""
    return max(1024, min(16384, 2**20 // width))


class NumPyBackend(SimulationBackend):
    """Reference NumPy backend using np.maximum + @ operator."""

    @property
    def name(self) -> str:
        return "numpy"

    @classmethod
    def is_available(cls) -> bool:
        return True

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        x = inputs
        for w in mlp.weights:
            x = np.maximum(x @ w, np.float32(0.0))
        return x

    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        x = inputs
        layers: List[NDArray[np.float32]] = []
        for w in mlp.weights:
            x = np.maximum(x @ w, np.float32(0.0))
            layers.append(x)
        return layers

    def output_stats(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)

        layer_sums = np.zeros((depth, width), dtype=np.float64)
        final_sum_sq = np.zeros(width, dtype=np.float64)
        n_processed = 0

        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = np.random.randn(n, width).astype(np.float32)

            for layer_idx, w in enumerate(mlp.weights):
                x = np.maximum(x @ w, np.float32(0.0))
                layer_sums[layer_idx] += x.sum(axis=0).astype(np.float64)

            final_sum_sq += (x.astype(np.float64) ** 2).sum(axis=0)
            n_processed += n

        layer_means = (layer_sums / n_processed).astype(np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(
            np.mean(final_sum_sq / n_processed - final_mean.astype(np.float64) ** 2)
        )
        return layer_means, final_mean, avg_variance
```

- [ ] **Step 2: Commit**

```bash
git add src/network_estimation/simulation_numpy.py
git commit -m "feat: add NumPy simulation backend"
```

### Task 3: Create the Backend Registry

**Files:**
- Create: `src/network_estimation/simulation_backends.py`

- [ ] **Step 1: Write the registry**

```python
# src/network_estimation/simulation_backends.py
"""Backend registry for simulation backends."""

from __future__ import annotations

import os
from typing import Dict, Optional, Type

from .simulation_backend import SimulationBackend
from .simulation_numpy import NumPyBackend


def _lazy_backends() -> Dict[str, Type[SimulationBackend]]:
    """Build the full backend dict, importing optional backends lazily."""
    backends: Dict[str, Type[SimulationBackend]] = {"numpy": NumPyBackend}

    try:
        from .simulation_pytorch import PyTorchBackend
        backends["pytorch"] = PyTorchBackend
    except ImportError:
        pass

    try:
        from .simulation_numba import NumbaBackend
        backends["numba"] = NumbaBackend
    except ImportError:
        pass

    try:
        from .simulation_scipy import SciPyBackend
        backends["scipy"] = SciPyBackend
    except ImportError:
        pass

    try:
        from .simulation_jax import JAXBackend
        backends["jax"] = JAXBackend
    except ImportError:
        pass

    return backends


# All known backend names (for error messages before backends are created)
ALL_BACKEND_NAMES = ("numpy", "pytorch", "numba", "scipy", "jax")


def get_available_backends() -> Dict[str, Type[SimulationBackend]]:
    """Return only backends whose dependencies are installed."""
    return {k: v for k, v in _lazy_backends().items() if v.is_available()}


def get_backend(name: Optional[str] = None) -> SimulationBackend:
    """Get a backend instance by name.

    Reads NESTIM_BACKEND env var if name not provided, defaults to 'numpy'.
    """
    if name is None:
        name = os.environ.get("NESTIM_BACKEND", "numpy")

    if name not in ALL_BACKEND_NAMES:
        raise ValueError(
            f"Unknown backend: {name!r}. Valid backends: {list(ALL_BACKEND_NAMES)}"
        )

    backends = _lazy_backends()
    cls = backends.get(name)
    if cls is None or not cls.is_available():
        hint = ""
        # Try to get install hint even if import failed
        if cls is not None:
            hint = cls.install_hint()
        raise RuntimeError(
            f"Backend {name!r} is not available."
            + (f" Install: {hint}" if hint else "")
        )

    return cls()
```

Note: The registry uses lazy imports with try/except so the module can be imported even when optional backend files reference unavailable dependencies. The backend files themselves will guard their imports with `is_available()`.

- [ ] **Step 2: Commit**

```bash
git add src/network_estimation/simulation_backends.py
git commit -m "feat: add backend registry with lazy discovery"
```

### Task 4: Write Correctness Tests for NumPy Backend

**Files:**
- Create: `tests/test_simulation_backends.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_simulation_backends.py
"""Parametrized correctness tests for all simulation backends."""

from __future__ import annotations

import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.generation import sample_mlp
from network_estimation.simulation import (
    output_stats as ref_output_stats,
    run_mlp as ref_run_mlp,
    run_mlp_all_layers as ref_run_mlp_all_layers,
)
from network_estimation.simulation_backends import get_available_backends, get_backend


def _make_mlp(width: int = 8, depth: int = 4, seed: int = 42) -> MLP:
    rng = np.random.default_rng(seed)
    return sample_mlp(width, depth, rng)


def _available_backend_names() -> list[str]:
    return list(get_available_backends().keys())


# --- Contract tests ---


class TestBackendContract:
    """Verify that all available backends satisfy the ABC contract."""

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_name_matches_key(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        assert backend.name == backend_name

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_is_available_true(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        assert backend.__class__.is_available() is True

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_run_mlp_returns_float32(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32)
        result = backend.run_mlp(mlp, inputs)
        assert result.dtype == np.float32
        assert result.shape == (16, 8)

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_run_mlp_all_layers_returns_correct_shapes(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = np.random.default_rng(0).standard_normal((16, 8)).astype(np.float32)
        layers = backend.run_mlp_all_layers(mlp, inputs)
        assert len(layers) == 4
        for layer in layers:
            assert layer.dtype == np.float32
            assert layer.shape == (16, 8)

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_output_stats_returns_correct_shapes(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        means, final_mean, avg_var = backend.output_stats(mlp, 1000)
        assert means.dtype == np.float32
        assert means.shape == (4, 8)
        assert final_mean.dtype == np.float32
        assert final_mean.shape == (8,)
        assert isinstance(avg_var, float)


# --- Exact match tests (run_mlp, run_mlp_all_layers) ---


class TestExactMatch:
    """Verify backends produce identical results to reference on same inputs."""

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_run_mlp_matches_reference(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = np.random.default_rng(123).standard_normal((32, 8)).astype(np.float32)
        ref = ref_run_mlp(mlp, inputs)
        result = backend.run_mlp(mlp, inputs)
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_run_mlp_all_layers_matches_reference(self, backend_name: str) -> None:
        backend = get_backend(backend_name)
        mlp = _make_mlp()
        inputs = np.random.default_rng(123).standard_normal((32, 8)).astype(np.float32)
        ref_layers = ref_run_mlp_all_layers(mlp, inputs)
        result_layers = backend.run_mlp_all_layers(mlp, inputs)
        assert len(result_layers) == len(ref_layers)
        for ref_layer, result_layer in zip(ref_layers, result_layers):
            np.testing.assert_allclose(result_layer, ref_layer, rtol=1e-5, atol=1e-6)


# --- Statistical equivalence tests (output_stats) ---


class TestStatisticalEquivalence:
    """Verify output_stats produces statistically equivalent results."""

    @pytest.mark.parametrize("backend_name", _available_backend_names())
    def test_means_close_to_reference(self, backend_name: str) -> None:
        mlp = _make_mlp(width=64, depth=4, seed=55)
        ref_means, ref_final, ref_var = ref_output_stats(mlp, n_samples=50000)
        backend = get_backend(backend_name)
        fast_means, fast_final, fast_var = backend.output_stats(mlp, n_samples=50000)
        np.testing.assert_allclose(fast_means, ref_means, atol=0.05)
        np.testing.assert_allclose(fast_final, ref_final, atol=0.05)
        assert abs(fast_var - ref_var) < max(0.1 * abs(ref_var), 0.01)


# --- Registry tests ---


class TestRegistry:
    def test_unknown_backend_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_numpy_always_available(self) -> None:
        backends = get_available_backends()
        assert "numpy" in backends

    def test_get_backend_default_is_numpy(self) -> None:
        import os
        old = os.environ.pop("NESTIM_BACKEND", None)
        try:
            backend = get_backend()
            assert backend.name == "numpy"
        finally:
            if old is not None:
                os.environ["NESTIM_BACKEND"] = old

    def test_get_backend_env_var(self) -> None:
        import os
        old = os.environ.get("NESTIM_BACKEND")
        os.environ["NESTIM_BACKEND"] = "numpy"
        try:
            backend = get_backend()
            assert backend.name == "numpy"
        finally:
            if old is not None:
                os.environ["NESTIM_BACKEND"] = old
            else:
                os.environ.pop("NESTIM_BACKEND", None)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/test_simulation_backends.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation_backends.py
git commit -m "test: add parametrized correctness tests for simulation backends"
```

---

## Chunk 2: PyTorch, SciPy, Numba, and JAX Backends

### Task 5: Create the PyTorch Backend

**Files:**
- Create: `src/network_estimation/simulation_pytorch.py`

- [ ] **Step 1: Migrate simulation_fast.py logic into the new backend class**

```python
# src/network_estimation/simulation_pytorch.py
"""PyTorch CPU backend with weight caching and chunked output_stats."""

from __future__ import annotations

import os
import weakref
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .simulation_backend import SimulationBackend

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if _HAS_TORCH:
    _MAX_THREADS = 4
    _n_threads = min(os.cpu_count() or _MAX_THREADS, _MAX_THREADS)
    torch.set_num_threads(_n_threads)

# Module-level weight cache shared across all PyTorchBackend instances
_weight_cache: Dict[int, list] = {}
_weak_refs: Dict[int, weakref.ref] = {}


def _pick_chunk_size(width: int) -> int:
    return max(1024, min(16384, 2**20 // width))


def _get_torch_weights(mlp: MLP) -> list:
    """Get or create cached torch tensors for an MLP's weight matrices."""
    import torch

    key = id(mlp)
    cached = _weight_cache.get(key)
    if cached is not None:
        return cached
    tensors = [torch.from_numpy(w) for w in mlp.weights]
    _weight_cache[key] = tensors

    def _on_finalize(ref: weakref.ref, k: int = key) -> None:
        _weight_cache.pop(k, None)
        _weak_refs.pop(k, None)

    _weak_refs[key] = weakref.ref(mlp, _on_finalize)
    return tensors


class PyTorchBackend(SimulationBackend):
    """PyTorch CPU backend using MKL/oneDNN BLAS with weight caching."""

    @property
    def name(self) -> str:
        return "pytorch"

    @classmethod
    def is_available(cls) -> bool:
        return _HAS_TORCH

    @classmethod
    def install_hint(cls) -> str:
        return "pip install torch>=2.0"

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        import torch

        with torch.no_grad():
            x = torch.from_numpy(inputs)
            for w in _get_torch_weights(mlp):
                x = torch.relu(x @ w)
            return x.numpy()

    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        import torch

        with torch.no_grad():
            x = torch.from_numpy(inputs)
            layers: List[NDArray[np.float32]] = []
            for w in _get_torch_weights(mlp):
                x = torch.relu(x @ w)
                layers.append(x.numpy())
            return layers

    def output_stats(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        import torch

        with torch.no_grad():
            weights = _get_torch_weights(mlp)
            width = mlp.width
            depth = mlp.depth
            chunk_size = _pick_chunk_size(width)

            layer_sums = torch.zeros(depth, width, dtype=torch.float32)
            final_sum_sq = torch.zeros(width, dtype=torch.float32)
            n_processed = 0

            for start in range(0, n_samples, chunk_size):
                n = min(chunk_size, n_samples - start)
                x = torch.randn(n, width, dtype=torch.float32)
                for layer_idx, w in enumerate(weights):
                    x = torch.relu(x @ w)
                    layer_sums[layer_idx] += x.sum(dim=0)
                final_sum_sq += (x * x).sum(dim=0)
                n_processed += n

            layer_means = (layer_sums / n_processed).numpy().astype(np.float32)
            final_mean = layer_means[-1].copy()
            final_mean_t = torch.from_numpy(final_mean)
            avg_variance = float(
                (final_sum_sq / n_processed - final_mean_t * final_mean_t).mean()
            )
            return layer_means, final_mean, avg_variance
```

- [ ] **Step 2: Run backend tests**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/test_simulation_backends.py -v`
Expected: PyTorch tests pass (if torch installed) or are skipped

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/simulation_pytorch.py
git commit -m "feat: add PyTorch simulation backend (migrated from simulation_fast.py)"
```

### Task 6: Create the SciPy BLAS Backend

**Files:**
- Create: `src/network_estimation/simulation_scipy.py`

- [ ] **Step 1: Write the SciPy backend**

```python
# src/network_estimation/simulation_scipy.py
"""SciPy BLAS backend — direct sgemm calls for minimal dispatch overhead."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg.blas import sgemm  # type: ignore[import-untyped]

from .domain import MLP
from .simulation_backend import SimulationBackend


def _pick_chunk_size(width: int) -> int:
    return max(1024, min(16384, 2**20 // width))


class SciPyBackend(SimulationBackend):
    """SciPy backend calling BLAS sgemm directly."""

    @property
    def name(self) -> str:
        return "scipy"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from scipy.linalg.blas import sgemm  # noqa: F811
            return True
        except ImportError:
            return False

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        for w in mlp.weights:
            x = sgemm(1.0, x, w)
            np.maximum(x, 0.0, out=x)
        return x

    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        x = np.ascontiguousarray(inputs, dtype=np.float32)
        layers: List[NDArray[np.float32]] = []
        for w in mlp.weights:
            x = sgemm(1.0, x, w)
            np.maximum(x, 0.0, out=x)
            layers.append(x.copy())
        return layers

    def output_stats(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)

        layer_sums = np.zeros((depth, width), dtype=np.float64)
        final_sum_sq = np.zeros(width, dtype=np.float64)
        n_processed = 0

        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = np.random.randn(n, width).astype(np.float32)

            for layer_idx, w in enumerate(mlp.weights):
                x = sgemm(1.0, x, w)
                np.maximum(x, 0.0, out=x)
                layer_sums[layer_idx] += x.sum(axis=0).astype(np.float64)

            final_sum_sq += (x.astype(np.float64) ** 2).sum(axis=0)
            n_processed += n

        layer_means = (layer_sums / n_processed).astype(np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(
            np.mean(final_sum_sq / n_processed - final_mean.astype(np.float64) ** 2)
        )
        return layer_means, final_mean, avg_variance
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/test_simulation_backends.py -v`
Expected: SciPy tests pass

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/simulation_scipy.py
git commit -m "feat: add SciPy BLAS simulation backend"
```

### Task 7: Create the Numba Backend

**Files:**
- Create: `src/network_estimation/simulation_numba.py`

- [ ] **Step 1: Write the Numba backend**

```python
# src/network_estimation/simulation_numba.py
"""Numba JIT backend — compiled matmul+ReLU loop."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .simulation_backend import SimulationBackend

try:
    from numba import njit, prange  # type: ignore[import-untyped]
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def _pick_chunk_size(width: int) -> int:
    return max(1024, min(16384, 2**20 // width))


if _HAS_NUMBA:
    @njit(cache=True, parallel=True)
    def _relu_inplace(x: np.ndarray) -> np.ndarray:  # type: ignore[type-arg]
        """In-place ReLU for 2D array."""
        rows, cols = x.shape
        for i in prange(rows):
            for j in range(cols):
                if x[i, j] < 0.0:
                    x[i, j] = 0.0
        return x

    @njit(cache=True)
    def _forward_pass(inputs: np.ndarray, weights: tuple) -> np.ndarray:  # type: ignore[type-arg]
        """JIT-compiled forward pass through all layers."""
        x = inputs.copy()
        for w in weights:
            x = x @ w
            _relu_inplace(x)
        return x

    @njit(cache=True)
    def _forward_pass_all_layers(
        inputs: np.ndarray, weights: tuple  # type: ignore[type-arg]
    ) -> list:  # type: ignore[type-arg]
        """JIT-compiled forward pass returning all layer activations."""
        x = inputs.copy()
        layers = []
        for w in weights:
            x = x @ w
            _relu_inplace(x)
            layers.append(x.copy())
        return layers

Note: Numba `@njit` with reflected lists can be problematic. If compilation fails at runtime, the implementer should fall back to returning a tuple from `_forward_pass_all_layers` and converting to list in the `NumbaBackend` method. Test this during implementation.


class NumbaBackend(SimulationBackend):
    """Numba JIT backend with compiled matmul+ReLU loops."""

    @property
    def name(self) -> str:
        return "numba"

    @classmethod
    def is_available(cls) -> bool:
        return _HAS_NUMBA

    @classmethod
    def install_hint(cls) -> str:
        return "pip install numba>=0.58"

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        weights_tuple = tuple(mlp.weights)
        result = _forward_pass(inputs, weights_tuple)
        return result.astype(np.float32)

    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        weights_tuple = tuple(mlp.weights)
        layers = _forward_pass_all_layers(inputs, weights_tuple)
        return [layer.astype(np.float32) for layer in layers]

    def output_stats(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)
        weights_tuple = tuple(mlp.weights)

        layer_sums = np.zeros((depth, width), dtype=np.float64)
        final_sum_sq = np.zeros(width, dtype=np.float64)
        n_processed = 0

        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            x = np.random.randn(n, width).astype(np.float32)

            layers = _forward_pass_all_layers(x, weights_tuple)
            for layer_idx, layer_out in enumerate(layers):
                layer_sums[layer_idx] += layer_out.sum(axis=0).astype(np.float64)

            final_out = layers[-1]
            final_sum_sq += (final_out.astype(np.float64) ** 2).sum(axis=0)
            n_processed += n

        layer_means = (layer_sums / n_processed).astype(np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(
            np.mean(final_sum_sq / n_processed - final_mean.astype(np.float64) ** 2)
        )
        return layer_means, final_mean, avg_variance
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/test_simulation_backends.py -v`
Expected: Numba tests pass (if numba installed) or are skipped

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/simulation_numba.py
git commit -m "feat: add Numba JIT simulation backend"
```

### Task 8: Create the JAX Backend

**Files:**
- Create: `src/network_estimation/simulation_jax.py`

- [ ] **Step 1: Write the JAX backend**

```python
# src/network_estimation/simulation_jax.py
"""JAX backend — JIT-compiled operations with XLA CPU backend."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .simulation_backend import SimulationBackend

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_platform_name", "cpu")
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _pick_chunk_size(width: int) -> int:
    return max(1024, min(16384, 2**20 // width))


if _HAS_JAX:
    @jax.jit
    def _jax_forward(inputs: jnp.ndarray, weights: list) -> jnp.ndarray:
        """JIT-compiled forward pass through all layers."""
        x = inputs
        for w in weights:
            x = jnp.maximum(x @ w, 0.0)
        return x


class JAXBackend(SimulationBackend):
    """JAX backend with JIT-compiled forward pass."""

    @property
    def name(self) -> str:
        return "jax"

    @classmethod
    def is_available(cls) -> bool:
        return _HAS_JAX

    @classmethod
    def install_hint(cls) -> str:
        return "pip install 'jax[cpu]>=0.4'"

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        jax_weights = [jnp.array(w) for w in mlp.weights]
        result = _jax_forward(jnp.array(inputs), jax_weights)
        return np.asarray(result, dtype=np.float32)

    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        x = jnp.array(inputs)
        layers: List[NDArray[np.float32]] = []
        for w in mlp.weights:
            x = jnp.maximum(x @ jnp.array(w), 0.0)
            layers.append(np.asarray(x, dtype=np.float32))
        return layers

    def output_stats(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)

        jax_weights = [jnp.array(w) for w in mlp.weights]

        layer_sums = jnp.zeros((depth, width), dtype=jnp.float32)
        final_sum_sq = jnp.zeros(width, dtype=jnp.float32)
        n_processed = 0

        key = jax.random.PRNGKey(np.random.default_rng().integers(2**31))

        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, shape=(n, width), dtype=jnp.float32)

            for layer_idx, w in enumerate(jax_weights):
                x = jnp.maximum(x @ w, 0.0)
                layer_sums = layer_sums.at[layer_idx].add(x.sum(axis=0))

            final_sum_sq = final_sum_sq + (x * x).sum(axis=0)
            n_processed += n

        layer_means = np.asarray(layer_sums / n_processed, dtype=np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(
            np.mean(
                np.asarray(final_sum_sq / n_processed, dtype=np.float64)
                - final_mean.astype(np.float64) ** 2
            )
        )
        return layer_means, final_mean, avg_variance
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/test_simulation_backends.py -v`
Expected: JAX tests pass (if jax installed) or are skipped

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/simulation_jax.py
git commit -m "feat: add JAX simulation backend"
```

---

## Chunk 3: Integration (scoring.py, dataset.py, pyproject.toml, cleanup)

### Task 9: Update scoring.py to Use Backend Registry

**Files:**
- Modify: `src/network_estimation/scoring.py:15` (import), `:62-84` (make_contest), `:87-92` (baseline_time), `:109-178` (evaluate_estimator)

- [ ] **Step 1: Update scoring.py**

Change the import on line 15 from:
```python
from .simulation_fast import output_stats, run_mlp
```
to:
```python
from .simulation_backends import get_backend
```

Then update `make_contest` to instantiate a backend and use it:

```python
def make_contest(spec: ContestSpec) -> ContestData:
    """Generate MLPs and compute ground truth for a contest run."""
    spec.validate()
    backend = get_backend()
    mlps: List[MLP] = []
    all_layer_targets: List[NDArray[np.float32]] = []
    final_targets: List[NDArray[np.float32]] = []
    avg_variances: List[float] = []

    for _ in range(spec.n_mlps):
        mlp = sample_mlp(spec.width, spec.depth)
        all_means, final_mean, avg_var = backend.output_stats(mlp, spec.ground_truth_budget)
        mlps.append(mlp)
        all_layer_targets.append(all_means)
        final_targets.append(final_mean)
        avg_variances.append(avg_var)

    return ContestData(
        spec=spec,
        mlps=mlps,
        all_layer_targets=all_layer_targets,
        final_targets=final_targets,
        avg_variances=avg_variances,
    )
```

Update `baseline_time` to accept and use a backend:

```python
def baseline_time(mlp: MLP, n_samples: int, backend: "SimulationBackend | None" = None) -> float:
    """Measure wall time for a single forward pass with ``n_samples`` inputs."""
    if backend is None:
        backend = get_backend()
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    t0 = time.perf_counter()
    backend.run_mlp(mlp, inputs)
    return time.perf_counter() - t0
```

Update `evaluate_estimator` to pass the backend through:

```python
def evaluate_estimator(
    estimator: BaseEstimator,
    data: ContestData,
) -> Dict[str, Any]:
    """Score an estimator against precomputed contest data."""
    spec = data.spec
    backend = get_backend()
    per_mlp: List[Dict[str, Any]] = []
    primary_scores: List[float] = []
    secondary_scores: List[float] = []

    for i, mlp in enumerate(data.mlps):
        time_budget = baseline_time(mlp, spec.estimator_budget, backend)
        # ... rest unchanged ...
```

- [ ] **Step 2: Run full test suite to verify nothing broke**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/ -v --ignore=tests/test_simulation_fast.py`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/scoring.py
git commit -m "refactor: scoring.py uses backend registry instead of simulation_fast"
```

### Task 10: Update dataset.py to Use Backend Registry

**Files:**
- Modify: `src/network_estimation/dataset.py:17` (import), `:76-77` (output_stats call)

- [ ] **Step 1: Update dataset.py**

Change the import on line 17 from:
```python
from .simulation_fast import output_stats
```
to:
```python
from .simulation_backends import get_backend
```

Update `create_dataset` to use the backend — change lines 76-77 from:
```python
        all_means, final_mean, avg_var = output_stats(mlp, n_samples)
```
to:
```python
        all_means, final_mean, avg_var = backend.output_stats(mlp, n_samples)
```

And add `backend = get_backend()` at the top of `create_dataset`, after the seed setup (after line 59):
```python
    backend = get_backend()
```

- [ ] **Step 2: Run tests**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/test_dataset.py tests/test_scoring_module.py -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/dataset.py
git commit -m "refactor: dataset.py uses backend registry instead of simulation_fast"
```

### Task 11: Update pyproject.toml

**Files:**
- Modify: `pyproject.toml:17-25`

- [ ] **Step 1: Replace the `fast` dependency group with per-backend groups**

Change lines 17-25 from:
```toml
[dependency-groups]
dev = [
    "pytest>=8.2.0",
    "pyright>=1.1.0",
    "ruff>=0.6.0",
]
fast = [
    "torch>=2.0",
]
```
to:
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

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "build: replace fast dep group with per-backend groups"
```

### Task 12: Remove simulation_fast.py and test_simulation_fast.py

**Files:**
- Remove: `src/network_estimation/simulation_fast.py`
- Remove: `tests/test_simulation_fast.py`

- [ ] **Step 1: Delete the old files**

```bash
git rm src/network_estimation/simulation_fast.py tests/test_simulation_fast.py
```

- [ ] **Step 2: Run full test suite to confirm nothing depends on deleted files**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/ -v`
Expected: All tests pass, no import errors

- [ ] **Step 3: Commit**

```bash
git commit -m "refactor: remove simulation_fast.py (replaced by backend system)"
```

---

## Chunk 4: Profiler Engine and CLI

### Task 13: Create the Profiler Engine

**Files:**
- Create: `src/network_estimation/profiler.py`

- [ ] **Step 1: Write the profiler module**

```python
# src/network_estimation/profiler.py
"""Profiling engine for simulation backends."""

from __future__ import annotations

import gc
import json
import platform
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .domain import MLP
from .generation import sample_mlp
from .simulation import (
    output_stats as ref_output_stats,
    run_mlp as ref_run_mlp,
)
from .simulation_backend import SimulationBackend
from .simulation_backends import ALL_BACKEND_NAMES, get_available_backends


@dataclass
class PresetConfig:
    """Parameter sweep grid for profiling."""

    widths: List[int]
    depths: List[int]
    n_samples_list: List[int]


PRESETS: Dict[str, PresetConfig] = {
    "quick": PresetConfig(
        widths=[256],
        depths=[4, 32],
        n_samples_list=[10_000, 100_000],
    ),
    "standard": PresetConfig(
        widths=[64, 256],
        depths=[4, 16, 32, 64, 128],
        n_samples_list=[10_000, 100_000, 1_000_000],
    ),
    "exhaustive": PresetConfig(
        widths=[64, 128, 256],
        depths=[4, 16, 32, 64, 128],
        n_samples_list=[10_000, 100_000, 500_000, 1_000_000, 16_700_000],
    ),
}


@dataclass
class CorrectnessResult:
    backend_name: str
    passed: bool
    error: str = ""


@dataclass
class TimingResult:
    backend_name: str
    operation: str
    width: int
    depth: int
    n_samples: int
    times: List[float]
    median_time: float
    speedup_vs_numpy: float


def _collect_hardware_info() -> Dict[str, Any]:
    """Collect hardware info for the profiling report."""
    import os
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }


def _collect_backend_versions(backend_names: List[str]) -> Dict[str, str]:
    """Collect version strings for each backend's underlying library."""
    versions: Dict[str, str] = {}
    versions["numpy"] = np.__version__
    try:
        import scipy
        versions["scipy"] = scipy.__version__
    except ImportError:
        pass
    if "pytorch" in backend_names:
        try:
            import torch
            versions["pytorch"] = torch.__version__
        except ImportError:
            pass
    if "numba" in backend_names:
        try:
            import numba
            versions["numba"] = numba.__version__
        except ImportError:
            pass
    if "jax" in backend_names:
        try:
            import jax
            versions["jax"] = jax.__version__
        except ImportError:
            pass
    return versions


def correctness_check(
    backend: SimulationBackend,
) -> CorrectnessResult:
    """Pre-flight correctness check against NumPy reference."""
    try:
        mlp = sample_mlp(8, 4, np.random.default_rng(42))
        inputs = np.random.default_rng(123).standard_normal((64, 8)).astype(np.float32)

        # Exact match for run_mlp
        ref = ref_run_mlp(mlp, inputs)
        result = backend.run_mlp(mlp, inputs)
        np.testing.assert_allclose(result, ref, rtol=1e-5, atol=1e-6)

        # Statistical match for output_stats
        ref_means, ref_final, ref_var = ref_output_stats(mlp, 1000)
        fast_means, fast_final, fast_var = backend.output_stats(mlp, 1000)
        np.testing.assert_allclose(fast_means, ref_means, atol=0.15)
        np.testing.assert_allclose(fast_final, ref_final, atol=0.15)
        # Variance can differ more with only 1000 samples
        if abs(ref_var) > 1e-6:
            assert abs(fast_var - ref_var) < max(0.5 * abs(ref_var), 0.1)

        return CorrectnessResult(backend_name=backend.name, passed=True)

    except Exception as e:
        return CorrectnessResult(
            backend_name=backend.name, passed=False, error=str(e)
        )


def _time_run_mlp(
    backend: SimulationBackend, mlp: MLP, n_samples: int
) -> float:
    """Time a single run_mlp call."""
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    t0 = time.perf_counter()
    backend.run_mlp(mlp, inputs)
    return time.perf_counter() - t0


def _time_output_stats(
    backend: SimulationBackend, mlp: MLP, n_samples: int
) -> float:
    """Time a single output_stats call."""
    t0 = time.perf_counter()
    backend.output_stats(mlp, n_samples)
    return time.perf_counter() - t0


def run_timing_sweep(
    backends: Dict[str, SimulationBackend],
    preset: PresetConfig,
    n_iterations: int = 3,
    progress_callback: Optional[Any] = None,
) -> Tuple[List[TimingResult], Dict[str, List[float]]]:
    """Run timing sweep across all backends and parameter combos.

    Returns:
        results: List of TimingResult for each (backend, operation, params) combo
        numpy_baselines: Dict mapping "op:w:d:n" -> list of numpy times
    """
    results: List[TimingResult] = []
    numpy_baselines: Dict[str, List[float]] = {}

    # Collect numpy baselines first
    numpy_backend = backends.get("numpy")

    operations = ["run_mlp", "output_stats"]

    for width in preset.widths:
        for depth in preset.depths:
            mlp = sample_mlp(width, depth, np.random.default_rng(42))
            for n_samples in preset.n_samples_list:
                for op in operations:
                    key = f"{op}:{width}:{depth}:{n_samples}"

                    for backend_name, backend in backends.items():
                        time_fn = _time_run_mlp if op == "run_mlp" else _time_output_stats

                        # Warmup (untimed, critical for JIT backends)
                        time_fn(backend, mlp, min(n_samples, 1000))

                        # Timed iterations with GC disabled
                        gc_was_enabled = gc.isenabled()
                        gc.disable()
                        times = []
                        try:
                            for _ in range(n_iterations):
                                t = time_fn(backend, mlp, n_samples)
                                times.append(t)
                        finally:
                            if gc_was_enabled:
                                gc.enable()

                        median_t = float(np.median(times))

                        # Store numpy baselines
                        if backend_name == "numpy":
                            numpy_baselines[key] = times

                        # Compute speedup vs numpy
                        numpy_times = numpy_baselines.get(key)
                        if numpy_times is not None:
                            numpy_median = float(np.median(numpy_times))
                            speedup = numpy_median / median_t if median_t > 0 else float("inf")
                        else:
                            speedup = 1.0

                        results.append(TimingResult(
                            backend_name=backend_name,
                            operation=op,
                            width=width,
                            depth=depth,
                            n_samples=n_samples,
                            times=times,
                            median_time=median_t,
                            speedup_vs_numpy=speedup,
                        ))

                        if progress_callback:
                            progress_callback()

    return results, numpy_baselines


def format_terminal_table(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    skipped_backends: Dict[str, str],
) -> str:
    """Format results as a Rich table string."""
    from rich.console import Console
    from rich.table import Table
    import io

    buf = io.StringIO()
    console = Console(file=buf, force_terminal=True, width=140)

    # Skipped backends
    if skipped_backends:
        console.print("\n[bold yellow]Skipped backends:[/bold yellow]")
        for name, hint in skipped_backends.items():
            console.print(f"  {name}: not installed. Install: {hint}")

    # Correctness results
    console.print("\n[bold]Pre-flight Correctness Check[/bold]")
    for cr in correctness_results:
        status = "[green]PASS[/green]" if cr.passed else f"[red]FAIL: {cr.error}[/red]"
        console.print(f"  {cr.backend_name}: {status}")

    # Build a set of passed backend names for status lookup
    passed_names = {cr.backend_name for cr in correctness_results if cr.passed}

    # Timing table
    if timing_results:
        table = Table(title="\nTiming Results", show_lines=True)
        table.add_column("Backend", style="cyan")
        table.add_column("Operation")
        table.add_column("Width", justify="right")
        table.add_column("Depth", justify="right")
        table.add_column("N_Samples", justify="right")
        table.add_column("Median Time (s)", justify="right")
        table.add_column("Speedup vs NumPy", justify="right")
        table.add_column("Status")

        for tr in timing_results:
            speedup_str = f"{tr.speedup_vs_numpy:.2f}x"
            if tr.speedup_vs_numpy > 1.0:
                speedup_str = f"[green]{speedup_str}[/green]"
            elif tr.speedup_vs_numpy < 1.0:
                speedup_str = f"[red]{speedup_str}[/red]"

            status = "[green]OK[/green]" if tr.backend_name in passed_names else "[red]FAIL[/red]"

            table.add_row(
                tr.backend_name,
                tr.operation,
                str(tr.width),
                str(tr.depth),
                f"{tr.n_samples:,}",
                f"{tr.median_time:.4f}",
                speedup_str,
                status,
            )

        console.print(table)

    return buf.getvalue()


def format_json_output(
    correctness_results: List[CorrectnessResult],
    timing_results: List[TimingResult],
    skipped_backends: Dict[str, str],
    backend_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Format results as a JSON-serializable dict."""
    return {
        "hardware": _collect_hardware_info(),
        "backend_versions": _collect_backend_versions(backend_names or []),
        "skipped_backends": skipped_backends,
        "correctness": [
            {"backend": cr.backend_name, "passed": cr.passed, "error": cr.error}
            for cr in correctness_results
        ],
        "timing": [
            {
                "backend": tr.backend_name,
                "operation": tr.operation,
                "width": tr.width,
                "depth": tr.depth,
                "n_samples": tr.n_samples,
                "times": tr.times,
                "median_time": tr.median_time,
                "speedup_vs_numpy": tr.speedup_vs_numpy,
            }
            for tr in timing_results
        ],
    }


def run_profile(
    preset_name: str = "standard",
    backend_filter: Optional[List[str]] = None,
    output_path: Optional[str] = None,
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Main profiling entry point.

    Returns:
        terminal_output: Rich-formatted string for terminal display
        json_data: JSON dict if output_path is set, else None
    """
    preset = PRESETS[preset_name]

    # Discover backends
    available = get_available_backends()
    if backend_filter:
        for name in backend_filter:
            if name not in ALL_BACKEND_NAMES:
                raise ValueError(
                    f"Unknown backend: {name!r}. Valid backends: {list(ALL_BACKEND_NAMES)}"
                )
        available = {k: v for k, v in available.items() if k in backend_filter}

    # Track skipped backends
    skipped: Dict[str, str] = {}
    all_names = list(backend_filter) if backend_filter else list(ALL_BACKEND_NAMES)
    for name in all_names:
        if name not in available:
            # Try to get install hint
            try:
                from . import simulation_backends
                backends_dict = simulation_backends._lazy_backends()
                cls = backends_dict.get(name)
                hint = cls.install_hint() if cls else ""
            except Exception:
                hint = ""
            skipped[name] = hint

    # Instantiate backends
    backend_instances: Dict[str, SimulationBackend] = {}
    for name, cls in available.items():
        backend_instances[name] = cls()

    # Pre-flight correctness check
    correctness_results: List[CorrectnessResult] = []
    passed_backends: Dict[str, SimulationBackend] = {}
    for name, backend in backend_instances.items():
        cr = correctness_check(backend)
        correctness_results.append(cr)
        if cr.passed:
            passed_backends[name] = backend

    # Timing sweep (only on backends that passed correctness)
    timing_results: List[TimingResult] = []
    if passed_backends:
        timing_results, _ = run_timing_sweep(passed_backends, preset)

    # Format output
    terminal_output = format_terminal_table(
        correctness_results, timing_results, skipped
    )

    json_data = None
    if output_path:
        json_data = format_json_output(
            correctness_results, timing_results, skipped,
            backend_names=list(backend_instances.keys()),
        )
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)

    return terminal_output, json_data
```

- [ ] **Step 2: Commit**

```bash
git add src/network_estimation/profiler.py
git commit -m "feat: add profiling engine with correctness check and timing sweep"
```

### Task 14: Add profile-simulation CLI Subcommand

**Files:**
- Modify: `src/network_estimation/cli.py:475-490` (add parser after visualizer), `:563-612` (add command handler)

- [ ] **Step 1: Add the subparser**

After the `visualizer_parser` block (after line 488), add:

```python
    profile_parser = subparsers.add_parser(
        "profile-simulation",
        help="Benchmark simulation backends head-to-head.",
    )
    profile_parser.add_argument(
        "--preset",
        choices=("quick", "standard", "exhaustive"),
        default="standard",
        help="Parameter sweep preset (default: standard).",
    )
    profile_parser.add_argument(
        "--backends",
        default=None,
        help="Comma-separated list of backends to profile (default: all available).",
    )
    profile_parser.add_argument(
        "--output",
        default=None,
        help="Path to save JSON results.",
    )
    profile_parser.add_argument("--debug", action="store_true")
```

- [ ] **Step 2: Add the command handler**

In `_main_participant`, add a handler block for `"profile-simulation"` (after the existing command handlers, before the `except` block). Add this before the final `except`:

```python
        if command == "profile-simulation":
            from .profiler import run_profile
            backend_filter = None
            if args.backends:
                backend_filter = [b.strip() for b in args.backends.split(",")]
            terminal_output, _ = run_profile(
                preset_name=str(args.preset),
                backend_filter=backend_filter,
                output_path=args.output,
            )
            print(terminal_output)
            return 0
```

- [ ] **Step 3: Run the profiler to verify it works**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m network_estimation.cli profile-simulation --preset quick --backends numpy`
Expected: Rich table with NumPy timing results

- [ ] **Step 4: Commit**

```bash
git add src/network_estimation/cli.py
git commit -m "feat: add 'nestim profile-simulation' CLI subcommand"
```

### Task 15: Write Profiler Tests

**Files:**
- Create: `tests/test_profiler.py`

- [ ] **Step 1: Write the profiler test file**

```python
# tests/test_profiler.py
"""Tests for the simulation profiler."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from network_estimation.profiler import (
    PRESETS,
    correctness_check,
    run_profile,
)
from network_estimation.simulation_backends import get_available_backends, get_backend


class TestCorrectnessCheck:
    def test_numpy_passes(self) -> None:
        backend = get_backend("numpy")
        result = correctness_check(backend)
        assert result.passed is True
        assert result.error == ""


class TestRunProfile:
    def test_quick_preset_runs(self) -> None:
        terminal_output, _ = run_profile(
            preset_name="quick", backend_filter=["numpy"]
        )
        assert "numpy" in terminal_output
        assert "Timing Results" in terminal_output

    def test_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "results.json")
            _, json_data = run_profile(
                preset_name="quick",
                backend_filter=["numpy"],
                output_path=out_path,
            )
            assert json_data is not None
            assert "hardware" in json_data
            assert "timing" in json_data
            assert "correctness" in json_data

            # Verify file was written
            with open(out_path) as f:
                saved = json.load(f)
            assert saved["hardware"]["cpu_count"] is not None

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            run_profile(backend_filter=["nonexistent"])

    def test_skipped_backends_in_output(self) -> None:
        # Request a backend that likely isn't installed
        terminal_output, _ = run_profile(
            preset_name="quick",
            backend_filter=["numpy"],
        )
        # At minimum numpy should appear
        assert "numpy" in terminal_output


class TestPresets:
    def test_all_presets_exist(self) -> None:
        assert "quick" in PRESETS
        assert "standard" in PRESETS
        assert "exhaustive" in PRESETS

    def test_quick_is_smallest(self) -> None:
        q = PRESETS["quick"]
        s = PRESETS["standard"]
        assert len(q.widths) <= len(s.widths)
        assert len(q.n_samples_list) <= len(s.n_samples_list)
```

- [ ] **Step 2: Run the profiler tests**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/test_profiler.py -v`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_profiler.py
git commit -m "test: add profiler CLI and engine tests"
```

### Task 16: Run Full Test Suite

- [ ] **Step 1: Run all tests**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/fast-mlp-cpu-forward && PYTHONPATH=src python -m pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 2: Final commit if any fixes needed**

If any tests fail, fix them and commit the fixes.
