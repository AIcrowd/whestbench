# Fast MLP Forward Pass Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optimized PyTorch CPU forward pass (`simulation_fast.py`) that is 2-5x faster and uses O(MB) memory instead of O(GB), while keeping the readable NumPy reference untouched.

**Architecture:** New module `simulation_fast.py` with identical API to `simulation.py`. Uses PyTorch CPU matmul+ReLU under `torch.no_grad()`, weight caching via `id()`-keyed dict with `weakref.ref` destructor callbacks for cleanup, and chunked streaming in `output_stats` to accumulate means/variances online. Falls back to NumPy reference when PyTorch is unavailable.

**Tech Stack:** PyTorch (CPU), NumPy, weakref, existing pytest suite

**Spec:** `docs/superpowers/specs/2026-03-20-fast-mlp-forward-design.md`

---

## Chunk 1: Core Module — `simulation_fast.py`

### Task 1: Add PyTorch optional dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `fast` dependency group to `pyproject.toml`**

In `pyproject.toml`, add a new dependency group after the existing `dev` group:

```toml
fast = [
    "torch>=2.0",
]
```

- [ ] **Step 2: Install the fast dependencies**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && pip install "torch>=2.0"`
Expected: PyTorch installs successfully

- [ ] **Step 3: Verify torch is importable**

Run: `python -c "import torch; print(torch.__version__)"`
Expected: Prints a version >= 2.0

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "build: add optional torch dependency group for fast simulation"
```

---

### Task 2: Create `simulation_fast.py` with fallback and helpers

**Files:**
- Create: `src/network_estimation/simulation_fast.py`

- [ ] **Step 1: Write the fallback skeleton and helper functions**

Create `src/network_estimation/simulation_fast.py`:

```python
"""Optimized MLP forward pass using PyTorch CPU backend.

Drop-in replacement for ``simulation.py`` with identical API. Falls back
to the reference NumPy implementation when PyTorch is not installed.

Key optimizations:
- PyTorch CPU BLAS (MKL/oneDNN) for matmul + fused ReLU
- Weight tensor caching keyed on id(mlp) with weakref cleanup
- Chunked streaming in output_stats (O(MB) memory instead of O(GB))
"""

from __future__ import annotations

import os
import weakref
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

if not _HAS_TORCH:
    from .simulation import output_stats, relu, run_mlp, run_mlp_all_layers
else:
    # Configure thread count at import time — cap at 4 to match target
    # 4-vCPU AWS compute-optimized instance.
    _MAX_THREADS = 4
    _n_threads = min(os.cpu_count() or _MAX_THREADS, _MAX_THREADS)
    torch.set_num_threads(_n_threads)

    # Weight cache: id(mlp) -> list of torch tensors.
    # MLP has frozen=True but contains a list field (unhashable), so we
    # cannot use WeakKeyDictionary. Instead we key on id() and register
    # a weak reference destructor to evict stale entries.
    _weight_cache: Dict[int, List[torch.Tensor]] = {}
    _weak_refs: Dict[int, weakref.ref[MLP]] = {}

    def _get_torch_weights(mlp: MLP) -> List[torch.Tensor]:
        """Get or create cached torch tensors for an MLP's weight matrices."""
        key = id(mlp)
        cached = _weight_cache.get(key)
        if cached is not None:
            return cached
        tensors = [torch.from_numpy(w) for w in mlp.weights]
        _weight_cache[key] = tensors

        def _on_finalize(ref: weakref.ref[MLP], k: int = key) -> None:
            _weight_cache.pop(k, None)
            _weak_refs.pop(k, None)

        _weak_refs[key] = weakref.ref(mlp, _on_finalize)
        return tensors

    def _pick_chunk_size(width: int) -> int:
        """Choose chunk size targeting a 2-8 MB working set for L2/L3 cache."""
        return max(1024, min(16384, 2**20 // width))

    def relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Element-wise ReLU activation."""
        return np.maximum(x, np.float32(0.0))

    @torch.no_grad()
    def run_mlp(mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        """Forward pass returning final-layer activations.

        Args:
            mlp: MLP to execute.
            inputs: Input matrix of shape ``(samples, mlp.width)``.

        Returns:
            Activations of shape ``(samples, mlp.width)`` after the last layer.
        """
        x = torch.from_numpy(inputs)
        for w in _get_torch_weights(mlp):
            x = torch.relu(x @ w)
        return x.numpy()

    @torch.no_grad()
    def run_mlp_all_layers(
        mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        """Forward pass returning activations after each layer.

        Args:
            mlp: MLP to execute.
            inputs: Input matrix of shape ``(samples, mlp.width)``.

        Returns:
            List of ``depth`` arrays, each shape ``(samples, mlp.width)``.
        """
        x = torch.from_numpy(inputs)
        layers: List[NDArray[np.float32]] = []
        for w in _get_torch_weights(mlp):
            x = torch.relu(x @ w)
            layers.append(x.numpy())
        return layers

    @torch.no_grad()
    def output_stats(
        mlp: MLP,
        n_samples: int,
        chunk_size: Optional[int] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """Compute per-layer means and average variance of the final layer.

        Uses chunked streaming to keep memory usage at O(chunk_size * width)
        instead of O(n_samples * width * depth).

        Args:
            mlp: MLP to evaluate.
            n_samples: Number of random Gaussian N(0,1) input vectors.
            chunk_size: Optional override for chunk size (for benchmarking).
                Defaults to an auto-tuned value based on width.

        Returns:
            all_layer_means: shape ``(depth, width)`` — mean activations per layer.
            final_mean: shape ``(width,)`` — mean activations at the final layer.
            avg_variance: scalar — average per-neuron variance at the final layer.
        """
        weights = _get_torch_weights(mlp)
        width = mlp.width
        depth = mlp.depth
        if chunk_size is None:
            chunk_size = _pick_chunk_size(width)

        # Online accumulators — only (depth, width) and (width,) sized
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

        # Compute final statistics
        layer_means = (layer_sums / n_processed).numpy().astype(np.float32)
        final_mean = layer_means[-1].copy()
        final_mean_t = torch.from_numpy(final_mean)
        avg_variance = float(
            (final_sum_sq / n_processed - final_mean_t * final_mean_t).mean()
        )
        return layer_means, final_mean, avg_variance
```

- [ ] **Step 2: Verify module imports without error**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && python -c "from network_estimation.simulation_fast import relu, run_mlp, run_mlp_all_layers, output_stats; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/simulation_fast.py
git commit -m "feat: add simulation_fast.py with PyTorch CPU backend and chunked output_stats"
```

---

## Chunk 2: Tests — `test_simulation_fast.py`

### Task 3: Write exact-match tests for `run_mlp` and `run_mlp_all_layers`

**Files:**
- Create: `tests/test_simulation_fast.py`

- [ ] **Step 1: Write the test file with exact-match tests**

Create `tests/test_simulation_fast.py`:

```python
"""Correctness tests for simulation_fast against the reference simulation."""

import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.simulation import (
    run_mlp as ref_run_mlp,
    run_mlp_all_layers as ref_run_mlp_all_layers,
    output_stats as ref_output_stats,
    relu as ref_relu,
)
from network_estimation.simulation_fast import (
    run_mlp as fast_run_mlp,
    run_mlp_all_layers as fast_run_mlp_all_layers,
    output_stats as fast_output_stats,
    relu as fast_relu,
)


def _make_mlp(width: int, depth: int, seed: int = 42) -> MLP:
    """Create a small deterministic MLP for testing."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(2.0 / width)
    weights = [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]
    return MLP(width=width, depth=depth, weights=weights)


class TestReluExactMatch:
    def test_matches_reference(self) -> None:
        x = np.array([-2.0, -1.0, 0.0, 0.5, 3.0], dtype=np.float32)
        np.testing.assert_array_equal(fast_relu(x), ref_relu(x))

    def test_all_negative(self) -> None:
        x = np.array([-5.0, -0.1, -100.0], dtype=np.float32)
        np.testing.assert_array_equal(fast_relu(x), ref_relu(x))

    def test_all_positive(self) -> None:
        x = np.array([0.1, 1.0, 99.0], dtype=np.float32)
        np.testing.assert_array_equal(fast_relu(x), ref_relu(x))


class TestRunMlpExactMatch:
    def test_small_mlp_matches_reference(self) -> None:
        mlp = _make_mlp(width=8, depth=4)
        rng = np.random.default_rng(123)
        inputs = rng.standard_normal((16, 8)).astype(np.float32)
        ref = ref_run_mlp(mlp, inputs)
        fast = fast_run_mlp(mlp, inputs)
        np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-6)

    def test_single_sample(self) -> None:
        mlp = _make_mlp(width=4, depth=2)
        inputs = np.ones((1, 4), dtype=np.float32)
        ref = ref_run_mlp(mlp, inputs)
        fast = fast_run_mlp(mlp, inputs)
        np.testing.assert_allclose(fast, ref, rtol=1e-5, atol=1e-6)

    def test_identity_weights(self) -> None:
        width = 4
        weights = [np.eye(width, dtype=np.float32)]
        mlp = MLP(width=width, depth=1, weights=weights)
        inputs = np.ones((2, width), dtype=np.float32)
        ref = ref_run_mlp(mlp, inputs)
        fast = fast_run_mlp(mlp, inputs)
        np.testing.assert_array_equal(fast, ref)


class TestRunMlpAllLayersExactMatch:
    def test_matches_reference(self) -> None:
        mlp = _make_mlp(width=8, depth=4)
        rng = np.random.default_rng(99)
        inputs = rng.standard_normal((16, 8)).astype(np.float32)
        ref_layers = ref_run_mlp_all_layers(mlp, inputs)
        fast_layers = fast_run_mlp_all_layers(mlp, inputs)
        assert len(fast_layers) == len(ref_layers)
        for i, (f, r) in enumerate(zip(fast_layers, ref_layers)):
            np.testing.assert_allclose(f, r, rtol=1e-5, atol=1e-6, err_msg=f"layer {i}")

    def test_last_layer_matches_run_mlp(self) -> None:
        mlp = _make_mlp(width=8, depth=3)
        rng = np.random.default_rng(7)
        inputs = rng.standard_normal((10, 8)).astype(np.float32)
        final = fast_run_mlp(mlp, inputs)
        all_layers = fast_run_mlp_all_layers(mlp, inputs)
        np.testing.assert_allclose(final, all_layers[-1], rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Run exact-match tests to verify they pass**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && python -m pytest tests/test_simulation_fast.py::TestReluExactMatch tests/test_simulation_fast.py::TestRunMlpExactMatch tests/test_simulation_fast.py::TestRunMlpAllLayersExactMatch -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation_fast.py
git commit -m "test: add exact-match tests for simulation_fast run_mlp and run_mlp_all_layers"
```

---

### Task 4: Write statistical equivalence and edge case tests for `output_stats`

**Files:**
- Modify: `tests/test_simulation_fast.py`

- [ ] **Step 1: Append statistical equivalence and edge case tests**

Add to the end of `tests/test_simulation_fast.py`:

```python
class TestOutputStatsStatisticalEquivalence:
    def test_means_close_to_reference(self) -> None:
        """Both paths should produce statistically similar means."""
        mlp = _make_mlp(width=8, depth=4, seed=55)
        # Different RNG streams (NumPy vs Torch) — rely on statistical convergence
        ref_means, ref_final, ref_var = ref_output_stats(mlp, n_samples=50000)
        fast_means, fast_final, fast_var = fast_output_stats(mlp, n_samples=50000)
        # Shapes must match exactly
        assert fast_means.shape == ref_means.shape
        assert fast_final.shape == ref_final.shape
        # Means should be close (different RNG streams, so use loose tolerance)
        np.testing.assert_allclose(fast_means, ref_means, atol=0.05)
        np.testing.assert_allclose(fast_final, ref_final, atol=0.05)
        # Variance should be in same ballpark
        assert abs(fast_var - ref_var) < max(0.1 * abs(ref_var), 0.01)

    def test_shapes_correct(self) -> None:
        mlp = _make_mlp(width=8, depth=2)
        means, final, var = fast_output_stats(mlp, n_samples=100)
        assert means.shape == (2, 8)
        assert final.shape == (8,)
        assert isinstance(var, float)
        assert var >= 0.0

    def test_final_mean_matches_last_layer(self) -> None:
        mlp = _make_mlp(width=8, depth=3)
        means, final, _ = fast_output_stats(mlp, n_samples=5000)
        np.testing.assert_allclose(final, means[-1], atol=1e-6)


class TestChunkBoundary:
    def test_non_divisible_n_samples(self) -> None:
        """n_samples not a multiple of chunk_size should not cause errors."""
        mlp = _make_mlp(width=8, depth=2)
        # 10007 is prime, won't divide evenly into any power-of-2 chunk
        means, final, var = fast_output_stats(mlp, n_samples=10007)
        assert means.shape == (2, 8)
        assert final.shape == (8,)
        assert isinstance(var, float)
        assert var >= 0.0

    def test_n_samples_smaller_than_chunk(self) -> None:
        """Should work even when n_samples < chunk_size."""
        mlp = _make_mlp(width=8, depth=2)
        means, final, var = fast_output_stats(mlp, n_samples=7)
        assert means.shape == (2, 8)
        assert final.shape == (8,)

    def test_n_samples_equals_one(self) -> None:
        mlp = _make_mlp(width=4, depth=2)
        means, final, var = fast_output_stats(mlp, n_samples=1)
        assert means.shape == (2, 4)
        assert final.shape == (4,)
```

- [ ] **Step 2: Run all simulation_fast tests**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && python -m pytest tests/test_simulation_fast.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation_fast.py
git commit -m "test: add statistical equivalence and chunk boundary tests for output_stats"
```

---

### Task 5: Write fallback test

**Files:**
- Modify: `tests/test_simulation_fast.py`

- [ ] **Step 1: Append fallback test**

Add to the end of `tests/test_simulation_fast.py`:

```python
class TestFallback:
    def test_fallback_exports_reference_functions(self) -> None:
        """When torch is unavailable, module should re-export simulation functions."""
        import importlib
        import sys
        from unittest.mock import patch

        import network_estimation.simulation as sim_mod
        import network_estimation.simulation_fast as fast_mod

        # Temporarily make torch unimportable and reload the module
        with patch.dict(sys.modules, {"torch": None}):
            importlib.reload(fast_mod)

        try:
            # After reload with torch blocked, the fallback path should have run.
            # The module's functions should be the exact same objects as simulation's.
            assert fast_mod.relu is sim_mod.relu
            assert fast_mod.run_mlp is sim_mod.run_mlp
            assert fast_mod.run_mlp_all_layers is sim_mod.run_mlp_all_layers
            assert fast_mod.output_stats is sim_mod.output_stats
            assert fast_mod._HAS_TORCH is False
        finally:
            # Restore the module to its torch-enabled state
            importlib.reload(fast_mod)
```

- [ ] **Step 2: Run the fallback test**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && python -m pytest tests/test_simulation_fast.py::TestFallback -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation_fast.py
git commit -m "test: add fallback test for simulation_fast without torch"
```

---

## Chunk 3: Integration — Wire Up Callers

### Task 6: Switch `scoring.py` to import from `simulation_fast`

**Files:**
- Modify: `src/network_estimation/scoring.py:15`

- [ ] **Step 1: Change the import in `scoring.py`**

In `src/network_estimation/scoring.py`, change line 15 from:

```python
from .simulation import output_stats, run_mlp
```

to:

```python
from .simulation_fast import output_stats, run_mlp
```

- [ ] **Step 2: Run existing scoring tests to verify nothing breaks**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && python -m pytest tests/test_scoring_module.py tests/test_evaluate.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/scoring.py
git commit -m "feat: switch scoring.py to use simulation_fast for optimized forward pass"
```

---

### Task 7: Switch `dataset.py` to import from `simulation_fast`

**Files:**
- Modify: `src/network_estimation/dataset.py:17`

- [ ] **Step 1: Change the import in `dataset.py`**

In `src/network_estimation/dataset.py`, change line 17 from:

```python
from .simulation import output_stats
```

to:

```python
from .simulation_fast import output_stats
```

- [ ] **Step 2: Run existing dataset tests to verify nothing breaks**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && python -m pytest tests/test_dataset.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/dataset.py
git commit -m "feat: switch dataset.py to use simulation_fast for optimized output_stats"
```

---

### Task 8: Run full test suite and verify

**Files:** (none modified — validation only)

- [ ] **Step 1: Run the complete test suite**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && python -m pytest tests/ -v`
Expected: All tests PASS, including the new `test_simulation_fast.py` tests and all existing tests unchanged.

- [ ] **Step 2: Run a quick smoke test via CLI**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && python -m network_estimation.cli smoke-test 2>&1 | head -20`
Expected: Smoke test completes without errors. Output shows scoring results.

- [ ] **Step 3: Verify `__init__.py` still exports from reference `simulation`**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/provo && python -c "from network_estimation import run_mlp; import inspect; print(inspect.getmodule(run_mlp).__name__)"`
Expected: `network_estimation.simulation` (NOT `simulation_fast`)

- [ ] **Step 4: Final commit if any fixups were needed**

Only if previous steps required fixes:
```bash
git add -A
git commit -m "fix: address test suite issues from simulation_fast integration"
```
