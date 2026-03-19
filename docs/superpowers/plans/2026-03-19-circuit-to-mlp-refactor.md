# Circuit-to-MLP Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the entire codebase from circuit estimation to MLP estimation, renaming the package from `circuit_estimation` to `network_estimation` and the CLI from `cestim` to `nestim`.

**Architecture:** In-place rename + rewrite (Approach A). The directory `src/circuit_estimation/` becomes `src/network_estimation/`. Every module is rewritten to use MLP domain objects (weight matrices + ReLU) instead of circuit objects (wire indices + bilinear gates). Streaming per-depth estimation is replaced by single-array returns.

**Tech Stack:** Python 3.10+, NumPy, SciPy (for `norm.pdf`/`norm.cdf` in estimators), Rich (CLI reporting), pytest

**Spec:** `docs/superpowers/specs/2026-03-19-circuit-to-mlp-refactor-design.md`

---

## Chunk 1: Foundation — Package Rename, Domain, Generation, Simulation

This chunk establishes the new package identity and core domain objects that everything else depends on.

### Task 1: Rename package directory and update pyproject.toml

**Files:**
- Rename: `src/circuit_estimation/` → `src/network_estimation/`
- Modify: `pyproject.toml`
- Modify: `main.py`

- [ ] **Step 1: Rename the package directory**

```bash
mv src/circuit_estimation src/network_estimation
```

- [ ] **Step 2: Update pyproject.toml**

Replace the full file contents:

```toml
[project]
name = "network-estimation"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "scipy",
    "plotext",
    "rich",
    "tqdm",
]

[project.scripts]
nestim = "network_estimation.cli:main"
network-estimation = "network_estimation.cli:main"

[dependency-groups]
dev = [
    "pytest>=8.2.0",
    "pyright>=1.1.0",
    "ruff>=0.6.0",
]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
"tests/conftest.py" = ["E402"]

[tool.pyright]
pythonVersion = "3.10"
typeCheckingMode = "standard"
include = [
    "src",
    "tests",
    "main.py",
]
```

- [ ] **Step 3: Update main.py**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from network_estimation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor: rename package circuit_estimation → network_estimation"
```

### Task 2: Rewrite domain.py — MLP dataclass

**Files:**
- Modify: `src/network_estimation/domain.py`
- Modify: `tests/test_domain.py`

- [ ] **Step 1: Write failing tests for MLP domain**

Replace `tests/test_domain.py` entirely:

```python
import numpy as np
import pytest

from network_estimation.domain import MLP


def test_mlp_validate_accepts_valid_mlp() -> None:
    weights = [np.zeros((4, 4), dtype=np.float32) for _ in range(3)]
    mlp = MLP(width=4, depth=3, weights=weights)
    mlp.validate()  # should not raise


def test_mlp_validate_rejects_zero_width() -> None:
    with pytest.raises(ValueError, match="width"):
        MLP(width=0, depth=1, weights=[np.zeros((0, 0), dtype=np.float32)]).validate()


def test_mlp_validate_rejects_zero_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        MLP(width=4, depth=0, weights=[]).validate()


def test_mlp_validate_rejects_depth_mismatch() -> None:
    weights = [np.zeros((4, 4), dtype=np.float32)]
    mlp = MLP(width=4, depth=2, weights=weights)
    with pytest.raises(ValueError, match="depth"):
        mlp.validate()


def test_mlp_validate_rejects_wrong_weight_shape() -> None:
    weights = [np.zeros((4, 3), dtype=np.float32)]
    mlp = MLP(width=4, depth=1, weights=weights)
    with pytest.raises(ValueError, match="shape"):
        mlp.validate()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_domain.py -v`
Expected: FAIL (old Circuit imports)

- [ ] **Step 3: Write MLP domain implementation**

Replace `src/network_estimation/domain.py` entirely:

```python
"""Core MLP data structure and invariant checks.

This module defines the canonical in-memory representation used throughout
generation, simulation, and scoring:

- ``MLP`` stores a sequence of weight matrices plus declared width/depth metadata.

All evaluator code assumes these objects pass validation before use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from numpy.typing import NDArray

Weights = List[NDArray[np.float32]]


@dataclass(frozen=True, slots=True)
class MLP:
    """Validated MLP container with fixed width and layer depth.

    Attributes:
        width: Number of neurons per layer.
        depth: Number of weight matrices (layers).
        weights: Ordered list of weight matrices, each shape ``(width, width)``.
    """

    width: int
    depth: int
    weights: Weights

    def validate(self) -> None:
        """Validate MLP metadata and weight matrix shapes.

        Raises:
            ValueError: if width/depth are invalid, if ``depth`` does not
                match ``len(weights)``, or if any weight matrix has wrong shape.
        """
        if self.width <= 0:
            raise ValueError("MLP width must be positive.")
        if self.depth <= 0:
            raise ValueError("MLP depth must be positive.")
        if len(self.weights) != self.depth:
            raise ValueError(
                f"MLP depth mismatch: declared depth={self.depth}, "
                f"got {len(self.weights)} weight matrices."
            )
        for i, w in enumerate(self.weights):
            if w.shape != (self.width, self.width):
                raise ValueError(
                    f"Weight matrix {i} has shape {w.shape}, "
                    f"expected ({self.width}, {self.width})."
                )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_domain.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/domain.py tests/test_domain.py
git commit -m "feat: rewrite domain.py with MLP dataclass replacing Circuit/Layer"
```

### Task 3: Rewrite generation.py — He-init MLP sampling

**Files:**
- Modify: `src/network_estimation/generation.py`
- Modify: `tests/test_generation.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_generation.py` entirely:

```python
import numpy as np
import pytest

from network_estimation.generation import sample_mlp


def test_sample_mlp_returns_valid_mlp() -> None:
    mlp = sample_mlp(width=8, depth=4)
    mlp.validate()
    assert mlp.width == 8
    assert mlp.depth == 4
    assert len(mlp.weights) == 4


def test_sample_mlp_weight_shapes() -> None:
    mlp = sample_mlp(width=16, depth=3)
    for w in mlp.weights:
        assert w.shape == (16, 16)
        assert w.dtype == np.float32


def test_sample_mlp_he_init_scale() -> None:
    """Verify weights have approximately correct He-init variance."""
    rng = np.random.default_rng(42)
    width = 256
    mlp = sample_mlp(width=width, depth=10, rng=rng)
    # He init: var = 2/width
    expected_var = 2.0 / width
    actual_var = np.var(np.concatenate([w.flatten() for w in mlp.weights]))
    assert abs(actual_var - expected_var) < 0.01 * expected_var


def test_sample_mlp_reproducible_with_rng() -> None:
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    mlp1 = sample_mlp(width=8, depth=2, rng=rng1)
    mlp2 = sample_mlp(width=8, depth=2, rng=rng2)
    for w1, w2 in zip(mlp1.weights, mlp2.weights):
        np.testing.assert_array_equal(w1, w2)


def test_sample_mlp_rejects_invalid_width() -> None:
    with pytest.raises(ValueError, match="width"):
        sample_mlp(width=0, depth=1)


def test_sample_mlp_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError, match="depth"):
        sample_mlp(width=4, depth=0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generation.py -v`
Expected: FAIL (old imports)

- [ ] **Step 3: Write implementation**

Replace `src/network_estimation/generation.py` entirely:

```python
"""Random MLP sampling utilities used by the evaluator.

This module produces synthetic MLPs with He-initialized weight matrices
for ReLU activation networks. The same sampling path is used by baseline
timing and score evaluation.
"""

from __future__ import annotations

import numpy as np

from .domain import MLP


def sample_mlp(
    width: int, depth: int, rng: np.random.Generator | None = None
) -> MLP:
    """Sample a random MLP with He-initialized weight matrices.

    Each weight matrix has shape ``(width, width)`` with entries drawn from
    ``N(0, 2/width)`` (He initialization for ReLU networks).

    Args:
        width: Number of neurons per layer.
        depth: Number of weight matrices (layers).
        rng: Optional NumPy generator for reproducible sampling.

    Returns:
        A validated ``MLP`` instance.
    """
    if width <= 0:
        raise ValueError("width must be positive.")
    if depth <= 0:
        raise ValueError("depth must be positive.")
    rng = rng or np.random.default_rng()
    scale = np.sqrt(2.0 / width)
    weights = [
        (rng.standard_normal((width, width)) * scale).astype(np.float32)
        for _ in range(depth)
    ]
    mlp = MLP(width=width, depth=depth, weights=weights)
    mlp.validate()
    return mlp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generation.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/generation.py tests/test_generation.py
git commit -m "feat: rewrite generation.py with He-init MLP sampling"
```

### Task 4: Rewrite simulation.py — MLP forward pass with ReLU

**Files:**
- Modify: `src/network_estimation/simulation.py`
- Modify: `tests/test_simulation.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_simulation.py` entirely:

```python
import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.simulation import (
    output_stats,
    relu,
    run_mlp,
    run_mlp_all_layers,
)


def test_relu_zeros_negatives() -> None:
    x = np.array([-1.0, 0.0, 1.0, -0.5, 2.0], dtype=np.float32)
    result = relu(x)
    np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 0.0, 2.0])


def test_run_mlp_identity_weights() -> None:
    """Identity weight matrices with ReLU should preserve positive inputs."""
    width = 4
    weights = [np.eye(width, dtype=np.float32)]
    mlp = MLP(width=width, depth=1, weights=weights)
    inputs = np.ones((2, width), dtype=np.float32)
    output = run_mlp(mlp, inputs)
    assert output.shape == (2, width)
    np.testing.assert_allclose(output, 1.0)


def test_run_mlp_all_layers_returns_per_layer() -> None:
    width = 4
    depth = 3
    weights = [np.eye(width, dtype=np.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    inputs = np.ones((5, width), dtype=np.float32)
    layers = run_mlp_all_layers(mlp, inputs)
    assert len(layers) == depth
    for layer_out in layers:
        assert layer_out.shape == (5, width)


def test_run_mlp_final_matches_all_layers_last() -> None:
    """run_mlp output should match last element of run_mlp_all_layers."""
    width = 8
    depth = 3
    rng = np.random.default_rng(42)
    weights = [(rng.standard_normal((width, width)) * 0.1).astype(np.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    inputs = rng.standard_normal((10, width)).astype(np.float32)
    final = run_mlp(mlp, inputs)
    all_layers = run_mlp_all_layers(mlp, inputs)
    np.testing.assert_array_equal(final, all_layers[-1])


def test_output_stats_returns_correct_shapes() -> None:
    width = 8
    depth = 2
    rng = np.random.default_rng(99)
    weights = [(rng.standard_normal((width, width)) * 0.1).astype(np.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    all_means, final_mean, avg_var = output_stats(mlp, n_samples=100)
    assert all_means.shape == (depth, width)
    assert final_mean.shape == (width,)
    assert isinstance(avg_var, float)
    assert avg_var >= 0.0
    np.testing.assert_allclose(final_mean, all_means[-1], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_simulation.py -v`
Expected: FAIL (old imports)

- [ ] **Step 3: Write implementation**

Replace `src/network_estimation/simulation.py` entirely:

```python
"""MLP execution helpers for batched forward passes and empirical moments.

These utilities run an MLP layer-by-layer over random inputs and expose
per-layer outputs/means used by score computation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .domain import MLP


def relu(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Element-wise ReLU activation."""
    return np.maximum(x, np.float32(0.0))


def run_mlp(mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
    """Forward pass returning final-layer activations.

    Args:
        mlp: MLP to execute.
        inputs: Input matrix of shape ``(samples, mlp.width)``.

    Returns:
        Activations of shape ``(samples, mlp.width)`` after the last layer.
    """
    x = inputs
    for w in mlp.weights:
        x = relu(x @ w)
    return x


def run_mlp_all_layers(
    mlp: MLP, inputs: NDArray[np.float32]
) -> list[NDArray[np.float32]]:
    """Forward pass returning activations after each layer.

    Args:
        mlp: MLP to execute.
        inputs: Input matrix of shape ``(samples, mlp.width)``.

    Returns:
        List of ``depth`` arrays, each shape ``(samples, mlp.width)``.
    """
    x = inputs
    layers: list[NDArray[np.float32]] = []
    for w in mlp.weights:
        x = relu(x @ w)
        layers.append(x)
    return layers


def output_stats(
    mlp: MLP, n_samples: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
    """Compute per-layer means and average variance of the final layer.

    Args:
        mlp: MLP to evaluate.
        n_samples: Number of random Gaussian N(0,1) input vectors.

    Returns:
        all_layer_means: shape ``(depth, width)`` — mean activations per layer.
        final_mean: shape ``(width,)`` — mean activations at the final layer.
        avg_variance: scalar — average per-neuron variance at the final layer,
            used for ``sampling_mse`` normalization.
    """
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    layer_outputs = run_mlp_all_layers(mlp, inputs)
    all_layer_means = np.stack(
        [np.mean(out, axis=0) for out in layer_outputs]
    ).astype(np.float32)
    final_outputs = layer_outputs[-1]
    final_mean = np.mean(final_outputs, axis=0).astype(np.float32)
    avg_variance = float(np.mean(np.var(final_outputs, axis=0)))
    return all_layer_means, final_mean, avg_variance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_simulation.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/simulation.py tests/test_simulation.py
git commit -m "feat: rewrite simulation.py with MLP forward pass and ReLU"
```

### Task 5: Rewrite sdk.py — participant interface

**Files:**
- Modify: `src/network_estimation/sdk.py`
- Modify: `tests/test_sdk.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_sdk.py` entirely:

```python
import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.sdk import BaseEstimator, SetupContext


def test_setup_context_fields() -> None:
    ctx = SetupContext(width=256, depth=16, estimator_budget=1000, api_version="1.0")
    assert ctx.width == 256
    assert ctx.depth == 16
    assert ctx.estimator_budget == 1000
    assert ctx.api_version == "1.0"
    assert ctx.scratch_dir is None


def test_base_estimator_requires_predict() -> None:
    """Subclass must implement predict."""
    class IncompleteEstimator(BaseEstimator):
        pass

    with pytest.raises(TypeError):
        IncompleteEstimator()


def test_base_estimator_default_setup_teardown() -> None:
    """setup and teardown should be callable without error."""
    class MinimalEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> np.ndarray:
            return np.zeros((mlp.depth, mlp.width), dtype=np.float32)

    est = MinimalEstimator()
    ctx = SetupContext(width=4, depth=2, estimator_budget=100, api_version="1.0")
    est.setup(ctx)  # should not raise
    est.teardown()  # should not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sdk.py -v`
Expected: FAIL (old imports)

- [ ] **Step 3: Write implementation**

Replace `src/network_estimation/sdk.py` entirely:

```python
"""Participant-facing estimator base interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .domain import MLP


@dataclass(frozen=True, slots=True)
class SetupContext:
    """Runtime context passed to ``BaseEstimator.setup``.

    This keeps participant setup hooks self-contained and future-proof without
    requiring direct imports from scoring internals.
    """

    width: int
    depth: int
    estimator_budget: int
    api_version: str
    scratch_dir: str | None = None


class BaseEstimator(ABC):
    """Estimator contract for participant implementations.

    Participants subclass this and implement ``predict`` to return
    predicted means for all layers as a single ``(depth, width)`` array.
    """

    @abstractmethod
    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        """Return predicted means for all layers, shape ``(depth, width)``."""
        raise NotImplementedError

    def setup(self, context: SetupContext) -> None:
        """Optional one-time setup hook before prediction calls."""
        return None

    def teardown(self) -> None:
        """Optional cleanup hook after scoring completes."""
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sdk.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/sdk.py tests/test_sdk.py
git commit -m "feat: rewrite sdk.py with MLP-based BaseEstimator interface"
```

### Task 6: Delete streaming.py and its tests

**Files:**
- Delete: `src/network_estimation/streaming.py`
- Delete: `tests/test_streaming.py`

- [ ] **Step 1: Delete streaming module and tests**

```bash
rm src/network_estimation/streaming.py tests/test_streaming.py
```

- [ ] **Step 2: Commit**

```bash
git add -u
git commit -m "refactor: delete streaming.py — no longer needed with single-array returns"
```

### Task 7: Update __init__.py exports

**Files:**
- Modify: `src/network_estimation/__init__.py`

- [ ] **Step 1: Write new __init__.py**

Replace `src/network_estimation/__init__.py` entirely:

```python
"""Core package for network estimation starter-kit runtime."""

from .domain import MLP
from .generation import sample_mlp
from .sdk import BaseEstimator, SetupContext
from .simulation import output_stats, relu, run_mlp, run_mlp_all_layers

__all__ = [
    "BaseEstimator",
    "SetupContext",
    "MLP",
    "sample_mlp",
    "relu",
    "run_mlp",
    "run_mlp_all_layers",
    "output_stats",
]
```

- [ ] **Step 2: Run foundation tests to verify everything works together**

Run: `pytest tests/test_domain.py tests/test_generation.py tests/test_simulation.py tests/test_sdk.py -v`
Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/__init__.py
git commit -m "feat: update __init__.py exports for MLP-based API"
```

---

## Chunk 2: Scoring, Estimators, and Reference Implementations

### Task 8: Rewrite scoring.py — single-budget MLP scoring

**Files:**
- Modify: `src/network_estimation/scoring.py`
- Modify: `tests/test_scoring_module.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_scoring_module.py` entirely:

```python
import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.scoring import (
    ContestData,
    ContestSpec,
    baseline_time,
    evaluate_estimator,
    make_contest,
)


def test_contest_spec_validates() -> None:
    spec = ContestSpec(width=8, depth=2, n_mlps=2, estimator_budget=100, ground_truth_budget=1000)
    spec.validate()


def test_contest_spec_rejects_zero_width() -> None:
    with pytest.raises(ValueError, match="width"):
        ContestSpec(width=0, depth=2, n_mlps=2, estimator_budget=100, ground_truth_budget=1000).validate()


def test_make_contest_produces_valid_data() -> None:
    spec = ContestSpec(width=8, depth=2, n_mlps=3, estimator_budget=64, ground_truth_budget=200)
    data = make_contest(spec)
    assert len(data.mlps) == 3
    assert len(data.all_layer_targets) == 3
    assert len(data.final_targets) == 3
    assert len(data.avg_variances) == 3
    for targets in data.all_layer_targets:
        assert targets.shape == (2, 8)
    for final in data.final_targets:
        assert final.shape == (8,)
    for var in data.avg_variances:
        assert var >= 0.0


def test_baseline_time_returns_positive() -> None:
    width = 8
    depth = 2
    weights = [np.eye(width, dtype=np.float32) for _ in range(depth)]
    mlp = MLP(width=width, depth=depth, weights=weights)
    t = baseline_time(mlp, n_samples=50)
    assert t > 0.0


def test_evaluate_estimator_with_zeros_estimator() -> None:
    """An estimator that always returns zeros should produce a finite score."""
    from network_estimation.sdk import BaseEstimator

    class ZerosEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> np.ndarray:
            return np.zeros((mlp.depth, mlp.width), dtype=np.float32)

    spec = ContestSpec(width=8, depth=2, n_mlps=2, estimator_budget=64, ground_truth_budget=200)
    data = make_contest(spec)
    result = evaluate_estimator(ZerosEstimator(), data)
    assert isinstance(result, dict)
    assert "primary_score" in result
    assert "secondary_score" in result
    assert np.isfinite(result["primary_score"])
    assert np.isfinite(result["secondary_score"])


def test_validate_predictions_rejects_wrong_shape() -> None:
    from network_estimation.scoring import validate_predictions
    with pytest.raises(ValueError, match="shape"):
        validate_predictions(np.zeros((3, 4), dtype=np.float32), depth=2, width=4)


def test_validate_predictions_rejects_nonfinite() -> None:
    from network_estimation.scoring import validate_predictions
    arr = np.zeros((2, 4), dtype=np.float32)
    arr[0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        validate_predictions(arr, depth=2, width=4)


def test_fraction_spent_floors_at_half() -> None:
    """Verify that very fast estimators get fraction_spent = 0.5, not less."""
    from network_estimation.sdk import BaseEstimator

    class InstantEstimator(BaseEstimator):
        def predict(self, mlp: MLP, budget: int) -> np.ndarray:
            return np.zeros((mlp.depth, mlp.width), dtype=np.float32)

    spec = ContestSpec(width=8, depth=2, n_mlps=1, estimator_budget=64, ground_truth_budget=200)
    data = make_contest(spec)
    result = evaluate_estimator(InstantEstimator(), data)
    # fraction_spent should be floored at 0.5
    assert result["per_mlp"][0]["fraction_spent"] >= 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scoring_module.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Replace `src/network_estimation/scoring.py` entirely:

```python
"""Scoring loop and baseline timing for MLP estimation contests."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .generation import sample_mlp
from .sdk import BaseEstimator
from .simulation import output_stats, run_mlp


@dataclass(slots=True)
class ContestSpec:
    """Evaluator configuration for one scoring run.

    Attributes:
        width: Number of neurons per MLP layer.
        depth: Number of layers per MLP.
        n_mlps: Number of MLPs to score over.
        estimator_budget: Sample count the estimator conceptually has available.
        ground_truth_budget: Sample count for computing ground truth means.
    """

    width: int
    depth: int
    n_mlps: int
    estimator_budget: int
    ground_truth_budget: int

    def validate(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be positive.")
        if self.depth <= 0:
            raise ValueError("depth must be positive.")
        if self.n_mlps <= 0:
            raise ValueError("n_mlps must be positive.")
        if self.estimator_budget <= 0:
            raise ValueError("estimator_budget must be positive.")
        if self.ground_truth_budget <= 0:
            raise ValueError("ground_truth_budget must be positive.")


default_spec = ContestSpec(
    width=256,
    depth=16,
    n_mlps=10,
    estimator_budget=256 * 256 * 4,
    ground_truth_budget=256 * 256 * 256,
)


@dataclass(slots=True)
class ContestData:
    """Precomputed contest data for scoring.

    Attributes:
        spec: Contest configuration.
        mlps: List of MLPs to score.
        all_layer_targets: Per-MLP ground truth means, each shape ``(depth, width)``.
        final_targets: Per-MLP final-layer ground truth means, each shape ``(width,)``.
        avg_variances: Per-MLP average final-layer variance for normalization.
    """

    spec: ContestSpec
    mlps: list[MLP]
    all_layer_targets: list[NDArray[np.float32]]
    final_targets: list[NDArray[np.float32]]
    avg_variances: list[float]


def make_contest(spec: ContestSpec) -> ContestData:
    """Generate MLPs and compute ground truth for a contest run."""
    spec.validate()
    mlps: list[MLP] = []
    all_layer_targets: list[NDArray[np.float32]] = []
    final_targets: list[NDArray[np.float32]] = []
    avg_variances: list[float] = []

    for _ in range(spec.n_mlps):
        mlp = sample_mlp(spec.width, spec.depth)
        all_means, final_mean, avg_var = output_stats(mlp, spec.ground_truth_budget)
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


def baseline_time(mlp: MLP, n_samples: int) -> float:
    """Measure wall time for a single forward pass with ``n_samples`` inputs.

    Returns:
        Wall time in seconds.
    """
    inputs = np.random.randn(n_samples, mlp.width).astype(np.float32)
    t0 = time.perf_counter()
    run_mlp(mlp, inputs)
    return time.perf_counter() - t0


def validate_predictions(
    predictions: NDArray[np.float32], *, depth: int, width: int
) -> NDArray[np.float32]:
    """Validate estimator prediction array shape and finiteness."""
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.shape != (depth, width):
        raise ValueError(
            f"Predictions must have shape ({depth}, {width}), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Predictions must contain only finite values.")
    return arr


def evaluate_estimator(
    estimator: BaseEstimator,
    data: ContestData,
) -> dict[str, Any]:
    """Score an estimator against precomputed contest data.

    Returns a dict with:
        primary_score: Average final-layer normalized MSE across MLPs.
        secondary_score: Average all-layer normalized MSE (diagnostic).
        per_mlp: List of per-MLP detail dicts.
    """
    spec = data.spec
    per_mlp: list[dict[str, Any]] = []
    primary_scores: list[float] = []
    secondary_scores: list[float] = []

    for i, mlp in enumerate(data.mlps):
        time_budget = baseline_time(mlp, spec.estimator_budget)
        time_budget = max(time_budget, 1e-9)

        t0 = time.perf_counter()
        try:
            raw_predictions = estimator.predict(mlp, spec.estimator_budget)
            predictions = validate_predictions(
                raw_predictions, depth=spec.depth, width=spec.width
            )
        except Exception as exc:
            predictions = np.zeros((spec.depth, spec.width), dtype=np.float32)
            per_mlp.append({"mlp_index": i, "error": str(exc)})
            # Use full time budget for errored predictions
            time_spent = time_budget
        else:
            time_spent = time.perf_counter() - t0

        # Time check: over budget -> zeros
        if time_spent > time_budget:
            predictions = np.zeros((spec.depth, spec.width), dtype=np.float32)

        # Time credit: floor at 50%
        fraction_spent = max(time_spent / time_budget, 0.5)

        # Normalization
        avg_var = data.avg_variances[i]
        sampling_mse = avg_var / (spec.estimator_budget * fraction_spent)
        sampling_mse = max(sampling_mse, 1e-30)  # avoid division by zero

        # Primary score: final layer
        final_pred = predictions[-1]
        final_target = data.final_targets[i]
        final_mse = float(np.mean((final_pred - final_target) ** 2))
        primary = final_mse / sampling_mse

        # Secondary score: all layers
        all_target = data.all_layer_targets[i]
        all_mse = float(np.mean((predictions - all_target) ** 2))
        secondary = all_mse / sampling_mse

        primary_scores.append(primary)
        secondary_scores.append(secondary)

        if not per_mlp or per_mlp[-1].get("mlp_index") != i:
            per_mlp.append({
                "mlp_index": i,
                "time_budget_s": time_budget,
                "time_spent_s": time_spent,
                "fraction_spent": fraction_spent,
                "final_mse": final_mse,
                "all_layer_mse": all_mse,
                "primary_score": primary,
                "secondary_score": secondary,
            })

    return {
        "primary_score": float(np.mean(primary_scores)),
        "secondary_score": float(np.mean(secondary_scores)),
        "per_mlp": per_mlp,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scoring_module.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/scoring.py tests/test_scoring_module.py
git commit -m "feat: rewrite scoring.py with single-budget MLP scoring"
```

### Task 9: Rewrite estimators.py — ReLU moment propagation

**Files:**
- Modify: `src/network_estimation/estimators.py`
- Modify: `tests/test_estimators_module.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_estimators_module.py` entirely:

```python
import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.estimators import (
    CombinedEstimator,
    CovariancePropagationEstimator,
    MeanPropagationEstimator,
)
from network_estimation.generation import sample_mlp


def _make_small_mlp() -> MLP:
    rng = np.random.default_rng(42)
    return sample_mlp(width=8, depth=3, rng=rng)


def test_mean_propagation_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    result = est.predict(mlp, budget=100)
    assert result.shape == (mlp.depth, mlp.width)
    assert result.dtype == np.float32


def test_covariance_propagation_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = CovariancePropagationEstimator()
    result = est.predict(mlp, budget=100)
    assert result.shape == (mlp.depth, mlp.width)
    assert result.dtype == np.float32


def test_combined_estimator_returns_correct_shape() -> None:
    mlp = _make_small_mlp()
    est = CombinedEstimator()
    result = est.predict(mlp, budget=100)
    assert result.shape == (mlp.depth, mlp.width)
    assert result.dtype == np.float32


def test_mean_propagation_all_finite() -> None:
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    result = est.predict(mlp, budget=100)
    assert np.all(np.isfinite(result))


def test_covariance_propagation_all_finite() -> None:
    mlp = _make_small_mlp()
    est = CovariancePropagationEstimator()
    result = est.predict(mlp, budget=100)
    assert np.all(np.isfinite(result))


def test_mean_propagation_nonnegative_outputs() -> None:
    """ReLU outputs are non-negative, so predicted means should be non-negative."""
    mlp = _make_small_mlp()
    est = MeanPropagationEstimator()
    result = est.predict(mlp, budget=100)
    assert np.all(result >= 0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_estimators_module.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Replace `src/network_estimation/estimators.py` entirely:

```python
"""Reference estimators for MLP mean prediction.

This module provides tutorial estimator classes that predict per-layer
output means using analytical moment propagation through ReLU networks.

- ``MeanPropagationEstimator``: first-moment propagation through ReLU.
- ``CovariancePropagationEstimator``: first + second moment propagation.
- ``CombinedEstimator``: budget-aware routing between the two.

For a ReLU unit z = max(0, w^T x), if x ~ N(mu, Sigma):
    E[z] = mu_pre * Phi(mu_pre/sigma_pre) + sigma_pre * phi(mu_pre/sigma_pre)

where mu_pre = w^T mu, sigma_pre^2 = w^T Sigma w, Phi is the normal CDF,
and phi is the normal PDF.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm  # type: ignore[import-untyped]

from .domain import MLP
from .sdk import BaseEstimator


class MeanPropagationEstimator(BaseEstimator):
    """Mean propagation estimator for ReLU MLPs.

    Propagates means through each layer using the ReLU expectation formula
    with a diagonal variance approximation (assumes independent neurons).

    For pre-activation z = W @ x, with x having mean mu and diagonal
    variance diag(v):
        mu_pre = W @ mu
        var_pre = (W**2) @ v
        E[ReLU(z)] = mu_pre * Phi(mu_pre / sigma_pre) + sigma_pre * phi(mu_pre / sigma_pre)

    Initial state: mu = 0, v = 1 (Gaussian N(0,1) inputs).
    """

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        _ = budget
        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        var = np.ones(width, dtype=np.float64)

        rows: list[NDArray[np.float32]] = []
        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu  # (width,)
            var_pre = (W ** 2).T @ var  # (width,)
            var_pre = np.maximum(var_pre, 1e-12)
            sigma_pre = np.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_alpha = norm.pdf(alpha)
            Phi_alpha = norm.cdf(alpha)

            # E[ReLU(z)]
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # Var[ReLU(z)] = E[z^2 | z>0]*P(z>0) - E[z]^2
            # E[z^2 | z>0]*P(z>0) = (mu_pre^2 + var_pre) * Phi(alpha) + mu_pre*sigma_pre*phi(alpha)
            ez2 = (mu_pre ** 2 + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var = np.maximum(ez2 - mu ** 2, 0.0)

            rows.append(mu.astype(np.float32))

        return np.stack(rows, axis=0)


class CovariancePropagationEstimator(BaseEstimator):
    """Full covariance propagation estimator for ReLU MLPs.

    Tracks both mean and full covariance matrix through each layer.
    More accurate than diagonal-variance mean propagation but O(n^2) per layer.

    Uses the same ReLU moment formulas but with full pre-activation covariance.
    """

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        _ = budget
        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        cov = np.eye(width, dtype=np.float64)

        rows: list[NDArray[np.float32]] = []
        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu
            cov_pre = W.T @ cov @ W
            var_pre = np.maximum(np.diag(cov_pre), 1e-12)
            sigma_pre = np.sqrt(var_pre)

            alpha = mu_pre / sigma_pre
            phi_alpha = norm.pdf(alpha)
            Phi_alpha = norm.cdf(alpha)

            # Post-ReLU means
            mu = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # Post-ReLU diagonal variance
            ez2 = (mu_pre ** 2 + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = np.maximum(ez2 - mu ** 2, 0.0)

            # Approximate post-ReLU covariance: scale pre-activation covariance
            # by the "gain" factors from the ReLU transfer
            gain = np.where(sigma_pre > 1e-12, Phi_alpha, 0.0)
            cov = np.outer(gain, gain) * cov_pre
            np.fill_diagonal(cov, var_post)

            rows.append(mu.astype(np.float32))

        return np.stack(rows, axis=0)


class CombinedEstimator(BaseEstimator):
    """Budget-aware hybrid estimator.

    Routes to covariance propagation for large budgets and mean propagation
    for small budgets. The routing threshold is configurable.
    """

    _COVARIANCE_BUDGET_MULTIPLIER = 30

    def __init__(
        self,
        *,
        mean_estimator: BaseEstimator | None = None,
        covariance_estimator: BaseEstimator | None = None,
    ) -> None:
        self._mean_estimator = mean_estimator or MeanPropagationEstimator()
        self._covariance_estimator = covariance_estimator or CovariancePropagationEstimator()

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        if budget >= self._COVARIANCE_BUDGET_MULTIPLIER * mlp.width:
            return self._covariance_estimator.predict(mlp, budget)
        return self._mean_estimator.predict(mlp, budget)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_estimators_module.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/estimators.py tests/test_estimators_module.py
git commit -m "feat: rewrite estimators.py with ReLU moment propagation"
```

### Task 10: Rewrite example estimators

**Files:**
- Modify: `examples/estimators/mean_propagation.py`
- Modify: `examples/estimators/covariance_propagation.py`
- Modify: `examples/estimators/combined_estimator.py`
- Modify: `examples/estimators/random_estimator.py`
- Modify: `tests/test_example_estimators.py`

- [ ] **Step 1: Rewrite examples/estimators/random_estimator.py**

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_estimation import BaseEstimator, SetupContext
from network_estimation.domain import MLP


class Estimator(BaseEstimator):
    """Random estimator: returns random predictions for all layers.

    This is a pedagogical example showing the estimator contract.
    Predictions are random and not meant to be accurate.
    """

    def __init__(self) -> None:
        self._predict_calls = 0
        self._context: SetupContext | None = None

    def setup(self, context: SetupContext) -> None:
        self._context = context
        self._predict_calls = 0

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        self._predict_calls += 1
        seed_text = f"random|call={self._predict_calls}|w={mlp.width}|d={mlp.depth}|b={budget}"
        seed_entropy = np.frombuffer(seed_text.encode("utf-8"), dtype=np.uint8).astype(np.uint32)
        rng = np.random.default_rng(seed_entropy)
        return rng.uniform(0.0, 1.0, size=(mlp.depth, mlp.width)).astype(np.float32)

    def teardown(self) -> None:
        self._context = None
        self._predict_calls = 0
```

- [ ] **Step 2: Rewrite examples/estimators/mean_propagation.py**

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_estimation import BaseEstimator
from network_estimation.domain import MLP


class Estimator(BaseEstimator):
    """Mean propagation estimator for ReLU MLPs.

    Propagates means through each layer using the ReLU expectation formula
    with a diagonal variance approximation.
    """

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        _ = budget
        from scipy.stats import norm  # type: ignore[import-untyped]

        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        var = np.ones(width, dtype=np.float64)
        rows: list[NDArray[np.float32]] = []

        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu
            var_pre = np.maximum((W ** 2).T @ var, 1e-12)
            sigma_pre = np.sqrt(var_pre)
            alpha = mu_pre / sigma_pre

            mu = mu_pre * norm.cdf(alpha) + sigma_pre * norm.pdf(alpha)
            ez2 = (mu_pre ** 2 + var_pre) * norm.cdf(alpha) + mu_pre * sigma_pre * norm.pdf(alpha)
            var = np.maximum(ez2 - mu ** 2, 0.0)
            rows.append(mu.astype(np.float32))

        return np.stack(rows, axis=0)
```

- [ ] **Step 3: Rewrite examples/estimators/covariance_propagation.py**

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_estimation import BaseEstimator
from network_estimation.domain import MLP


class Estimator(BaseEstimator):
    """Covariance propagation estimator for ReLU MLPs.

    Tracks both mean and full covariance matrix through each layer.
    """

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        _ = budget
        from scipy.stats import norm  # type: ignore[import-untyped]

        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        cov = np.eye(width, dtype=np.float64)
        rows: list[NDArray[np.float32]] = []

        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu
            cov_pre = W.T @ cov @ W
            var_pre = np.maximum(np.diag(cov_pre), 1e-12)
            sigma_pre = np.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi = norm.pdf(alpha)
            Phi = norm.cdf(alpha)

            mu = mu_pre * Phi + sigma_pre * phi
            ez2 = (mu_pre ** 2 + var_pre) * Phi + mu_pre * sigma_pre * phi
            var_post = np.maximum(ez2 - mu ** 2, 0.0)
            gain = np.where(sigma_pre > 1e-12, Phi, 0.0)
            cov = np.outer(gain, gain) * cov_pre
            np.fill_diagonal(cov, var_post)
            rows.append(mu.astype(np.float32))

        return np.stack(rows, axis=0)
```

- [ ] **Step 4: Rewrite examples/estimators/combined_estimator.py**

```python
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from network_estimation import BaseEstimator
from network_estimation.domain import MLP


class Estimator(BaseEstimator):
    """Budget-aware hybrid estimator: routes between mean and covariance propagation."""

    _COVARIANCE_BUDGET_MULTIPLIER = 30

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        if budget >= self._COVARIANCE_BUDGET_MULTIPLIER * mlp.width:
            return self._covariance_path(mlp)
        return self._mean_path(mlp)

    def _mean_path(self, mlp: MLP) -> NDArray[np.float32]:
        from scipy.stats import norm  # type: ignore[import-untyped]

        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        var = np.ones(width, dtype=np.float64)
        rows: list[NDArray[np.float32]] = []
        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu
            var_pre = np.maximum((W ** 2).T @ var, 1e-12)
            sigma_pre = np.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            mu = mu_pre * norm.cdf(alpha) + sigma_pre * norm.pdf(alpha)
            ez2 = (mu_pre ** 2 + var_pre) * norm.cdf(alpha) + mu_pre * sigma_pre * norm.pdf(alpha)
            var = np.maximum(ez2 - mu ** 2, 0.0)
            rows.append(mu.astype(np.float32))
        return np.stack(rows, axis=0)

    def _covariance_path(self, mlp: MLP) -> NDArray[np.float32]:
        from scipy.stats import norm  # type: ignore[import-untyped]

        width = mlp.width
        mu = np.zeros(width, dtype=np.float64)
        cov = np.eye(width, dtype=np.float64)
        rows: list[NDArray[np.float32]] = []
        for w in mlp.weights:
            W = w.astype(np.float64)
            mu_pre = W.T @ mu
            cov_pre = W.T @ cov @ W
            var_pre = np.maximum(np.diag(cov_pre), 1e-12)
            sigma_pre = np.sqrt(var_pre)
            alpha = mu_pre / sigma_pre
            phi = norm.pdf(alpha)
            Phi = norm.cdf(alpha)
            mu = mu_pre * Phi + sigma_pre * phi
            ez2 = (mu_pre ** 2 + var_pre) * Phi + mu_pre * sigma_pre * phi
            var_post = np.maximum(ez2 - mu ** 2, 0.0)
            gain = np.where(sigma_pre > 1e-12, Phi, 0.0)
            cov = np.outer(gain, gain) * cov_pre
            np.fill_diagonal(cov, var_post)
            rows.append(mu.astype(np.float32))
        return np.stack(rows, axis=0)
```

- [ ] **Step 5: Rewrite tests/test_example_estimators.py**

```python
import numpy as np
import pytest

from network_estimation.generation import sample_mlp


@pytest.fixture
def small_mlp():
    return sample_mlp(width=8, depth=3, rng=np.random.default_rng(42))


@pytest.mark.parametrize("estimator_module", [
    "examples.estimators.random_estimator",
    "examples.estimators.mean_propagation",
    "examples.estimators.covariance_propagation",
    "examples.estimators.combined_estimator",
])
def test_example_estimator_returns_correct_shape(small_mlp, estimator_module) -> None:
    import importlib
    mod = importlib.import_module(estimator_module)
    est = mod.Estimator()
    result = est.predict(small_mlp, budget=100)
    assert result.shape == (small_mlp.depth, small_mlp.width)
    assert result.dtype == np.float32
    assert np.all(np.isfinite(result))
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_example_estimators.py -v`
Expected: All 4 tests PASS

- [ ] **Step 7: Commit**

```bash
git add examples/estimators/ tests/test_example_estimators.py
git commit -m "feat: rewrite example estimators for MLP interface"
```

---

## Chunk 3: Runner, Subprocess Worker, Loader, Dataset

### Task 11: Rewrite runner.py — simplified predict returning NDArray

**Files:**
- Modify: `src/network_estimation/runner.py`
- Modify: `tests/test_inprocess_runner.py`
- Modify: `tests/test_runner_types.py`

- [ ] **Step 1: Write failing tests for InProcessRunner**

Replace `tests/test_inprocess_runner.py` entirely:

```python
import numpy as np
import pytest

from network_estimation.domain import MLP
from network_estimation.generation import sample_mlp
from network_estimation.runner import (
    EstimatorEntrypoint,
    InProcessRunner,
    ResourceLimits,
    RunnerError,
)
from network_estimation.sdk import SetupContext


@pytest.fixture
def small_mlp():
    return sample_mlp(width=8, depth=2, rng=np.random.default_rng(42))


@pytest.fixture
def limits():
    return ResourceLimits(setup_timeout_s=5.0, predict_timeout_s=30.0, memory_limit_mb=4096)


def test_inprocess_runner_predict_returns_array(small_mlp, limits, tmp_path) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import numpy as np\n"
        "from network_estimation.sdk import BaseEstimator\n"
        "from network_estimation.domain import MLP\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp: MLP, budget: int) -> np.ndarray:\n"
        "        return np.zeros((mlp.depth, mlp.width), dtype=np.float32)\n"
    )
    runner = InProcessRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, estimator_budget=100, api_version="1.0")
    runner.start(entry, ctx, limits)
    result = runner.predict(small_mlp, budget=100)
    assert result.shape == (2, 8)
    assert result.dtype == np.float32
    runner.close()


def test_inprocess_runner_predict_before_start_raises(small_mlp) -> None:
    runner = InProcessRunner()
    with pytest.raises(RunnerError):
        runner.predict(small_mlp, budget=100)
```

- [ ] **Step 2: Write failing tests for runner types**

Replace `tests/test_runner_types.py` with basic structural tests:

```python
import pytest

from network_estimation.runner import (
    EstimatorEntrypoint,
    ResourceLimits,
    RunnerError,
    RunnerErrorDetail,
)


def test_resource_limits_rejects_nonpositive_setup_timeout() -> None:
    with pytest.raises(ValueError):
        ResourceLimits(setup_timeout_s=0, predict_timeout_s=1.0, memory_limit_mb=1024)


def test_runner_error_carries_stage_and_detail() -> None:
    detail = RunnerErrorDetail(code="TEST", message="test error")
    err = RunnerError("predict", detail)
    assert err.stage == "predict"
    assert err.detail.code == "TEST"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_inprocess_runner.py tests/test_runner_types.py -v`
Expected: FAIL

- [ ] **Step 4: Write implementation**

Replace `src/network_estimation/runner.py` entirely:

```python
"""Runner interfaces for estimator isolation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .loader import load_estimator_from_path
from .sdk import BaseEstimator, SetupContext

try:
    import resource
except ImportError:  # pragma: no cover
    resource = None

try:
    import psutil  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover
    psutil = None

RunnerStage = Literal["load", "setup", "predict", "validate", "package", "submit"]


@dataclass(frozen=True, slots=True)
class EstimatorEntrypoint:
    file_path: Path
    class_name: str | None = None


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    setup_timeout_s: float
    predict_timeout_s: float
    memory_limit_mb: int
    cpu_time_limit_s: float | None = None

    def __post_init__(self) -> None:
        if self.setup_timeout_s <= 0:
            raise ValueError("setup_timeout_s must be positive.")
        if self.predict_timeout_s <= 0:
            raise ValueError("predict_timeout_s must be positive.")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mb must be positive.")
        if self.cpu_time_limit_s is not None and self.cpu_time_limit_s <= 0:
            raise ValueError("cpu_time_limit_s must be positive when provided.")


@dataclass(frozen=True, slots=True)
class RunnerErrorDetail:
    code: str
    message: str
    details: dict[str, str | int | float | bool] | None = None
    traceback: str | None = None


class RunnerError(RuntimeError):
    def __init__(self, stage: RunnerStage, detail: RunnerErrorDetail):
        super().__init__(detail.message)
        self.stage = stage
        self.detail = detail


class EstimatorRunner(Protocol):
    def start(
        self, entrypoint: EstimatorEntrypoint, context: SetupContext, limits: ResourceLimits,
    ) -> None: ...

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]: ...

    def close(self) -> None: ...


def _mlp_to_payload(mlp: MLP) -> dict[str, Any]:
    return {
        "width": int(mlp.width),
        "depth": int(mlp.depth),
        "weights": [w.tolist() for w in mlp.weights],
    }


def validate_predictions(
    predictions: object, *, depth: int, width: int,
) -> NDArray[np.float32]:
    arr = np.asarray(predictions, dtype=np.float32)
    if arr.shape != (depth, width):
        raise ValueError(
            f"Predictions must have shape ({depth}, {width}), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("Predictions must contain only finite values.")
    return arr


class InProcessRunner:
    def __init__(self) -> None:
        self._estimator: BaseEstimator | None = None
        self._limits: ResourceLimits | None = None
        self._context: SetupContext | None = None
        self._started = False

    def start(
        self, entrypoint: EstimatorEntrypoint, context: SetupContext, limits: ResourceLimits,
    ) -> None:
        self.close()
        self._limits = limits
        self._context = context
        start_wall = time.time()
        estimator, _ = load_estimator_from_path(
            entrypoint.file_path, class_name=entrypoint.class_name
        )
        self._estimator = estimator
        try:
            estimator.setup(context)
        except Exception as exc:
            raise RunnerError(
                "setup", RunnerErrorDetail(code="SETUP_ERROR", message=str(exc)),
            ) from exc
        setup_elapsed = time.time() - start_wall
        if setup_elapsed > limits.setup_timeout_s:
            raise RunnerError(
                "setup",
                RunnerErrorDetail(
                    code="SETUP_TIMEOUT",
                    message=f"setup exceeded timeout ({setup_elapsed:.6f}s > {limits.setup_timeout_s:.6f}s)",
                ),
            )
        self._started = True

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        if not self._started or self._estimator is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="RUNNER_NOT_STARTED",
                    message="Runner must be started before calling predict.",
                ),
            )
        raw = self._estimator.predict(mlp, budget)
        return validate_predictions(raw, depth=mlp.depth, width=mlp.width)

    def close(self) -> None:
        if self._estimator is not None:
            teardown = getattr(self._estimator, "teardown", None)
            if callable(teardown):
                teardown()
        self._estimator = None
        self._limits = None
        self._context = None
        self._started = False


class SubprocessRunner:
    def __init__(self, *, worker_command: list[str] | None = None) -> None:
        self._worker_command = (
            worker_command
            if worker_command is not None
            else [sys.executable, "-m", "network_estimation.subprocess_worker"]
        )
        self._process: subprocess.Popen[str] | None = None
        self._limits: ResourceLimits | None = None
        self._context: SetupContext | None = None
        self._started = False

    def start(
        self, entrypoint: EstimatorEntrypoint, context: SetupContext, limits: ResourceLimits,
    ) -> None:
        self.close()
        self._limits = limits
        self._context = context
        self._process = subprocess.Popen(
            self._worker_command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, env=self._worker_env(),
        )
        self._send_request({
            "command": "start",
            "entrypoint": {
                "file_path": str(entrypoint.file_path),
                "class_name": entrypoint.class_name,
            },
            "context": {
                "width": context.width,
                "depth": context.depth,
                "estimator_budget": context.estimator_budget,
                "api_version": context.api_version,
                "scratch_dir": context.scratch_dir,
            },
        })
        try:
            response = self._read_response(timeout_s=limits.setup_timeout_s)
        except TimeoutError as exc:
            self._terminate_process()
            raise RunnerError(
                "setup",
                RunnerErrorDetail(code="SETUP_TIMEOUT", message="worker setup timed out."),
            ) from exc
        except RunnerError as exc:
            stderr_tail = self._read_stderr_tail()
            msg = exc.detail.message
            if stderr_tail:
                msg = f"{msg} stderr: {stderr_tail}"
            self._terminate_process()
            raise RunnerError(
                "setup", RunnerErrorDetail(code="SETUP_PROTOCOL_ERROR", message=msg),
            ) from exc
        if response.get("status") != "ok":
            raise RunnerError(
                "setup",
                RunnerErrorDetail(
                    code="SETUP_ERROR",
                    message=str(response.get("error_message", "worker setup failed")),
                ),
            )
        self._started = True

    def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
        if not self._started or self._process is None or self._limits is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="RUNNER_NOT_STARTED", message="Runner must be started."),
            )
        self._send_request({
            "command": "predict",
            "budget": int(budget),
            "mlp": _mlp_to_payload(mlp),
        })
        try:
            response = self._read_response(timeout_s=self._limits.predict_timeout_s)
        except TimeoutError:
            self._terminate_process()
            self._started = False
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="PREDICT_TIMEOUT", message="predict timed out."),
            )
        except RunnerError:
            raise

        if response.get("status") == "error":
            raise RunnerError(
                "predict",
                RunnerErrorDetail(
                    code="PREDICT_ERROR",
                    message=str(response.get("error_message", "unknown error")),
                ),
            )
        predictions_data = response.get("predictions")
        if predictions_data is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="PREDICT_NO_DATA", message="No predictions in response."),
            )
        return np.asarray(predictions_data, dtype=np.float32)

    def close(self) -> None:
        if self._process is None:
            self._started = False
            return
        if self._process.poll() is None:
            try:
                self._send_request({"command": "close"})
                self._read_response(timeout_s=0.5)
            except Exception:
                pass
            self._terminate_process()
        self._process = None
        self._limits = None
        self._context = None
        self._started = False

    def _send_request(self, payload: dict[str, Any]) -> None:
        if self._process is None or self._process.stdin is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_IO_ERROR", message="Worker stdin unavailable."),
            )
        try:
            self._process.stdin.write(json.dumps(payload) + "\n")
            self._process.stdin.flush()
        except BrokenPipeError as exc:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_BROKEN_PIPE", message="Worker stdin closed."),
            ) from exc

    def _read_response(self, timeout_s: float) -> dict[str, Any]:
        if self._process is None or self._process.stdout is None:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_IO_ERROR", message="Worker stdout unavailable."),
            )
        import threading
        result: list[str] = []

        def _read() -> None:
            assert self._process is not None and self._process.stdout is not None
            result.append(self._process.stdout.readline())

        reader = threading.Thread(target=_read, daemon=True)
        reader.start()
        reader.join(timeout=timeout_s)
        if reader.is_alive():
            raise TimeoutError("worker response timed out")
        if not result or result[0] == "":
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_EOF", message="Worker closed stdout."),
            )
        try:
            payload = json.loads(result[0])
        except json.JSONDecodeError as exc:
            raise RunnerError(
                "predict",
                RunnerErrorDetail(code="WORKER_PROTOCOL_ERROR", message="Invalid JSON."),
            ) from exc
        return payload

    def _terminate_process(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.kill()
            self._process.wait(timeout=1.0)

    def _read_stderr_tail(self) -> str:
        if self._process is None or self._process.stderr is None:
            return ""
        if self._process.poll() is None:
            return ""
        stderr = self._process.stderr.read().strip()
        return stderr.splitlines()[-1] if stderr else ""

    def _worker_env(self) -> dict[str, str]:
        env = dict(os.environ)
        src_root = str(Path(__file__).resolve().parents[1])
        current = env.get("PYTHONPATH")
        env["PYTHONPATH"] = src_root if not current else f"{src_root}{os.pathsep}{current}"
        return env
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_inprocess_runner.py tests/test_runner_types.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/network_estimation/runner.py tests/test_inprocess_runner.py tests/test_runner_types.py
git commit -m "feat: rewrite runner.py with single-array predict interface"
```

### Task 12: Rewrite subprocess_worker.py

**Files:**
- Modify: `src/network_estimation/subprocess_worker.py`
- Modify: `tests/test_subprocess_runner.py`

- [ ] **Step 1: Write failing test**

Replace `tests/test_subprocess_runner.py` with the integration test shown in Step 3 below (write the test file first).

- [ ] **Step 2: Write subprocess_worker.py**

Replace `src/network_estimation/subprocess_worker.py` entirely:

```python
"""Subprocess worker for running participant estimators in isolation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .domain import MLP
from .loader import load_estimator_from_path
from .sdk import BaseEstimator, SetupContext


def _payload_to_mlp(payload: dict[str, Any]) -> MLP:
    weights = [
        np.asarray(w, dtype=np.float32) for w in payload["weights"]
    ]
    mlp = MLP(
        width=int(payload["width"]),
        depth=int(payload["depth"]),
        weights=weights,
    )
    mlp.validate()
    return mlp


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _handle_predict(estimator: BaseEstimator, request: dict[str, Any]) -> None:
    try:
        mlp = _payload_to_mlp(request["mlp"])
        budget = int(request["budget"])
    except Exception as exc:
        _write_response({"status": "error", "error_message": str(exc)})
        return

    try:
        predictions = estimator.predict(mlp, budget)
        arr = np.asarray(predictions, dtype=np.float32)
        if arr.shape != (mlp.depth, mlp.width):
            _write_response({
                "status": "error",
                "error_message": f"Predictions shape {arr.shape} != ({mlp.depth}, {mlp.width})",
            })
            return
        if not np.all(np.isfinite(arr)):
            _write_response({"status": "error", "error_message": "Non-finite predictions."})
            return
        _write_response({"status": "ok", "predictions": arr.tolist()})
    except Exception as exc:
        _write_response({"status": "error", "error_message": str(exc)})


def main() -> int:
    estimator: BaseEstimator | None = None
    for line in sys.stdin:
        raw = line.strip()
        if not raw:
            continue
        try:
            request = json.loads(raw)
        except json.JSONDecodeError:
            _write_response({"status": "protocol_error", "error_message": "Invalid JSON."})
            continue

        command = request.get("command")
        if command == "start":
            try:
                entrypoint = request["entrypoint"]
                ctx_payload = request["context"]
                estimator, _ = load_estimator_from_path(
                    Path(entrypoint["file_path"]),
                    class_name=entrypoint.get("class_name"),
                )
                context = SetupContext(
                    width=int(ctx_payload["width"]),
                    depth=int(ctx_payload["depth"]),
                    estimator_budget=int(ctx_payload["estimator_budget"]),
                    api_version=str(ctx_payload["api_version"]),
                    scratch_dir=(
                        str(ctx_payload["scratch_dir"])
                        if ctx_payload.get("scratch_dir") is not None
                        else None
                    ),
                )
                estimator.setup(context)
                _write_response({"status": "ok"})
            except Exception as exc:
                _write_response({"status": "runtime_error", "error_message": str(exc)})
        elif command == "predict":
            if estimator is None:
                _write_response({"status": "error", "error_message": "Estimator not initialized."})
                continue
            _handle_predict(estimator, request)
        elif command == "close":
            if estimator is not None:
                estimator.teardown()
            _write_response({"status": "ok"})
            break
        else:
            _write_response({"status": "protocol_error", "error_message": "Unknown command."})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Verify test passes**

The test file for Step 1 is:

```python
import numpy as np
import pytest

from network_estimation.generation import sample_mlp
from network_estimation.runner import (
    EstimatorEntrypoint,
    ResourceLimits,
    SubprocessRunner,
)
from network_estimation.sdk import SetupContext


@pytest.fixture
def small_mlp():
    return sample_mlp(width=8, depth=2, rng=np.random.default_rng(42))


def test_subprocess_runner_predict(small_mlp, tmp_path) -> None:
    est_file = tmp_path / "est.py"
    est_file.write_text(
        "import numpy as np\n"
        "from network_estimation.sdk import BaseEstimator\n"
        "from network_estimation.domain import MLP\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp: MLP, budget: int) -> np.ndarray:\n"
        "        return np.zeros((mlp.depth, mlp.width), dtype=np.float32)\n"
    )
    runner = SubprocessRunner()
    entry = EstimatorEntrypoint(file_path=est_file)
    ctx = SetupContext(width=8, depth=2, estimator_budget=100, api_version="1.0")
    limits = ResourceLimits(setup_timeout_s=10.0, predict_timeout_s=10.0, memory_limit_mb=4096)
    runner.start(entry, ctx, limits)
    result = runner.predict(small_mlp, budget=100)
    assert result.shape == (2, 8)
    runner.close()
```

Run: `pytest tests/test_subprocess_runner.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/network_estimation/subprocess_worker.py tests/test_subprocess_runner.py
git commit -m "feat: rewrite subprocess_worker.py for MLP interface"
```

### Task 13: Update loader.py

**Files:**
- Modify: `src/network_estimation/loader.py`

- [ ] **Step 1: Update the module naming prefix**

In `src/network_estimation/loader.py`, find the string `_circuit_estimation_submission_` and replace it with `_network_estimation_submission_`.

- [ ] **Step 2: Run loader tests**

Run: `pytest tests/test_loader.py -v`
Expected: Tests should pass (loader logic is otherwise unchanged). If any tests reference old circuit imports, update those imports.

- [ ] **Step 3: Commit**

```bash
git add src/network_estimation/loader.py tests/test_loader.py
git commit -m "refactor: update loader.py module naming prefix"
```

### Task 14: Rewrite dataset.py

**Files:**
- Modify: `src/network_estimation/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write failing tests**

Replace `tests/test_dataset.py` entirely:

```python
import numpy as np
import pytest

from network_estimation.dataset import create_dataset, load_dataset


def test_create_and_load_roundtrip(tmp_path) -> None:
    out = create_dataset(
        n_mlps=2,
        n_samples=50,
        width=8,
        depth=2,
        estimator_budget=32,
        seed=42,
        output_path=tmp_path / "test.npz",
    )
    bundle = load_dataset(out)
    assert bundle.n_mlps == 2
    assert len(bundle.mlps) == 2
    assert bundle.all_layer_means.shape == (2, 2, 8)
    assert bundle.final_means.shape == (2, 8)
    assert len(bundle.avg_variances) == 2
    for mlp in bundle.mlps:
        mlp.validate()
        assert mlp.width == 8
        assert mlp.depth == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

Replace `src/network_estimation/dataset.py` entirely:

```python
"""Create, save, and load pre-computed evaluation datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .generation import sample_mlp
from .hardware import collect_hardware_fingerprint
from .simulation import output_stats

SCHEMA_VERSION = "2.0"


def dataset_file_hash(path: Path | str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    metadata: dict[str, Any]
    mlps: list[MLP]
    all_layer_means: NDArray[np.float32]
    final_means: NDArray[np.float32]
    avg_variances: list[float]

    @property
    def n_mlps(self) -> int:
        return len(self.mlps)


def create_dataset(
    *,
    n_mlps: int,
    n_samples: int,
    width: int,
    depth: int,
    estimator_budget: int,
    seed: int | None = None,
    output_path: Path | str,
    progress: Any | None = None,
) -> Path:
    """Generate MLPs, compute ground truth, and save to .npz."""
    output_path = Path(output_path)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy)  # type: ignore[arg-type]
    rng = np.random.default_rng(seed)

    mlps: list[MLP] = []
    for i in range(n_mlps):
        mlps.append(sample_mlp(width, depth, rng))
        if progress is not None:
            progress({"phase": "generating", "completed": i + 1, "total": n_mlps})

    # Pack weight matrices: shape (n_mlps, depth, width, width)
    weights_array = np.stack(
        [np.stack(mlp.weights) for mlp in mlps]
    ).astype(np.float32)

    # Compute ground truth
    all_means_list: list[NDArray[np.float32]] = []
    final_means_list: list[NDArray[np.float32]] = []
    avg_variances: list[float] = []
    for i, mlp in enumerate(mlps):
        all_means, final_mean, avg_var = output_stats(mlp, n_samples)
        all_means_list.append(all_means)
        final_means_list.append(final_mean)
        avg_variances.append(avg_var)
        if progress is not None:
            progress({"phase": "sampling", "completed": i + 1, "total": n_mlps})

    all_layer_means = np.stack(all_means_list).astype(np.float32)
    final_means = np.stack(final_means_list).astype(np.float32)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "n_mlps": n_mlps,
        "n_samples": n_samples,
        "width": width,
        "depth": depth,
        "estimator_budget": estimator_budget,
        "hardware": collect_hardware_fingerprint(),
    }

    np.savez(
        output_path,
        metadata=np.array(json.dumps(metadata)),
        weights=weights_array,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=np.array(avg_variances, dtype=np.float64),
    )
    return output_path


def load_dataset(path: Path | str) -> DatasetBundle:
    path = Path(path)
    data = np.load(path, allow_pickle=False)
    metadata = json.loads(str(data["metadata"]))

    if "schema_version" not in metadata:
        raise ValueError("Invalid dataset: missing schema_version.")

    weights_array = data["weights"].astype(np.float32)
    all_layer_means = data["all_layer_means"].astype(np.float32)
    final_means = data["final_means"].astype(np.float32)
    avg_variances = data["avg_variances"].astype(np.float64).tolist()

    n_mlps = int(weights_array.shape[0])
    depth = int(weights_array.shape[1])
    width = int(weights_array.shape[2])

    mlps: list[MLP] = []
    for i in range(n_mlps):
        layer_weights = [weights_array[i, j] for j in range(depth)]
        mlp = MLP(width=width, depth=depth, weights=layer_weights)
        mlp.validate()
        mlps.append(mlp)

    return DatasetBundle(
        metadata=metadata,
        mlps=mlps,
        all_layer_means=all_layer_means,
        final_means=final_means,
        avg_variances=avg_variances,
    )
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_dataset.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/network_estimation/dataset.py tests/test_dataset.py
git commit -m "feat: rewrite dataset.py for MLP weight matrix packing"
```

---

## Chunk 4: CLI, Reporting, Protocol, Remaining Tests, Docs

### Task 15: Update protocol.py

**Files:**
- Modify: `src/network_estimation/protocol.py`
- Modify: `tests/test_protocol.py`

- [ ] **Step 1: Update protocol.py field names if needed**

The protocol module is a simple request/response schema. Update field names to match new terminology if any references to circuit-specific fields exist. The current `ScoreRequest`/`ScoreResponse` are fairly generic — update the `budget` field description if needed.

- [ ] **Step 2: Update test imports**

In `tests/test_protocol.py`, change `from circuit_estimation.protocol` to `from network_estimation.protocol`.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_protocol.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/network_estimation/protocol.py tests/test_protocol.py
git commit -m "refactor: update protocol.py imports for network_estimation"
```

### Task 16: Rewrite cli.py

**Files:**
- Modify: `src/network_estimation/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_cli_fallback.py`
- Modify: `tests/test_cli_participant_commands.py`

This is the largest single file. The CLI needs to:
1. Replace all `cestim` references with `nestim`
2. Replace `circuit` references with `mlp`/`network`
3. Update `ContestParams` → `ContestSpec`
4. Update `_default_contest_params()` → `_default_contest_spec()`
5. Remove streaming validation (`_consume_prediction_stream`)
6. Update `validate_submission_entrypoint` to use MLP
7. Update progress phases and labels
8. Update dataset commands for new field names

- [ ] **Step 1: Rewrite cli.py imports and defaults**

Read the current `src/network_estimation/cli.py`. Apply these concrete changes:

**Imports (top of file):** Replace:
- `from .domain import Circuit, Layer` → delete this line
- `from .scoring import ContestParams, score_estimator_report` → `from .scoring import ContestSpec, evaluate_estimator, make_contest, validate_predictions`
- `from .streaming import validate_depth_row` → delete this line
- All remaining `circuit_estimation` references → `network_estimation`

**Defaults:** Replace `_default_contest_params()` with:
```python
def _default_contest_spec() -> ContestSpec:
    return ContestSpec(
        width=100,
        depth=16,
        n_mlps=10,
        estimator_budget=100 * 100 * 4,
        ground_truth_budget=100 * 100 * 256,
    )
```

**Delete:** Remove `_consume_prediction_stream` function entirely (was ~lines 425-451).

**`validate_submission_entrypoint`:** Replace the body to use MLP instead of Circuit:
```python
def validate_submission_entrypoint(estimator_path, *, class_name=None):
    from .generation import sample_mlp
    estimator, metadata = load_estimator_from_path(estimator_path, class_name=class_name)
    context = SetupContext(width=4, depth=2, estimator_budget=100, api_version="1.0")
    mlp = sample_mlp(width=4, depth=2)
    try:
        estimator.setup(context)
        predictions = estimator.predict(mlp, 100)
        arr = validate_predictions(predictions, depth=mlp.depth, width=mlp.width)
    finally:
        estimator.teardown()
    return {"ok": True, "class_name": metadata.class_name, "module_name": metadata.module_name, "output_shape": list(arr.shape)}
```

- [ ] **Step 1b: Update parser and command handlers**

**Parser:** In `_build_participant_parser`:
- Change description from "circuit-estimation" to "network-estimation"
- Replace `--max-depth` args with `--depth`
- Replace `--budgets` args with `--estimator-budget` (single int)
- Update help text: "circuit" → "network"/"MLP", "cestim" → "nestim"

**`run` command handler:** The current handler uses `score_estimator_report` with multi-budget `ContestParams`. Replace with:
1. Build a `ContestSpec` from CLI args
2. Call `make_contest(spec)` to get `ContestData`
3. Call `evaluate_estimator(estimator, data)` to get results
4. For runner-based execution: use `runner.start()` / `runner.predict()` / `runner.close()` directly, or simplify to just use `InProcessRunner` for now

**`create-dataset` handler:** Update to pass new field names (`n_mlps`, `depth`, `estimator_budget` instead of `n_circuits`, `max_depth`, `budgets`).

**String replacements throughout the file:**
- "Circuit Estimation" → "Network Estimation"
- "cestim" → "nestim"
- "circuit" → "network" (in user-facing strings)
- "circuits" → "MLPs" (in labels and progress descriptions)
- "adjusted_mse" → "primary_score"

**`_render_plain_text_report`:** Update field names to match new report structure (`primary_score` instead of `adjusted_mse`, `depth` instead of `max_depth`, etc.).

**`_error_code`:** Update error code strings (remove `ESTIMATOR_STREAM_*` codes, add `ESTIMATOR_BAD_SHAPE` etc.).

- [ ] **Step 2: Update CLI test files**

In `tests/test_cli.py`, `tests/test_cli_fallback.py`, and `tests/test_cli_participant_commands.py`:
- Change all `from circuit_estimation` imports to `from network_estimation`
- Update any circuit-specific test fixtures to use MLP objects
- Update CLI command strings from `cestim` to `nestim`

- [ ] **Step 3: Run CLI tests**

Run: `pytest tests/test_cli.py tests/test_cli_fallback.py tests/test_cli_participant_commands.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/network_estimation/cli.py tests/test_cli.py tests/test_cli_fallback.py tests/test_cli_participant_commands.py
git commit -m "feat: rewrite cli.py for nestim with MLP interface"
```

### Task 17: Update reporting.py terminology

**Files:**
- Modify: `src/network_estimation/reporting.py`
- Modify: `tests/test_reporting.py`

- [ ] **Step 1: Replace circuit terminology in reporting.py**

Search and replace in `src/network_estimation/reporting.py`:
- "Circuit Estimation" → "Network Estimation"
- "cestim" → "nestim"
- "circuit" → "network" (in user-facing strings)
- "circuits" → "MLPs" (in labels)
- "wire" → "neuron" (in labels)
- "layer_count" → "depth" (in field references)
- "max_depth" → "depth"
- "adjusted_mse" → "primary_score"

Also update any report field accessors to match the new scoring report structure.

- [ ] **Step 2: Update test imports**

In `tests/test_reporting.py`, change imports from `circuit_estimation` to `network_estimation`.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_reporting.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/network_estimation/reporting.py tests/test_reporting.py
git commit -m "refactor: update reporting.py terminology for network estimation"
```

### Task 18: Update visualizer.py references

**Files:**
- Modify: `src/network_estimation/visualizer.py`
- Modify: `tests/test_visualizer.py`

- [ ] **Step 1: Update string references**

In `src/network_estimation/visualizer.py`, replace "circuit-explorer" references with "network-explorer" or equivalent.

- [ ] **Step 2: Update test imports**

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_visualizer.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/network_estimation/visualizer.py tests/test_visualizer.py
git commit -m "refactor: update visualizer.py references for network estimation"
```

### Task 19: Fix remaining test files

**Files:**
- Modify all remaining `tests/test_*.py` files that still import `circuit_estimation`

- [ ] **Step 1: Find and fix all remaining test imports**

Run: `grep -r "circuit_estimation" tests/` to find all remaining references.

For each file found:
- Change `from circuit_estimation` to `from network_estimation`
- Update any circuit-specific fixtures/assertions to use MLP objects
- Files likely affected: `test_circuit.py`, `test_evaluate.py`, `test_exhaustive_harness.py`, `test_hardware.py`, `test_packaging.py`, `test_docs_quality.py`

- [ ] **Step 2: Delete tests/test_circuit.py**

This file tests the old `Circuit`/`Layer` domain — it's been replaced by `test_domain.py`.

```bash
rm tests/test_circuit.py
```

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "fix: update all remaining test imports to network_estimation"
```

### Task 20: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/concepts/problem-setup.md`
- Modify: `docs/concepts/scoring-model.md`
- Modify: `docs/getting-started/first-local-run.md`
- Modify: `docs/getting-started/install-and-cli-quickstart.md`
- Modify: `docs/how-to/write-an-estimator.md`
- Modify: `docs/how-to/inspect-circuit-structure.md`
- Modify: `docs/how-to/use-circuit-explorer.md`
- Modify: `docs/how-to/use-evaluation-datasets.md`
- Modify: `docs/how-to/validate-run-package.md`
- Modify: `docs/reference/cli-reference.md`
- Modify: `docs/reference/estimator-contract.md`
- Modify: `docs/reference/score-report-fields.md`
- Modify: `docs/index.md`
- Modify: `docs/troubleshooting/common-participant-errors.md`

- [ ] **Step 1: Update all docs**

For every doc file:
1. Replace "circuit estimation" → "network estimation"
2. Replace "cestim" → "nestim"
3. Replace "Circuit" → "MLP" (the data type)
4. Replace "circuit" → "MLP" / "network" (in descriptions)
5. Replace "wire" → "neuron"
6. Replace "gate" → "layer"
7. Update code examples to use new API (`MLP`, `sample_mlp`, non-streaming `predict`)
8. Update scoring descriptions to match new formula
9. Rename `docs/how-to/inspect-circuit-structure.md` → consider renaming or rewriting for MLP

- [ ] **Step 2: Run docs quality tests if they exist**

Run: `pytest tests/test_docs_quality.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add README.md docs/
git commit -m "docs: update all documentation for network estimation / MLP refactor"
```

### Task 21: Final full test suite run

- [ ] **Step 1: Run the complete test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Run ruff linting**

Run: `ruff check src/ tests/ examples/`
Expected: No errors

- [ ] **Step 3: Run ruff formatting**

Run: `ruff format --check src/ tests/ examples/`
Expected: No formatting issues (or fix them)

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -A
git commit -m "style: fix linting and formatting issues"
```
