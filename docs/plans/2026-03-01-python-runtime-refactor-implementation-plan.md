# Python Runtime Refactor Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the Python runtime into a documented package with strict quality gates, richer tests, and targeted behavior fixes while preserving starter-kit usability.

**Architecture:** Move runtime logic into `src/circuit_estimation/*` with transport-agnostic core modules (`domain`, `generation`, `simulation`, `estimators`, `scoring`) plus a thin `cli` boundary and compatibility wrappers. Use test-first incremental migration so behavior changes are explicit and justified.

**Tech Stack:** Python 3.10+, NumPy, Pytest, Ruff, Pyright, uv

---

## Execution Notes

- Use `@test-driven-development` for all code changes.
- Use `@verification-before-completion` before claiming done.
- Keep commits small and frequent (one task per commit).
- Do not touch `CHALLENGE-CONTEXT.md`.

### Task 1: Introduce package scaffold and validated domain entities

**Files:**
- Create: `src/circuit_estimation/__init__.py`
- Create: `src/circuit_estimation/domain.py`
- Create: `tests/test_domain.py`
- Modify: `pyproject.toml`

**Step 1: Write the failing test**

```python
# tests/test_domain.py
import numpy as np
import pytest

from circuit_estimation.domain import Circuit, Layer

def test_layer_validate_rejects_mismatched_shapes():
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1], dtype=np.int32),
        first_coeff=np.array([1.0, 1.0], dtype=np.float32),
        second_coeff=np.array([1.0, 1.0], dtype=np.float32),
        const=np.array([0.0, 0.0], dtype=np.float32),
        product_coeff=np.array([0.0, 0.0], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="same shape"):
        layer.validate(n=2)

def test_circuit_validate_rejects_depth_mismatch():
    layer = Layer.identity(n=2)
    circuit = Circuit(n=2, d=2, gates=[layer])
    with pytest.raises(ValueError, match="depth"):
        circuit.validate()
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_domain.py -v`  
Expected: FAIL with `ModuleNotFoundError` for `circuit_estimation.domain`.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/domain.py
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray

@dataclass(slots=True)
class Layer:
    first: NDArray[np.int32]
    second: NDArray[np.int32]
    first_coeff: NDArray[np.float32]
    second_coeff: NDArray[np.float32]
    const: NDArray[np.float32]
    product_coeff: NDArray[np.float32]

    def validate(self, n: int) -> None:
        shapes = {
            self.first.shape,
            self.second.shape,
            self.first_coeff.shape,
            self.second_coeff.shape,
            self.const.shape,
            self.product_coeff.shape,
        }
        if len(shapes) != 1:
            raise ValueError("All layer vectors must have the same shape.")
        if np.any(self.first < 0) or np.any(self.first >= n):
            raise ValueError("first indices out of bounds.")
        if np.any(self.second < 0) or np.any(self.second >= n):
            raise ValueError("second indices out of bounds.")

    @staticmethod
    def identity(n: int) -> "Layer":
        return Layer(
            first=np.arange(n, dtype=np.int32),
            second=np.roll(np.arange(n, dtype=np.int32), -1),
            first_coeff=np.ones(n, dtype=np.float32),
            second_coeff=np.zeros(n, dtype=np.float32),
            const=np.zeros(n, dtype=np.float32),
            product_coeff=np.zeros(n, dtype=np.float32),
        )

@dataclass(slots=True)
class Circuit:
    n: int
    d: int
    gates: list[Layer]

    def validate(self) -> None:
        if len(self.gates) != self.d:
            raise ValueError("Circuit depth mismatch between d and number of gates.")
        for layer in self.gates:
            layer.validate(self.n)
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_domain.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml src/circuit_estimation/__init__.py src/circuit_estimation/domain.py tests/test_domain.py
git commit -m "feat: add circuit domain models with validation"
```

### Task 2: Add deterministic generation module

**Files:**
- Create: `src/circuit_estimation/generation.py`
- Create: `tests/test_generation.py`

**Step 1: Write the failing test**

```python
# tests/test_generation.py
import numpy as np
from circuit_estimation.generation import random_circuit, random_gates

def test_random_gates_disallow_duplicate_inputs():
    layer = random_gates(64, np.random.default_rng(123))
    assert np.all(layer.first != layer.second)

def test_random_circuit_is_reproducible_with_seeded_rng():
    c1 = random_circuit(8, 3, np.random.default_rng(7))
    c2 = random_circuit(8, 3, np.random.default_rng(7))
    for l1, l2 in zip(c1.gates, c2.gates):
        np.testing.assert_array_equal(l1.first, l2.first)
        np.testing.assert_array_equal(l1.second, l2.second)
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_generation.py -v`  
Expected: FAIL because `generation.py` does not exist yet.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/generation.py
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .domain import Circuit, Layer

def random_gates(n: int, rng: np.random.Generator | None = None) -> Layer:
    rng = rng or np.random.default_rng()
    first: NDArray[np.int32] = rng.integers(0, n, size=n, dtype=np.int32)
    second_raw: NDArray[np.int32] = rng.integers(0, n - 1, size=n, dtype=np.int32)
    second: NDArray[np.int32] = (second_raw + (second_raw >= first).astype(np.int32)).astype(np.int32)
    # keep current polynomial generation behavior from existing code
    ...

def random_circuit(n: int, d: int, rng: np.random.Generator | None = None) -> Circuit:
    rng = rng or np.random.default_rng()
    circuit = Circuit(n=n, d=d, gates=[random_gates(n, rng) for _ in range(d)])
    circuit.validate()
    return circuit
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_generation.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/generation.py tests/test_generation.py
git commit -m "feat: add deterministic circuit generation module"
```

### Task 3: Add simulation module and migrate execution primitives

**Files:**
- Create: `src/circuit_estimation/simulation.py`
- Create: `tests/test_simulation.py`

**Step 1: Write the failing test**

```python
# tests/test_simulation.py
import numpy as np
from circuit_estimation.domain import Circuit, Layer
from circuit_estimation.simulation import run_batched

def test_run_batched_matches_manual_layer_equation():
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.array([1.0, -1.0], dtype=np.float32),
        second_coeff=np.array([0.5, 0.25], dtype=np.float32),
        const=np.array([0.0, 1.0], dtype=np.float32),
        product_coeff=np.array([0.0, -0.5], dtype=np.float32),
    )
    circuit = Circuit(n=2, d=1, gates=[layer])
    out = list(run_batched(circuit, np.array([[-1.0, 1.0]], dtype=np.float16)))[0]
    expected = layer.const + layer.first_coeff * np.array([[-1.0, 1.0]])[:, layer.first] + layer.second_coeff * np.array([[-1.0, 1.0]])[:, layer.second] + layer.product_coeff * np.array([[-1.0, 1.0]])[:, layer.first] * np.array([[-1.0, 1.0]])[:, layer.second]
    np.testing.assert_allclose(out, expected)
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_simulation.py -v`  
Expected: FAIL due to missing `simulation.py`.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/simulation.py
from __future__ import annotations
from collections.abc import Iterator
import numpy as np
from numpy.typing import NDArray
from .domain import Circuit

def run_batched(circuit: Circuit, inputs: NDArray[np.float16]) -> Iterator[NDArray[np.float16]]:
    x = inputs
    for layer in circuit.gates:
        x = (
            layer.const
            + layer.first_coeff * x[:, layer.first]
            + layer.second_coeff * x[:, layer.second]
            + layer.product_coeff * x[:, layer.first] * x[:, layer.second]
        )
        yield x
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_simulation.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/simulation.py tests/test_simulation.py
git commit -m "feat: add simulation module for batched execution"
```

### Task 4: Add estimator module with explicit invariants

**Files:**
- Create: `src/circuit_estimation/estimators.py`
- Create: `tests/test_estimators.py`

**Step 1: Write the failing test**

```python
# tests/test_estimators.py
import numpy as np
from circuit_estimation.estimators import clip

def test_clip_enforces_correlation_bounds():
    mean = np.array([2.0, -2.0], dtype=np.float32)
    cov = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    clip(mean, cov)
    assert np.all(mean <= 1.0) and np.all(mean >= -1.0)
    np.testing.assert_allclose(np.diag(cov), 1.0 - mean * mean)
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_estimators.py -v`  
Expected: FAIL because estimator module does not exist.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/estimators.py
from __future__ import annotations
from collections.abc import Iterator
import numpy as np
from numpy.typing import NDArray
from .domain import Circuit

def clip(mean: NDArray[np.float32], cov: NDArray[np.float32]) -> None:
    n = len(mean)
    np.clip(mean, -1.0, 1.0, out=mean)
    var = 1.0 - mean * mean
    cov[np.arange(n), np.arange(n)] = var
    std = np.sqrt(np.clip(var, 0.0, None))
    limit = np.outer(std, std)
    np.clip(cov, -limit, limit, out=cov)
```

Then migrate current `mean_propagation`, covariance helpers, `covariance_propagation`, and `combined_estimator` with module-level documentation for assumptions.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_estimators.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/estimators.py tests/test_estimators.py
git commit -m "feat: migrate estimator math into package module"
```

### Task 5: Add scoring module and targeted behavior fixes

**Files:**
- Create: `src/circuit_estimation/scoring.py`
- Create: `tests/test_scoring.py`

**Step 1: Write the failing test**

```python
# tests/test_scoring.py
import numpy as np
import pytest
from circuit_estimation.scoring import ContestParams, score_estimator
from circuit_estimation.domain import Circuit, Layer

def test_score_estimator_rejects_wrong_output_width():
    params = ContestParams(width=2, max_depth=1, budgets=[10], time_tolerance=0.1)
    layer = Layer.identity(2)
    circuit = Circuit(n=2, d=1, gates=[layer])

    def bad_estimator(_circuit, _budget):
        yield np.array([0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="output width"):
        score_estimator(bad_estimator, n_circuits=1, n_samples=4, contest_params=params, circuits=[circuit])
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_scoring.py -v`  
Expected: FAIL because `scoring.py` and validation are missing.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/scoring.py
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable, Iterator, Sequence
import numpy as np
from numpy.typing import NDArray
from .domain import Circuit
from .generation import random_circuit
from .simulation import empirical_mean, run_batched

@dataclass(slots=True)
class ContestParams:
    width: int
    max_depth: int
    budgets: list[int]
    time_tolerance: float

    def validate(self) -> None:
        ...

EstimatorFn = Callable[[Circuit, int], Iterator[NDArray[np.float32]]]

def score_estimator(..., circuits: Sequence[Circuit] | None = None) -> float:
    # keep prior semantics, add explicit shape/depth validation
    ...
```

Include targeted fixes with tests for:
- estimator output count less than `max_depth` (error),
- shape mismatch (error),
- deterministic test seam (`circuits` injection) for reproducible unit tests.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_scoring.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring.py
git commit -m "feat: add scoring module with explicit estimator validation"
```

### Task 6: Add RPC-ready protocol schemas (no RPC server)

**Files:**
- Create: `src/circuit_estimation/protocol.py`
- Create: `tests/test_protocol.py`

**Step 1: Write the failing test**

```python
# tests/test_protocol.py
from circuit_estimation.protocol import ScoreRequest

def test_score_request_has_versioned_schema():
    req = ScoreRequest(schema_version="1.0", n_circuits=2, n_samples=16, budget=100)
    assert req.schema_version == "1.0"
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_protocol.py -v`  
Expected: FAIL because module does not exist.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/protocol.py
from dataclasses import dataclass

@dataclass(slots=True)
class ScoreRequest:
    schema_version: str
    n_circuits: int
    n_samples: int
    budget: int
```

Add request/response dataclasses and serialization helpers for core DTOs.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_protocol.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/protocol.py tests/test_protocol.py
git commit -m "feat: add rpc-ready protocol schema module"
```

### Task 7: Wire CLI and backward-compatible top-level wrappers

**Files:**
- Create: `src/circuit_estimation/cli.py`
- Modify: `main.py`
- Modify: `circuit.py`
- Modify: `estimators.py`
- Modify: `evaluate.py`
- Create: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
# tests/test_cli.py
from circuit_estimation.cli import run_default_score

def test_run_default_score_returns_float():
    score = run_default_score()
    assert isinstance(score, float)
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_cli.py -v`  
Expected: FAIL because `cli.py` missing.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/cli.py
from .estimators import combined_estimator
from .scoring import ContestParams, score_estimator

def run_default_score() -> float:
    return score_estimator(
        combined_estimator,
        n_circuits=10,
        n_samples=10000,
        contest_params=ContestParams(width=100, max_depth=30, budgets=[10, 100, 1000, 10000], time_tolerance=0.1),
    )
```

Then:
- make `main.py` print `run_default_score()`,
- keep root modules as compatibility re-exports from package modules.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_cli.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py main.py circuit.py estimators.py evaluate.py tests/test_cli.py
git commit -m "feat: add cli and preserve compatibility wrappers"
```

### Task 8: Add strict lint/type tooling and enforce in project config

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`

**Step 1: Write the failing test/check**

Add missing-tooling checks to docs and run commands:

Run: `uv run --group dev ruff check . && uv run --group dev ruff format --check . && uv run --group dev pyright`  
Expected: FAIL initially because dependencies/config are missing.

**Step 2: Run check to verify it fails**

Run: `uv run --group dev ruff check .`  
Expected: command/dependency/config failure.

**Step 3: Write minimal implementation**

Add to `pyproject.toml`:

```toml
[dependency-groups]
dev = ["pytest>=8.2.0", "ruff>=0.6.0", "pyright>=1.1.0"]

[tool.ruff]
line-length = 100
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B"]

[tool.pyright]
pythonVersion = "3.10"
typeCheckingMode = "standard"
include = ["src", "tests", "main.py", "circuit.py", "estimators.py", "evaluate.py"]
```

Update README with exact quality-gate commands.

**Step 4: Run checks to verify they pass**

Run:
- `uv run --group dev ruff check .`
- `uv run --group dev ruff format --check .`
- `uv run --group dev pyright`

Expected: all PASS.

**Step 5: Commit**

```bash
git add pyproject.toml README.md
git commit -m "chore: add strict lint and type-check quality gates"
```

### Task 9: Enrich test comments and add regression coverage for behavior fixes

**Files:**
- Modify: `tests/test_domain.py`
- Modify: `tests/test_generation.py`
- Modify: `tests/test_simulation.py`
- Modify: `tests/test_estimators.py`
- Modify: `tests/test_scoring.py`

**Step 1: Write the failing regression test**

```python
def test_score_estimator_raises_when_estimator_stops_early():
    # Regression: estimator yielding fewer layers than max_depth must fail loudly.
    ...
```

**Step 2: Run targeted tests to verify failure**

Run: `uv run --group dev pytest tests/test_scoring.py::test_score_estimator_raises_when_estimator_stops_early -v`  
Expected: FAIL before guard is implemented.

**Step 3: Write minimal implementation**

Implement missing guard in `src/circuit_estimation/scoring.py` and add explanatory comments in tests:
- What contest rule each test enforces.
- Why this failure mode matters for participant submissions.

**Step 4: Run full test suite**

Run:
- `uv run --group dev pytest -m "not exhaustive" -q`
- `uv run --group dev pytest -m exhaustive -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_domain.py tests/test_generation.py tests/test_simulation.py tests/test_estimators.py tests/test_scoring.py src/circuit_estimation/scoring.py
git commit -m "test: enrich docs and add scoring regression coverage"
```

### Task 10: Update docs with key decisions and finalize release verification

**Files:**
- Modify: `README.md`
- Create: `docs/context/python-runtime-refactor-decisions.md`
- Modify: `docs/context/mvp-technical-snapshot.md`

**Step 1: Write the failing doc verification check**

Run: `rg -n "quality gate|failure semantics|extension point|deterministic seed" README.md docs/context/*.md`  
Expected: missing required sections before edits.

**Step 2: Run check to verify it fails**

Run the `rg` command above.  
Expected: at least one required phrase not found.

**Step 3: Write minimal implementation**

Add:
- README sections: participant contract, extension points, failure semantics, deterministic seed policy, strict verification commands.
- Decision log in `docs/context/python-runtime-refactor-decisions.md` including date/owner/decision.
- Snapshot update referencing new package layout.

**Step 4: Run end-to-end verification**

Run:
- `uv run --group dev pytest -q`
- `uv run --group dev ruff check .`
- `uv run --group dev ruff format --check .`
- `uv run --group dev pyright`
- `uv run main.py`

Expected: all checks pass; `main.py` prints a numeric score.

**Step 5: Commit**

```bash
git add README.md docs/context/python-runtime-refactor-decisions.md docs/context/mvp-technical-snapshot.md
git commit -m "docs: publish runtime refactor decisions and release guidance"
```

## Final Verification Gate

Run once before merge:

```bash
uv run --group dev pytest -q
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pyright
uv run main.py
```

Expected: all green, deterministic command behavior, clear failure messages.
