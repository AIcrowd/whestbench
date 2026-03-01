# Streaming-First Estimator API Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make estimator `predict` streaming-only (yield one row per depth), remove `predict_batch`, and align scoring exactly with reference-style per-depth timing semantics.

**Architecture:** Use one explicit participant contract: `predict(circuit, budget)` yields depth rows progressively. Scoring measures cumulative elapsed time at each yield boundary, applies timeout/floor against `time_budget_by_depth_s[i]`, and computes runtime-adjusted loss per depth. In-repo estimators, docs, and tests are migrated to this single-path API.

**Tech Stack:** Python 3.13, NumPy, dataclasses, pytest, ruff, pyright, uv

---

### Task 1: Define Streaming-Only SDK Contract

**Files:**
- Create: `src/circuit_estimation/sdk.py`
- Modify: `src/circuit_estimation/__init__.py`
- Test: `tests/test_sdk.py`

**Step 1: Write the failing test**

```python
from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from circuit_estimation.domain import Circuit, Layer
from circuit_estimation.sdk import BaseEstimator


def _circuit(n: int, d: int) -> Circuit:
    layer = Layer(
        first=np.array([0, 1], dtype=np.int32),
        second=np.array([1, 0], dtype=np.int32),
        first_coeff=np.zeros(n, dtype=np.float32),
        second_coeff=np.zeros(n, dtype=np.float32),
        const=np.zeros(n, dtype=np.float32),
        product_coeff=np.zeros(n, dtype=np.float32),
    )
    return Circuit(n=n, d=d, gates=[layer for _ in range(d)])


class ExampleEstimator(BaseEstimator):
    def predict(self, circuit: Circuit, budget: int) -> Iterator[np.ndarray]:
        for _ in range(circuit.d):
            yield np.zeros((circuit.n,), dtype=np.float32)


def test_predict_streaming_signature_and_depth_rows() -> None:
    est = ExampleEstimator()
    rows = list(est.predict(_circuit(2, 3), budget=10))
    assert len(rows) == 3
    assert all(row.shape == (2,) for row in rows)
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_sdk.py -q`
Expected: FAIL because SDK file and class do not exist.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/sdk.py
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .domain import Circuit


class BaseEstimator(ABC):
    @abstractmethod
    def predict(self, circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
        raise NotImplementedError

    def setup(self, context: object) -> None:
        return None

    def teardown(self) -> None:
        return None
```

Export `BaseEstimator` from `src/circuit_estimation/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_sdk.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/sdk.py src/circuit_estimation/__init__.py tests/test_sdk.py
git commit -m "feat: add streaming-only base estimator contract"
```

---

### Task 2: Add Stream Validation Helpers For Depth Rows

**Files:**
- Create: `src/circuit_estimation/streaming.py`
- Test: `tests/test_streaming.py`

**Step 1: Write the failing test**

```python
from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest

from circuit_estimation.streaming import validate_depth_row


def test_validate_depth_row_accepts_float_vector() -> None:
    row = validate_depth_row(np.array([0.1, -0.2], dtype=np.float32), width=2, depth_index=0)
    assert row.shape == (2,)


def test_validate_depth_row_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        validate_depth_row(np.zeros((2, 1), dtype=np.float32), width=2, depth_index=1)


def test_validate_depth_row_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_depth_row(np.array([np.nan, 0.0], dtype=np.float32), width=2, depth_index=2)
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_streaming.py -q`
Expected: FAIL because helper module does not exist.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/streaming.py
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_depth_row(row: object, *, width: int, depth_index: int) -> NDArray[np.float32]:
    arr = np.asarray(row, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != width:
        raise ValueError(
            f"Estimator row at depth {depth_index} must have shape ({width},), got {arr.shape}."
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Estimator row at depth {depth_index} must contain finite values.")
    return arr
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_streaming.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/streaming.py tests/test_streaming.py
git commit -m "feat: add depth-row validation helpers for streaming estimators"
```

---

### Task 3: Refactor Scoring To Strict Per-Depth Stream Timing

**Files:**
- Modify: `src/circuit_estimation/scoring.py`
- Test: `tests/test_evaluate.py`
- Test: `tests/test_scoring_module.py`

**Step 1: Write failing tests**

```python
def test_streaming_predict_uses_per_depth_cumulative_timing(monkeypatch):
    # baseline depth times [1.0, 2.0], tolerance 0.1
    ...


def test_streaming_predict_too_few_rows_raises() -> None:
    ...


def test_streaming_predict_too_many_rows_raises() -> None:
    ...
```

Also add regression preserving existing scalar report keys plus depth arrays.

**Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_evaluate.py tests/test_scoring_module.py -q`
Expected: FAIL because scorer still assumes single return timing path.

**Step 3: Write minimal implementation**

Implementation outline:

```python
# in per-circuit loop
start_wall = time.time()
start_cpu = time.process_time()
rows: list[np.ndarray] = []

for depth_index, raw_row in enumerate(estimator(circuit, budget)):
    if depth_index >= depth:
        raise ValueError("Estimator emitted more than max_depth rows.")

    elapsed = time.time() - start_wall
    row = validate_depth_row(raw_row, width=width, depth_index=depth_index)

    baseline_time = float(baseline_times[depth_index])
    timed_out = elapsed > (1.0 + tolerance) * baseline_time
    floored = elapsed < (1.0 - tolerance) * baseline_time
    effective_time = max(elapsed, (1.0 - tolerance) * baseline_time)

    if timed_out:
        row = np.zeros_like(row)

    rows.append(row)
    effective_time_sums_by_depth[depth_index] += effective_time
    timeout_counts_by_depth[depth_index] += float(timed_out)
    floor_counts_by_depth[depth_index] += float(floored)

if len(rows) != depth:
    raise ValueError("Estimator must emit exactly max_depth rows.")

cpu_elapsed = time.process_time() - start_cpu
output_tensor = np.stack(rows, axis=0).astype(np.float32)
```

Type alias update:

```python
EstimatorFn = Callable[[Circuit, int], Iterator[NDArray[np.float32]]]
```

Keep `by_budget_raw` field names stable where possible, plus depth runtime arrays.

**Step 4: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_evaluate.py tests/test_scoring_module.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_evaluate.py tests/test_scoring_module.py
git commit -m "refactor: score estimators from streamed depth outputs"
```

---

### Task 4: Migrate In-Repo Estimators To Streaming Predict

**Files:**
- Modify: `src/circuit_estimation/estimators.py`
- Test: `tests/test_estimators.py`
- Test: `tests/test_estimators_module.py`

**Step 1: Write failing tests**

```python
def test_mean_propagation_yields_one_row_per_depth() -> None:
    ...


def test_covariance_propagation_yields_one_row_per_depth() -> None:
    ...


def test_combined_estimator_is_streaming_and_budget_switched() -> None:
    ...
```

**Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_estimators.py tests/test_estimators_module.py -q`
Expected: FAIL because estimators currently return full tensors.

**Step 3: Write minimal implementation**

- Convert estimators to generators yielding `(width,)` rows per depth.
- Keep any existing function names to minimize import churn.
- Ensure combined estimator uses explicit budget-switch `if` in `predict` path and `yield from` chosen strategy.

**Step 4: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_estimators.py tests/test_estimators_module.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/estimators.py tests/test_estimators.py tests/test_estimators_module.py
git commit -m "refactor: migrate reference estimators to streaming outputs"
```

---

### Task 5: Remove `predict_batch` References Everywhere

**Files:**
- Modify: `README.md`
- Modify: `docs/context/mvp-technical-snapshot.md`
- Modify: `docs/context/python-runtime-refactor-decisions.md`
- Modify: `docs/plans/2026-03-01-estimator-sdk-and-runner-decoupling-design.md`
- Modify: `tests/test_docs_quality.py`

**Step 1: Write failing test**

```python
def test_docs_do_not_reference_predict_batch_contract() -> None:
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "predict_batch" not in text
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_docs_quality.py::test_docs_do_not_reference_predict_batch_contract -q`
Expected: FAIL before docs are updated.

**Step 3: Write minimal implementation**

- Remove all participant-facing mentions of `predict_batch`.
- Update contract to streaming-only `predict`.
- Standardize terminology to "budget-by-depth".

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_docs_quality.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add README.md docs/context/mvp-technical-snapshot.md docs/context/python-runtime-refactor-decisions.md docs/plans/2026-03-01-estimator-sdk-and-runner-decoupling-design.md tests/test_docs_quality.py
git commit -m "docs: remove predict_batch and standardize streaming estimator contract"
```

---

### Task 6: Update Reporting/CLI Expectations For Streamed Runtime Semantics

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Modify: `tests/test_reporting.py`
- Modify: `tests/test_cli.py`

**Step 1: Write failing tests**

```python
def test_human_report_describes_streamed_depth_runtime_semantics() -> None:
    rendered = render_human_report(sample_report())
    assert "budget-by-depth" in rendered.lower()


def test_agent_mode_schema_keeps_stream_runtime_fields() -> None:
    payload = json.loads(render_agent_report(sample_report()))
    row = payload["results"]["by_budget_raw"][0]
    assert "time_budget_by_depth_s" in row
```

**Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_reporting.py tests/test_cli.py -q`
Expected: FAIL before rendering language/schema assertions are aligned.

**Step 3: Write minimal implementation**

- Keep human dashboard readable and compact.
- Ensure runtime plot labels and explanatory text align with streaming depth semantics.
- Preserve strict JSON-only behavior in `--agent-mode`.

**Step 4: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_reporting.py tests/test_cli.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py tests/test_cli.py
git commit -m "refactor: align reporting and cli expectations with streaming scoring semantics"
```

---

### Task 6A: Add CLI Contract Validation For Streaming Predict

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_cli_fallback.py`

**Step 1: Write failing tests**

```python
def test_cli_human_mode_surfaces_stream_contract_errors_readably() -> None:
    ...


def test_cli_agent_mode_surfaces_stream_contract_errors_as_json() -> None:
    ...
```

Cases to cover:
- estimator emits too few rows,
- estimator emits too many rows,
- estimator emits wrong row shape.

**Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_cli.py tests/test_cli_fallback.py -q`
Expected: FAIL before stream-contract error mapping is wired.

**Step 3: Write minimal implementation**

- Ensure CLI catches stream-contract `ValueError`s from scorer and renders:
  - human mode: concise actionable message including depth index/context.
  - agent mode: JSON object with stable fields (`stage`, `code`, `message`).
- Keep current default dashboard behavior unchanged on success.

**Step 4: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_cli.py tests/test_cli_fallback.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py tests/test_cli.py tests/test_cli_fallback.py
git commit -m "feat: add cli stream-contract validation error surfacing"
```

---

### Task 6B: Add Participant-Facing Streaming Guide

**Files:**
- Create: `docs/context/participant-streaming-estimator-guide.md`
- Modify: `README.md`
- Modify: `tests/test_docs_quality.py`

**Step 1: Write failing test**

```python
def test_readme_links_streaming_participant_guide() -> None:
    text = Path("README.md").read_text(encoding="utf-8").lower()
    assert "participant-streaming-estimator-guide.md" in text
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_docs_quality.py::test_readme_links_streaming_participant_guide -q`
Expected: FAIL before guide is added and linked.

**Step 3: Write minimal implementation**

Create `participant-streaming-estimator-guide.md` with:
- minimal estimator skeleton using `yield`,
- common mistakes (flush-at-end, wrong row shape, wrong row count),
- budget-by-depth tuning workflow,
- debugging checklist for local runs.

Link this guide from README under estimator extension section.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_docs_quality.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/context/participant-streaming-estimator-guide.md README.md tests/test_docs_quality.py
git commit -m "docs: add participant streaming estimator guide"
```

---

### Task 7: Add Edge-Case Guardrails For Generator Misbehavior

**Files:**
- Modify: `src/circuit_estimation/scoring.py`
- Test: `tests/test_scoring_module.py`

**Step 1: Write failing tests**

```python
def test_generator_exception_after_partial_rows_is_structured_error() -> None:
    ...


def test_non_iterable_predict_output_raises_clear_error() -> None:
    ...


def test_row_dtype_cast_and_finite_validation_are_enforced() -> None:
    ...
```

**Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_scoring_module.py -q`
Expected: FAIL before hardened validation/error handling.

**Step 3: Write minimal implementation**

- Wrap iterator consumption with clear stage-aware error messages.
- Differentiate validation failures from runtime exceptions.
- Keep structured report behavior unchanged in agent mode.

**Step 4: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_scoring_module.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring_module.py
git commit -m "fix: harden streaming estimator edge-case handling"
```

---

### Task 8: Verification Before Completion

**Files:**
- Modify: none (verification-only)

**Step 1: Run full quality gates**

Run:
- `uv run --group dev ruff check .`
- `uv run --group dev ruff format --check .`
- `uv run --group dev pyright`
- `uv run --group dev pytest -m "not exhaustive"`
- `uv run --group dev pytest -m exhaustive`

Expected: all PASS.

**Step 2: Run focused streaming parity suite**

Run: `uv run --group dev pytest tests/test_evaluate.py tests/test_scoring_module.py tests/test_estimators.py tests/test_estimators_module.py -q`
Expected: PASS with streaming semantics coverage.

**Step 3: Request review before merge**

Use @requesting-code-review with summary:
- streaming-only `predict` API
- no `predict_batch`
- scorer uses per-depth emitted timing
- docs and reports updated to budget-by-depth semantics

**Step 4: Commit verification marker (optional)**

```bash
git commit --allow-empty -m "chore: verify streaming-only estimator api refactor"
```

---

## Scope Guards (YAGNI)

- Do not implement subprocess/cloud runner streaming transport in this slice.
- Do not build a compatibility wrapper for old tensor-return participant code.
- Do not alter competition budget policy beyond streaming semantics alignment.

## Risks And Mitigations

- Risk: participants unfamiliar with generators.
  - Mitigation: include 2-3 fully commented starter estimator examples.
- Risk: shape/count runtime bugs increase support burden.
  - Mitigation: strict validation with explicit depth-indexed error messages.
- Risk: docs drift from scorer behavior.
  - Mitigation: add doc contract tests and keep wording assertions in CI.

## Rollout Notes

- Phase 1: local runtime + scorer migrated to streaming-only predict.
- Phase 2: sandbox/cloud runner can reuse same scoring semantics by timing stream frame arrivals.
- Phase 3: challenge release with clear streaming examples and budget-by-depth tuning docs.
