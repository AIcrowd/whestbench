# Streaming Scoring Unification Implementation Plan

> **For Antigravity:** REQUIRED WORKFLOW: Use `.agent/workflows/execute-plan.md` to execute this plan in single-flow mode.

**Goal:** Make all runners stream per-depth prediction outcomes. Remove the call-level `score_submission_report`. Unify on a single per-depth streaming scoring path. Update all docs.

**Architecture:** Change `EstimatorRunner.predict()` to return `Iterator[DepthRowOutcome]` instead of `PredictOutcome`. Both `InProcessRunner` and `SubprocessRunner` yield one outcome per depth row with cumulative wall time. The scoring function iterates this stream, applying per-depth timeout/floor. `score_submission_report` is deleted. `cestim run` routes through the unified `score_estimator_report`.

**Tech Stack:** Python 3.13, NumPy, dataclasses, pytest, ruff, pyright, uv

---

### Task 1: Add `DepthRowOutcome` Data Type

**Files:**
- Modify: `src/circuit_estimation/runner.py`
- Test: `tests/test_runner_types.py`

**Step 1: Write the failing test**

```python
# tests/test_runner_types.py — append to existing file
import numpy as np
from circuit_estimation.runner import DepthRowOutcome

def test_depth_row_outcome_ok_status():
    row = np.array([0.1, -0.2], dtype=np.float32)
    outcome = DepthRowOutcome(depth_index=0, row=row, wall_time_s=0.01, status="ok")
    assert outcome.status == "ok"
    assert outcome.row is not None
    assert outcome.depth_index == 0
    assert outcome.wall_time_s == 0.01
    assert outcome.error_message is None

def test_depth_row_outcome_error_status():
    outcome = DepthRowOutcome(
        depth_index=2, row=None, wall_time_s=0.5,
        status="error", error_message="boom"
    )
    assert outcome.status == "error"
    assert outcome.row is None
    assert outcome.error_message == "boom"
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_runner_types.py::test_depth_row_outcome_ok_status tests/test_runner_types.py::test_depth_row_outcome_error_status -q`
Expected: FAIL because `DepthRowOutcome` doesn't exist.

**Step 3: Write minimal implementation**

Add to `src/circuit_estimation/runner.py`:

```python
DepthRowStatus = Literal["ok", "error"]

@dataclass(slots=True)
class DepthRowOutcome:
    """Per-depth-row outcome yielded by streaming runner predict."""
    depth_index: int
    row: NDArray[np.float32] | None
    wall_time_s: float
    status: DepthRowStatus
    error_message: str | None = None
```

Export `DepthRowOutcome` and `DepthRowStatus` from `src/circuit_estimation/__init__.py`.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_runner_types.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/runner.py src/circuit_estimation/__init__.py tests/test_runner_types.py
git commit -m "feat: add DepthRowOutcome data type for streaming runner predict"
```

---

### Task 2: Change `EstimatorRunner` Protocol to Streaming

**Files:**
- Modify: `src/circuit_estimation/runner.py`

**Step 1: No test needed** (protocol change only, implementations follow in Tasks 3-4)

**Step 2: Write minimal implementation**

Update the `EstimatorRunner` protocol:

```python
class EstimatorRunner(Protocol):
    """Protocol for in-process, subprocess, and future cloud runners."""

    def start(
        self,
        entrypoint: EstimatorEntrypoint,
        context: SetupContext,
        limits: ResourceLimits,
    ) -> None: ...

    def predict(self, circuit: Circuit, budget: int) -> Iterator[DepthRowOutcome]: ...

    def close(self) -> None: ...
```

Remove `predict_batch` from the protocol. Add `from collections.abc import Iterator` import if not present.

**Step 3: Commit**

```bash
git add src/circuit_estimation/runner.py
git commit -m "refactor: change EstimatorRunner.predict to return Iterator[DepthRowOutcome]"
```

---

### Task 3: Refactor `InProcessRunner.predict` to Stream

**Files:**
- Modify: `src/circuit_estimation/runner.py`
- Modify: `tests/test_inprocess_runner.py`

**Step 1: Write the failing test**

Replace existing tests to expect streaming:

```python
def test_inprocess_runner_streams_depth_rows(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int):
                for i in range(circuit.d):
                    yield np.full((circuit.n,), float(i), dtype=np.float32)
        """,
    )
    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), _setup_context(), _limits())
    outcomes = list(runner.predict(_sample_circuit(), budget=10))

    assert len(outcomes) == 1  # _sample_circuit has d=1
    assert outcomes[0].status == "ok"
    assert outcomes[0].depth_index == 0
    assert outcomes[0].row is not None
    np.testing.assert_allclose(outcomes[0].row, [0.0, 0.0])
    assert outcomes[0].wall_time_s >= 0.0


def test_inprocess_runner_streams_error_on_runtime_exception(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int):
                raise RuntimeError("boom")
        """,
    )
    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), _setup_context(), _limits())
    outcomes = list(runner.predict(_sample_circuit(), budget=10))

    assert len(outcomes) == 1
    assert outcomes[0].status == "error"
    assert outcomes[0].row is None
    assert "boom" in outcomes[0].error_message


def test_inprocess_runner_streams_cumulative_wall_times(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import time
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int):
                for _ in range(circuit.d):
                    time.sleep(0.01)
                    yield np.zeros((circuit.n,), dtype=np.float32)
        """,
    )
    ctx = SetupContext(width=2, max_depth=3, budgets=(10,), time_tolerance=0.1, api_version="1.0")
    circuit = make_circuit(2, [make_layer([0,1],[1,0],[1,1],[0,0],[0,0],[0,0])] * 3)
    runner = InProcessRunner()
    runner.start(EstimatorEntrypoint(file_path=module_path), ctx, _limits())
    outcomes = list(runner.predict(circuit, budget=10))

    assert len(outcomes) == 3
    # wall times should be monotonically increasing (cumulative)
    for i in range(1, len(outcomes)):
        assert outcomes[i].wall_time_s >= outcomes[i-1].wall_time_s
```

**Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_inprocess_runner.py -q`
Expected: FAIL because `predict` still returns `PredictOutcome`.

**Step 3: Write minimal implementation**

Rewrite `InProcessRunner.predict`:

```python
def predict(self, circuit: Circuit, budget: int) -> Iterator[DepthRowOutcome]:
    if not self._started or self._estimator is None or self._limits is None:
        raise RunnerError(
            "predict",
            RunnerErrorDetail(
                code="RUNNER_NOT_STARTED",
                message="Runner must be started before calling predict.",
            ),
        )

    start_wall = time.time()
    try:
        raw_predictions = self._estimator.predict(circuit, budget)
    except Exception as exc:
        yield DepthRowOutcome(
            depth_index=0, row=None,
            wall_time_s=time.time() - start_wall,
            status="error", error_message=str(exc),
        )
        return

    try:
        output_iter = iter(raw_predictions)
    except TypeError as exc:
        yield DepthRowOutcome(
            depth_index=0, row=None,
            wall_time_s=time.time() - start_wall,
            status="error",
            error_message="Estimator must return an iterator of depth-row outputs.",
        )
        return

    for depth_index in range(circuit.d):
        try:
            raw_row = next(output_iter)
        except StopIteration:
            yield DepthRowOutcome(
                depth_index=depth_index, row=None,
                wall_time_s=time.time() - start_wall,
                status="error",
                error_message="Estimator must emit exactly max_depth rows.",
            )
            return
        except Exception as exc:
            yield DepthRowOutcome(
                depth_index=depth_index, row=None,
                wall_time_s=time.time() - start_wall,
                status="error",
                error_message=f"Estimator stream failed at depth {depth_index}: {exc}",
            )
            return

        elapsed = time.time() - start_wall
        try:
            row = validate_depth_row(raw_row, width=circuit.n, depth_index=depth_index)
        except ValueError as exc:
            yield DepthRowOutcome(
                depth_index=depth_index, row=None,
                wall_time_s=elapsed, status="error",
                error_message=str(exc),
            )
            return

        yield DepthRowOutcome(
            depth_index=depth_index, row=row,
            wall_time_s=elapsed, status="ok",
        )

    # Check for extra rows
    try:
        _extra = next(output_iter)
    except StopIteration:
        pass
    except Exception:
        pass
    else:
        pass  # Silently ignore extra rows (scoring will only use first d)
```

Remove `predict_batch` from `InProcessRunner`. Remove old `_collect_prediction_tensor` from `runner.py`.

**Step 4: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_inprocess_runner.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/runner.py tests/test_inprocess_runner.py
git commit -m "refactor: InProcessRunner.predict streams DepthRowOutcome per depth"
```

---

### Task 4: Refactor `SubprocessRunner` + Worker for Per-Depth Streaming

**Files:**
- Modify: `src/circuit_estimation/subprocess_worker.py`
- Modify: `src/circuit_estimation/runner.py` (SubprocessRunner)
- Modify: `tests/test_subprocess_runner.py`

**Step 1: Write the failing test**

```python
def test_subprocess_runner_streams_depth_rows(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int):
                for i in range(circuit.d):
                    yield np.full((circuit.n,), float(i), dtype=np.float32)
        """,
    )
    runner = SubprocessRunner()
    runner.start(
        EstimatorEntrypoint(file_path=module_path), _context(),
        ResourceLimits(setup_timeout_s=2.0, predict_timeout_s=5.0, memory_limit_mb=256),
    )
    outcomes = list(runner.predict(_sample_circuit(), budget=10))
    runner.close()

    assert len(outcomes) == 1  # d=1
    assert outcomes[0].status == "ok"
    assert outcomes[0].depth_index == 0
    assert outcomes[0].row is not None
    np.testing.assert_allclose(outcomes[0].row, [0.0, 0.0])
    assert outcomes[0].wall_time_s >= 0.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_subprocess_runner.py::test_subprocess_runner_streams_depth_rows -q`
Expected: FAIL.

**Step 3: Write minimal implementation**

**subprocess_worker.py** — change predict handler:

```python
# In main(), replace the predict block with:
elif command == "predict":
    if estimator is None:
        _write_response({"status": "error", "depth_index": 0, "error_message": "Estimator not initialized."})
        _write_response({"status": "done"})
        continue
    try:
        circuit = _payload_to_circuit(request["circuit"])
        budget = int(request["budget"])
        raw_predictions = estimator.predict(circuit, budget)
        try:
            output_iter = iter(raw_predictions)
        except TypeError:
            _write_response({"status": "error", "depth_index": 0, "error_message": "Estimator must return an iterator."})
            _write_response({"status": "done"})
            continue
        for depth_index in range(circuit.d):
            try:
                raw_row = next(output_iter)
            except StopIteration:
                _write_response({"status": "error", "depth_index": depth_index, "error_message": "Too few rows."})
                _write_response({"status": "done"})
                break
            except Exception as exc:
                _write_response({"status": "error", "depth_index": depth_index, "error_message": str(exc)})
                _write_response({"status": "done"})
                break
            try:
                row = validate_depth_row(raw_row, width=circuit.n, depth_index=depth_index)
            except ValueError as exc:
                _write_response({"status": "error", "depth_index": depth_index, "error_message": str(exc)})
                _write_response({"status": "done"})
                break
            _write_response({"status": "row", "depth_index": depth_index, "row": row.tolist()})
        else:
            _write_response({"status": "done"})
    except Exception as exc:
        _write_response({"status": "error", "depth_index": 0, "error_message": str(exc)})
        _write_response({"status": "done"})
```

Remove `_collect_prediction_tensor` from `subprocess_worker.py`.

**SubprocessRunner.predict** — read streaming responses:

```python
def predict(self, circuit: Circuit, budget: int) -> Iterator[DepthRowOutcome]:
    # Send predict request
    start_wall = time.time()
    try:
        self._send_request({
            "command": "predict",
            "budget": int(budget),
            "circuit": _circuit_to_payload(circuit),
        })
    except RunnerError as exc:
        yield DepthRowOutcome(
            depth_index=0, row=None,
            wall_time_s=time.time() - start_wall,
            status="error", error_message=exc.detail.message,
        )
        return

    # Read streaming responses
    while True:
        try:
            response = self._read_response(timeout_s=self._limits.predict_timeout_s)
        except TimeoutError:
            self._terminate_process()
            self._started = False
            yield DepthRowOutcome(
                depth_index=-1, row=None,
                wall_time_s=time.time() - start_wall,
                status="error", error_message="predict timed out.",
            )
            return
        except RunnerError as exc:
            yield DepthRowOutcome(
                depth_index=-1, row=None,
                wall_time_s=time.time() - start_wall,
                status="error", error_message=exc.detail.message,
            )
            return

        status = response.get("status")
        if status == "done":
            return
        elif status == "row":
            elapsed = time.time() - start_wall
            row_data = response.get("row")
            row = np.asarray(row_data, dtype=np.float32) if row_data is not None else None
            yield DepthRowOutcome(
                depth_index=int(response.get("depth_index", -1)),
                row=row,
                wall_time_s=elapsed,
                status="ok",
            )
        elif status == "error":
            elapsed = time.time() - start_wall
            yield DepthRowOutcome(
                depth_index=int(response.get("depth_index", -1)),
                row=None,
                wall_time_s=elapsed,
                status="error",
                error_message=str(response.get("error_message", "unknown worker error")),
            )
            # Continue reading until "done"
        else:
            yield DepthRowOutcome(
                depth_index=-1, row=None,
                wall_time_s=time.time() - start_wall,
                status="error",
                error_message=f"Unknown worker response status: {status}",
            )
            return
```

Remove `predict_batch` from `SubprocessRunner`.

**Step 4: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_subprocess_runner.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/subprocess_worker.py src/circuit_estimation/runner.py tests/test_subprocess_runner.py
git commit -m "refactor: SubprocessRunner streams DepthRowOutcome via per-depth JSON lines"
```

---

### Task 5: Unify Scoring — Delete `score_submission_report`, Adapt `score_estimator_report` to Accept Runners

**Files:**
- Modify: `src/circuit_estimation/scoring.py`
- Modify: `tests/test_scoring_module.py`

**Step 1: Write the failing test**

```python
def test_score_estimator_report_accepts_runner_stream():
    """score_estimator_report works with an EstimatorRunner that streams DepthRowOutcome."""
    from circuit_estimation.runner import DepthRowOutcome

    class FakeStreamingRunner:
        def start(self, entrypoint, context, limits):
            self.started = True
        def predict(self, circuit, budget):
            for i in range(circuit.d):
                row = np.zeros(circuit.n, dtype=np.float32)
                yield DepthRowOutcome(depth_index=i, row=row, wall_time_s=0.001 * (i + 1), status="ok")
        def close(self):
            pass

    params = ContestParams(width=4, max_depth=2, budgets=[10], time_tolerance=0.1)
    circuit = _constant_circuit(4, 2)
    report = score_estimator_report(
        FakeStreamingRunner(),
        n_circuits=1, n_samples=10,
        contest_params=params, circuits=[circuit],
    )
    assert "final_score" in report["results"]
    row = report["results"]["by_budget_raw"][0]
    assert "time_budget_by_depth_s" in row
    assert "timeout_rate_by_depth" in row
    assert len(row["mse_by_layer"]) == 2
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_scoring_module.py::test_score_estimator_report_accepts_runner_stream -q`
Expected: FAIL because `score_estimator_report` currently only accepts `EstimatorFn`.

**Step 3: Write minimal implementation**

Refactor `score_estimator_report` to accept either an `EstimatorFn` (raw callable) or an `EstimatorRunner`. When given a runner, iterate its streaming `predict()`. When given a callable, wrap it in a lightweight adapter that yields `DepthRowOutcome`.

Delete `score_submission_report` entirely.

Update the per-circuit loop in `score_estimator_report`:

```python
# Instead of directly calling estimator(circuit, budget) and iterating raw rows,
# accept a runner and iterate runner.predict(circuit, budget) for DepthRowOutcome.
# Apply the same per-depth timeout/floor logic using outcome.wall_time_s.
```

Key scoring logic (unchanged from existing `score_estimator_report`):
- Compare `outcome.wall_time_s` against `baseline_times[depth_index]`
- Timeout: if wall_time > baseline * (1 + tolerance), zero the row
- Floor: if wall_time < baseline * (1 - tolerance), floor effective time
- MSE computed per-depth as before

**Step 4: Run all scoring tests to verify they pass**

Run: `uv run --group dev pytest tests/test_scoring_module.py -q`
Expected: PASS (some old tests referencing `score_submission_report` need removal/adaptation).

**Step 5: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring_module.py
git commit -m "refactor: unify scoring on streaming per-depth path, remove score_submission_report"
```

---

### Task 6: Rewire CLI to Use Unified Scoring

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_cli_fallback.py`
- Modify: `tests/test_cli_participant_commands.py`

**Step 1: Write the failing test**

```python
def test_cli_run_uses_unified_scoring(monkeypatch, tmp_path):
    """cestim run calls score_estimator_report, not score_submission_report."""
    import circuit_estimation.cli as cli
    assert not hasattr(cli, "score_submission_report") or "score_submission_report" not in dir(cli)
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_cli.py::test_cli_run_uses_unified_scoring -q`
Expected: FAIL because cli still imports `score_submission_report`.

**Step 3: Write minimal implementation**

In `cli.py`:
- Change import: `from .scoring import ContestParams, score_estimator_report` (remove `score_submission_report`)
- In the `run` command block (~line 318-332): replace `score_submission_report(runner, ...)` with `score_estimator_report(runner, ...)` (same args, the function now accepts runners)

Update test monkeypatches in `test_cli.py`, `test_cli_fallback.py`, `test_cli_participant_commands.py` to patch `score_estimator_report` instead of `score_submission_report`.

**Step 4: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_cli.py tests/test_cli_fallback.py tests/test_cli_participant_commands.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py tests/test_cli.py tests/test_cli_fallback.py tests/test_cli_participant_commands.py
git commit -m "refactor: rewire cestim run to unified streaming scoring"
```

---

### Task 7: Clean Up Dead Code

**Files:**
- Modify: `src/circuit_estimation/runner.py`
- Modify: `src/circuit_estimation/__init__.py`
- Modify: `tests/test_scoring_module.py`

**Step 1: Remove leftover dead code**

- Remove `PredictOutcome` class from `runner.py` (replaced by `DepthRowOutcome`)
- Remove `_collect_prediction_tensor` from `runner.py`
- Remove any remaining `predict_batch` references
- Remove old `_FakeRunner` class from `test_scoring_module.py` that used `PredictOutcome`
- Update `__init__.py` exports

**Step 2: Run full test suite**

Run: `uv run --group dev pytest -m "not exhaustive" -q`
Expected: PASS.

**Step 3: Commit**

```bash
git add -A
git commit -m "chore: remove PredictOutcome, _collect_prediction_tensor, predict_batch dead code"
```

---

### Task 8: Update Docs to Per-Depth Streaming Semantics

**Files:**
- Modify: `docs/concepts/scoring-model.md`
- Modify: `docs/reference/score-report-fields.md`
- Modify: `docs/reference/estimator-contract.md`
- Modify: `README.md`

**Step 1: Update scoring-model.md**

Restore per-depth streaming language:
- Line 26: "Your `predict(circuit, budget)` is called, and each yielded depth row is timed against the corresponding sampling baseline."
- Line 48: "If your estimator's cumulative runtime at depth `i` exceeds the sampling baseline at that depth (above tolerance), that depth row is zeroed — incurring maximum error for that depth."

**Step 2: Update score-report-fields.md**

Add back per-depth timing fields to the table:

| Field | Description |
|---|---|
| `time_budget_by_depth_s` | Per-depth sampling baseline time (array of length `d`) |
| `time_ratio_by_depth_mean` | Your time / baseline time at each depth (averaged across circuits) |
| `effective_time_s_by_depth_mean` | Your effective runtime per depth after floor (averaged) |
| `timeout_rate_by_depth` | Fraction of circuits timed out at each depth |
| `time_floor_rate_by_depth` | Fraction of circuits floored at each depth |

Keep the existing call-level aggregate fields too (they are computed from per-depth data).

**Step 3: Update estimator-contract.md**

Add a note: "Timing is measured at each `yield` boundary — your cumulative wall time at depth `i` is compared against the sampling baseline at depth `i`."

**Step 4: Update README.md scoring description if needed**

Verify the scoring summary in README still makes sense with per-depth semantics.

**Step 5: Commit**

```bash
git add docs/ README.md
git commit -m "docs: update all docs to per-depth streaming scoring semantics"
```

---

### Task 9: Full Verification

**Step 1: Run quality gates**

Run:
- `uv run --group dev ruff check .`
- `uv run --group dev ruff format --check .`
- `uv run --group dev pyright`
- `uv run --group dev pytest -m "not exhaustive" -q`

Expected: all PASS.

**Step 2: Run focused streaming tests**

Run: `uv run --group dev pytest tests/test_inprocess_runner.py tests/test_subprocess_runner.py tests/test_scoring_module.py tests/test_cli.py -q`
Expected: PASS.

**Step 3: Run an actual estimator end-to-end**

Run: `uv run cestim run --estimator examples/estimators/mean_propagation.py --n-circuits 1`
Expected: Report with per-depth fields visible.

**Step 4: Commit verification marker**

```bash
git commit --allow-empty -m "chore: verify streaming scoring unification complete"
```
