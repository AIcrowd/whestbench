# Budget & Time Exhaustion Warning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn silent budget/time exhaustion in `scoring.evaluate_estimator()` into visible Python warnings + per-MLP tracebacks, while honoring `--fail-fast` and keeping default exit code 0.

**Architecture:** Add three `UserWarning` subclasses (`ScoringExhaustionWarning` base + `BudgetExhaustionWarning`/`TimeExhaustionWarning` leaves) co-located with the only emitter in `scoring.py`. Modify the two silent `except` blocks in `evaluate_estimator()` to (a) re-raise under `fail_fast`, (b) capture the traceback into a new `exhaustion_traceback` local, and (c) call `warnings.warn()` with a one-line message. Extend the success-path per-MLP dict with a unified `"traceback"` key. In `cli.py`, install a Rich-aware `showwarning` handler around the `evaluate_estimator` call site, suppress warnings entirely in `--json` mode, and emit an end-of-run summary line in non-JSON modes.

**Tech Stack:** Python `warnings` module, pytest (`pytest.warns`, `capsys`), Rich (`rich.console.Console.log` above active `Live` regions), existing `_tiny_run_argv` / `_write_tiny_dataset` test helpers.

**Spec:** [docs/superpowers/specs/2026-05-20-budget-exhaustion-warning-design.md](../specs/2026-05-20-budget-exhaustion-warning-design.md)

---

## File Structure

- **Modify** `src/whestbench/scoring.py`
  - Add `import warnings` (top).
  - Add three warning classes near other module-level types (above `ContestSpec`).
  - Add `exhaustion_traceback: Optional[str] = None` to the per-iteration locals at line ~495.
  - Replace `BudgetExhaustedError` handler at lines 523-537 with the new version.
  - Replace `TimeExhaustedError` handler at lines 538-551 with the new version.
  - Add `"traceback": exhaustion_traceback` key to the success-path per-MLP dict at lines 716-734.

- **Modify** `src/whestbench/__init__.py`
  - Re-export the three new warning classes from `scoring`.

- **Modify** `src/whestbench/cli.py`
  - In `_run_estimator_with_runner` (line 1007), wrap the `evaluate_estimator` call with a context that installs a Rich-aware `showwarning` handler. Suppress warnings under `--json`.
  - After the call returns, in non-JSON modes only, emit a summary line if any MLP exhausted.

- **Modify** `tests/test_cli_participant_commands.py`
  - Update existing `test_run_budget_exhausted_does_not_set_exit_1` (line 1256).

- **Create** `tests/test_scoring_warnings.py`
  - Library-level tests (no CLI): `test_budget_exhaustion_emits_warning`, `test_time_exhaustion_emits_warning`, `test_fail_fast_reraises_budget_exhausted`, `test_fail_fast_reraises_time_exhausted`, `test_traceback_in_per_mlp_entry_on_exhaustion`.

- **Create** new tests in `tests/test_cli_participant_commands.py`:
  - `test_json_mode_silences_exhaustion_warning`
  - `test_plain_mode_emits_summary_line`
  - `test_fail_fast_exits_nonzero_on_budget_exhaustion`

---

## Task 1: Add warning classes and public re-exports

**Files:**
- Create: `tests/test_scoring_warnings.py`
- Modify: `src/whestbench/scoring.py` (top of file: add `import warnings`, then three classes before `ContestSpec`)
- Modify: `src/whestbench/__init__.py` (add re-exports)

- [ ] **Step 1: Write the failing test**

Create `tests/test_scoring_warnings.py`:

```python
from __future__ import annotations

import whestbench
from whestbench.scoring import (
    BudgetExhaustionWarning,
    ScoringExhaustionWarning,
    TimeExhaustionWarning,
)


def test_warning_class_hierarchy() -> None:
    """ScoringExhaustionWarning is a UserWarning; budget/time inherit from it."""
    assert issubclass(ScoringExhaustionWarning, UserWarning)
    assert issubclass(BudgetExhaustionWarning, ScoringExhaustionWarning)
    assert issubclass(TimeExhaustionWarning, ScoringExhaustionWarning)


def test_warning_classes_reexported_from_package() -> None:
    """Library users can import the classes from the top-level package."""
    assert whestbench.ScoringExhaustionWarning is ScoringExhaustionWarning
    assert whestbench.BudgetExhaustionWarning is BudgetExhaustionWarning
    assert whestbench.TimeExhaustionWarning is TimeExhaustionWarning
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_warnings.py -v`

Expected: FAIL with `ImportError: cannot import name 'BudgetExhaustionWarning' from 'whestbench.scoring'` (or similar).

- [ ] **Step 3: Add the warning classes to `scoring.py`**

At the top of `src/whestbench/scoring.py`, after the existing imports, add `import warnings` if not already present:

```python
import traceback as _tb
import warnings  # NEW
from dataclasses import dataclass
```

Then add the three classes near the top of the file, before the `ContestSpec` dataclass (just below the imports block, around line 18):

```python
class ScoringExhaustionWarning(UserWarning):
    """Base class for budget/time exhaustion warnings raised during scoring."""


class BudgetExhaustionWarning(ScoringExhaustionWarning):
    """Raised when an estimator exhausts its FLOP budget on a single MLP."""


class TimeExhaustionWarning(ScoringExhaustionWarning):
    """Raised when an estimator exhausts its wall-clock budget on a single MLP."""
```

- [ ] **Step 4: Re-export from `__init__.py`**

Replace the contents of `src/whestbench/__init__.py` with:

```python
"""Core package for WhestBench starter-kit runtime."""

from .domain import MLP
from .generation import sample_mlp
from .scoring import (
    BudgetExhaustionWarning,
    ScoringExhaustionWarning,
    TimeExhaustionWarning,
)
from .sdk import BaseEstimator, SetupContext
from .simulation import relu, run_mlp, run_mlp_all_layers, sample_layer_statistics

__all__ = [
    "BaseEstimator",
    "SetupContext",
    "MLP",
    "BudgetExhaustionWarning",
    "ScoringExhaustionWarning",
    "TimeExhaustionWarning",
    "sample_mlp",
    "relu",
    "run_mlp",
    "run_mlp_all_layers",
    "sample_layer_statistics",
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_scoring_warnings.py -v`

Expected: both tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/whestbench/scoring.py src/whestbench/__init__.py tests/test_scoring_warnings.py
git commit -m "feat(scoring): add ScoringExhaustionWarning hierarchy (#24)"
```

---

## Task 2: BudgetExhaustedError handler emits warning + captures traceback

**Files:**
- Modify: `tests/test_scoring_warnings.py` (add three tests)
- Modify: `src/whestbench/scoring.py:495` (add `exhaustion_traceback` local)
- Modify: `src/whestbench/scoring.py:523-537` (rewrite BudgetExhaustedError handler)
- Modify: `src/whestbench/scoring.py:716-734` (add `"traceback"` key to per-MLP dict)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scoring_warnings.py`. The test helpers mirror the canonical pattern used by [tests/test_scoring_subprocess_residual.py](../../../tests/test_scoring_subprocess_residual.py):

```python
import flopscope as flops
import flopscope.numpy as fnp
import pytest

from whestbench.domain import MLP
from whestbench.sdk import BaseEstimator
from whestbench.scoring import (
    BudgetExhaustionWarning,
    ContestData,
    ContestSpec,
    evaluate_estimator,
)


def _make_tiny_data(width: int = 4, depth: int = 2, n_mlps: int = 2) -> ContestData:
    """Build minimal ContestData for `n_mlps` trivially-tiny MLPs."""
    spec = ContestSpec(
        width=width,
        depth=depth,
        n_mlps=n_mlps,
        flop_budget=1_000,
        ground_truth_samples=10,
        wall_time_limit_s=5.0,
        residual_wall_time_limit_s=5.0,
    )
    weights = [fnp.zeros((width, width), dtype=fnp.float32) for _ in range(depth)]
    mlps = [MLP(width=width, depth=depth, weights=weights) for _ in range(n_mlps)]
    target = fnp.zeros((depth, width), dtype=fnp.float32)
    return ContestData(
        spec=spec,
        mlps=mlps,
        all_layer_targets=[target for _ in range(n_mlps)],
        final_targets=[target[-1] for _ in range(n_mlps)],
        avg_variances=[0.0 for _ in range(n_mlps)],
    )


class _HungryEstimator(BaseEstimator):
    """Estimator that always exhausts the FLOP budget immediately."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        raise flops.BudgetExhaustedError(
            "test", flop_cost=budget + 1, flops_remaining=0
        )


def test_budget_exhaustion_emits_warning() -> None:
    """evaluate_estimator emits BudgetExhaustionWarning once per exhausted MLP."""
    data = _make_tiny_data(n_mlps=2)
    with pytest.warns(BudgetExhaustionWarning) as records:
        result = evaluate_estimator(_HungryEstimator(), data)

    assert len(records) == 2
    msgs = [str(r.message) for r in records]
    assert all("exhausted FLOP budget" in m for m in msgs)
    assert all("estimator output set to zeros" in m for m in msgs)
    # The two warnings reference MLP 0 and MLP 1 respectively.
    assert any("MLP 0" in m for m in msgs)
    assert any("MLP 1" in m for m in msgs)
    # All MLPs are flagged exhausted.
    for entry in result["per_mlp"]:
        assert entry["budget_exhausted"] is True


def test_traceback_in_per_mlp_entry_on_budget_exhaustion() -> None:
    """The per-MLP entry stores the BudgetExhaustedError traceback as a string."""
    data = _make_tiny_data(n_mlps=2)
    with pytest.warns(BudgetExhaustionWarning):
        result = evaluate_estimator(_HungryEstimator(), data)

    for entry in result["per_mlp"]:
        assert entry["budget_exhausted"] is True
        assert isinstance(entry["traceback"], str)
        assert "BudgetExhaustedError" in entry["traceback"]


def test_fail_fast_reraises_budget_exhausted() -> None:
    """Under fail_fast=True, BudgetExhaustedError propagates instead of being captured."""
    data = _make_tiny_data(n_mlps=2)
    with pytest.raises(flops.BudgetExhaustedError):
        evaluate_estimator(_HungryEstimator(), data, fail_fast=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scoring_warnings.py -v`

Expected: the three new tests FAIL.
- `test_budget_exhaustion_emits_warning` → `Failed: DID NOT WARN`
- `test_traceback_in_per_mlp_entry_on_budget_exhaustion` → `KeyError: 'traceback'`
- `test_fail_fast_reraises_budget_exhausted` → `DID NOT RAISE`

If `make_contest` or `ContestSpec` signatures differ, adjust `_tiny_spec()` and `_build_tiny_data()` accordingly — read `src/whestbench/scoring.py` to find the actual signatures and update the helpers. The test intent does not change.

- [ ] **Step 3: Add the `exhaustion_traceback` local**

In `src/whestbench/scoring.py`, locate the per-iteration init block at lines 493-499 (the `for i, mlp in enumerate(data.mlps):` loop body). Add one new line:

```python
    for i, mlp in enumerate(data.mlps):
        flops_used = 0
        budget_exhausted = False
        time_exhausted = False
        residual_wall_time_exhausted = False
        raw_breakdown: Optional[Dict[str, Any]] = None
        normalized_breakdown: Optional[Dict[str, Any]] = None
        exhaustion_traceback: Optional[str] = None  # NEW
```

- [ ] **Step 4: Rewrite the BudgetExhaustedError handler**

Replace lines 523-537 of `src/whestbench/scoring.py`:

```python
        except flops.BudgetExhaustedError:
            if fail_fast:
                raise
            exhaustion_traceback = _tb.format_exc()
            predictions = fnp.zeros((spec.depth, spec.width))
            budget_exhausted = True
            stats = _predict_stats_to_dict(
                last_predict_stats() if callable(last_predict_stats) else None
            )
            if stats is not None:
                flops_used = int(stats.get("flops_used", spec.flop_budget))
                raw_breakdown = stats.get("budget_breakdown")
            if raw_breakdown is None:
                raw_breakdown = budget_ctx.summary_dict(by_namespace=True)
            normalized_breakdown = _normalize_estimator_budget_breakdown(raw_breakdown)
            flops_used = flops_used or spec.flop_budget
            if normalized_breakdown is not None:
                normalized_breakdowns.append(normalized_breakdown)
            warnings.warn(
                f"MLP {i} (depth={spec.depth}, width={spec.width}) exhausted FLOP "
                f"budget after {flops_used:,} FLOPs (budget={spec.flop_budget:,}); "
                f"estimator output set to zeros.",
                BudgetExhaustionWarning,
                stacklevel=2,
            )
```

- [ ] **Step 5: Add `"traceback"` key to the success-path per-MLP dict**

In `src/whestbench/scoring.py`, the `per_mlp.append({...})` call at lines 716-734. Add one new key inside the dict literal (place it right after `"breakdowns"`):

```python
        per_mlp.append(
            {
                "mlp_index": i,
                "final_layer_mse": final_layer_mse,
                "all_layers_mse": all_layers_mse,
                "adjusted_final_layer_score": adjusted_final_layer_score,
                "flops_used": flops_used,
                "effective_compute": effective_compute,
                "budget_exhausted": budget_exhausted,
                "time_exhausted": time_exhausted,
                "residual_wall_time_exhausted": residual_wall_time_exhausted,
                "combined_budget_exhausted": combined_budget_exhausted,
                "wall_time_s": wall_time_s,
                "flopscope_backend_time_s": flopscope_backend_time_s,
                "flopscope_overhead_time_s": flopscope_overhead_time_s,
                "residual_wall_time_s": residual_wall_time_s,
                "breakdowns": {"estimator": normalized_breakdown},
                "traceback": exhaustion_traceback,  # NEW
            }
        )
```

- [ ] **Step 6: Run tests to verify all three pass**

Run: `pytest tests/test_scoring_warnings.py -v`

Expected: 5 tests PASS (2 from Task 1 + 3 new).

- [ ] **Step 7: Run the existing test suite to confirm no regression**

Run: `pytest tests/test_cli_participant_commands.py -k budget -v`

Expected: `test_run_budget_exhausted_does_not_set_exit_1` PASSES (the new `traceback` key is in the JSON but the existing assertions don't check it). If it fails because the existing assertion `all("error" not in entry for entry in per_mlp)` somehow trips — investigate; "error" should still be absent (we only added "traceback").

- [ ] **Step 8: Commit**

```bash
git add src/whestbench/scoring.py tests/test_scoring_warnings.py
git commit -m "feat(scoring): warn + capture traceback on BudgetExhaustedError (#24)

Honor fail_fast by re-raising. Store traceback in per-MLP entry under
the unified 'traceback' key. Library users opt out via
warnings.simplefilter('ignore', BudgetExhaustionWarning)."
```

---

## Task 3: TimeExhaustedError handler — same treatment

**Files:**
- Modify: `tests/test_scoring_warnings.py` (add two tests)
- Modify: `src/whestbench/scoring.py:538-551` (rewrite TimeExhaustedError handler)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_scoring_warnings.py`:

```python
from whestbench.scoring import TimeExhaustionWarning


class _SlowEstimator(BaseEstimator):
    """Estimator that always exhausts the wall-clock budget immediately."""

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        raise flops.TimeExhaustedError("test", elapsed_s=999.0, limit_s=1.0)


def test_time_exhaustion_emits_warning() -> None:
    """evaluate_estimator emits TimeExhaustionWarning on TimeExhaustedError."""
    data = _make_tiny_data(n_mlps=2)
    with pytest.warns(TimeExhaustionWarning) as records:
        result = evaluate_estimator(_SlowEstimator(), data)

    assert len(records) == 2
    msgs = [str(r.message) for r in records]
    assert all("exhausted wall-clock budget" in m for m in msgs)
    assert all("estimator output set to zeros" in m for m in msgs)
    for entry in result["per_mlp"]:
        assert entry["time_exhausted"] is True
        assert isinstance(entry["traceback"], str)
        assert "TimeExhaustedError" in entry["traceback"]


def test_fail_fast_reraises_time_exhausted() -> None:
    """Under fail_fast=True, TimeExhaustedError propagates."""
    data = _make_tiny_data(n_mlps=2)
    with pytest.raises(flops.TimeExhaustedError):
        evaluate_estimator(_SlowEstimator(), data, fail_fast=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scoring_warnings.py -v`

Expected: both new tests FAIL — `test_time_exhaustion_emits_warning` with "DID NOT WARN", `test_fail_fast_reraises_time_exhausted` with "DID NOT RAISE".

- [ ] **Step 3: Rewrite the TimeExhaustedError handler**

Replace lines 538-551 of `src/whestbench/scoring.py`:

```python
        except flops.TimeExhaustedError:
            if fail_fast:
                raise
            exhaustion_traceback = _tb.format_exc()
            predictions = fnp.zeros((spec.depth, spec.width))
            time_exhausted = True
            stats = _predict_stats_to_dict(
                last_predict_stats() if callable(last_predict_stats) else None
            )
            if stats is not None:
                flops_used = int(stats.get("flops_used", budget_ctx.flops_used))
                raw_breakdown = stats.get("budget_breakdown")
            if raw_breakdown is None:
                raw_breakdown = budget_ctx.summary_dict(by_namespace=True)
            normalized_breakdown = _normalize_estimator_budget_breakdown(raw_breakdown)
            if normalized_breakdown is not None:
                normalized_breakdowns.append(normalized_breakdown)
            elapsed_s = float(budget_ctx.wall_time_s or 0.0)
            warnings.warn(
                f"MLP {i} (depth={spec.depth}, width={spec.width}) exhausted "
                f"wall-clock budget after {elapsed_s:.2f}s "
                f"(limit={spec.wall_time_limit_s:.2f}s); "
                f"estimator output set to zeros.",
                TimeExhaustionWarning,
                stacklevel=2,
            )
```

Note: the codebase already reads `budget_ctx.wall_time_s` in the success path (see [scoring.py:631](../../../src/whestbench/scoring.py)). Use the same attribute here.

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_scoring_warnings.py -v`

Expected: 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/whestbench/scoring.py tests/test_scoring_warnings.py
git commit -m "feat(scoring): warn + capture traceback on TimeExhaustedError (#24)"
```

---

## Task 4: Update existing CLI test for new JSON contract

**Files:**
- Modify: `tests/test_cli_participant_commands.py:1256-1290`

- [ ] **Step 1: Update the existing test**

In `tests/test_cli_participant_commands.py`, replace the body of `test_run_budget_exhausted_does_not_set_exit_1` (lines 1256-1290) with:

```python
def test_run_budget_exhausted_does_not_set_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Budget exhaustion is a legitimate scoring outcome that is surfaced via a
    traceback in the per-MLP JSON entry; without --fail-fast the run still
    completes cleanly with exit code 0. In --json mode, no warning or summary
    leaks to stderr.

    Uses a trivially tiny budget and a predict that raises
    BudgetExhaustedError directly (no real FLOP loop), so the test stays
    fast.
    """
    estimator = tmp_path / "hungry.py"
    estimator.write_text(
        dedent(
            """
            import flopscope as flops
            import flopscope.numpy as fnp
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    raise flops.BudgetExhaustedError('test', flop_cost=0, flops_remaining=0)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset) + ["--json"])
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    payload = json.loads(captured.out)
    per_mlp = payload["results"]["per_mlp"]
    assert all("error" not in entry for entry in per_mlp)
    assert all(entry["budget_exhausted"] for entry in per_mlp)
    # NEW: traceback captured per exhausted MLP.
    assert all(
        isinstance(entry.get("traceback"), str)
        and "BudgetExhaustedError" in entry["traceback"]
        for entry in per_mlp
    )
    # NEW: --json mode keeps stderr clean (no warning preamble, no summary).
    assert "exhausted" not in captured.err.lower(), captured.err
```

- [ ] **Step 2: Run the test to verify the new traceback assertion passes and the stderr assertion fails**

Run: `pytest tests/test_cli_participant_commands.py::test_run_budget_exhausted_does_not_set_exit_1 -v`

Expected: FAIL on the new `"exhausted" not in captured.err.lower()` assertion — because warnings currently leak to stderr in --json mode. (The traceback assertion should already pass after Task 2.) If the traceback assertion fails first, debug that before continuing.

- [ ] **Step 3: Confirm the failure mode is the stderr-leak**

If the test fails on the stderr assertion (showing a `BudgetExhaustionWarning` preamble), that's expected. Proceed to Task 5 — fixing it requires the CLI change. If it fails on the traceback assertion, return to Task 2 to investigate.

- [ ] **Step 4: Commit the test change**

```bash
git add tests/test_cli_participant_commands.py
git commit -m "test: update budget-exhaustion CLI test for new contract (#24)

The test still pins exit code 0 and absence of 'error' key, but now also
asserts the per-MLP 'traceback' is populated with BudgetExhaustedError
content, and that --json mode does not leak warning output to stderr.
The stderr assertion will fail until the CLI change in Task 5 lands."
```

---

## Task 5: CLI — suppress exhaustion warnings in `--json` mode

**Files:**
- Modify: `src/whestbench/cli.py:1007-1075` (`_run_estimator_with_runner` body around the `evaluate_estimator` call)
- Add: a new test in `tests/test_cli_participant_commands.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_participant_commands.py` (after `test_run_budget_exhausted_does_not_set_exit_1`):

```python
def test_json_mode_silences_exhaustion_warning(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """In --json mode, exhaustion warnings must not leak to stderr; stdout
    must remain pure JSON; per-MLP entries still carry the traceback.
    """
    estimator = tmp_path / "hungry.py"
    estimator.write_text(
        dedent(
            """
            import flopscope as flops
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    raise flops.BudgetExhaustedError('test', flop_cost=0, flops_remaining=0)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset) + ["--json"])
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    # stdout is pure JSON.
    payload = json.loads(captured.out)
    # stderr contains no warning preamble and no summary line.
    assert captured.err == "" or "exhausted" not in captured.err.lower(), captured.err
    # Tracebacks still present in per-MLP entries.
    for entry in payload["results"]["per_mlp"]:
        assert entry["budget_exhausted"] is True
        assert isinstance(entry["traceback"], str)
        assert "BudgetExhaustedError" in entry["traceback"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cli_participant_commands.py::test_json_mode_silences_exhaustion_warning -v`

Expected: FAIL on the stderr-cleanliness assertion (warning preamble leaks).

- [ ] **Step 3: Add `from contextlib import ExitStack` and the suppression context**

In `src/whestbench/cli.py`, ensure `from contextlib import ExitStack` is imported at the top (check current imports first; if missing, add it alongside other `contextlib` imports).

Then add a private helper function near `_run_estimator_with_runner` (around line 1000, before the function):

```python
@contextmanager
def _route_scoring_warnings(*, output_format: str) -> Iterator[None]:
    """Route ScoringExhaustionWarnings appropriately for the current output mode.

    - json: suppress entirely (stdout must be pure JSON).
    - rich/plain: route to Rich's active console so messages render cleanly
      above any Live region; falls back to default formatting if Rich is
      unavailable.
    """
    from .scoring import ScoringExhaustionWarning

    if output_format == "json":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ScoringExhaustionWarning)
            yield
        return

    from rich import get_console as _get_console
    console = _get_console()
    original_showwarning = warnings.showwarning

    def _routed(message, category, filename, lineno, file=None, line=None):
        if isinstance(message, ScoringExhaustionWarning):
            console.log(f"[yellow]⚠[/]  {message}")
        else:
            original_showwarning(message, category, filename, lineno, file, line)

    warnings.showwarning = _routed
    try:
        yield
    finally:
        warnings.showwarning = original_showwarning
```

Ensure `from contextlib import contextmanager` and `from typing import Iterator` are imported (likely already present; check).

- [ ] **Step 4: Wrap the `evaluate_estimator` call**

In `_run_estimator_with_runner` at line 1061, wrap the existing call:

```python
    t0 = _time.time()
    runner.start(entrypoint, context, limits)

    try:
        with _route_scoring_warnings(output_format=output_format):
            results = evaluate_estimator(
                _RunnerEstimator(runner),
                data,
                on_mlp_scored=lambda i: (
                    progress({"phase": "scoring", "completed": i, "total": n_mlps})
                    if progress is not None
                    else None
                ),
                fail_fast=fail_fast,
            )
    finally:
        runner.close()
```

For this to work, `_run_estimator_with_runner` must know `output_format`. Check the function signature at line 1007. If `output_format` is not already a parameter, add it:

```python
def _run_estimator_with_runner(
    runner: "Any",
    *,
    entrypoint: EstimatorEntrypoint,
    contest_spec: ContestSpec,
    n_mlps: int,
    profile: bool,
    detail: str,
    output_format: str,  # NEW
    progress: Optional[ProgressCallback] = None,
    contest_data: "Optional[Any]" = None,
    fail_fast: bool = False,
) -> Dict[str, Any]:
```

Update every call site (search `_run_estimator_with_runner(`) to pass `output_format=output_format`.

- [ ] **Step 5: Run the JSON-mode test**

Run: `pytest tests/test_cli_participant_commands.py::test_json_mode_silences_exhaustion_warning -v`

Expected: PASS.

- [ ] **Step 6: Re-run the updated test from Task 4**

Run: `pytest tests/test_cli_participant_commands.py::test_run_budget_exhausted_does_not_set_exit_1 -v`

Expected: PASS (the stderr assertion now holds).

- [ ] **Step 7: Run the full CLI participant test file to catch any signature-change regressions**

Run: `pytest tests/test_cli_participant_commands.py -v`

Expected: all tests PASS. If a test fails because `_run_estimator_with_runner` is called without the new `output_format` parameter, fix the call site and re-run.

- [ ] **Step 8: Commit**

```bash
git add src/whestbench/cli.py tests/test_cli_participant_commands.py
git commit -m "feat(cli): route ScoringExhaustionWarnings per output mode (#24)

--json suppresses warnings entirely (stdout stays pure JSON). Rich and
plain modes route through rich.get_console().log() so messages render
above any active Live progress region without flicker."
```

---

## Task 6: CLI — emit fail-fast traceback test

**Files:**
- Modify: `tests/test_cli_participant_commands.py` (new test)

- [ ] **Step 1: Write the test**

Append to `tests/test_cli_participant_commands.py`:

```python
def test_fail_fast_exits_nonzero_on_budget_exhaustion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With --fail-fast, BudgetExhaustedError propagates: nonzero exit and
    traceback on stderr.
    """
    estimator = tmp_path / "hungry.py"
    estimator.write_text(
        dedent(
            """
            import flopscope as flops
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    raise flops.BudgetExhaustedError('test', flop_cost=0, flops_remaining=0)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset)

    exit_code = cli.main(_tiny_run_argv(estimator, dataset) + ["--fail-fast"])
    captured = capsys.readouterr()

    assert exit_code != 0
    assert "BudgetExhaustedError" in captured.err
```

- [ ] **Step 2: Run the test**

Run: `pytest tests/test_cli_participant_commands.py::test_fail_fast_exits_nonzero_on_budget_exhaustion -v`

Expected: PASS — the CLI's existing top-level exception handler should already exit nonzero and print the traceback when `evaluate_estimator` raises `BudgetExhaustedError`. If the test fails:
- Check the CLI's top-level `try/except` for the `run` command (around `cli.py:1559-1565` per the spec). If `BudgetExhaustedError` is not in the caught chain, no code change is needed because Python's default behavior on uncaught exception is exit-nonzero with traceback.
- If the test fails because exit_code IS 0 but stderr has the traceback, investigate the CLI's outer wrapper.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli_participant_commands.py
git commit -m "test: --fail-fast exits nonzero on BudgetExhaustedError (#24)"
```

---

## Task 7: CLI — end-of-run summary line in Rich and plain modes

**Files:**
- Modify: `tests/test_cli_participant_commands.py` (new test)
- Modify: `src/whestbench/cli.py` (`_run_estimator_with_runner` or its caller)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cli_participant_commands.py`:

```python
def test_plain_mode_emits_summary_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """In plain mode, after the run a one-line summary names the number of
    MLPs that exhausted budget. Per-MLP warning text also appears in stderr.
    """
    estimator = tmp_path / "hungry.py"
    estimator.write_text(
        dedent(
            """
            import flopscope as flops
            from whestbench import BaseEstimator

            class Estimator(BaseEstimator):
                def predict(self, mlp, budget):
                    raise flops.BudgetExhaustedError('test', flop_cost=0, flops_remaining=0)
            """
        ).lstrip(),
        encoding="utf-8",
    )
    dataset = tmp_path / "ds.npz"
    _write_tiny_dataset(dataset, n_mlps=3)

    # _tiny_run_argv already sets --format plain.
    exit_code = cli.main(_tiny_run_argv(estimator, dataset))
    captured = capsys.readouterr()

    assert exit_code == 0, captured.err
    # Per-MLP warnings appear in stderr (Python default formatter or Rich console).
    assert "exhausted FLOP budget" in captured.err
    # Summary line names the count of exhausted MLPs.
    assert "3 of 3 MLPs exhausted" in captured.err or "3/3 MLPs exhausted" in captured.err
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_cli_participant_commands.py::test_plain_mode_emits_summary_line -v`

Expected: FAIL — no summary line is emitted today. The per-MLP warning text assertion may pass (Python's default formatter writes warnings to stderr in plain mode); confirm by reading the failure message.

- [ ] **Step 3: Find the right summary-emission site**

The summary should fire **after** scoring completes but **before** the per-MLP table is rendered. The cleanest place is inside `_run_estimator_with_runner` immediately after the `evaluate_estimator(...)` call returns and the runner is closed:

```python
    elapsed = _time.time() - t0
    if output_format != "json":
        _emit_exhaustion_summary(results, output_format=output_format)
    return {
        "schema_version": "1.0",
        "mode": "human",
        "results": results,
        ...
    }
```

- [ ] **Step 4: Add the `_emit_exhaustion_summary` helper**

Add this private helper near `_route_scoring_warnings`:

```python
def _emit_exhaustion_summary(results: Dict[str, Any], *, output_format: str) -> None:
    """If any MLP exhausted budget or time, print a one-line summary to stderr.

    Suppressed for --json mode (caller's responsibility to check).
    """
    per_mlp = results.get("per_mlp") or []
    total = len(per_mlp)
    if total == 0:
        return
    budget_n = sum(
        1 for entry in per_mlp
        if isinstance(entry, dict) and bool(entry.get("budget_exhausted"))
    )
    time_n = sum(
        1 for entry in per_mlp
        if isinstance(entry, dict) and bool(entry.get("time_exhausted"))
    )
    if budget_n == 0 and time_n == 0:
        return

    exhausted_total = budget_n + time_n
    parts = []
    if budget_n:
        parts.append(f"{budget_n} FLOP")
    if time_n:
        parts.append(f"{time_n} time")
    breakdown = " and ".join(parts) if parts else ""
    msg = (
        f"{exhausted_total} of {total} MLPs exhausted budget ({breakdown}). "
        f"Pass --fail-fast to stop on first exhaustion; per-MLP tracebacks are "
        f"in the JSON output."
    )

    if output_format == "rich":
        from rich import get_console as _get_console
        _get_console().log(f"[yellow]{msg}[/]")
    else:
        print(msg, file=sys.stderr)
```

Ensure `import sys` is at the top of `cli.py` (it likely already is).

- [ ] **Step 5: Run the test**

Run: `pytest tests/test_cli_participant_commands.py::test_plain_mode_emits_summary_line -v`

Expected: PASS.

- [ ] **Step 6: Re-run the JSON-mode test to confirm the summary is still suppressed there**

Run: `pytest tests/test_cli_participant_commands.py::test_json_mode_silences_exhaustion_warning -v`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/whestbench/cli.py tests/test_cli_participant_commands.py
git commit -m "feat(cli): summary line for exhausted MLPs in non-JSON modes (#24)"
```

---

## Task 8: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `pytest -x`

Expected: all tests PASS. If anything fails, debug — likely a missed call site of `_run_estimator_with_runner` that needs the new `output_format` keyword argument.

- [ ] **Step 2: Sanity-check the CLI manually with a hungry estimator**

Create a tmp estimator and dataset, then run:

```bash
python -c "
import tempfile, pathlib, subprocess
d = pathlib.Path(tempfile.mkdtemp())
(d / 'hungry.py').write_text('''
import flopscope as flops
from whestbench import BaseEstimator
class Estimator(BaseEstimator):
    def predict(self, mlp, budget):
        raise flops.BudgetExhaustedError(\"test\", flop_cost=0, flops_remaining=0)
''')
# Use the existing tiny-dataset helper from tests if necessary, or skip if
# the smoke-test command does the job:
print(d)
"
```

(Or just rely on `pytest tests/test_cli_participant_commands.py -v` as the manual proof.)

Expected: in plain mode, you see per-MLP warnings followed by a summary line. In `--json` mode, stdout is pure JSON and stderr is empty.

- [ ] **Step 3: Verify pyright / type checks pass (if the repo runs them)**

Run: `pyright src/whestbench/scoring.py src/whestbench/cli.py src/whestbench/__init__.py` (skip if pyright is not configured).

Expected: no new errors.

- [ ] **Step 4: Close the issue**

Mark `#24` as ready-to-close in the PR description (do NOT close the issue from this plan — that's the merger's call):

```
Closes #24
```
