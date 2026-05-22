# Budget & Time Exhaustion: Warn and Fail-Fast

**Issue:** [whestbench#24](https://github.com/AIcrowd/whestbench/issues/24)
**Date:** 2026-05-20
**Status:** Approved (pending user review of this spec)

## Problem

Today, when `scoring.score()` catches `flops.BudgetExhaustedError` or `flops.TimeExhaustedError`
at [src/whestbench/scoring.py:523-551](../../../src/whestbench/scoring.py), it silently:

1. Sets the per-MLP prediction to zeros.
2. Records `budget_exhausted=True` (or `time_exhausted=True`).
3. Continues scoring the next MLP.

No warning is emitted, no traceback is captured, and the `--fail-fast` flag is ignored
because the exhaustion handlers run **before** the generic `except Exception` block that
checks `fail_fast`.

This produces three concrete problems:

- **Silent failures hide bugs.** Participants ship estimators that quietly hit the budget cap
  and don't realize their estimator is returning zeros instead of real predictions.
- **Hard to debug exhaustion.** Even when participants notice `budget_exhausted: True`, they
  have no information about *where* in their code the exhaustion happened.
- **Inconsistency surface area.** Two error-handling paths with different semantics is a
  maintenance smell — the silent path skips traceback capture, `fail_fast`, and CLI rendering
  that the generic path already implements.

There is also an existing test
[`test_run_budget_exhausted_does_not_set_exit_1`](../../../tests/test_cli_participant_commands.py)
whose docstring asserts *"Budget exhaustion is a legitimate scoring outcome, not an error."*
The design below honors that intent — exit code 0 remains the default — while adding
visibility and a fail-fast escape hatch.

## Scope

In scope:

- Both `flops.BudgetExhaustedError` and `flops.TimeExhaustedError` (same silent-handling
  pattern, same UX problem).
- Library users (callers of `scoring.score()`).
- CLI users in all three output modes: Rich (default), plain text, JSON.

Out of scope:

- `RunnerError` and other exceptions raised from the subprocess runner — these already
  flow through the generic exception path correctly.
- Renaming `budget_exhausted` / `time_exhausted` result fields.
- Changing the budget-adjusted scoring formula or zero-prediction MSE fallback.

## Design

### Architecture: warning subclasses

Add three new classes near the top of `src/whestbench/scoring.py` (alongside other
module-level types). Co-locating with the only code site that raises them keeps the
classes discoverable without creating a `warnings.py` module that would shadow the
stdlib. Inheriting from `UserWarning` puts them in the default-visible category without
being errors:

```python
class ScoringExhaustionWarning(UserWarning):
    """Base class for budget/time exhaustion warnings during scoring."""

class BudgetExhaustionWarning(ScoringExhaustionWarning):
    """An estimator exhausted its FLOP budget."""

class TimeExhaustionWarning(ScoringExhaustionWarning):
    """An estimator exhausted its wall-clock budget."""
```

Two leaf classes with a shared parent gives users granular control:

- `warnings.simplefilter("error", ScoringExhaustionWarning)` — escalate either kind.
- `warnings.simplefilter("error", BudgetExhaustionWarning)` — escalate FLOP-only.
- `warnings.simplefilter("ignore", ScoringExhaustionWarning)` — fully silence (matches
  library users who already check the `budget_exhausted` flag themselves).

Public re-export from `whestbench/__init__.py` so users can write
`from whestbench import BudgetExhaustionWarning`.

### Behavior: `scoring.score()` exhaustion handlers

Replace [scoring.py:523-551](../../../src/whestbench/scoring.py).

**`BudgetExhaustedError` handler** becomes:

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
        f"MLP {i} (depth={spec.depth}, width={spec.width}) exhausted FLOP budget "
        f"after {flops_used:,} FLOPs (budget={spec.flop_budget:,}); "
        f"estimator output set to zeros.",
        BudgetExhaustionWarning,
        stacklevel=2,
    )
```

Two changes from today's body:

- New `if fail_fast: raise` at the top (mirrors the generic handler's behavior).
- New `exhaustion_traceback` local + `warnings.warn(...)` at the bottom.

**`TimeExhaustedError` handler:** symmetric. Message uses `wall_time_limit` instead of
`spec.flop_budget`, warning subclass is `TimeExhaustionWarning`, sets
`exhaustion_traceback = _tb.format_exc()` and `time_exhausted = True`.

### Data shape: per-MLP result

The success-path per-MLP dict at [scoring.py:700-730](../../../src/whestbench/scoring.py)
gains the existing `"traceback"` key (currently only set in the generic-failure path at
line 595):

```python
{
    "mlp_index": i,
    "final_layer_mse": ...,
    "all_layers_mse": ...,
    # ... existing fields ...
    "budget_exhausted": budget_exhausted,
    "time_exhausted": time_exhausted,
    "combined_budget_exhausted": combined_budget_exhausted,
    # ... existing fields ...
    "traceback": exhaustion_traceback,  # NEW: None unless exhausted
}
```

**Contract:** `traceback is not None` iff something went wrong — whether expected
(budget/time exhaustion) or unexpected (generic error). Consumers discriminate by also
reading `budget_exhausted`, `time_exhausted`, and `error_code`:

| Outcome              | `budget_exhausted` | `time_exhausted` | `error_code` | `traceback`  |
| -------------------- | ------------------ | ---------------- | ------------ | ------------ |
| Clean run            | `False`            | `False`          | absent       | `None`       |
| FLOP exhaustion      | `True`             | `False`          | absent       | non-`None`   |
| Time exhaustion      | `False`            | `True`           | absent       | non-`None`   |
| Unexpected error     | `False`            | `False`          | non-`None`   | non-`None`   |

### Behavior: `fail_fast`

Under `--fail-fast`, `BudgetExhaustedError` and `TimeExhaustedError` re-raise unchanged from
the new `if fail_fast: raise` at the top of each handler. They propagate out of `score()`,
through the existing CLI top-level handler at
[cli.py:1559-1565](../../../src/whestbench/cli.py), which prints the traceback and exits
non-zero. **No CLI changes are required for the fail-fast path.**

The existing test
[`test_run_budget_exhausted_does_not_set_exit_1`](../../../tests/test_cli_participant_commands.py)
(which does not pass `--fail-fast`) continues to assert exit code 0. A new test asserts
fail-fast behavior — see Tests below.

### CLI: warning routing in three render modes

**Plain text mode (`--output-format plain` or `--log-progress`):**
No Rich Live is active; Python's default `warnings.showwarning` writes to stderr. We do
nothing special — the default format (with file:line preamble) is acceptable for plain
mode.

**Rich mode (default):**
During the scoring Live block (`_LiveTopPaneSession.__enter__` / `__exit__` around
[cli.py:455 and cli.py:635](../../../src/whestbench/cli.py)), install a custom
`warnings.showwarning` via `contextlib.ExitStack` that routes `ScoringExhaustionWarning`
through Rich's `console.log()` (which writes above the Live region without flicker):

```python
def _rich_warning_handler(message, category, filename, lineno, file=None, line=None):
    if isinstance(message, ScoringExhaustionWarning):
        console.log(f"[yellow]⚠[/]  {message}")
    else:
        _original_showwarning(message, category, filename, lineno, file, line)

stack.callback(setattr, warnings, "showwarning", warnings.showwarning)  # restore
warnings.showwarning = _rich_warning_handler
```

Tear-down restores the original via the ExitStack callback.

**JSON mode (`--json`):**
stdout must be parseable JSON, so warnings must not leak there. Wrap the scoring call in:

```python
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ScoringExhaustionWarning)
    result = score(...)
```

The traceback is still captured in the per-MLP entry — `--json` consumers read it from
there. The end-of-run summary line (next section) is also suppressed in `--json` mode.

### CLI: end-of-run summary line

After scoring finishes, in Rich and plain modes (NOT in `--json`), inspect the per-MLP
results and emit one summary line if any exhaustion occurred:

```
3 of 64 MLPs exhausted budget (1 time, 2 FLOPs). Pass --debug or read the JSON for tracebacks.
```

Rendered by the CLI via `console.print()` after the Live block exits. The exact counts come
from existing aggregate fields in the suite-level result
(see [scoring.py:785-797](../../../src/whestbench/scoring.py)) so no extra plumbing is
needed.

## Tests

All test paths under `tests/`. Six tests total: one updated, five new.

### Updated: `test_run_budget_exhausted_does_not_set_exit_1`

File: [tests/test_cli_participant_commands.py:1256](../../../tests/test_cli_participant_commands.py)

The existing test runs with `--json`, so it covers the **machine-output** path. Keep all
existing assertions (exit 0, no `error` key, `budget_exhausted: True`). Add:

- `assert entry["traceback"]` is non-`None` and contains `"BudgetExhaustedError"` for every
  exhausted MLP.
- `assert captured.err == ""` (or equivalently, no `"exhausted budget"` substring) to verify
  the summary line **is suppressed** in `--json` mode.

Update docstring to: *"Budget exhaustion is a legitimate scoring outcome that is surfaced
via a traceback in the per-MLP JSON entry; without `--fail-fast` the run still completes
cleanly with exit code 0. In --json mode, no warning or summary leaks to stderr."*

### New: `test_budget_exhaustion_emits_warning`

Direct library-level test against `scoring.score()` (no CLI). Build a minimal estimator
that raises `flops.BudgetExhaustedError` on `predict()`. Assert with
`pytest.warns(BudgetExhaustionWarning)` that:

- The warning is emitted exactly once per exhausted MLP.
- The warning message contains the MLP index, FLOPs used, and budget.

### New: `test_budget_exhaustion_fail_fast_raises`

Run the hungry-estimator fixture from the existing test with `--fail-fast`. Assert:

- `exit_code != 0`.
- `captured.err` contains `"BudgetExhaustedError"` (full traceback propagated).
- No `UnboundLocalError` or similar from a half-initialized Live region — i.e., the Rich
  Live tore down cleanly.

### New: `test_time_exhaustion_parity`

Mirror of `test_run_budget_exhausted_does_not_set_exit_1`, but the estimator raises
`flops.TimeExhaustedError`. Assert:

- Exit code 0.
- `entry["time_exhausted"]` is `True` for every entry.
- `entry["traceback"]` contains `"TimeExhaustedError"`.
- A `TimeExhaustionWarning` (subclass of `ScoringExhaustionWarning`) was emitted.

### New: `test_json_mode_silences_exhaustion_warning`

Run the hungry-estimator fixture with `--json` (no `--fail-fast`). Assert:

- `json.loads(captured.out)` parses successfully — no warning preamble polluting stdout.
- `captured.err` is empty (summary line is suppressed in `--json` mode).
- For every per-MLP entry: `budget_exhausted is True` and `traceback` is non-`None`.

Note: this test overlaps with the updated `test_run_budget_exhausted_does_not_set_exit_1`
on the JSON-cleanliness assertion. They differ in intent: the updated test pins the exit-code
contract; this test pins the stream-isolation contract. Keep both — they document distinct
guarantees.

### New: `test_rich_mode_emits_summary_line`

Run the hungry-estimator fixture **without** `--json` (so the Rich/plain path renders the
summary). Use `--output-format plain` to avoid Rich ANSI escapes interfering with
substring assertions in CI. Assert:

- Exit code 0.
- `captured.err` contains the substring `"exhausted budget"` (the summary line).
- `captured.err` also contains the per-MLP warning text (e.g.,
  `"MLP 0 (depth=..., width=...) exhausted FLOP budget"`).

## Migration notes

For downstream consumers of `score()` JSON output:

- **New field:** Per-MLP entries (successful and exhausted) now include a `"traceback"`
  key. Consumers that JSON-schema-validate result payloads need to add this field. It is
  `None` for clean runs and a string otherwise.
- **Warning emission:** `score()` now calls `warnings.warn()` from library code. Consumers
  that treat all warnings as errors via `-W error` will now see `BudgetExhaustionWarning`
  propagate as an exception. Filter explicitly with
  `-W ignore::whestbench.ScoringExhaustionWarning` to preserve old behavior.
- **`--fail-fast` semantics expanded:** Previously, `--fail-fast` only stopped on generic
  exceptions. It now also stops on budget and time exhaustion, matching the docstring of
  the flag (*"Stop on the first estimator error"*).

## Open questions

None at this time.

## File-by-file change list

For the implementation plan:

- `src/whestbench/scoring.py` — add three warning classes; replace exhaustion handlers
  (lines 523-551); extend success-path per-MLP dict with `"traceback"` key.
- `src/whestbench/__init__.py` — public re-export of the three warning classes.
- `src/whestbench/cli.py` — install Rich warning handler around the Live block; suppress
  warnings in `--json` mode; emit end-of-run summary line (Rich + plain only).
- `tests/test_cli_participant_commands.py` — update existing test; add four new tests.
- `tests/` — possibly a new file `test_scoring_warnings.py` for the library-level
  `pytest.warns` test if mixing test styles in `test_cli_participant_commands.py` feels
  off.
