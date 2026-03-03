# Streaming Scoring Unification Design

## Problem

The codebase has two scoring paths:

1. **`score_estimator_report`** — iterates the estimator's stream directly, times each depth row against a per-depth baseline, applies per-depth timeout/floor. This is the intended scoring model per the design doc.
2. **`score_submission_report`** — routes through the runner boundary, which collects the full `(d,n)` tensor and returns total wall time. Uses call-level timing only.

The runner boundary (`EstimatorRunner.predict()`) returns a single `PredictOutcome` with the complete tensor, discarding per-depth timing information. This means `cestim run` (which uses runners) cannot enforce per-depth streaming scoring.

**This is unacceptable.** Per-depth streaming IS the scoring model. All runners must support it.

## Decision

**Approach A: Streaming Runner Protocol.**

Change `EstimatorRunner.predict()` to return `Iterator[DepthRowOutcome]`. Each yielded item carries one depth row and its cumulative wall time. The scoring function iterates this stream and applies per-depth timeout/floor logic.

Remove `score_submission_report` entirely. There is one scoring path.

## Design

### New Data Type

```python
DepthRowStatus = Literal["ok", "error"]

@dataclass(slots=True)
class DepthRowOutcome:
    depth_index: int
    row: NDArray[np.float32] | None   # None on error
    wall_time_s: float                 # cumulative since predict() start
    status: DepthRowStatus
    error_message: str | None = None
```

### Runner Protocol

```python
class EstimatorRunner(Protocol):
    def start(self, entrypoint, context, limits) -> None: ...
    def predict(self, circuit: Circuit, budget: int) -> Iterator[DepthRowOutcome]: ...
    def close(self) -> None: ...
```

- `predict_batch` is removed (callers loop themselves)
- Stream must yield exactly `d` items
- `wall_time_s` is cumulative from `predict()` start

### InProcessRunner

- Calls `estimator.predict(circuit, budget)` to get the iterator
- For each `next()` call, measures `time.time() - start_wall`
- Validates row shape/finiteness via `validate_depth_row`
- On exception or bad row: yields error `DepthRowOutcome`, stops

### SubprocessRunner + Worker

**Worker protocol change:**

```
-> parent:  {"command": "predict", "circuit": {...}, "budget": 100}
<- worker:  {"status": "row", "depth_index": 0, "row": [0.1, -0.2, ...]}
<- worker:  {"status": "row", "depth_index": 1, "row": [0.3, 0.5, ...]}
...
<- worker:  {"status": "done"}
```

On error:
```
<- worker:  {"status": "error", "depth_index": 2, "error_message": "..."}
```

**Worker implementation:** Instead of calling `_collect_prediction_tensor`, iterate the estimator's stream directly. Validate each row, write one JSON line per depth row, flush after each line.

**Runner implementation:** Read one JSON line per depth, measure `time.time()` at each arrival, yield `DepthRowOutcome`. Handle timeout by killing the process if no line arrives within `predict_timeout_s`.

### Scoring Unification

- **Delete** `score_submission_report` from `scoring.py`
- **Adapt** `score_estimator_report` to accept `EstimatorRunner` (or wrap raw `EstimatorFn` in a lightweight `InProcessRunner`)
- Scoring loop iterates `runner.predict(circuit, budget)`, gets `DepthRowOutcome` stream, applies per-depth timing logic (timeout/floor/effective time) exactly as the current `score_estimator_report` does
- `cestim run` calls the unified function with the chosen runner
- All JSON report fields use the per-depth arrays: `time_budget_by_depth_s`, `time_ratio_by_depth_mean`, `effective_time_s_by_depth_mean`, `timeout_rate_by_depth`, `time_floor_rate_by_depth`

### CLI Wiring

- `cestim run` currently calls `score_submission_report` with `InProcessRunner` or `SubprocessRunner` — rewire to call the unified `score_estimator_report` (adapted to accept runners)
- `cestim smoke-test` already uses `score_estimator_report` — no change needed (or it can also go through a runner)

### Docs

All participant-facing docs updated to describe per-depth streaming:

- **`scoring-model.md`**: Restore per-depth timeout/floor language (reverse the earlier "fix" that changed it to call-level)
- **`score-report-fields.md`**: Add back per-depth fields (`time_budget_by_depth_s`, `time_ratio_by_depth_mean`, `effective_time_s_by_depth_mean`, `timeout_rate_by_depth`, `time_floor_rate_by_depth`)
- **`estimator-contract.md`**: Emphasize that timing is measured at each yield boundary

### What Gets Removed

- `score_submission_report` function
- `PredictOutcome` class (replaced by `DepthRowOutcome`)
- `_collect_prediction_tensor` in both `runner.py` and `subprocess_worker.py` (no longer needed — streaming replaces batch collection)
- `predict_batch` from runner protocol and implementations

## Clock Synchronization

For subprocess runners on the same machine: `time.time()` is synchronized. The runner measures arrival time of each JSON line, which naturally includes compute time + IPC transport. This is the correct wall-clock cost.

For future cloud runners (out of scope): arrival-time measurement at the scorer side provides the same semantics without requiring NTP sync.

## Risks

- **Participants with slow generators:** If a participant's estimator does all computation upfront and then yields rows quickly, per-depth timing will show the first row as very slow and subsequent rows as very fast. This is correct behavior — it incentivizes progressive computation.
- **JSON-line flush latency:** stdout buffering could add artificial delay. Worker must `flush()` after each line. Already handled by `_write_response`.
- **Test surface:** Both runner implementations need streaming tests. Existing `test_scoring_module.py` tests cover per-depth semantics; they need to route through runners now.
