# Black-Box Runtime Metrics Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove invalid per-layer runtime-derived metrics and update scoring/reporting/docs to a strict black-box runtime contract while keeping runtime-aware scoring at scalar budget level.

**Architecture:** Keep estimator execution as black-box call boundaries. Compute prediction metrics per layer from returned tensors, but compute runtime effects only at call/budget level. Remove all schema/reporting artifacts that imply measured per-layer runtime from participant estimator internals.

**Tech Stack:** Python 3.13, NumPy, Rich, pytest, ruff, pyright.

---

### Task 1: Add Contract Guard Tests (Fail First)

**Files:**
- Modify: `tests/test_scoring_module.py`
- Test: `tests/test_scoring_module.py`

**Step 1: Write the failing tests**

```python
def test_by_budget_raw_forbids_layer_runtime_fields() -> None:
    report = score_estimator_report(..., detail="raw")
    for row in report["results"]["by_budget_raw"]:
        assert "time_ratio_by_layer" not in row
        assert "adjusted_mse_by_layer" not in row
        assert "baseline_time_s_by_layer" not in row
        assert "effective_time_s_by_layer" not in row


def test_by_budget_raw_contains_scalar_runtime_fields() -> None:
    report = score_estimator_report(..., detail="raw")
    row = report["results"]["by_budget_raw"][0]
    assert "mse_mean" in row
    assert "adjusted_mse" in row
    assert "call_time_ratio_mean" in row
    assert "call_effective_time_s_mean" in row
    assert "timeout_rate" in row
    assert "time_floor_rate" in row
```

**Step 2: Run tests to verify failure**

Run: `uv run pytest tests/test_scoring_module.py -q`  
Expected: FAIL on missing/new keys and presence of removed keys.

**Step 3: Commit test scaffold**

```bash
git add tests/test_scoring_module.py
git commit -m "test: add blackbox runtime field contract guards"
```

---

### Task 2: Refactor `by_budget_raw` Scoring Output to Scalar Runtime Fields

**Files:**
- Modify: `src/circuit_estimation/scoring.py`
- Test: `tests/test_scoring_module.py`

**Step 1: Implement minimal scoring changes**

```python
# keep per-layer prediction error
mse_by_layer = ...
mse_mean = float(np.mean(mse_by_layer))

# runtime at call/budget level only
call_time_ratio_mean = float(np.mean(call_time_ratios))
adjusted_mse = mse_mean * call_time_ratio_mean

by_budget_raw.append({
    "budget": int(budget),
    "mse_by_layer": mse_by_layer.tolist(),
    "mse_mean": mse_mean,
    "adjusted_mse": adjusted_mse,
    "call_time_ratio_mean": call_time_ratio_mean,
    "call_effective_time_s_mean": float(np.mean(call_effective_times)),
    "timeout_rate": timeout_rate,
    "time_floor_rate": time_floor_rate,
})
```

**Step 2: Update final score aggregation**

```python
final_score = float(np.mean([entry["adjusted_mse"] for entry in by_budget_raw]))
```

**Step 3: Run scoring tests**

Run: `uv run pytest tests/test_scoring_module.py -q`  
Expected: PASS for contract tests, with any remaining failures in `detail=full` tests.

**Step 4: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring_module.py
git commit -m "refactor: remove layer runtime fields from by_budget_raw"
```

---

### Task 3: Clean `detail=full` Runtime-By-Layer Aggregates

**Files:**
- Modify: `src/circuit_estimation/scoring.py`
- Modify: `tests/test_scoring_module.py`

**Step 1: Write/adjust failing full-detail tests**

```python
def test_detail_full_excludes_runtime_by_layer_aggregates() -> None:
    report = score_estimator_report(..., detail="full")
    results = report["results"]
    assert "time_ratio_mean_by_layer" not in results.get("by_layer_overall", {})
    assert "baseline_time_s_mean_by_layer" not in results.get("by_layer_overall", {})
    assert "effective_time_s_mean_by_layer" not in results.get("by_layer_overall", {})
    assert "time_ratio_by_budget_layer" not in results.get("by_budget_layer_matrix", {})
```

**Step 2: Implement minimal `detail=full` cleanup**

- Remove runtime-by-layer arrays/matrices from `_compute_full_detail`.
- Keep prediction-derived layer summaries and budget summaries.

**Step 3: Run tests**

Run: `uv run pytest tests/test_scoring_module.py -q`  
Expected: PASS.

**Step 4: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring_module.py
git commit -m "refactor: drop runtime-by-layer aggregates from detail full"
```

---

### Task 4: Update Reporting Data Dependencies (No Layer Runtime Metrics)

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Modify: `tests/test_reporting.py`

**Step 1: Write/adjust failing reporting tests**

```python
def test_layer_diagnostics_uses_prediction_metrics_only() -> None:
    rendered = render_human_report(report_without_layer_runtime_fields)
    assert "MSE by Layer" in rendered
    assert "Time Ratio by Layer" not in rendered
    assert "Adjusted MSE by Layer" not in rendered
```

**Step 2: Minimal reporting refactor**

- Budget table reads scalar fields (`adjusted_mse`, `mse_mean`, `call_time_ratio_mean`, `call_effective_time_s_mean`).
- Layer section displays prediction-centric rows only.
- Remove code paths requiring removed fields.

**Step 3: Run reporting tests**

Run: `uv run pytest tests/test_reporting.py -q`  
Expected: PASS.

**Step 4: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py
git commit -m "refactor: align reporting with blackbox runtime metric contract"
```

---

### Task 5: Update CLI Integration Tests for New Human Report Contract

**Files:**
- Modify: `tests/test_cli.py`
- Test: `tests/test_cli.py`

**Step 1: Add/adjust failing CLI tests**

```python
def test_human_mode_defaults_without_layer_runtime_plot_dependencies() -> None:
    ...
```

**Step 2: Minimal CLI glue updates (if needed)**

- Ensure `render_human_report(...)` receives unchanged inputs but does not assume removed fields.

**Step 3: Run CLI tests**

Run: `uv run pytest tests/test_cli.py -q`  
Expected: PASS.

**Step 4: Commit**

```bash
git add tests/test_cli.py src/circuit_estimation/cli.py
git commit -m "test: align CLI contract checks with blackbox metric cleanup"
```

---

### Task 6: Documentation Updates (Contract + Decision Log)

**Files:**
- Modify: `README.md`
- Modify: `docs/context/python-runtime-refactor-decisions.md`
- Modify: `docs/context/mvp-technical-snapshot.md` (if contract fields listed there)

**Step 1: Update README scoring contract section**

- Remove mentions of per-layer runtime-derived fields.
- Document scalar runtime-aware fields and black-box boundary.

**Step 2: Update decision log**

- Add decision entry describing removal of invalid per-layer runtime-derived metrics.

**Step 3: Run doc-targeted grep checks**

Run:

```bash
rg -n "time_ratio_by_layer|adjusted_mse_by_layer|baseline_time_s_by_layer|effective_time_s_by_layer" README.md docs/context
```

Expected: no stale contract references (except historical plan docs if intentionally retained).

**Step 4: Commit**

```bash
git add README.md docs/context/python-runtime-refactor-decisions.md docs/context/mvp-technical-snapshot.md
git commit -m "docs: enforce blackbox runtime observability contract"
```

---

### Task 7: Full Repo Audit and Final Verification Gate

**Files:**
- Modify: any remaining `src/` / `tests/` files discovered by audit

**Step 1: Forbidden field audit**

Run:

```bash
rg -n "time_ratio_by_layer|adjusted_mse_by_layer|baseline_time_s_by_layer|effective_time_s_by_layer" src tests README.md docs/context
```

Expected: no active contract usage in `src/`, `tests`, `README.md`, `docs/context`.

**Step 2: Run full quality gate**

Run:

```bash
uv run pytest tests/test_scoring_module.py tests/test_reporting.py tests/test_cli.py -q
uv run ruff check src/circuit_estimation/scoring.py src/circuit_estimation/reporting.py src/circuit_estimation/cli.py tests/test_scoring_module.py tests/test_reporting.py tests/test_cli.py
uv run pyright src/circuit_estimation/scoring.py src/circuit_estimation/reporting.py src/circuit_estimation/cli.py tests/test_scoring_module.py tests/test_reporting.py tests/test_cli.py
```

Expected: all pass.

**Step 3: Final commit**

```bash
git add src/circuit_estimation/scoring.py src/circuit_estimation/reporting.py src/circuit_estimation/cli.py tests/test_scoring_module.py tests/test_reporting.py tests/test_cli.py README.md docs/context/python-runtime-refactor-decisions.md docs/context/mvp-technical-snapshot.md
git commit -m "refactor: remove invalid layer runtime metrics for blackbox estimator contract"
```

---

## Notes for Executor

- Use @test-driven-development per task.
- Keep changes minimal and contract-focused (YAGNI).
- Avoid touching historical plan docs unless explicitly requested.
- Do not introduce schema version changes in this plan.
