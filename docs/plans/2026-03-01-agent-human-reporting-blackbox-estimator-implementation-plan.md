# Agent/Human Reporting + Black-Box Estimator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace scalar-only CLI output with dual-mode reporting (agent JSON default + rich human report) and enforce a strict black-box estimator contract that returns full-layer `np.ndarray` outputs.

**Architecture:** Keep existing scorer math semantics but change estimator integration from iterator/yield to single-call tensor return. Add a detailed report path in scoring for raw machine-consumable metrics, then render this report in either agent JSON mode or human rich mode. Treat estimator execution as untrusted and profile only at evaluator call boundaries.

**Tech Stack:** Python 3.10+, NumPy, Pytest, Ruff, Pyright, uv

---

## Execution Notes

- Use `@test-driven-development` for each behavior change.
- Use `@verification-before-completion` before each completion claim/commit.
- Keep agent mode stdout as pretty JSON only.
- Treat estimator implementations as untrusted black boxes.

### Task 1: Lock the new estimator output contract with failing tests

**Files:**
- Modify: `tests/test_scoring_module.py`
- Test: `tests/test_scoring_module.py`

**Step 1: Write the failing test**

Add tests:

```python
def test_score_estimator_rejects_non_ndarray_output():
    ...

def test_score_estimator_rejects_wrong_tensor_shape():
    ...
```

Validate that estimator must return `np.ndarray` with shape `(max_depth, width)`.

**Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_scoring_module.py -q`  
Expected: FAIL because scorer still accepts iterator-style output.

**Step 3: Write minimal implementation**

Update scorer typing and validation path to enforce strict tensor return shape.

**Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_scoring_module.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_scoring_module.py src/circuit_estimation/scoring.py
git commit -m "feat: enforce ndarray estimator output contract"
```

### Task 2: Add black-box call-level profiling/report capture in scorer

**Files:**
- Modify: `src/circuit_estimation/scoring.py`
- Test: `tests/test_scoring_module.py`

**Step 1: Write the failing test**

Add tests for call-level profile entries:

```python
def test_profile_records_one_entry_per_circuit_budget_call():
    ...
```

Assert profile records include:
- `budget`, `circuit_index`, `wire_count`, `layer_count`
- `wall_time_s`, `cpu_time_s`, `rss_bytes`, `peak_rss_bytes`

**Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_scoring_module.py::test_profile_records_one_entry_per_circuit_budget_call -q`  
Expected: FAIL before new report path.

**Step 3: Write minimal implementation**

Add a detailed scorer entrypoint, e.g.:

```python
def score_estimator_report(..., profile: bool = False, detail: str = "raw") -> dict[str, Any]:
    ...
```

Keep `score_estimator(...) -> float` as compatibility wrapper.

**Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_scoring_module.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring_module.py
git commit -m "feat: add call-level blackbox profiling report path"
```

### Task 3: Update example estimators to single-pass tensor output

**Files:**
- Modify: `src/circuit_estimation/estimators.py`
- Modify: `estimators.py`
- Test: `tests/test_estimators_module.py`
- Test: `tests/test_estimators.py`

**Step 1: Write the failing test**

Add/adjust tests:

```python
def test_combined_estimator_returns_depth_width_tensor():
    ...
```

**Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_estimators_module.py tests/test_estimators.py -q`  
Expected: FAIL until estimator return shape is migrated.

**Step 3: Write minimal implementation**

Make `mean_propagation`, `covariance_propagation`, and `combined_estimator` return:
- `np.ndarray` shape `(depth, width)`  
in one pass.

**Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_estimators_module.py tests/test_estimators.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/estimators.py estimators.py tests/test_estimators_module.py tests/test_estimators.py
git commit -m "feat: migrate example estimators to single-pass tensor output"
```

### Task 4: Add report rendering module (agent JSON + human rich sections)

**Files:**
- Create: `src/circuit_estimation/reporting.py`
- Create: `tests/test_reporting.py`

**Step 1: Write the failing test**

Add tests:

```python
def test_render_agent_mode_returns_pretty_json_only():
    ...

def test_render_human_mode_includes_expected_sections():
    ...
```

**Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_reporting.py -q`  
Expected: FAIL because module missing.

**Step 3: Write minimal implementation**

Implement:
- `render_agent_report(report: dict[str, Any]) -> str`
- `render_human_report(report: dict[str, Any]) -> str`

Human output must include:
- run context,
- score summary,
- budget breakdown,
- layer/depth section,
- optional profile section.

**Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_reporting.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py
git commit -m "feat: add dual-mode report renderers"
```

### Task 5: Wire CLI flags (`--mode`, `--detail`, `--profile`) and output contract

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Modify: `main.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

Add tests:

```python
def test_agent_mode_stdout_is_json_only(...):
    ...

def test_human_mode_outputs_rich_sections(...):
    ...
```

**Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_cli.py -q`  
Expected: FAIL until CLI wiring is updated.

**Step 3: Write minimal implementation**

CLI behavior:
- default `--mode agent`
- `--detail raw|full` (default `raw`)
- `--profile` toggles call-level diagnostics in report
- stdout formatting delegated to reporting module

**Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_cli.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py main.py tests/test_cli.py
git commit -m "feat: add agent/human CLI modes with detail controls"
```

### Task 6: Add full-detail computed metrics path

**Files:**
- Modify: `src/circuit_estimation/scoring.py`
- Test: `tests/test_scoring_module.py`

**Step 1: Write the failing test**

Add tests:

```python
def test_detail_full_includes_budget_and_layer_aggregates():
    ...
```

**Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_scoring_module.py::test_detail_full_includes_budget_and_layer_aggregates -q`  
Expected: FAIL until computed sections are implemented.

**Step 3: Write minimal implementation**

For `detail="full"`, add:
- `results.by_budget_summary`
- `results.by_layer_overall`
- `results.by_budget_layer_matrix`
- `profile_summary` (if profiling enabled)

**Step 4: Run test to verify it passes**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest tests/test_scoring_module.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring_module.py
git commit -m "feat: add full-detail computed reporting metrics"
```

### Task 7: Update docs and future-agent black-box note

**Files:**
- Modify: `README.md`
- Modify: `docs/context/mvp-technical-snapshot.md`
- Modify: `docs/context/python-runtime-refactor-decisions.md`

**Step 1: Write the failing docs check**

Run:

```bash
rg -n "black box|adversarial|malicious|mode agent|mode human|detail full|ndarray" README.md docs/context/*.md
```

Expected: missing phrases before update.

**Step 2: Run check to verify it fails**

Run the `rg` command above.

**Step 3: Write minimal implementation**

Document:
- agent/human mode behavior,
- strict ndarray contract,
- black-box/untrusted estimator policy,
- note that in-repo estimators are examples only.

**Step 4: Run check to verify it passes**

Run the same `rg` command; expected required terms present.

**Step 5: Commit**

```bash
git add README.md docs/context/mvp-technical-snapshot.md docs/context/python-runtime-refactor-decisions.md
git commit -m "docs: define dual-mode reporting and black-box estimator policy"
```

### Task 8: Final verification gate

**Files:**
- Verify repository state only

**Step 1: Run full checks**

Run:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --group dev ruff check .
UV_CACHE_DIR=/tmp/uv-cache uv run --group dev ruff format --check .
UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pyright
UV_CACHE_DIR=/tmp/uv-cache uv run --group dev pytest -q
UV_CACHE_DIR=/tmp/uv-cache uv run main.py
UV_CACHE_DIR=/tmp/uv-cache uv run main.py --mode human
UV_CACHE_DIR=/tmp/uv-cache uv run main.py --profile
UV_CACHE_DIR=/tmp/uv-cache uv run main.py --profile --detail full
```

Expected:
- static checks clean,
- tests pass,
- agent mode stdout is pretty JSON only,
- human mode shows rich multi-section output,
- profile outputs include call-level diagnostics only.

**Step 2: Commit final stabilization changes (if any)**

```bash
git add -A
git commit -m "chore: finalize dual-mode reporting verification adjustments"
```
