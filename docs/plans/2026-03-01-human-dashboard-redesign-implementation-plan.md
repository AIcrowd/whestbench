# Human Dashboard Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a professional tri-objective human CLI dashboard that clearly covers submission readiness, interpretability, and performance diagnostics in a multi-column, multi-pane Rich interface.

**Architecture:** Refactor `render_human_report()` into width-aware layout orchestration plus pure pane-builder helpers. Keep the existing agent JSON contract unchanged, enforce sparse-vs-dense plot rendering policy in one chart helper, and drive behavior with test-first changes in `tests/test_reporting.py` and `tests/test_cli.py`.

**Tech Stack:** Python 3.13, Rich, plotext, pytest, ruff, pyright.

---

### Task 1: Introduce Width-Aware Layout Spec (wide/medium/narrow)

**Skill refs:** @superpowers:test-driven-development

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Test: `tests/test_reporting.py`

**Step 1: Write the failing test**

```python
def test_human_report_uses_three_column_top_row_on_wide_layout() -> None:
    report = _sample_report(include_profile=False)
    rendered = render_human_report(report)
    assert "Run Context" in rendered
    assert "Readiness" in rendered
    assert "Hardware & Runtime" in rendered
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_reporting.py::test_human_report_uses_three_column_top_row_on_wide_layout -q`
Expected: FAIL because `Readiness` / `Hardware & Runtime` pane titles are not yet aligned with the new layout contract.

**Step 3: Write minimal implementation**

```python
# reporting.py

def _layout_mode(console_width: int) -> str:
    if console_width >= 120:
        return "wide"
    if console_width >= 90:
        return "medium"
    return "narrow"

# Use _layout_mode(console.width) to branch top-row Columns composition.
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_reporting.py::test_human_report_uses_three_column_top_row_on_wide_layout -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py
git commit -m "feat: add width-aware dashboard layout modes"
```

### Task 2: Implement Top Row Triad Panes (Run Context, Readiness, Hardware)

**Skill refs:** @superpowers:test-driven-development

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Test: `tests/test_reporting.py`

**Step 1: Write the failing test**

```python
def test_hardware_pane_renders_only_captured_host_fields() -> None:
    rendered = render_human_report(_sample_report(include_profile=False))
    assert "Hardware & Runtime" in rendered
    assert "[host.hostname]" in rendered
    assert "[host.os]" in rendered
    assert "[host.os_release]" in rendered
    assert "[host.platform]" in rendered
    assert "[host.machine]" in rendered
    assert "[host.python_version]" in rendered
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_reporting.py::test_hardware_pane_renders_only_captured_host_fields -q`
Expected: FAIL until dedicated hardware pane is wired.

**Step 3: Write minimal implementation**

```python
# reporting.py

def _hardware_runtime_panel(report: dict[str, Any]) -> Panel:
    host = report.get("run_meta", {}).get("host", {})
    # rows for host.hostname, host.os, host.os_release, host.platform, host.machine, host.python_version
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_reporting.py::test_hardware_pane_renders_only_captured_host_fields -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py
git commit -m "feat: add dedicated hardware and readiness top-row panes"
```

### Task 3: Build Budget Intelligence Lane (Table + Frontier + Runtime)

**Skill refs:** @superpowers:test-driven-development

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Test: `tests/test_reporting.py`

**Step 1: Write the failing test**

```python
def test_budget_lane_contains_table_and_two_plots() -> None:
    rendered = render_human_report(_sample_report(include_profile=False))
    assert "Budget Table" in rendered
    assert "Budget Frontier Plot" in rendered
    assert "Budget Runtime Plot" in rendered
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_reporting.py::test_budget_lane_contains_table_and_two_plots -q`
Expected: FAIL if pane names/structure are still transitional.

**Step 3: Write minimal implementation**

```python
# reporting.py

def _budget_lane_panel(report: dict[str, Any]) -> Panel:
    # Group(
    #   Panel(budget_table, title="Budget Table"),
    #   _budget_frontier_plot_panel(...),
    #   _budget_runtime_plot_panel(...),
    # )
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_reporting.py::test_budget_lane_contains_table_and_two_plots -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py
git commit -m "feat: add budget intelligence lane composition"
```

### Task 4: Build Layer Intelligence Lane (Stats + Accuracy Trend + Runtime Trend)

**Skill refs:** @superpowers:test-driven-development

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Test: `tests/test_reporting.py`

**Step 1: Write the failing test**

```python
def test_layer_lane_contains_stats_and_trend_plots() -> None:
    rendered = render_human_report(_sample_report(include_profile=False))
    assert "Layer Metric Table" in rendered
    assert "Layer Trend Plot" in rendered
    assert "Layer Runtime Plot" in rendered
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_reporting.py::test_layer_lane_contains_stats_and_trend_plots -q`
Expected: FAIL if lane grouping/titles diverge.

**Step 3: Write minimal implementation**

```python
# reporting.py

def _layer_lane_panel(report: dict[str, Any]) -> Panel:
    # Build stats table with p05/min/p95/max/mean and include both layer plots.
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_reporting.py::test_layer_lane_contains_stats_and_trend_plots -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py
git commit -m "feat: add layer intelligence lane composition"
```

### Task 5: Finalize Plot Policy (Sparse scatter-first, Dense line-based)

**Skill refs:** @superpowers:test-driven-development

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Test: `tests/test_reporting.py`

**Step 1: Write the failing test**

```python
def test_plot_policy_uses_sparse_scatter_and_dense_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    # spy object for plotext
    # assert sparse x path calls scatter(marker="●")
    # assert dense x path calls plot(marker="hd")
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_reporting.py::test_plot_policy_uses_sparse_scatter_and_dense_lines -q`
Expected: FAIL before policy is fully asserted.

**Step 3: Write minimal implementation**

```python
# reporting.py
if len(x) <= 12:
    scatter_fn(x, values, color=color, marker="●")
else:
    _plotext.plot(x, values, color=color, marker="hd")
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_reporting.py::test_plot_policy_uses_sparse_scatter_and_dense_lines -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py
git commit -m "feat: enforce sparse-vs-dense plot rendering policy"
```

### Task 6: Profile Row Conditional Layout + No-Profile Fallback Narrative

**Skill refs:** @superpowers:test-driven-development

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Test: `tests/test_reporting.py`

**Step 1: Write the failing test**

```python
def test_profile_row_omitted_when_profile_data_missing() -> None:
    rendered = render_human_report(_sample_report(include_profile=False))
    assert "Profile Runtime Plot" not in rendered
    assert "Profile Memory Plot" not in rendered


def test_profile_row_present_when_profile_data_available() -> None:
    rendered = render_human_report(_sample_report(include_profile=True))
    assert "Profile Runtime Plot" in rendered
    assert "Profile Memory Plot" in rendered
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_reporting.py::test_profile_row_omitted_when_profile_data_missing tests/test_reporting.py::test_profile_row_present_when_profile_data_available -q`
Expected: FAIL if row gating is not fully explicit.

**Step 3: Write minimal implementation**

```python
# reporting.py
if profile_calls:
    # render profile row
else:
    # render compact key-notes pane or skip row cleanly
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_reporting.py::test_profile_row_omitted_when_profile_data_missing tests/test_reporting.py::test_profile_row_present_when_profile_data_available -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py
git commit -m "feat: make profile lane conditional and clean"
```

### Task 7: CLI/README Contract and End-to-End Verification

**Skill refs:** @superpowers:verification-before-completion

**Files:**
- Modify: `README.md`
- Test: `tests/test_cli.py`

**Step 1: Write/update failing test assertions**

```python
def test_default_mode_outputs_human_dashboard_sections(...):
    # assert new pane names appear in human output
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_cli.py::test_default_mode_outputs_human_dashboard_sections -q`
Expected: FAIL before docs/strings are aligned.

**Step 3: Write minimal implementation and docs**

```markdown
# README
- explain tri-objective multi-column human dashboard
- explain width adaptation and profile row behavior
```

**Step 4: Run full verification to verify it passes**

Run: `uv run --group dev ruff check . && uv run --group dev pyright && uv run --group dev pytest -q`
Expected: all commands PASS.

**Step 5: Commit**

```bash
git add README.md tests/test_cli.py
git commit -m "docs: describe redesigned human dashboard and verify cli contract"
```

### Task 8: Final Visual QA Snapshot Pass

**Skill refs:** @superpowers:verification-before-completion

**Files:**
- Modify (if needed): `src/circuit_estimation/reporting.py`

**Step 1: Capture wide/medium/narrow outputs**

Run:

```bash
COLUMNS=140 uv run main.py --profile > /tmp/ce_dash_wide.out
COLUMNS=100 uv run main.py --profile > /tmp/ce_dash_medium.out
COLUMNS=80 uv run main.py --profile > /tmp/ce_dash_narrow.out
```

Expected: all outputs render with readable pane ordering and no broken sections.

**Step 2: Validate tri-objective coverage manually**

- Readiness is obvious in top row.
- Interpretability visible via budget/layer lanes.
- Performance diagnostics visible when profile enabled.

**Step 3: Apply minimal polish fixes if needed**

```python
# reporting.py
# tighten labels, spacing, or legend wording only if required
```

**Step 4: Re-run full verification**

Run: `uv run --group dev ruff check . && uv run --group dev pyright && uv run --group dev pytest -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py
git commit -m "chore: finalize dashboard readability polish"
```

---

Plan complete and saved to `docs/plans/2026-03-01-human-dashboard-redesign-implementation-plan.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
