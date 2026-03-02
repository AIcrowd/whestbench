# Textual Dashboard Pane + Plotext Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current partially-usable Textual UI with a production-grade dark pane-based dashboard where Summary fully matches legacy Rich coverage and all substantive charts render via `plotext`.

**Architecture:** Keep CLI routing unchanged (`--agent-mode` JSON, human Textual default, static fallback path), but refactor Textual rendering into strict pane builders driven by a richer `DashboardState` and a centralized `plots.py` module. Summary will follow the approved legacy matrix contract and deep tabs will become structured drill-down surfaces.

**Tech Stack:** Python 3.10+, Textual, plotext, Rich (fallback), pytest, ruff, pyright.

---

**Required implementation skills:** `@test-driven-development`, `@systematic-debugging`, `@verification-before-completion`.

### Task 1: Add plot module skeleton and failing contract tests

**Files:**
- Create: `src/circuit_estimation/textual_dashboard/plots.py`
- Create: `tests/test_textual_plotext_rendering.py`

**Step 1: Write the failing test**

```python
def test_budget_frontier_plot_returns_chart_and_legend() -> None:
    chart, legend = build_budget_frontier_plot([10, 100], [0.22, 0.036], [0.2, 0.04], width=60, height=12)
    assert chart.strip()
    assert "adjusted_mse" in legend
```

Add additional failing test for fallback:

```python
def test_plot_builder_fallback_when_plotext_raises(monkeypatch) -> None:
    monkeypatch.setattr(plot_module, "_render_plotext", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    chart, legend = build_budget_frontier_plot(...)
    assert "plot unavailable" in legend.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_plotext_rendering.py -q`  
Expected: FAIL (missing module/functions).

**Step 3: Write minimal implementation**

Create `plots.py` with initial APIs:
- `build_budget_frontier_plot(...) -> tuple[str, str]`
- `build_budget_runtime_plot(...) -> tuple[str, str]`
- `build_layer_trend_plot(...) -> tuple[str, str]`
- `build_profile_runtime_plot(...) -> tuple[str, str]`
- `build_profile_memory_plot(...) -> tuple[str, str]`

Implement with `plotext` and fallback handling.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_plotext_rendering.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/plots.py tests/test_textual_plotext_rendering.py
git commit -m "feat: add plotext chart builders with fallback"
```

### Task 2: Expand dashboard state contract for legacy-complete panes

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/state.py`
- Modify: `tests/test_textual_dashboard_state.py`

**Step 1: Write the failing test**

Add tests asserting derived state includes:
- host metadata fields
- budget arrays for scores/mse/runtime
- profile arrays for wall/cpu/rss/peak
- layer summary stats helpers

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_dashboard_state.py -q`  
Expected: FAIL on missing/new fields.

**Step 3: Write minimal implementation**

Augment `DashboardDerivedState` and derivation logic to expose all Summary matrix inputs.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_dashboard_state.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/state.py tests/test_textual_dashboard_state.py
git commit -m "feat: enrich dashboard state for legacy pane coverage"
```

### Task 3: Implement strict Summary layout matrix (status strip + 3 rows)

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/views/summary.py`
- Modify: `src/circuit_estimation/textual_dashboard/widgets.py`
- Modify: `tests/test_textual_summary_view.py`
- Create: `tests/test_textual_summary_layout_contract.py`

**Step 1: Write the failing test**

Add layout-contract tests for pane ids/anchors:
- `summary-status-strip`
- `summary-run-context`
- `summary-readiness`
- `summary-hardware-runtime`
- `summary-budget-table`
- `summary-budget-frontier`
- `summary-budget-runtime`
- `summary-layer-diagnostics`
- `summary-layer-trend`
- `summary-profile-*` (or explicit unavailable panel)

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_summary_view.py tests/test_textual_summary_layout_contract.py -q`  
Expected: FAIL on missing pane ids/content.

**Step 3: Write minimal implementation**

Rebuild Summary to exactly match approved matrix for all width tiers.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_summary_view.py tests/test_textual_summary_layout_contract.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/views/summary.py src/circuit_estimation/textual_dashboard/widgets.py tests/test_textual_summary_view.py tests/test_textual_summary_layout_contract.py
git commit -m "feat: implement legacy-complete summary pane matrix"
```

### Task 4: Wire Summary plots to plotext charts (no placeholder text)

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/views/summary.py`
- Modify: `tests/test_textual_dashboard_layout.py`

**Step 1: Write the failing test**

Add assertion that Summary plot panes render chart-like content from plot builder (not only titles):

```python
assert "┤" in summary_plot_text or "|" in summary_plot_text
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_dashboard_layout.py -q`  
Expected: FAIL with placeholder-only output.

**Step 3: Write minimal implementation**

Use `plots.py` builders in Summary panes and add legends.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_dashboard_layout.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/views/summary.py tests/test_textual_dashboard_layout.py
git commit -m "feat: render summary pane charts via plotext"
```

### Task 5: Upgrade Budgets tab to structured pane + plotext contract

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/views/budgets.py`
- Modify: `tests/test_textual_budget_layer_views.py`

**Step 1: Write the failing test**

Add assertions for panes:
- budgets-table-panel
- budgets-frontier-panel
- budgets-runtime-panel
- budgets-insight-panel

Plus chart body content assertions.

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_budget_layer_views.py -q`  
Expected: FAIL on missing pane ids/content.

**Step 3: Write minimal implementation**

Render Budget tab with dedicated table + two `plotext` plots + insight panel.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_budget_layer_views.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/views/budgets.py tests/test_textual_budget_layer_views.py
git commit -m "feat: implement budget tab pane contract with plotext"
```

### Task 6: Upgrade Layers tab to structured pane + plotext contract

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/views/layers.py`
- Modify: `tests/test_textual_budget_layer_views.py`

**Step 1: Write the failing test**

Add assertions for panes:
- layers-stats-panel
- layers-trend-panel
- layers-insight-panel

and chart body content.

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_budget_layer_views.py -q`  
Expected: FAIL.

**Step 3: Write minimal implementation**

Render layer stats + primary trend plot + optional smoothed/overlay plot.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_budget_layer_views.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/views/layers.py tests/test_textual_budget_layer_views.py
git commit -m "feat: implement layer tab pane contract with plotext"
```

### Task 7: Upgrade Performance tab to profile composite contract

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/views/performance.py`
- Modify: `tests/test_textual_performance_data_views.py`

**Step 1: Write the failing test**

Add assertions for:
- performance-summary-panel
- performance-runtime-plot
- performance-memory-plot
- performance-outlier-panel
- explicit unavailable panel when profile missing

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_performance_data_views.py -q`  
Expected: FAIL.

**Step 3: Write minimal implementation**

Implement full profile composite and missing-profile fallback pane.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_performance_data_views.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/views/performance.py tests/test_textual_performance_data_views.py
git commit -m "feat: implement performance tab composite panes"
```

### Task 8: Data tab structure and navigation polish

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/views/data.py`
- Modify: `src/circuit_estimation/textual_dashboard/app.py`
- Modify: `tests/test_textual_dashboard_navigation.py`

**Step 1: Write the failing test**

Add tests that Data tab exposes segmented sections (run_meta/run_config/results/profile_calls) with section ids and keyboard navigation stability.

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_dashboard_navigation.py tests/test_textual_performance_data_views.py -q`  
Expected: FAIL on missing sections/ids.

**Step 3: Write minimal implementation**

Build segmented Data pane and verify tab focus/shortcuts remain intact.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_dashboard_navigation.py tests/test_textual_performance_data_views.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/views/data.py src/circuit_estimation/textual_dashboard/app.py tests/test_textual_dashboard_navigation.py tests/test_textual_performance_data_views.py
git commit -m "feat: segment data tab and preserve navigation ergonomics"
```

### Task 9: Dark visual system finalization (black baseline)

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/dashboard.tcss`
- Modify: `tests/test_textual_dashboard_style_contract.py`

**Step 1: Write the failing test**

Add strict palette/contrast/layout token checks for dark baseline and pane readability.

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_dashboard_style_contract.py -q`  
Expected: FAIL.

**Step 3: Write minimal implementation**

Finalize theme, spacing, borders, typography hierarchy for skimability.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_dashboard_style_contract.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/dashboard.tcss tests/test_textual_dashboard_style_contract.py
git commit -m "feat: finalize dark pane visual system for textual dashboard"
```

### Task 10: Update static fallback copy and docs consistency

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Modify: `README.md`
- Modify: `.aicrowd/docs/context/mvp-technical-snapshot.md`
- Modify: `.aicrowd/docs/context/python-runtime-refactor-decisions.md`

**Step 1: Write the failing test**

Add/adjust docs assertions so removed flags are not referenced and fallback copy matches current behavior.

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_docs_quality.py tests/test_cli.py tests/test_reporting.py -q`  
Expected: FAIL on stale wording.

**Step 3: Write minimal implementation**

Align docs + fallback text with final UI behavior and plotting stack.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_docs_quality.py tests/test_cli.py tests/test_reporting.py -q`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py README.md .aicrowd/docs/context/mvp-technical-snapshot.md .aicrowd/docs/context/python-runtime-refactor-decisions.md
git commit -m "docs: align dashboard contract and fallback messaging"
```

### Task 11: Full verification and release-quality check

**Files:**
- Modify: none (unless fixes needed)

**Step 1: Run quality gates**

Run:
- `uv run --group dev ruff check .`
- `uv run --group dev ruff format --check .`
- `uv run --group dev pyright`
- `uv run --group dev pytest -m "not exhaustive"`

Expected: all PASS.

**Step 2: Manual smoke checks**

Run:
- `uv run main.py` (Textual dashboard)
- `uv run main.py --agent-mode` (JSON only)
- `TERM=dumb uv run main.py` (static fallback)

Expected:
- Summary includes all legacy sections/panes,
- plots render via `plotext`,
- quit keys (`Esc`, `Ctrl+C`, `q`) work,
- fallback path remains robust.

**Step 3: Final commit if needed**

```bash
git add -A
git commit -m "chore: finalize textual pane+plotext redesign verification"
```

Skip if no files changed.

---

Plan complete and saved to `.aicrowd/docs/plans/2026-03-01-textual-dashboard-pane-plotext-redesign-implementation-plan.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
