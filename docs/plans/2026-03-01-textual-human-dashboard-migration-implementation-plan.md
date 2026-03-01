# Textual Human Dashboard Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace default human-mode Rich output with a Textual multi-tab dashboard (summary-first), simplify CLI to `--agent-mode` only, and preserve robust static fallback behavior.

**Architecture:** Introduce a new Textual UI package under `src/circuit_estimation/textual_dashboard/` with an app shell, shared state adapter, and tab view modules. Keep scoring as the source of truth and route CLI output to either agent JSON, Textual app, or static fallback depending on runtime capabilities. Use strict TDD for CLI contracts, state derivations, and Textual view anchors.

**Tech Stack:** Python 3.10+, Textual, Rich (static fallback), NumPy, pytest, ruff, pyright.

---

**Required implementation skills:** `@test-driven-development`, `@verification-before-completion`, `@systematic-debugging` (if tests fail unexpectedly).

### Task 1: Add Textual dependency and dashboard package skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `src/circuit_estimation/textual_dashboard/__init__.py`
- Create: `src/circuit_estimation/textual_dashboard/app.py`
- Create: `src/circuit_estimation/textual_dashboard/state.py`
- Create: `src/circuit_estimation/textual_dashboard/views/summary.py`
- Create: `src/circuit_estimation/textual_dashboard/views/budgets.py`
- Create: `src/circuit_estimation/textual_dashboard/views/layers.py`
- Create: `src/circuit_estimation/textual_dashboard/views/performance.py`
- Create: `src/circuit_estimation/textual_dashboard/views/data.py`
- Test: `tests/test_textual_dashboard_bootstrap.py`

**Step 1: Write the failing test**

```python
from circuit_estimation.textual_dashboard.app import DashboardApp


def test_dashboard_app_importable() -> None:
    app = DashboardApp(report={})
    assert app is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_dashboard_bootstrap.py::test_dashboard_app_importable -q`
Expected: FAIL with import/module error.

**Step 3: Write minimal implementation**

```python
# src/circuit_estimation/textual_dashboard/app.py
from textual.app import App


class DashboardApp(App[None]):
    def __init__(self, report: dict, **kwargs):
        super().__init__(**kwargs)
        self.report = report
```

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_dashboard_bootstrap.py::test_dashboard_app_importable -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add pyproject.toml src/circuit_estimation/textual_dashboard tests/test_textual_dashboard_bootstrap.py
git commit -m "feat: scaffold textual dashboard package"
```

### Task 2: Introduce dashboard state adapter with derived metrics

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/state.py`
- Test: `tests/test_textual_dashboard_state.py`

**Step 1: Write the failing test**

```python
from circuit_estimation.textual_dashboard.state import build_dashboard_state


def test_state_derives_budget_extrema(sample_report: dict) -> None:
    state = build_dashboard_state(sample_report)
    assert state.derived.best_budget_score <= state.derived.worst_budget_score
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_dashboard_state.py::test_state_derives_budget_extrema -q`
Expected: FAIL with missing function/attributes.

**Step 3: Write minimal implementation**

```python
def build_dashboard_state(report: dict) -> DashboardState:
    # Compute shared metrics once and store in typed structures.
    ...
```

Include derivations for:
- final score
- best/worst budget score
- budget score spread
- layer mse aggregates
- profile summary presence/absence

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_dashboard_state.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/state.py tests/test_textual_dashboard_state.py
git commit -m "feat: add dashboard state derivation layer"
```

### Task 3: Build tab shell and default Summary-first navigation

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/app.py`
- Modify: `src/circuit_estimation/textual_dashboard/__init__.py`
- Test: `tests/test_textual_dashboard_navigation.py`

**Step 1: Write the failing test**

```python
def test_default_tab_is_summary(textual_pilot):
    app = DashboardApp(report=sample_report())
    with textual_pilot(app) as pilot:
        assert app.active_tab == "summary"
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_dashboard_navigation.py::test_default_tab_is_summary -q`
Expected: FAIL with missing tab state.

**Step 3: Write minimal implementation**

Implement tab container and keys:
- tabs: summary, budgets, layers, performance, data
- key bindings: `1`-`5`, `q`, `?`, `r`
- default active tab set to summary

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_dashboard_navigation.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/app.py src/circuit_estimation/textual_dashboard/__init__.py tests/test_textual_dashboard_navigation.py
git commit -m "feat: add summary-first tab navigation shell"
```

### Task 4: Implement Summary view with full legacy-coverage anchors

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/views/summary.py`
- Modify: `src/circuit_estimation/textual_dashboard/app.py`
- Test: `tests/test_textual_summary_view.py`

**Step 1: Write the failing test**

```python
def test_summary_contains_legacy_sections(textual_pilot):
    app = DashboardApp(report=sample_report_with_profile())
    with textual_pilot(app):
        rendered = app.export_plain_text()
    assert "Run Context" in rendered
    assert "Readiness Scorecard" in rendered
    assert "Budget" in rendered
    assert "Layer Diagnostics" in rendered
    assert "Profile" in rendered
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_summary_view.py::test_summary_contains_legacy_sections -q`
Expected: FAIL with missing section anchors.

**Step 3: Write minimal implementation**

Implement a scrollable Summary layout that includes:
- executive top strip
- interesting plots area (placeholder widgets allowed initially)
- all legacy section anchors and key metrics

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_summary_view.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/views/summary.py src/circuit_estimation/textual_dashboard/app.py tests/test_textual_summary_view.py
git commit -m "feat: implement summary view with legacy metric coverage"
```

### Task 5: Implement Budgets and Layers deep-dive tabs

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/views/budgets.py`
- Modify: `src/circuit_estimation/textual_dashboard/views/layers.py`
- Modify: `src/circuit_estimation/textual_dashboard/app.py`
- Test: `tests/test_textual_budget_layer_views.py`

**Step 1: Write the failing test**

```python
def test_budget_and_layer_tabs_render_expected_headers(textual_pilot):
    app = DashboardApp(report=sample_report())
    with textual_pilot(app) as pilot:
        pilot.press("2")
        assert "Budget Analysis" in app.export_plain_text()
        pilot.press("3")
        assert "Layer Analysis" in app.export_plain_text()
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_budget_layer_views.py::test_budget_and_layer_tabs_render_expected_headers -q`
Expected: FAIL due missing views.

**Step 3: Write minimal implementation**

Add:
- budget table + trend summary block in `Budgets` tab
- layer diagnostics table + trend block in `Layers` tab

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_budget_layer_views.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/views/budgets.py src/circuit_estimation/textual_dashboard/views/layers.py src/circuit_estimation/textual_dashboard/app.py tests/test_textual_budget_layer_views.py
git commit -m "feat: add budgets and layers deep-dive views"
```

### Task 6: Implement Performance and Data tabs

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/views/performance.py`
- Modify: `src/circuit_estimation/textual_dashboard/views/data.py`
- Modify: `src/circuit_estimation/textual_dashboard/app.py`
- Test: `tests/test_textual_performance_data_views.py`

**Step 1: Write the failing test**

```python
def test_performance_and_data_tabs_render(textual_pilot):
    app = DashboardApp(report=sample_report_with_profile())
    with textual_pilot(app) as pilot:
        pilot.press("4")
        assert "Performance" in app.export_plain_text()
        pilot.press("5")
        assert "Raw Data" in app.export_plain_text()
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_performance_data_views.py::test_performance_and_data_tabs_render -q`
Expected: FAIL due missing tab content.

**Step 3: Write minimal implementation**

Implement:
- performance summary + runtime/memory diagnostics widgets
- raw/structured data view with key JSON anchors

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_performance_data_views.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard/views/performance.py src/circuit_estimation/textual_dashboard/views/data.py src/circuit_estimation/textual_dashboard/app.py tests/test_textual_performance_data_views.py
git commit -m "feat: add performance and data tabs"
```

### Task 7: Simplify CLI surface and route human mode to Textual

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Modify: `src/circuit_estimation/reporting.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
def test_removed_human_flags_are_no_longer_accepted(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--profile"])
```

Add tests for:
- default path calls Textual launcher
- `--agent-mode` still bypasses Textual and prints JSON

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_cli.py -q`
Expected: FAIL on old flag handling and routing expectations.

**Step 3: Write minimal implementation**

- Remove parser args: `--detail`, `--profile`, `--show-diagnostic-plots`.
- Always call scorer in human mode with profiling enabled and full dashboard payload.
- Keep `--agent-mode` path unchanged in contract.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_cli.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py src/circuit_estimation/reporting.py tests/test_cli.py
git commit -m "feat: route default human mode to textual and simplify cli flags"
```

### Task 8: Add Textual capability checks and static fallback behavior

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Modify: `src/circuit_estimation/reporting.py`
- Test: `tests/test_cli_fallback.py`

**Step 1: Write the failing test**

```python
def test_non_tty_human_mode_falls_back_to_static(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdout.isatty", lambda: False)
    exit_code = cli.main([])
    out = capsys.readouterr()
    assert exit_code == 0
    assert "Textual UI unavailable" in out.err
    assert "Circuit Estimation Report" in out.out
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_cli_fallback.py::test_non_tty_human_mode_falls_back_to_static -q`
Expected: FAIL (fallback not implemented).

**Step 3: Write minimal implementation**

- Add `supports_textual_dashboard()` gate.
- Wrap Textual launch in safe `try/except` and fallback to static renderer.
- Emit one-line fallback notice to stderr.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_cli_fallback.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py src/circuit_estimation/reporting.py tests/test_cli_fallback.py
git commit -m "feat: add resilient textual fallback to static report"
```

### Task 9: Apply scientific-editorial styling and responsive layout polish

**Files:**
- Modify: `src/circuit_estimation/textual_dashboard/app.py`
- Modify: `src/circuit_estimation/textual_dashboard/views/summary.py`
- Modify: `src/circuit_estimation/textual_dashboard/views/budgets.py`
- Modify: `src/circuit_estimation/textual_dashboard/views/layers.py`
- Modify: `src/circuit_estimation/textual_dashboard/views/performance.py`
- Test: `tests/test_textual_dashboard_style_contract.py`

**Step 1: Write the failing test**

```python
def test_style_contract_includes_scientific_editorial_tokens() -> None:
    css = load_dashboard_css()
    assert "--accent-accuracy" in css
    assert "--accent-runtime" in css
    assert "--accent-score" in css
```

Add a layout test asserting summary stays readable in narrow width.

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_textual_dashboard_style_contract.py -q`
Expected: FAIL because style tokens/layout rules are missing.

**Step 3: Write minimal implementation**

- Add centralized theme tokens/CSS.
- Enforce responsive layout tiers (wide/medium/narrow).
- Keep focus outlines and keyboard affordances visible.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_textual_dashboard_style_contract.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/textual_dashboard tests/test_textual_dashboard_style_contract.py
git commit -m "feat: apply scientific editorial textual theme and responsive layout"
```

### Task 10: Documentation updates for new human-mode contract

**Files:**
- Modify: `README.md`
- Modify: `docs/context/mvp-technical-snapshot.md`
- Modify: `docs/context/python-runtime-refactor-decisions.md`
- Test: `tests/test_docs_quality.py`

**Step 1: Write the failing test**

```python
def test_readme_documents_textual_default() -> None:
    readme = Path("README.md").read_text()
    assert "Textual" in readme
    assert "--agent-mode" in readme
    assert "--profile" not in readme
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_docs_quality.py -q`
Expected: FAIL because docs still mention removed flags.

**Step 3: Write minimal implementation**

- Document Textual default human UX.
- Remove removed-flag references.
- Document fallback behavior and machine mode continuity.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_docs_quality.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add README.md docs/context/mvp-technical-snapshot.md docs/context/python-runtime-refactor-decisions.md tests/test_docs_quality.py
git commit -m "docs: describe textual-first human mode and updated cli contract"
```

### Task 11: Final verification before completion

**Files:**
- Modify: none
- Test: full changed surface

**Step 1: Run lint/type/test verification**

Run:
- `uv run --group dev ruff check .`
- `uv run --group dev ruff format --check .`
- `uv run --group dev pyright`
- `uv run --group dev pytest -m "not exhaustive"`

Expected: all PASS.

**Step 2: Manual behavior checks**

Run:
- `uv run main.py` (Textual launches)
- `uv run main.py --agent-mode` (JSON only)

Expected:
- human default enters Textual UI
- agent mode prints JSON without UI framing

**Step 3: Final commit (if verification fixes were needed)**

```bash
git add -A
git commit -m "chore: finalize textual dashboard migration with verification"
```

Skip this commit if no code changed during final verification.

---

Plan complete and saved to `docs/plans/2026-03-01-textual-human-dashboard-migration-implementation-plan.md`. Two execution options:

1. Subagent-Driven (this session) - I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) - Open new session with executing-plans, batch execution with checkpoints

Which approach?
