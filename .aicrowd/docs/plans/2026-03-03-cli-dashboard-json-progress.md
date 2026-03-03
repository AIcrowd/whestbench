# CLI JSON Contract + Append-Only Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Ensure `--json` always emits JSON-only output while human mode gets append-only startup context + progress + polished dashboard sections.

**Architecture:** Introduce strict mode routing in CLI, add progress-event hooks in scoring loops, and split human rendering into pre-run and post-run appendable sections. Keep rendering and scoring decoupled by using callbacks and small reporting helpers.

**Tech Stack:** Python 3.10, argparse, rich, tqdm, pytest.

---

### Task 1: Fix argv Dispatch Contract (entrypoint bug)

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

Add a regression test asserting `main()` (no args) respects `sys.argv` and honors `--json`.

```python
def test_main_without_argv_uses_sys_argv(monkeypatch, capsys):
    monkeypatch.setattr(cli, "score_estimator_report", lambda *_a, **_k: _sample_report(profile_enabled=False, detail="raw"))
    monkeypatch.setattr(cli, "render_agent_report", lambda _r: '{\n  "mode": "agent"\n}\n')
    monkeypatch.setattr(cli.sys, "argv", ["cestim", "--json"])

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert json.loads(captured.out) == {"mode": "agent"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py::test_main_without_argv_uses_sys_argv -v`
Expected: FAIL because current `main()` ignores `sys.argv`.

**Step 3: Write minimal implementation**

In `main`, replace:

```python
args_list = list(argv or [])
```

with:

```python
args_list = list(sys.argv[1:] if argv is None else argv)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py::test_main_without_argv_uses_sys_argv -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py tests/test_cli.py
git commit -m "fix(cli): honor sys.argv when main() called without argv"
```

### Task 2: Add Progress Hook in Scoring Engine

**Files:**
- Modify: `src/circuit_estimation/scoring.py`
- Test: `tests/test_scoring_module.py`

**Step 1: Write the failing test**

Add tests for both scoring paths asserting progress callback receives exactly `budgets * circuits` events.

```python
def test_score_estimator_report_emits_progress_events():
    events = []
    report = score_estimator_report(..., progress=lambda e: events.append(e))
    assert report["results"]["final_score"] >= 0.0
    assert len(events) == expected_units
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_scoring_module.py::test_score_estimator_report_emits_progress_events -v`
Expected: FAIL (`progress` kwarg unsupported).

**Step 3: Write minimal implementation**

- Add optional `progress: Callable[[dict[str, int]], None] | None = None` to:
  - `score_estimator_report`
  - `score_submission_report`
- Emit callback after each completed circuit within each budget loop.
- Include fields at minimum: `budget_index`, `budget`, `circuit_index`, `completed`, `total`.

**Step 4: Run test to verify it passes**

Run:
- `pytest tests/test_scoring_module.py::test_score_estimator_report_emits_progress_events -v`
- `pytest tests/test_scoring_module.py::test_score_submission_report_emits_progress_events -v`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring_module.py
git commit -m "feat(scoring): emit progress callbacks for budget x circuit units"
```

### Task 3: Pre-Run Rendering Helpers + Score Panel Rename

**Files:**
- Modify: `src/circuit_estimation/reporting.py`
- Test: `tests/test_reporting.py`

**Step 1: Write the failing test**

Add/adjust tests to assert:
- panel title is `Scores` (not `Readiness Scorecard`)
- no `Rich Dashboard` subtitle in human report
- no hint lines embedded in rendered post-run dashboard sections
- new helper renders `Run Context` and `Hardware & Runtime` adjacent.

**Step 2: Run test to verify it fails**

Run:
- `pytest tests/test_reporting.py::test_render_human_mode_includes_expected_sections_without_profile -v`
- `pytest tests/test_reporting.py::test_human_report_uses_two_column_plus_stack_layout_on_medium_width -v`

Expected: FAIL on old strings/layout.

**Step 3: Write minimal implementation**

- Rename `_score_summary_panel` title to `Scores`.
- Remove `Rich Dashboard` subtitle.
- Remove hint lines from `render_human_report`.
- Add helper for pre-run two-pane row (Run Context + Hardware & Runtime) that always renders side-by-side.
- Add helper to render post-run sections appendably (`Scores`, `Budget`, `Layer Diagnostics`, `Profile`).

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_reporting.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/reporting.py tests/test_reporting.py
git commit -m "feat(reporting): split pre-run panes and rename score panel"
```

### Task 4: CLI Human Flow (append-only, estimator identity, rich/classic tqdm)

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Test: `tests/test_cli.py`, `tests/test_cli_fallback.py`

**Step 1: Write the failing test**

Add tests for:
- `--json` mode produces JSON-only output with no human hints/panels.
- human mode prints pre-run content before scoring.
- estimator identity line includes class and estimator path in participant `run`.
- `--disable-rich` selects classic tqdm path.

**Step 2: Run test to verify it fails**

Run:
- `pytest tests/test_cli.py::test_json_flag_stdout_is_json_only -v`
- `pytest tests/test_cli.py::test_participant_run_prints_estimator_identity -v`
- `pytest tests/test_cli.py::test_disable_rich_uses_classic_tqdm -v`

Expected: FAIL on new behavior assertions.

**Step 3: Write minimal implementation**

- Add `--disable-rich` to legacy parser and participant `run` parser.
- Human-mode flow:
  - print title + hints
  - print pre-run two-pane context row
  - print estimator identity line (`class=... path=...`)
  - run scoring with progress callback hooked to tqdm (rich/classic)
  - append post-run dashboard sections
- JSON flow bypasses all above and prints JSON only.

**Step 4: Run test to verify it passes**

Run:
- `pytest tests/test_cli.py tests/test_cli_fallback.py -q`

Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py tests/test_cli.py tests/test_cli_fallback.py
git commit -m "feat(cli): append-only human flow with estimator identity and tqdm progress"
```

### Task 5: Docs Updates for New CLI UX

**Files:**
- Modify: `docs/reference/cli-reference.md`
- Modify: `docs/getting-started/install-and-cli-quickstart.md`
- Test: `tests/test_docs_quality.py`

**Step 1: Write the failing test**

Add/update docs quality expectations for:
- `--disable-rich`
- strict `--json` output contract
- progress behavior notes in human mode.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_docs_quality.py -q`
Expected: FAIL due missing docs updates.

**Step 3: Write minimal implementation**

Document:
- `--json`: machine-only JSON output, no dashboard.
- `--disable-rich`: plain output + classic tqdm.
- human flow ordering: startup context -> progress -> appended results sections.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_docs_quality.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/reference/cli-reference.md docs/getting-started/install-and-cli-quickstart.md tests/test_docs_quality.py
git commit -m "docs(cli): describe strict json mode and rich/classic progress behavior"
```

### Task 6: Final Verification Before Completion

**Files:**
- Verify repository state and outputs.

**Step 1: Run focused suite**

Run:
```bash
pytest tests/test_cli.py tests/test_cli_fallback.py tests/test_reporting.py tests/test_scoring_module.py -q
```
Expected: all pass.

**Step 2: Run full test suite**

Run:
```bash
pytest -q
```
Expected: all pass.

**Step 3: Manual smoke commands**

Run:
```bash
uv tool install -e .
cestim --json
cestim --profile --show-diagnostic-plots
cestim --disable-rich
```
Expected:
- first command succeeds
- `--json` returns valid JSON only
- rich command shows startup context, progress, then appended panes
- `--disable-rich` uses classic tqdm and plain rendering path.

**Step 4: Verify diff and status**

Run:
```bash
git status --short
git diff -- src/circuit_estimation/cli.py src/circuit_estimation/reporting.py src/circuit_estimation/scoring.py
```
Expected: only intended changes present.

**Step 5: Commit final integration if needed**

```bash
git add -A
git commit -m "feat(cli): strict json contract and append-only human dashboard flow"
```
