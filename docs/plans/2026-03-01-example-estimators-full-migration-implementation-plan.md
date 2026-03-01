# Example Estimators Full Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fully replace function-style starter estimators with class-based file-entrypoint examples under `examples/estimators/`, and make the local default evaluation path use those examples through the runner/loader contract.

**Architecture:** Move algorithm implementations into participant-style example files (`Estimator(BaseEstimator)`), switch default CLI/scoring execution to entrypoint-driven runner calls, then remove function-based estimator dependencies in tests/docs. Keep scoring math and report schema behavior stable while changing estimator integration boundaries.

**Tech Stack:** Python 3.10+, NumPy, argparse, pytest, ruff, pyright, uv

---

### Task 1: Add class-based starter estimator examples in `examples/estimators/`

**Files:**
- Create: `examples/estimators/mean_propagation.py`
- Create: `examples/estimators/covariance_propagation.py`
- Create: `examples/estimators/combined_estimator.py`
- Test: `tests/test_example_estimators.py`

**Step 1: Write the failing test**

```python
def test_mean_example_estimator_returns_depth_width_tensor(tmp_path):
    ...

def test_covariance_example_estimator_returns_depth_width_tensor(tmp_path):
    ...

def test_combined_example_switches_mode_by_budget(tmp_path):
    ...
```

Use loader-based class execution from file paths and assert shape/behavior.

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_example_estimators.py -q`
Expected: FAIL because example files do not exist.

**Step 3: Write minimal implementation**

Create each file with `Estimator(BaseEstimator)`:

- `mean_propagation.py`: first-moment propagation in `predict`
- `covariance_propagation.py`: covariance propagation in `predict`
- `combined_estimator.py`: budget switch in `predict` (`cov` when `budget >= 30 * circuit.n`)

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_example_estimators.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add examples/estimators/mean_propagation.py examples/estimators/covariance_propagation.py examples/estimators/combined_estimator.py tests/test_example_estimators.py
git commit -m "feat: add class-based starter estimator examples"
```

### Task 2: Remove function-style estimator API usage in tests and runtime surfaces

**Files:**
- Modify: `tests/test_estimators.py`
- Modify: `tests/test_estimators_module.py`
- Modify: `tests/test_docs_quality.py`
- Modify: `src/circuit_estimation/estimators.py`

**Step 1: Write the failing test update**

Convert estimator tests from direct function imports to class/file-based example tests (or remove obsolete function assertions).

Add guard check:

```python
def test_no_function_style_estimator_entrypoints_exposed():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_estimators.py tests/test_estimators_module.py tests/test_docs_quality.py -q`
Expected: FAIL until imports/expectations are migrated.

**Step 3: Write minimal implementation**

- Replace old function-based test calls with class-based example calls.
- Update docs-quality module list/phrases as needed.
- Replace `src/circuit_estimation/estimators.py` with a minimal module doc explaining migration and no participant callable API.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_estimators.py tests/test_estimators_module.py tests/test_docs_quality.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_estimators.py tests/test_estimators_module.py tests/test_docs_quality.py src/circuit_estimation/estimators.py
git commit -m "refactor: remove function-style estimator api dependencies"
```

### Task 3: Switch default local run/report path to file-entrypoint runner flow

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Modify: `src/circuit_estimation/scoring.py`
- Test: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
def test_run_default_report_uses_combined_example_entrypoint_via_runner(...):
    ...
```

Assert default path no longer calls function-style `combined_estimator` directly.

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_cli.py -q`
Expected: FAIL until `run_default_report` path is switched.

**Step 3: Write minimal implementation**

- Update `run_default_report` to call `score_submission_report` with:
  - `InProcessRunner` (or configured default runner)
  - `EstimatorEntrypoint` pointing to `examples/estimators/combined_estimator.py`
- Keep output mode semantics unchanged.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_cli.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py src/circuit_estimation/scoring.py tests/test_cli.py
git commit -m "feat: route default local runs through example estimator entrypoint"
```

### Task 4: Ensure participant commands default to example-first guidance

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Test: `tests/test_cli_participant_commands.py`

**Step 1: Write the failing test**

```python
def test_init_and_run_help_text_reference_examples_estimators_path(...):
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_cli_participant_commands.py -q`
Expected: FAIL until help/defaults mention `examples/estimators`.

**Step 3: Write minimal implementation**

- Update CLI descriptions/help to point users to `examples/estimators/` starter files.
- Ensure run/validate docs in command output align with class-based API messaging.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_cli_participant_commands.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py tests/test_cli_participant_commands.py
git commit -m "docs(cli): point participant workflow to examples estimators"
```

### Task 5: Update README + context docs for full migration

**Files:**
- Modify: `README.md`
- Modify: `docs/context/mvp-technical-snapshot.md`
- Modify: `docs/context/python-runtime-refactor-decisions.md`

**Step 1: Write the failing docs check**

Run:

```bash
rg -n "examples/estimators|BaseEstimator|Estimator\(BaseEstimator\)|function-style|full migration" README.md docs/context/*.md
```

Expected: missing wording before update.

**Step 2: Run check to verify it fails**

Run the same `rg` command and confirm missing required phrases.

**Step 3: Write minimal implementation**

Document:

- starter examples now live in `examples/estimators/`
- estimator contract is class-based only
- old function-style API is no longer participant path
- default local run uses example file entrypoint and runner

**Step 4: Run docs check to verify it passes**

Run the same `rg` command; expected required phrases present.

**Step 5: Commit**

```bash
git add README.md docs/context/mvp-technical-snapshot.md docs/context/python-runtime-refactor-decisions.md
git commit -m "docs: define full class-based estimator migration"
```

### Task 6: Run targeted migration verification suite

**Files:**
- Modify as needed from earlier tasks

**Step 1: Run migration-focused tests**

Run:

```bash
uv run --group dev pytest tests/test_example_estimators.py tests/test_estimators.py tests/test_estimators_module.py tests/test_cli.py tests/test_cli_participant_commands.py tests/test_scoring_module.py tests/test_docs_quality.py -q
```

Expected: PASS.

**Step 2: Run typing + lint checks**

Run:

```bash
uv run --group dev pyright
uv run --group dev ruff check .
```

Expected: PASS.

**Step 3: Run formatting check**

Run:

```bash
uv run --group dev ruff format --check .
```

Expected: PASS (or document unrelated pre-existing format drift if any).

**Step 4: Commit final migration fixes**

```bash
git add -A
git commit -m "chore: finalize estimator examples full migration"
```

### Task 7: Request explicit code review before branch integration

**Files:**
- No code files required

**Step 1: Capture review range**

Run:

```bash
BASE_SHA=$(git merge-base HEAD main)
HEAD_SHA=$(git rev-parse HEAD)
```

**Step 2: Invoke `@requesting-code-review` workflow**

Use code-reviewer prompt with:

- What was implemented: full estimator API migration to example class files
- Plan reference: this migration implementation plan
- `BASE_SHA` and `HEAD_SHA`

**Step 3: Address review findings**

- Fix critical/important findings before merge.
- Re-run relevant tests.

**Step 4: Final commit for review-driven fixes (if needed)**

```bash
git add -A
git commit -m "fix: address code review findings for estimator migration"
```

