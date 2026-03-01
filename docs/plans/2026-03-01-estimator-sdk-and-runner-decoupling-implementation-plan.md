# Estimator SDK and Runner Decoupling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce a participant-first estimator SDK and installable CLI that decouple estimator implementations from core scorer internals, while adding a runner abstraction that supports in-process and subprocess execution with resource controls.

**Architecture:** Add a participant-facing API layer (`BaseEstimator`, context, loader), then route scoring through runner interfaces that return structured outcomes. Keep a temporary compatibility wrapper for current callable-based scoring paths. Add CLI commands for init/validate/run/package with strict JSON mode and structured error reporting.

**Tech Stack:** Python 3.10+, NumPy, dataclasses, argparse (or existing CLI framework), pytest, ruff, pyright, uv

---

### Task 1: Add estimator SDK interfaces and context types

**Files:**
- Create: `src/circuit_estimation/sdk.py`
- Modify: `src/circuit_estimation/__init__.py`
- Test: `tests/test_sdk.py`

**Step 1: Write the failing test**

```python
def test_base_estimator_default_predict_batch_stacks_predict_outputs():
    ...

def test_setup_context_is_immutable_and_contains_required_fields():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_sdk.py -q`
Expected: FAIL because `sdk.py` and exports do not exist.

**Step 3: Write minimal implementation**

Implement in `sdk.py`:

- `SetupContext` dataclass (`frozen=True`, `slots=True`)
- `BaseEstimator` abstract class with:
  - optional `setup`
  - required `predict`
  - default `predict_batch` sequential stack
  - optional `teardown`

Export these symbols from package init.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_sdk.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/sdk.py src/circuit_estimation/__init__.py tests/test_sdk.py
git commit -m "feat: add participant estimator sdk interfaces"
```

### Task 2: Implement estimator module/class loader with deterministic resolution

**Files:**
- Create: `src/circuit_estimation/loader.py`
- Test: `tests/test_loader.py`

**Step 1: Write the failing test**

```python
def test_loader_prefers_default_estimator_class_name():
    ...

def test_loader_allows_explicit_class_override():
    ...

def test_loader_errors_on_ambiguous_multiple_classes():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_loader.py -q`
Expected: FAIL because loader does not exist.

**Step 3: Write minimal implementation**

Implement loader behavior:

- import module from file path
- find subclasses of `BaseEstimator`
- resolve class by rule order:
  1. explicit class name
  2. class named `Estimator`
  3. exactly one subclass fallback
  4. otherwise error

Return instantiated estimator and resolved class metadata.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_loader.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/loader.py tests/test_loader.py
git commit -m "feat: add deterministic estimator loader"
```

### Task 3: Add runner abstractions and structured outcome/error types

**Files:**
- Create: `src/circuit_estimation/runner.py`
- Test: `tests/test_runner_types.py`

**Step 1: Write the failing test**

```python
def test_predict_outcome_supports_required_status_and_metrics_fields():
    ...

def test_resource_limits_require_setup_and_predict_caps():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_runner_types.py -q`
Expected: FAIL because runner types do not exist.

**Step 3: Write minimal implementation**

Implement:

- `ResourceLimits`
- `PredictOutcome`
- `RunnerError` / structured error dataclasses
- runner protocol/interface (`start`, `predict`, `predict_batch`, `close`)

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_runner_types.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/runner.py tests/test_runner_types.py
git commit -m "feat: define runner interfaces and structured outcomes"
```

### Task 4: Implement InProcessRunner with setup exclusion and predict metrics

**Files:**
- Modify: `src/circuit_estimation/runner.py`
- Test: `tests/test_inprocess_runner.py`

**Step 1: Write the failing test**

```python
def test_inprocess_runner_calls_setup_once_before_predicts():
    ...

def test_inprocess_runner_collects_wall_cpu_and_memory_metrics():
    ...

def test_inprocess_runner_returns_structured_runtime_error_status():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_inprocess_runner.py -q`
Expected: FAIL before implementation.

**Step 3: Write minimal implementation**

Add concrete `InProcessRunner`:

- construct estimator via loader
- call `setup` in `start`
- call `predict` in `predict`
- convert exceptions to structured statuses
- capture wall/cpu/rss/peak metrics per predict call

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_inprocess_runner.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/runner.py tests/test_inprocess_runner.py
git commit -m "feat: add in-process estimator runner"
```

### Task 5: Add SubprocessRunner skeleton with timeout and protocol framing

**Files:**
- Create: `src/circuit_estimation/subprocess_worker.py`
- Modify: `src/circuit_estimation/runner.py`
- Test: `tests/test_subprocess_runner.py`

**Step 1: Write the failing test**

```python
def test_subprocess_runner_times_out_predict_calls_and_returns_timeout_status():
    ...

def test_subprocess_runner_reports_protocol_errors_cleanly():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_subprocess_runner.py -q`
Expected: FAIL before subprocess protocol exists.

**Step 3: Write minimal implementation**

Implement JSON-line or msgpack protocol between runner and worker:

- worker loads estimator and performs setup once
- runner sends predict requests and waits with timeout
- timeout/worker crash/protocol decode failures map to structured statuses

Implement best-effort local resource controls where feasible.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_subprocess_runner.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/subprocess_worker.py src/circuit_estimation/runner.py tests/test_subprocess_runner.py
git commit -m "feat: add subprocess runner protocol with timeout handling"
```

### Task 6: Refactor scoring to consume runner outcomes

**Files:**
- Modify: `src/circuit_estimation/scoring.py`
- Test: `tests/test_scoring_module.py`
- Test: `tests/test_evaluate.py`

**Step 1: Write the failing test**

```python
def test_score_submission_report_uses_runner_predict_outcomes_for_metrics():
    ...

def test_predict_failure_statuses_are_reflected_in_budget_failure_counts():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_scoring_module.py tests/test_evaluate.py -q`
Expected: FAIL because scorer still expects direct callable contract.

**Step 3: Write minimal implementation**

Add `score_submission_report(...)` path:

- accepts runner + entrypoint metadata
- runs setup via runner start
- scores predict outcomes with existing zero-output failure semantics
- keeps setup timing excluded from score

Retain temporary compatibility wrapper for existing `score_estimator(...)`.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_scoring_module.py tests/test_evaluate.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/scoring.py tests/test_scoring_module.py tests/test_evaluate.py
git commit -m "feat: route scoring through runner outcomes"
```

### Task 7: Add installable CLI command surface (`init`, `validate`, `run`, `package`)

**Files:**
- Modify: `src/circuit_estimation/cli.py`
- Modify: `main.py`
- Modify: `pyproject.toml`
- Create: `tests/test_cli_participant_commands.py`
- Modify: `tests/test_cli.py`

**Step 1: Write the failing test**

```python
def test_validate_command_returns_json_only_in_agent_mode():
    ...

def test_run_command_renders_human_report_in_non_agent_mode():
    ...

def test_package_command_writes_manifest_with_entrypoint_and_hashes():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_cli.py tests/test_cli_participant_commands.py -q`
Expected: FAIL until new CLI subcommands exist.

**Step 3: Write minimal implementation**

Implement subcommands:

- `init`
- `validate`
- `run`
- `package`

Add script entrypoint for `cestim` in `pyproject.toml`.

Enforce:

- `--agent-mode` JSON-only output
- human mode rich output on success
- structured errors in both modes
- traceback only with `--debug`

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_cli.py tests/test_cli_participant_commands.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py main.py pyproject.toml tests/test_cli.py tests/test_cli_participant_commands.py
git commit -m "feat: add participant-first installable cli workflow"
```

### Task 8: Implement packaging schema and metadata split

**Files:**
- Create: `src/circuit_estimation/packaging.py`
- Test: `tests/test_packaging.py`

**Step 1: Write the failing test**

```python
def test_package_includes_generated_manifest_json_and_estimator_file():
    ...

def test_manifest_records_resolved_entrypoint_and_sha256_hashes():
    ...

def test_optional_submission_yaml_and_approach_md_are_included_when_present():
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_packaging.py -q`
Expected: FAIL before packaging module exists.

**Step 3: Write minimal implementation**

Implement packager:

- validates estimator first
- creates deterministic artifact archive
- generates `manifest.json`
- includes optional metadata files if present

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_packaging.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/packaging.py tests/test_packaging.py
git commit -m "feat: add submission packaging with manifest contract"
```

### Task 9: Add starter template and docs for participant workflow

**Files:**
- Create: `src/circuit_estimation/templates/estimator.py.tmpl`
- Modify: `README.md`
- Modify: `docs/context/mvp-technical-snapshot.md`
- Modify: `docs/context/python-runtime-refactor-decisions.md`
- Test: `tests/test_docs_quality.py`

**Step 1: Write the failing docs/test check**

Add assertions for new CLI names and participant workflow keywords.

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_docs_quality.py -q`
Expected: FAIL before docs update.

**Step 3: Write minimal implementation**

Document:

- installable `cestim` usage
- single-file participant contract
- setup-time exclusion policy
- structured error behavior and `--debug`

Add init template used by `cestim init`.

**Step 4: Run test to verify it passes**

Run: `uv run --group dev pytest tests/test_docs_quality.py -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/templates/estimator.py.tmpl README.md docs/context/mvp-technical-snapshot.md docs/context/python-runtime-refactor-decisions.md tests/test_docs_quality.py
git commit -m "docs: define participant estimator lifecycle and cli contracts"
```

### Task 10: End-to-end verification gate

**Files:**
- Modify as needed from earlier tasks

**Step 1: Run lint and format checks**

Run: `uv run --group dev ruff check . && uv run --group dev ruff format --check .`
Expected: PASS.

**Step 2: Run static type checks**

Run: `uv run --group dev pyright`
Expected: PASS.

**Step 3: Run non-exhaustive tests**

Run: `uv run --group dev pytest -m "not exhaustive"`
Expected: PASS.

**Step 4: Run exhaustive tests**

Run: `uv run --group dev pytest -m exhaustive`
Expected: PASS.

**Step 5: Commit any final fixes**

```bash
git add -A
git commit -m "chore: finalize estimator sdk and runner decoupling rollout"
```

