# Core Runtime + README Documentation Uplift Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade core runtime docs and root README so the repository is a strong onboarding resource for mixed practitioner/research audiences.

**Architecture:** Add documentation quality tests first, then iteratively improve README and core module docs to satisfy those tests. Keep code behavior stable, allow only tiny readability refactors, and preserve green runtime tests as a regression guard.

**Tech Stack:** Python 3.10+, pytest, Ruff, existing `src/circuit_estimation` modules, Markdown README/docs.

---

### Task 1: Add README onboarding contract tests

**Files:**
- Create: `tests/test_docs_quality.py`
- Test: `tests/test_docs_quality.py`

**Step 1: Write the failing test**

```python
from pathlib import Path


def test_readme_contains_onboarding_sections() -> None:
    readme = Path(__file__).resolve().parents[1] / "README.md"
    text = readme.read_text(encoding="utf-8")
    required = [
        "## What This Repository Teaches",
        "## Conceptual Problem Overview",
        "## How Evaluation Works (End-to-End)",
        "## Codebase Map (Suggested Reading Order)",
        "## Quickstart",
        "## Extending the Estimator",
        "## Verification Commands",
    ]
    for heading in required:
        assert heading in text
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_docs_quality.py::test_readme_contains_onboarding_sections -v`  
Expected: FAIL because at least one required heading is missing.

**Step 3: Commit**

```bash
git add tests/test_docs_quality.py
git commit -m "test: add README onboarding structure contract"
```

### Task 2: Rewrite README as onboarding narrative

**Files:**
- Modify: `README.md`
- Test: `tests/test_docs_quality.py`

**Step 1: Write minimal implementation for README sections**

```markdown
## What This Repository Teaches
## Conceptual Problem Overview
## How Evaluation Works (End-to-End)
## Codebase Map (Suggested Reading Order)
## Quickstart
## Extending the Estimator
## Verification Commands
```

**Step 2: Run README contract test**

Run: `uv run --group dev pytest tests/test_docs_quality.py::test_readme_contains_onboarding_sections -v`  
Expected: PASS.

**Step 3: Strengthen section content**

Add concise but rigorous prose for:
- evaluator flow (generation -> empirical mean -> estimator -> runtime adjustment -> final score),
- runtime timeout/floor semantics,
- guided module reading order,
- estimator output contract and shape checks.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README as onboarding introduction"
```

### Task 3: Add core runtime docstring quality tests

**Files:**
- Modify: `tests/test_docs_quality.py`
- Test: `tests/test_docs_quality.py`

**Step 1: Add failing tests for module/class/function docstrings**

```python
import importlib


def _doc_len(doc: str | None) -> int:
    return len((doc or "").strip())


def test_core_modules_have_descriptive_module_docstrings() -> None:
    modules = [
        "circuit_estimation.domain",
        "circuit_estimation.generation",
        "circuit_estimation.simulation",
        "circuit_estimation.estimators",
        "circuit_estimation.scoring",
        "circuit_estimation.reporting",
        "circuit_estimation.cli",
        "circuit_estimation.protocol",
    ]
    for name in modules:
        module = importlib.import_module(name)
        assert _doc_len(module.__doc__) >= 40, name
```

```python
def test_critical_public_apis_have_docstrings() -> None:
    from circuit_estimation import domain, generation, simulation
    from circuit_estimation import estimators, scoring, reporting, cli, protocol

    required = [
        domain.Layer.identity,
        domain.Layer.validate,
        domain.Circuit.validate,
        generation.random_gates,
        generation.random_circuit,
        simulation.run_batched,
        simulation.run_on_random,
        simulation.empirical_mean,
        estimators.mean_propagation,
        estimators.covariance_propagation,
        estimators.combined_estimator,
        scoring.ContestParams.validate,
        scoring.score_estimator_report,
        scoring.score_estimator,
        reporting.render_agent_report,
        reporting.render_human_report,
        cli.run_default_report,
        cli.main,
        protocol.ScoreRequest.to_dict,
        protocol.ScoreResponse.to_dict,
    ]
    for obj in required:
        assert _doc_len(getattr(obj, "__doc__", None)) >= 30, repr(obj)
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_docs_quality.py -v`  
Expected: FAIL because several docstrings are too short or missing.

**Step 3: Commit**

```bash
git add tests/test_docs_quality.py
git commit -m "test: add core runtime docstring quality checks"
```

### Task 4: Document foundational runtime modules

**Files:**
- Modify: `src/circuit_estimation/domain.py`
- Modify: `src/circuit_estimation/generation.py`
- Modify: `src/circuit_estimation/simulation.py`
- Test: `tests/test_docs_quality.py`
- Test: `tests/test_domain.py`
- Test: `tests/test_generation.py`
- Test: `tests/test_simulation.py`

**Step 1: Expand module/class/function docstrings**

Include:
- invariants for width/depth and index validity,
- gate sampling assumptions and RNG behavior,
- simulation dtype expectations and yielded tensor semantics.

**Step 2: Add moderate inline comments at non-obvious logic blocks**

Examples:
- distinct-index trick for `second` wire selection in `random_gates`,
- why simulation casts to `float16` during batched propagation.

**Step 3: Run targeted tests**

Run: `uv run --group dev pytest tests/test_docs_quality.py tests/test_domain.py tests/test_generation.py tests/test_simulation.py -v`  
Expected: PASS.

**Step 4: Commit**

```bash
git add src/circuit_estimation/domain.py src/circuit_estimation/generation.py src/circuit_estimation/simulation.py
git commit -m "docs: enrich foundational runtime docstrings and comments"
```

### Task 5: Add estimator tutorial-comment contract tests

**Files:**
- Modify: `tests/test_docs_quality.py`
- Test: `tests/test_docs_quality.py`

**Step 1: Add failing tests for estimator walkthrough clarity**

```python
def test_estimators_module_has_tutorial_walkthrough_markers() -> None:
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "src/circuit_estimation/estimators.py").read_text(
        encoding="utf-8"
    )
    required_phrases = [
        "first-moment propagation",
        "pairwise moment closure",
        "covariance",
        "budget",
    ]
    for phrase in required_phrases:
        assert phrase in text.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run --group dev pytest tests/test_docs_quality.py::test_estimators_module_has_tutorial_walkthrough_markers -v`  
Expected: FAIL before tutorial commentary is added.

**Step 3: Commit**

```bash
git add tests/test_docs_quality.py
git commit -m "test: require estimator tutorial commentary markers"
```

### Task 6: Document estimator, scoring, reporting, CLI, and protocol modules

**Files:**
- Modify: `src/circuit_estimation/estimators.py`
- Modify: `src/circuit_estimation/scoring.py`
- Modify: `src/circuit_estimation/reporting.py`
- Modify: `src/circuit_estimation/cli.py`
- Modify: `src/circuit_estimation/protocol.py`
- Optional tiny refactor: same files above
- Test: `tests/test_docs_quality.py`
- Test: `tests/test_estimators.py`
- Test: `tests/test_scoring_module.py`
- Test: `tests/test_reporting.py`
- Test: `tests/test_cli.py`
- Test: `tests/test_protocol.py`

**Step 1: Expand module/function docstrings with assumptions and contracts**

Cover:
- scoring runtime normalization + timeout/floor semantics,
- report modes and schema expectations,
- CLI mode/detail/profile behavior,
- protocol serialization boundaries.

**Step 2: Add tutorial walkthrough commentary in `estimators.py`**

Commentary targets:
- propagation state (`x_mean`, `x_cov`),
- one-vs-two and two-vs-two covariance approximations,
- clipping to feasible ranges,
- budget threshold design in `combined_estimator`.

**Step 3: Apply only tiny readability refactors where they improve explanation**

Examples:
- local variable names for intermediate tensors,
- extracting tiny helper logic if it removes repeated expressions.

**Step 4: Run targeted tests**

Run: `uv run --group dev pytest tests/test_docs_quality.py tests/test_estimators.py tests/test_scoring_module.py tests/test_reporting.py tests/test_cli.py tests/test_protocol.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add src/circuit_estimation/estimators.py src/circuit_estimation/scoring.py src/circuit_estimation/reporting.py src/circuit_estimation/cli.py src/circuit_estimation/protocol.py
git commit -m "docs: add tutorial estimator walkthrough and runtime API docstrings"
```

### Task 7: Final verification and cleanup

**Files:**
- Modify: (none expected; only if verification surfaces issues)
- Test: `tests/test_docs_quality.py`
- Test: `tests/` (full suite)

**Step 1: Run formatting and lint checks**

Run: `uv run --group dev ruff check .`  
Expected: PASS.

Run: `uv run --group dev ruff format --check .`  
Expected: PASS.

**Step 2: Run documentation-specific and full test suites**

Run: `uv run --group dev pytest tests/test_docs_quality.py -v`  
Expected: PASS.

Run: `./scripts/run-test-harness.sh full`  
Expected: PASS (quick + exhaustive checks).

**Step 3: Commit final polish (if needed)**

```bash
git add -A
git commit -m "chore: finalize documentation onboarding uplift"
```

**Step 4: Prepare handoff summary**

Include:
- files changed,
- doc quality guarantees added by tests,
- verification command results,
- any deferred follow-ups.
