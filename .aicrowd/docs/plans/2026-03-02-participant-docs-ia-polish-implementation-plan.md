# Participant Docs IA + Professional Polish Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rebuild participant-facing documentation into a FastAPI-style GitHub Markdown information architecture that is clear for first-time participants and robust under sanitization.

**Architecture:** Move from one large README + flat guides into typed docs buckets (`getting-started`, `concepts`, `how-to`, `reference`, `troubleshooting`) with a concise front-door README and canonical `docs/index.md`. Use TDD against `tests/test_docs_quality.py` to enforce structure, no-Mermaid policy, and no sensitive-path leakage in participant-facing docs.

**Tech Stack:** Markdown, Python test suite (`pytest`), ripgrep, existing CLI contract in `src/circuit_estimation/cli.py`.

---

### Task 1: Prepare Isolated Workspace And Baseline

**Files:**
- Modify: none
- Test: none

**Step 1: Create isolated worktree for execution**

Run:
```bash
git worktree add .worktrees/docs-ia-polish -b docs-ia-polish
```
Expected: new worktree created at `.worktrees/docs-ia-polish`.

**Step 2: Enter worktree and confirm clean state**

Run:
```bash
cd .worktrees/docs-ia-polish
git status --short
```
Expected: no staged or unstaged changes.

**Step 3: Commit (workspace setup marker)**

Run:
```bash
git commit --allow-empty -m "chore: start docs IA polish workspace"
```

### Task 2: Write Failing Docs-Architecture Tests

**Files:**
- Modify: `tests/test_docs_quality.py`
- Test: `tests/test_docs_quality.py`

**Step 1: Write failing test for required docs taxonomy**

```python
def test_docs_taxonomy_directories_exist() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    required = [
        "docs/getting-started",
        "docs/concepts",
        "docs/how-to",
        "docs/reference",
        "docs/troubleshooting",
    ]
    for rel in required:
        assert (repo_root / rel).is_dir(), rel
```

**Step 2: Write failing test for canonical docs index**

```python
def test_docs_index_exists_and_links_taxonomy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs/index.md").read_text(encoding="utf-8").lower()
    for token in [
        "getting started",
        "concepts",
        "how-to",
        "reference",
        "troubleshooting",
    ]:
        assert token in text
```

**Step 3: Run test to verify failure**

Run:
```bash
uv run --group dev pytest tests/test_docs_quality.py -k "taxonomy or docs_index" -v
```
Expected: FAIL on missing directories/index.

**Step 4: Commit failing tests**

```bash
git add tests/test_docs_quality.py
git commit -m "test: add failing checks for docs taxonomy and index"
```

### Task 3: Create New Docs Taxonomy Skeleton

**Files:**
- Create: `docs/index.md`
- Create: `docs/getting-started/quickstart.md`
- Create: `docs/concepts/problem-and-scoring.md`
- Create: `docs/how-to/validate-run-package.md`
- Create: `docs/reference/cli-reference.md`
- Create: `docs/troubleshooting/common-errors.md`
- Test: `tests/test_docs_quality.py`

**Step 1: Add minimal `docs/index.md` with category links**

```markdown
# Documentation

## Getting Started
- [Quickstart](./getting-started/quickstart.md)

## Concepts
- [Problem and Scoring](./concepts/problem-and-scoring.md)

## How-To
- [Validate, Run, Package](./how-to/validate-run-package.md)

## Reference
- [CLI Reference](./reference/cli-reference.md)

## Troubleshooting
- [Common Errors](./troubleshooting/common-errors.md)
```

**Step 2: Add minimal stub pages for each category**

Use this template for each created file:
```markdown
# <Title>

Draft page for docs IA migration.
```

**Step 3: Run tests to verify pass**

Run:
```bash
uv run --group dev pytest tests/test_docs_quality.py -k "taxonomy or docs_index" -v
```
Expected: PASS.

**Step 4: Commit**

```bash
git add docs/index.md docs/getting-started/quickstart.md docs/concepts/problem-and-scoring.md docs/how-to/validate-run-package.md docs/reference/cli-reference.md docs/troubleshooting/common-errors.md
git commit -m "docs: add new participant docs taxonomy skeleton"
```

### Task 4: Enforce No-Mermaid + No-Internal-Reference Policy With Tests

**Files:**
- Modify: `tests/test_docs_quality.py`
- Test: `tests/test_docs_quality.py`

**Step 1: Add failing test banning Mermaid in participant docs**

```python
def test_participant_docs_do_not_use_mermaid_blocks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = [repo_root / "README.md", *sorted((repo_root / "docs").rglob("*.md"))]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "```mermaid" not in text, str(path)
```

**Step 2: Add failing test banning sanitized/internal references**

```python
def test_participant_docs_do_not_reference_sanitized_internal_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    banned = [
        ".aicrowd/",
        "docs/context/",
        "docs/plans/",
        "challenge-context.md",
        "worktrees-and-cli.md",
        "style_guide.md",
    ]
    paths = [repo_root / "README.md", *sorted((repo_root / "docs").rglob("*.md")), repo_root / "tools/circuit-explorer/README.md"]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            assert token not in text, f"{path}: {token}"
```

**Step 3: Run tests to verify failure**

Run:
```bash
uv run --group dev pytest tests/test_docs_quality.py -k "mermaid or sanitized_internal" -v
```
Expected: FAIL on current README Mermaid blocks.

**Step 4: Commit**

```bash
git add tests/test_docs_quality.py
git commit -m "test: enforce no-mermaid and no-internal-path docs policy"
```

### Task 5: Rewrite README As Front Door

**Files:**
- Modify: `README.md`
- Test: `tests/test_docs_quality.py`

**Step 1: Replace README structure with front-door sections**

Required sections:
- Challenge overview (short)
- 5-minute quickstart commands
- docs map by category
- current platform status (submission/upload wording without TODO markers)

**Step 2: Remove all Mermaid blocks and duplicated deep dives**

Delete:
- all ````mermaid` blocks
- repeated long-form guide content already moved under `docs/*`.

**Step 3: Run targeted tests**

Run:
```bash
uv run --group dev pytest tests/test_docs_quality.py -k "mermaid or onboarding or links" -v
```
Expected: PASS for updated checks.

**Step 4: Commit**

```bash
git add README.md tests/test_docs_quality.py
git commit -m "docs: rewrite README as concise participant front door"
```

### Task 6: Migrate Existing Guides Into New Category Pages

**Files:**
- Create: `docs/getting-started/install-and-cli-quickstart.md`
- Create: `docs/getting-started/first-local-run.md`
- Create: `docs/concepts/problem-setup.md`
- Create: `docs/concepts/scoring-model.md`
- Create: `docs/how-to/write-an-estimator.md`
- Create: `docs/how-to/validate-run-package.md`
- Create: `docs/reference/estimator-contract.md`
- Create: `docs/reference/cli-reference.md`
- Create: `docs/troubleshooting/common-participant-errors.md`
- Modify: existing files under `docs/guides/*.md` (moved stubs or redirects)

**Step 1: Move content by page intent (not filename parity)**

Example shell:
```bash
mkdir -p docs/getting-started docs/concepts docs/how-to docs/reference docs/troubleshooting
```

**Step 2: Ensure each page has consistent page contract**

Minimum page template:
```markdown
# <Title>

## When To Use This Page
## Steps / Explanation
## Next
```

**Step 3: Add moved-page stubs for legacy guide paths**

Example:
```markdown
# Moved

This content moved to:
- [New Location](../how-to/write-an-estimator.md)
```

**Step 4: Run doc link + docs tests**

Run:
```bash
uv run --group dev pytest tests/test_docs_quality.py -v
```
Expected: PASS.

**Step 5: Commit**

```bash
git add docs
git commit -m "docs: migrate guides into fastapi-style category architecture"
```

### Task 7: Align Circuit Explorer README Messaging

**Files:**
- Modify: `tools/circuit-explorer/README.md`
- Test: `tests/test_docs_quality.py`

**Step 1: Update wording for participant clarity**

Keep consistent phrasing:
- Explorer is optional
- Explorer is educational/debugging aid
- official scoring path is `cestim run`

**Step 2: Add explicit link back to participant docs index**

```markdown
See [Participant Docs](../../docs/index.md) for the core workflow.
```

**Step 3: Run targeted tests**

Run:
```bash
uv run --group dev pytest tests/test_docs_quality.py -k "links or sanitized_internal" -v
```
Expected: PASS.

**Step 4: Commit**

```bash
git add tools/circuit-explorer/README.md tests/test_docs_quality.py
git commit -m "docs: align circuit explorer messaging with participant workflow"
```

### Task 8: Expand Docs Tests For New IA Quality Guarantees

**Files:**
- Modify: `tests/test_docs_quality.py`

**Step 1: Replace old heading-coupled assertions with IA-coupled assertions**

Add checks for:
- required category dirs/files,
- presence of category links in `docs/index.md`,
- absence of Mermaid,
- absence of sensitive internal references,
- README docs map points to new pages.

**Step 2: Keep existing important contract checks**

Preserve useful tests for:
- estimator examples presence,
- CLI install/use command correctness,
- key guide links (updated paths).

**Step 3: Run full docs test file**

Run:
```bash
uv run --group dev pytest tests/test_docs_quality.py -v
```
Expected: all PASS.

**Step 4: Commit**

```bash
git add tests/test_docs_quality.py
git commit -m "test: update docs quality suite for new information architecture"
```

### Task 9: Verification Before Completion (@superpowers:verification-before-completion)

**Files:**
- Modify: none
- Test: all verification commands below

**Step 1: Run docs-focused grep audits**

Run:
```bash
rg -n "```mermaid|\\.aicrowd/|docs/context/|docs/plans/|CHALLENGE-CONTEXT\\.md|worktrees-and-cli\\.md|STYLE_GUIDE\\.md" README.md docs tools/circuit-explorer/README.md
```
Expected: no matches.

**Step 2: Run quality/test checks**

Run:
```bash
uv run --group dev ruff check .
uv run --group dev ruff format --check .
uv run --group dev pyright
uv run --group dev pytest tests/test_docs_quality.py -v
```
Expected: all commands succeed.

**Step 3: Optional onboarding smoke test**

Run:
```bash
uv tool install -e .
cestim --json
cestim init ./tmp-estimator
cestim validate --estimator ./tmp-estimator/estimator.py
```
Expected: commands complete successfully with valid output.

**Step 4: Commit final integration**

```bash
git add README.md docs tools/circuit-explorer/README.md tests/test_docs_quality.py
git commit -m "docs: deliver participant-first docs IA and polish pass"
```

### Task 10: Request Review (@superpowers:requesting-code-review)

**Files:**
- Modify: none

**Step 1: Produce concise reviewer checklist**

Include:
- first-time participant path clarity,
- professionalism/tone consistency,
- sanitization safety,
- command accuracy.

**Step 2: Share changed file map + key decisions**

Run:
```bash
git show --name-only --oneline -n 1
git diff --name-status HEAD~1..HEAD
```

**Step 3: Address findings in follow-up commits if needed**

Run:
```bash
git commit -m "docs: address review feedback on participant docs polish"
```
Only if changes are required.
