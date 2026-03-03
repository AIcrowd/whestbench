import importlib
from pathlib import Path


def _doc_len(doc: str | None) -> int:
    return len((doc or "").strip())


def _normalize_h2_heading(line: str) -> str:
    if not line.startswith("## "):
        return line
    heading = line[3:].strip()
    first_token, sep, rest = heading.partition(" ")
    if sep and not first_token.isascii():
        heading = rest.strip()
    return f"## {heading}"


def _participant_markdown_paths(repo_root: Path) -> list[Path]:
    paths = [repo_root / "README.md", repo_root / "tools/circuit-explorer/README.md"]
    docs_root = repo_root / "docs"
    for path in sorted(docs_root.rglob("*.md")):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith("internal plans/"):
            continue
        paths.append(path)
    # Dedupe while preserving order.
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


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


def test_docs_index_exists_and_links_taxonomy() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs/index.md").read_text(encoding="utf-8").lower()

    required_tokens = [
        "getting started",
        "concepts",
        "how-to",
        "reference",
        "troubleshooting",
        "./getting-started/",
        "./concepts/",
        "./how-to/",
        "./reference/",
        "./troubleshooting/",
    ]
    for token in required_tokens:
        assert token in text


def test_readme_is_front_door_with_expected_sections() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    headings = {
        _normalize_h2_heading(line.strip()) for line in text.splitlines() if line.startswith("## ")
    }
    required = [
        "## 60-Second Overview",
        "## 5-Minute Quickstart",
        "## Documentation",
        "## Example Estimators",
        "## Current Platform Status",
    ]
    for heading in required:
        assert heading in headings


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


def test_critical_public_apis_have_docstrings() -> None:
    from circuit_estimation import (
        cli,
        domain,
        estimators,
        generation,
        protocol,
        reporting,
        scoring,
        simulation,
    )

    required = [
        domain.Layer.identity,
        domain.Layer.validate,
        domain.Circuit.validate,
        generation.random_gates,
        generation.random_circuit,
        simulation.run_batched,
        simulation.run_on_random,
        simulation.empirical_mean,
        estimators.MeanPropagationEstimator.predict,
        estimators.CovariancePropagationEstimator.predict,
        estimators.CombinedEstimator.predict,
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


def test_estimators_module_has_tutorial_walkthrough_markers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "src/circuit_estimation/estimators.py").read_text(encoding="utf-8")
    required_phrases = [
        "first-moment propagation",
        "pairwise moment closure",
        "covariance",
        "budget",
    ]
    lowered = text.lower()
    for phrase in required_phrases:
        assert phrase in lowered


def test_docs_do_not_reference_predict_batch_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        repo_root / "README.md",
        repo_root / "docs/how-to/write-an-estimator.md",
        repo_root / "docs/reference/estimator-contract.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "predict_batch" not in text, str(path)


def test_readme_links_primary_docs_pages() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8").lower()
    required_links = [
        "docs/index.md",
        "docs/getting-started/install-and-cli-quickstart.md",
        "docs/getting-started/first-local-run.md",
        "docs/concepts/problem-setup.md",
        "docs/concepts/scoring-model.md",
        "docs/how-to/write-an-estimator.md",
        "docs/how-to/validate-run-package.md",
        "docs/reference/estimator-contract.md",
        "docs/reference/cli-reference.md",
        "docs/troubleshooting/common-participant-errors.md",
    ]
    for link in required_links:
        assert link in text


def test_readme_documents_cestim_install_and_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8").lower()
    assert "uv tool install -e ." in text
    assert "cestim smoke-test" in text
    assert "uv run --with-editable . cestim" in text
    assert "uv run cestim --" not in text


def test_participant_docs_do_not_use_mermaid_blocks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for path in _participant_markdown_paths(repo_root):
        text = path.read_text(encoding="utf-8").lower()
        assert "```mermaid" not in text, str(path)


def test_participant_docs_do_not_reference_sanitized_internal_paths() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    banned = [
        "internal docs/",
        "internal context docs/",
        "internal plans/",
        "internal context document",
        "internal CLI note",
        "internal style note",
        "internal docs",
        "internal plans",
        "internal context docs",
    ]
    for path in _participant_markdown_paths(repo_root):
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            assert token not in text, f"{path}: {token}"


def test_legacy_guides_directory_is_removed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "docs/guides").exists()


def test_examples_estimators_folder_contains_starter_classes() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples/estimators"
    assert (examples_dir / "random_estimator.py").exists()
    assert (examples_dir / "mean_propagation.py").exists()
    assert (examples_dir / "covariance_propagation.py").exists()
    assert (examples_dir / "combined_estimator.py").exists()


def test_onboarding_docs_recommend_random_estimator_first() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    how_to = (repo_root / "docs/how-to/write-an-estimator.md").read_text(encoding="utf-8")

    assert "random_estimator.py" in readme
    assert "random_estimator.py" in how_to

    if "mean_propagation.py" in readme:
        assert readme.index("random_estimator.py") < readme.index("mean_propagation.py")
    if "mean_propagation.py" in how_to:
        assert how_to.index("random_estimator.py") < how_to.index("mean_propagation.py")
