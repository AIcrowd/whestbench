from __future__ import annotations

import importlib
from pathlib import Path


def _doc_len(doc):
    # type: (Optional[str]) -> int
    return len((doc or "").strip())


def _normalize_h2_heading(line):
    # type: (str) -> str
    if not line.startswith("## "):
        return line
    heading = line[3:].strip()
    first_token, sep, rest = heading.partition(" ")
    if sep and not first_token.isascii():
        heading = rest.strip()
    return "## {}".format(heading)


def _participant_markdown_paths(repo_root):
    # type: (Path) -> List[Path]
    paths = [repo_root / "README.md", repo_root / "tools/network-explorer/README.md"]  # type: List[Path]
    docs_root = repo_root / "docs"
    for path in sorted(docs_root.rglob("*.md")):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith("docs/plans/"):
            continue
        paths.append(path)
    # Dedupe while preserving order.
    deduped = []  # type: List[Path]
    seen = set()  # type: Set[Path]
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def test_docs_taxonomy_directories_exist():
    # type: () -> None
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


def test_docs_index_exists_and_links_taxonomy():
    # type: () -> None
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


def test_readme_is_front_door_with_expected_sections():
    # type: () -> None
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


def test_core_modules_have_descriptive_module_docstrings():
    # type: () -> None
    modules = [
        "network_estimation.domain",
        "network_estimation.generation",
        "network_estimation.simulation",
        "network_estimation.estimators",
        "network_estimation.scoring",
        "network_estimation.reporting",
        "network_estimation.cli",
        "network_estimation.protocol",
    ]
    for name in modules:
        module = importlib.import_module(name)
        assert _doc_len(module.__doc__) >= 40, name


def test_critical_public_apis_have_docstrings():
    # type: () -> None
    from network_estimation import (
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
        domain.MLP.validate,
        generation.sample_mlp,
        simulation.run_mlp,
        simulation.run_mlp_all_layers,
        simulation.sample_layer_statistics,
        estimators.MeanPropagationEstimator.predict,
        estimators.CovariancePropagationEstimator.predict,
        estimators.CombinedEstimator.predict,
        scoring.ContestSpec.validate,
        scoring.evaluate_estimator,
        reporting.render_agent_report,
        reporting.render_human_report,
        cli.run_default_report,
        cli.main,
        protocol.ScoreRequest.to_dict,
        protocol.ScoreResponse.to_dict,
    ]
    for obj in required:
        assert _doc_len(getattr(obj, "__doc__", None)) >= 30, repr(obj)


def test_estimators_module_has_tutorial_walkthrough_markers():
    # type: () -> None
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "src/network_estimation/estimators.py").read_text(encoding="utf-8")
    required_phrases = [
        "first-moment propagation",
        "covariance",
        "budget",
    ]
    lowered = text.lower()
    for phrase in required_phrases:
        assert phrase in lowered


def test_docs_do_not_reference_predict_batch_contract():
    # type: () -> None
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        repo_root / "README.md",
        repo_root / "docs/how-to/write-an-estimator.md",
        repo_root / "docs/reference/estimator-contract.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "predict_batch" not in text, str(path)


def test_readme_links_primary_docs_pages():
    # type: () -> None
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


def test_readme_documents_nestim_install_and_usage():
    # type: () -> None
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8").lower()
    assert "uv tool install -e ." in text
    assert "nestim smoke-test" in text
    assert "uv run --with-editable . nestim" in text
    assert "uv run nestim --" not in text


def test_participant_docs_do_not_use_mermaid_blocks():
    # type: () -> None
    repo_root = Path(__file__).resolve().parents[1]
    for path in _participant_markdown_paths(repo_root):
        text = path.read_text(encoding="utf-8").lower()
        assert "```mermaid" not in text, str(path)


def test_participant_docs_do_not_reference_sanitized_internal_paths():
    # type: () -> None
    repo_root = Path(__file__).resolve().parents[1]
    banned = [
        ".aicrowd/",
        "docs/context/",
        "docs/plans/",
        "challenge-context.md",
        "worktrees-and-cli.md",
        "style_guide.md",
        "internal docs",
        "internal plans",
        "internal context docs",
    ]
    for path in _participant_markdown_paths(repo_root):
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            assert token not in text, "{}: {}".format(path, token)


def test_legacy_guides_directory_is_removed():
    # type: () -> None
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "docs/guides").exists()


def test_examples_estimators_folder_contains_starter_classes():
    # type: () -> None
    repo_root = Path(__file__).resolve().parents[1]
    examples_dir = repo_root / "examples/estimators"
    assert (examples_dir / "random_estimator.py").exists()
    assert (examples_dir / "mean_propagation.py").exists()
    assert (examples_dir / "covariance_propagation.py").exists()
    assert (examples_dir / "combined_estimator.py").exists()


def test_onboarding_docs_recommend_random_estimator_first():
    # type: () -> None
    repo_root = Path(__file__).resolve().parents[1]
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    how_to = (repo_root / "docs/how-to/write-an-estimator.md").read_text(encoding="utf-8")

    assert "random_estimator.py" in readme
    assert "random_estimator.py" in how_to

    if "mean_propagation.py" in readme:
        assert readme.index("random_estimator.py") < readme.index("mean_propagation.py")
    if "mean_propagation.py" in how_to:
        assert how_to.index("random_estimator.py") < how_to.index("mean_propagation.py")
