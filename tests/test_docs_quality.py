from __future__ import annotations

import importlib
from pathlib import Path


def _doc_len(doc):
    return len((doc or "").strip())


def _library_markdown_paths(repo_root):
    paths = [repo_root / "README.md", repo_root / "tools/whestbench-explorer/README.md"]
    docs_root = repo_root / "docs"
    for path in sorted(docs_root.rglob("*.md")):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith("internal plans/"):
            continue
        paths.append(path)
    deduped, seen = [], set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    return deduped


def test_docs_reference_directory_exists():
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "docs/reference").is_dir()


def test_docs_index_points_to_starterkit_and_reference():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "docs/index.md").read_text(encoding="utf-8")
    assert "whest-starterkit" in text
    assert "reference/" in text


def test_readme_points_to_starterkit_for_participants():
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    assert "whest-starterkit" in text
    assert "BaseEstimator" in text


def test_core_modules_have_descriptive_module_docstrings():
    modules = [
        "whestbench.domain",
        "whestbench.generation",
        "whestbench.simulation",
        "whestbench.estimators",
        "whestbench.scoring",
        "whestbench.reporting",
        "whestbench.cli",
        "whestbench.protocol",
    ]
    for name in modules:
        module = importlib.import_module(name)
        assert _doc_len(module.__doc__) >= 40, name


def test_critical_public_apis_have_docstrings():
    from whestbench import (
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
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "src/whestbench/estimators.py").read_text(encoding="utf-8")
    required_phrases = [
        "first-moment propagation",
        "covariance",
        "budget",
    ]
    lowered = text.lower()
    for phrase in required_phrases:
        assert phrase in lowered


def test_docs_do_not_reference_predict_batch_contract():
    repo_root = Path(__file__).resolve().parents[1]
    paths = [
        repo_root / "README.md",
        repo_root / "docs/reference/estimator-contract.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "predict_batch" not in text, str(path)


def test_library_docs_do_not_use_mermaid_blocks():
    repo_root = Path(__file__).resolve().parents[1]
    for path in _library_markdown_paths(repo_root):
        text = path.read_text(encoding="utf-8").lower()
        assert "```mermaid" not in text, str(path)


def test_library_docs_do_not_reference_sanitized_internal_paths():
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
    for path in _library_markdown_paths(repo_root):
        text = path.read_text(encoding="utf-8").lower()
        for token in banned:
            assert token not in text, "{}: {}".format(path, token)


def test_legacy_guides_directory_is_removed():
    repo_root = Path(__file__).resolve().parents[1]
    assert not (repo_root / "docs/guides").exists()


def test_participant_taxonomy_removed():
    repo_root = Path(__file__).resolve().parents[1]
    for gone in (
        "docs/getting-started",
        "docs/concepts",
        "docs/how-to",
        "docs/troubleshooting",
        "examples/estimators",
    ):
        assert not (repo_root / gone).exists(), gone
