import importlib
from pathlib import Path


def _doc_len(doc: str | None) -> int:
    return len((doc or "").strip())


def test_readme_contains_onboarding_sections() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8")
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
        repo_root / "internal context docs/mvp-technical-snapshot.md",
        repo_root / "internal context docs/python-runtime-refactor-decisions.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "predict_batch" not in text, str(path)


def test_readme_links_streaming_participant_guide() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8").lower()
    assert "participant-streaming-estimator-guide.md" in text


def test_readme_documents_cestim_install_and_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8").lower()
    assert "uv tool install -e ." in text
    assert "cestim --agent-mode" in text
