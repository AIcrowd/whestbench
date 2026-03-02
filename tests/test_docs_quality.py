import importlib
from pathlib import Path


def _doc_len(doc: str | None) -> int:
    return len((doc or "").strip())


def test_readme_contains_onboarding_sections() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8")
    required = [
        "## What Is The Problem?",
        "## How Scoring Works",
        "## Install And Get The CLI Working (Short Version)",
        "## Participant Workflow",
        "## Circuit Explorer: Build Intuition Fast",
        "## How To Write Your Own Estimator",
        "## Validate, Run, Package, And Local Modes",
        "## What The Scores Mean",
        "## Included Example Estimators",
        "## Documentation Map",
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
        repo_root / "docs/guides/how-to-write-your-own-estimator.md",
        repo_root / "docs/guides/example-estimators-and-how-to-run-them.md",
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert "predict_batch" not in text, str(path)


def test_readme_links_primary_guides() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8").lower()
    required_links = [
        "docs/guides/install-and-cli-quickstart.md",
        "docs/guides/what-is-the-problem-and-how-is-it-scored.md",
        "docs/guides/how-to-use-circuit-explorer.md",
        "docs/guides/how-to-write-your-own-estimator.md",
        "docs/guides/how-to-validate-run-and-package.md",
        "docs/guides/example-estimators-and-how-to-run-them.md",
    ]
    for link in required_links:
        assert link in text


def test_readme_documents_cestim_install_and_usage() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    text = (repo_root / "README.md").read_text(encoding="utf-8").lower()
    assert "uv tool install -e ." in text
    assert "cestim --agent-mode" in text
    assert "uv run --with-editable . cestim" in text
    assert "uv run cestim --" not in text


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
    guide = (repo_root / "docs/guides/example-estimators-and-how-to-run-them.md").read_text(
        encoding="utf-8"
    )

    assert "random_estimator.py" in readme
    assert "random_estimator.py" in guide

    if "mean_propagation.py" in readme:
        assert readme.index("random_estimator.py") < readme.index("mean_propagation.py")
    if "mean_propagation.py" in guide:
        assert guide.index("random_estimator.py") < guide.index("mean_propagation.py")
