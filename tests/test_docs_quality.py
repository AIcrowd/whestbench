from pathlib import Path


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
