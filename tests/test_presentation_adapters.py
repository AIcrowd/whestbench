from __future__ import annotations

from whestbench.presentation.adapters import build_smoke_test_presentation
from whestbench.presentation.models import StepItem, StepsSection


def test_build_smoke_test_presentation_includes_structured_next_steps() -> None:
    report = {
        "run_meta": {"run_duration_s": 1.0, "host": {}},
        "run_config": {"n_mlps": 3, "width": 100, "depth": 16, "flop_budget": 10_000_000},
        "results": {"primary_score": 0.42, "secondary_score": 0.55, "per_mlp": []},
    }

    doc = build_smoke_test_presentation(report, debug=False)

    next_steps = next(
        section
        for section in doc.sections
        if isinstance(section, StepsSection) and section.title == "Next Steps"
    )
    assert all(isinstance(step, StepItem) for step in next_steps.steps)
    assert [(step.purpose, step.command) for step in next_steps.steps] == [
        ("Create starter files you can edit.", "whest init ./my-estimator"),
        (
            "Validate an Estimator implementation.",
            "whest validate --estimator ./my-estimator/estimator.py",
        ),
        (
            "Run local evaluation with isolation.",
            "whest run --estimator ./my-estimator/estimator.py --runner local",
        ),
        (
            "Build submission artifacts for AIcrowd.",
            "whest package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz",
        ),
    ]
