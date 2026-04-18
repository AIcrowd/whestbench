from __future__ import annotations

from whestbench.presentation.adapters import build_smoke_test_presentation


def test_build_smoke_test_presentation_includes_next_steps() -> None:
    report = {
        "run_meta": {"run_duration_s": 1.0, "host": {}},
        "run_config": {"n_mlps": 3, "width": 100, "depth": 16, "flop_budget": 10_000_000},
        "results": {"primary_score": 0.42, "secondary_score": 0.55, "per_mlp": []},
    }

    doc = build_smoke_test_presentation(report, debug=False)

    assert any(section.title == "Next Steps" for section in doc.sections)
    assert any(
        "whest init ./my-estimator" in step
        for section in doc.sections
        if hasattr(section, "steps")
        for step in section.steps
    )
