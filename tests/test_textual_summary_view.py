from __future__ import annotations

from typing import Any

from circuit_estimation.textual_dashboard.state import build_dashboard_state
from circuit_estimation.textual_dashboard.views.summary import render_summary_view


def _sample_report(*, include_profile: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "run_meta": {
            "run_started_at_utc": "2026-03-01T00:00:00+00:00",
            "run_finished_at_utc": "2026-03-01T00:00:01+00:00",
            "run_duration_s": 1.0,
        },
        "run_config": {
            "n_circuits": 2,
            "n_samples": 100,
            "width": 4,
            "max_depth": 3,
            "budgets": [10, 100],
        },
        "results": {
            "final_score": 0.123,
            "by_budget_raw": [
                {"budget": 10, "adjusted_mse": 0.22, "mse_by_layer": [0.1, 0.2, 0.3]},
                {"budget": 100, "adjusted_mse": 0.036, "mse_by_layer": [0.05, 0.04, 0.03]},
            ],
        },
    }
    if include_profile:
        report["profile_calls"] = [{"wall_time_s": 0.05, "cpu_time_s": 0.04}]
    return report


def test_summary_contains_legacy_sections_with_profile() -> None:
    state = build_dashboard_state(_sample_report(include_profile=True))

    rendered = render_summary_view(state)

    assert "Interesting Plots" in rendered
    assert "Run Context" in rendered
    assert "Readiness Scorecard" in rendered
    assert "Budget" in rendered
    assert "Layer Diagnostics" in rendered
    assert "Profile" in rendered


def test_summary_omits_profile_section_when_profile_unavailable() -> None:
    state = build_dashboard_state(_sample_report(include_profile=False))

    rendered = render_summary_view(state)

    assert "Run Context" in rendered
    assert "Readiness Scorecard" in rendered
    assert "Profile\n" not in rendered
