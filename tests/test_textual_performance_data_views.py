from __future__ import annotations

from typing import Any

from circuit_estimation.textual_dashboard.app import DashboardApp


def _sample_report(*, include_profile: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "results": {
            "final_score": 0.123,
            "by_budget_raw": [
                {"budget": 10, "adjusted_mse": 0.22, "mse_by_layer": [0.1, 0.2, 0.3]},
                {"budget": 100, "adjusted_mse": 0.036, "mse_by_layer": [0.05, 0.04, 0.03]},
            ],
        },
    }
    if include_profile:
        report["profile_calls"] = [
            {
                "budget": 10,
                "wall_time_s": 0.05,
                "cpu_time_s": 0.04,
                "rss_bytes": 12_345_678,
                "peak_rss_bytes": 15_000_000,
            }
        ]
    return report


def test_performance_and_data_tabs_render() -> None:
    app = DashboardApp(report=_sample_report(include_profile=True))

    app.action_tab_performance()
    performance_text = app._tab_content()
    assert "Performance" in performance_text
    assert "Profile Runtime Plot" in performance_text
    assert "Profile Memory Plot" in performance_text

    app.action_tab_data()
    data_text = app._tab_content()
    assert "Raw Data" in data_text
    assert '"results"' in data_text


def test_performance_view_handles_missing_profile() -> None:
    app = DashboardApp(report=_sample_report(include_profile=False))

    app.action_tab_performance()
    performance_text = app._tab_content()
    assert "No profiling calls available" in performance_text
