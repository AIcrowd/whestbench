from __future__ import annotations

from typing import Any

from circuit_estimation.textual_dashboard.app import DashboardApp


def _sample_report() -> dict[str, Any]:
    return {
        "run_config": {
            "budgets": [10, 100],
        },
        "results": {
            "final_score": 0.123,
            "by_budget_raw": [
                {
                    "budget": 10,
                    "adjusted_mse": 0.22,
                    "mse_mean": 0.2,
                    "call_time_ratio_mean": 1.1,
                    "call_effective_time_s_mean": 0.011,
                    "mse_by_layer": [0.1, 0.2, 0.3],
                },
                {
                    "budget": 100,
                    "adjusted_mse": 0.036,
                    "mse_mean": 0.04,
                    "call_time_ratio_mean": 0.9,
                    "call_effective_time_s_mean": 0.018,
                    "mse_by_layer": [0.05, 0.04, 0.03],
                },
            ],
        },
    }


def test_budget_and_layer_tabs_render_expected_headers() -> None:
    app = DashboardApp(report=_sample_report())

    app.action_tab_budgets()
    budget_text = app._tab_content()
    assert "Budget Analysis" in budget_text
    assert "Budget Table" in budget_text
    assert "Budget Frontier Plot" in budget_text

    app.action_tab_layers()
    layer_text = app._tab_content()
    assert "Layer Analysis" in layer_text
    assert "Layer Diagnostics" in layer_text
    assert "Layer Trend Plot" in layer_text
