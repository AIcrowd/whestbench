from __future__ import annotations

import asyncio
from typing import Any

from textual.widgets import Static

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


def test_budget_and_layer_tabs_follow_pane_contract() -> None:
    async def _run() -> None:
        app = DashboardApp(report=_sample_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#budgets-table-panel")
            app.query_one("#budgets-frontier-panel")
            app.query_one("#budgets-runtime-panel")
            app.query_one("#budgets-insight-panel")
            app.query_one("#layers-stats-panel")
            app.query_one("#layers-trend-panel")
            app.query_one("#layers-insight-panel")

    asyncio.run(_run())


def test_budget_and_layer_plot_panes_render_chart_like_content() -> None:
    async def _run() -> None:
        app = DashboardApp(report=_sample_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            for selector in (
                "#budgets-frontier-panel .plot-body",
                "#budgets-runtime-panel .plot-body",
                "#layers-trend-panel .plot-body",
            ):
                content = app.query_one(selector, Static)
                chart_text = str(content.render())
                assert chart_text.strip()
                assert ("┤" in chart_text) or ("|" in chart_text) or (
                    "plot unavailable" in chart_text.lower()
                )

    asyncio.run(_run())
