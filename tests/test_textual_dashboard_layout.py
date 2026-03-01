from __future__ import annotations

import asyncio
from typing import Any

from textual.widgets import Static, TabbedContent

from circuit_estimation.textual_dashboard.app import DashboardApp


def _sample_report() -> dict[str, Any]:
    return {
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
        "profile_calls": [
            {
                "budget": 10,
                "circuit_index": 0,
                "wall_time_s": 0.05,
                "cpu_time_s": 0.04,
                "rss_bytes": 12_345_678,
                "peak_rss_bytes": 15_000_000,
            }
        ],
    }


def test_dashboard_uses_tabbed_pane_layout_with_plotext_summary_charts() -> None:
    async def _run() -> None:
        app = DashboardApp(report=_sample_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one(TabbedContent)
            assert len(list(app.query(".panel"))) >= 10
            frontier_plot = app.query_one("#summary-budget-frontier .plot-body", Static)
            runtime_plot = app.query_one("#summary-budget-runtime .plot-body", Static)
            layer_plot = app.query_one("#summary-layer-trend .plot-body", Static)
            for plot_body in (frontier_plot, runtime_plot, layer_plot):
                content = str(plot_body.render())
                assert content.strip()
                assert ("┤" in content) or ("|" in content) or ("plot unavailable" in content.lower())

    asyncio.run(_run())
