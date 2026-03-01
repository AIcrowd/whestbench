from __future__ import annotations

import asyncio
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


def test_performance_and_data_tabs_render_structured_panes() -> None:
    async def _run() -> None:
        app = DashboardApp(report=_sample_report(include_profile=True))
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#performance-summary-panel")
            app.query_one("#performance-runtime-plot")
            app.query_one("#performance-memory-plot")
            app.query_one("#performance-outlier-panel")
            app.query_one("#data-run-meta-section")
            app.query_one("#data-run-config-section")
            app.query_one("#data-results-section")
            app.query_one("#data-profile-calls-section")

    asyncio.run(_run())


def test_performance_view_handles_missing_profile() -> None:
    async def _run() -> None:
        app = DashboardApp(report=_sample_report(include_profile=False))
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#performance-unavailable-panel")

    asyncio.run(_run())
