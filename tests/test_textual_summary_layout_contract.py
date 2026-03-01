from __future__ import annotations

import asyncio
from typing import Any

from circuit_estimation.textual_dashboard.app import DashboardApp


def _sample_report(*, include_profile: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {
        "run_meta": {
            "run_started_at_utc": "2026-03-01T00:00:00+00:00",
            "run_finished_at_utc": "2026-03-01T00:00:01+00:00",
            "run_duration_s": 1.0,
            "host": {
                "hostname": "example-host",
                "os": "Darwin",
                "os_release": "25.3.0",
                "platform": "macOS-26.3-arm64-arm-64bit-Mach-O",
                "machine": "arm64",
                "python_version": "3.13.7",
            },
        },
        "run_config": {
            "n_circuits": 2,
            "n_samples": 100,
            "width": 4,
            "max_depth": 3,
            "layer_count": 3,
            "budgets": [10, 100],
            "time_tolerance": 0.1,
        },
        "results": {
            "final_score": 0.123,
            "by_budget_raw": [
                {
                    "budget": 10,
                    "mse_by_layer": [0.1, 0.2, 0.3],
                    "mse_mean": 0.2,
                    "adjusted_mse": 0.22,
                    "call_time_ratio_mean": 1.1,
                    "call_effective_time_s_mean": 0.011,
                },
                {
                    "budget": 100,
                    "mse_by_layer": [0.05, 0.04, 0.03],
                    "mse_mean": 0.04,
                    "adjusted_mse": 0.036,
                    "call_time_ratio_mean": 0.9,
                    "call_effective_time_s_mean": 0.018,
                },
            ],
        },
    }
    if include_profile:
        report["profile_calls"] = [
            {
                "budget": 10,
                "circuit_index": 0,
                "wall_time_s": 0.05,
                "cpu_time_s": 0.04,
                "rss_bytes": 12_345_678,
                "peak_rss_bytes": 15_000_000,
            }
        ]
    return report


def test_summary_tab_contains_approved_pane_matrix() -> None:
    async def _run() -> None:
        app = DashboardApp(report=_sample_report(include_profile=True))
        async with app.run_test() as pilot:
            await pilot.pause()
            for pane_id in (
                "summary-status-strip",
                "summary-run-context",
                "summary-readiness",
                "summary-hardware-runtime",
                "summary-budget-table",
                "summary-budget-frontier",
                "summary-budget-runtime",
                "summary-layer-diagnostics",
                "summary-layer-trend",
                "summary-profile-summary",
                "summary-profile-runtime",
                "summary-profile-memory",
            ):
                app.query_one(f"#{pane_id}")

    asyncio.run(_run())


def test_summary_shows_profile_unavailable_pane_when_profile_missing() -> None:
    async def _run() -> None:
        app = DashboardApp(report=_sample_report(include_profile=False))
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#summary-profile-unavailable")

    asyncio.run(_run())
