from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from circuit_estimation.textual_dashboard.app import DashboardApp, layout_mode_for_width

DASHBOARD_CSS = Path("src/circuit_estimation/textual_dashboard/dashboard.tcss")


def test_style_contract_includes_scientific_editorial_palette() -> None:
    css = DASHBOARD_CSS.read_text(encoding="utf-8")

    assert "#060b16" in css  # dark base canvas
    assert "#0f172a" in css  # panel surface
    assert "#38bdf8" in css  # emphasis accent
    assert "#fbbf24" in css  # runtime accent


def test_dashboard_stylesheet_parses_in_textual_runtime() -> None:
    async def _run() -> None:
        report: dict[str, Any] = {"results": {"final_score": 0.0, "by_budget_raw": []}}
        app = DashboardApp(report=report)
        async with app.run_test() as pilot:
            await pilot.pause()

    asyncio.run(_run())


def test_layout_mode_breakpoints() -> None:
    assert layout_mode_for_width(180) == "wide"
    assert layout_mode_for_width(120) == "medium"
    assert layout_mode_for_width(85) == "narrow"
