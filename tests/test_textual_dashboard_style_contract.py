from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from circuit_estimation.textual_dashboard.app import DashboardApp

DASHBOARD_CSS = Path("src/circuit_estimation/textual_dashboard/dashboard.tcss")


def test_style_contract_includes_black_baseline_palette_and_pane_tokens() -> None:
    css = DASHBOARD_CSS.read_text(encoding="utf-8")

    assert "#000000" in css  # black base canvas
    assert "#080f1f" in css  # elevated panel surface
    assert "#38bdf8" in css  # cyan emphasis accent
    assert "#f59e0b" in css  # amber runtime accent
    assert ".panel-title" in css
    assert ".plot-body" in css
    assert ".pane-row" in css


def test_dashboard_stylesheet_parses_in_textual_runtime() -> None:
    async def _run() -> None:
        report: dict[str, Any] = {"results": {"final_score": 0.0, "by_budget_raw": []}}
        app = DashboardApp(report=report)
        async with app.run_test() as pilot:
            await pilot.pause()

    asyncio.run(_run())
