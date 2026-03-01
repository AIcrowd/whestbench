from __future__ import annotations

import asyncio
from typing import Any

from textual.binding import Binding

from circuit_estimation.textual_dashboard.app import DashboardApp, layout_mode_for_width


def _sample_report() -> dict[str, Any]:
    return {
        "results": {
            "final_score": 0.123,
            "by_budget_raw": [
                {"budget": 10, "adjusted_mse": 0.22, "mse_by_layer": [0.1, 0.2]},
                {"budget": 100, "adjusted_mse": 0.036, "mse_by_layer": [0.04, 0.03]},
            ],
        }
    }


def test_default_tab_is_summary() -> None:
    app = DashboardApp(report=_sample_report())

    assert app.active_tab == "summary"


def test_tab_actions_switch_active_tab() -> None:
    app = DashboardApp(report=_sample_report())

    app.action_tab_budgets()
    assert app.active_tab == "budgets"
    app.action_tab_layers()
    assert app.active_tab == "layers"
    app.action_tab_performance()
    assert app.active_tab == "performance"
    app.action_tab_data()
    assert app.active_tab == "data"


def test_bindings_include_numeric_tab_shortcuts() -> None:
    bindings = [binding for binding in DashboardApp.BINDINGS if isinstance(binding, Binding)]
    keys = {binding.key: binding.action for binding in bindings}

    assert keys["1"] == "tab_summary"
    assert keys["2"] == "tab_budgets"
    assert keys["3"] == "tab_layers"
    assert keys["4"] == "tab_performance"
    assert keys["5"] == "tab_data"
    assert keys["escape"] == "quit"
    assert keys["ctrl+c"] == "quit"


def test_data_tab_sections_present_after_tab_switch() -> None:
    async def _run() -> None:
        app = DashboardApp(report=_sample_report())
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_tab_data()
            await pilot.pause()
            app.query_one("#data-run-meta-section")
            app.query_one("#data-run-config-section")
            app.query_one("#data-results-section")
            app.query_one("#data-profile-calls-section")

    asyncio.run(_run())


def test_layout_mode_breakpoints_match_dashboard_contract() -> None:
    assert layout_mode_for_width(160) == "wide"
    assert layout_mode_for_width(159) == "medium"
    assert layout_mode_for_width(110) == "medium"
    assert layout_mode_for_width(109) == "narrow"
