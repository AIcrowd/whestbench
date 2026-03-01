from __future__ import annotations

from typing import Any, cast

from circuit_estimation.textual_dashboard.app import DashboardApp
from textual.binding import Binding


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
    keys: dict[str, str] = {}
    for binding in DashboardApp.BINDINGS:
        if isinstance(binding, tuple) and len(binding) >= 2:
            keys[str(binding[0])] = str(binding[1])
            continue
        normalized = cast(Binding, binding)
        keys[str(normalized.key)] = str(normalized.action)

    assert keys["1"] == "tab_summary"
    assert keys["2"] == "tab_budgets"
    assert keys["3"] == "tab_layers"
    assert keys["4"] == "tab_performance"
    assert keys["5"] == "tab_data"
