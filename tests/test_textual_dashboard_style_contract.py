from __future__ import annotations

from pathlib import Path

from circuit_estimation.textual_dashboard.app import layout_mode_for_width


DASHBOARD_CSS = Path("src/circuit_estimation/textual_dashboard/dashboard.tcss")


def test_style_contract_includes_scientific_editorial_tokens() -> None:
    css = DASHBOARD_CSS.read_text(encoding="utf-8")

    assert "--accent-accuracy" in css
    assert "--accent-runtime" in css
    assert "--accent-score" in css
    assert "--surface-base" in css


def test_layout_mode_breakpoints() -> None:
    assert layout_mode_for_width(180) == "wide"
    assert layout_mode_for_width(120) == "medium"
    assert layout_mode_for_width(85) == "narrow"
