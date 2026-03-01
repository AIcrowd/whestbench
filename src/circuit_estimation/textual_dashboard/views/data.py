"""Raw data inspection tab view for the Textual dashboard."""

from __future__ import annotations

import json

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widget import Widget
from textual.widgets import Pretty

from ..state import DashboardState
from ..widgets import panel


def render_data_view(state: DashboardState) -> str:
    """Render a raw data inspection view."""

    payload = json.dumps(state.raw_report, indent=2)
    return f"Raw Data\n\n{payload}"


def build_data_pane(state: DashboardState) -> Widget:
    """Build the Data tab for structured raw payload inspection."""

    run_meta = _as_dict(state.raw_report.get("run_meta"))
    run_config = _as_dict(state.raw_report.get("run_config"))
    results = _as_dict(state.raw_report.get("results"))
    profile_calls = state.raw_report.get("profile_calls", [])

    return VerticalScroll(
        Horizontal(
            panel(
                "Run Meta",
                Pretty(run_meta, id="data-run-meta-pretty"),
                id="data-run-meta-section",
            ),
            panel(
                "Run Config",
                Pretty(run_config, id="data-run-config-pretty"),
                id="data-run-config-section",
            ),
            classes="pane-row",
            id="data-top-row",
        ),
        Horizontal(
            panel(
                "Results",
                Pretty(results, id="data-results-pretty"),
                id="data-results-section",
            ),
            panel(
                "Profile Calls",
                Pretty(profile_calls, id="data-profile-calls-pretty"),
                id="data-profile-calls-section",
            ),
            classes="pane-row",
            id="data-bottom-row",
        ),
        panel(
            "Raw Payload",
            Vertical(
                Pretty(state.raw_report, id="data-pretty"),
                classes="data-raw-body",
            ),
            id="data-raw-payload-section",
        ),
        classes="tab-scroll",
        id="data-pane",
    )


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    return {}
