"""Raw data inspection tab view for the Textual dashboard."""

from __future__ import annotations

import json

from textual.containers import VerticalScroll
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

    return VerticalScroll(
        panel(
            "Raw Payload",
            Pretty(state.raw_report, id="data-pretty"),
            id="data-pane-content",
        ),
        classes="tab-scroll",
        id="data-pane",
    )
