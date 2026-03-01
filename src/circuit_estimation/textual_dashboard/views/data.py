"""Raw data inspection tab view for the Textual dashboard."""

from __future__ import annotations

import json

from ..state import DashboardState


def render_data_view(state: DashboardState) -> str:
    """Render a raw data inspection view."""

    payload = json.dumps(state.raw_report, indent=2)
    return f"Raw Data\n\n{payload}"
