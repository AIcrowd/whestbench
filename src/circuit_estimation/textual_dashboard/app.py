"""Textual app shell for the human dashboard."""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.widgets import Static


class DashboardApp(App[None]):
    """Minimal dashboard app scaffold; views are added incrementally."""

    def __init__(self, report: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.report = report
        self.active_tab = "summary"

    def compose(self) -> ComposeResult:
        yield Static("Circuit Estimation Dashboard", id="dashboard-shell")
