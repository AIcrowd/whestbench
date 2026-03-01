"""Textual app shell for the human dashboard."""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Header, Static

from .state import DashboardState, build_dashboard_state


class DashboardApp(App[None]):
    """Dashboard shell with tab routing and global key bindings."""

    TITLE = "Circuit Estimation Dashboard"
    BINDINGS = [
        Binding("1", "tab_summary", "Summary", show=True),
        Binding("2", "tab_budgets", "Budgets", show=True),
        Binding("3", "tab_layers", "Layers", show=True),
        Binding("4", "tab_performance", "Performance", show=True),
        Binding("5", "tab_data", "Data", show=True),
        Binding("r", "reload_report", "Reload", show=False),
        Binding("?", "toggle_help", "Help", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    def __init__(self, report: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.report = report
        self.state: DashboardState = build_dashboard_state(report)
        self.active_tab = "summary"
        self.show_help_overlay = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static("", id="dashboard-tabs")
        yield Static("", id="dashboard-content")
        yield Footer()

    def on_mount(self) -> None:
        self._refresh_screen()

    def action_tab_summary(self) -> None:
        self.active_tab = "summary"
        self._refresh_screen()

    def action_tab_budgets(self) -> None:
        self.active_tab = "budgets"
        self._refresh_screen()

    def action_tab_layers(self) -> None:
        self.active_tab = "layers"
        self._refresh_screen()

    def action_tab_performance(self) -> None:
        self.active_tab = "performance"
        self._refresh_screen()

    def action_tab_data(self) -> None:
        self.active_tab = "data"
        self._refresh_screen()

    def action_reload_report(self) -> None:
        self.state = build_dashboard_state(self.report)
        self._refresh_screen()

    def action_toggle_help(self) -> None:
        self.show_help_overlay = not self.show_help_overlay
        self._refresh_screen()

    def _refresh_screen(self) -> None:
        if not self.is_mounted or not self._screen_stack:
            return
        tabs = self.query_one("#dashboard-tabs", Static)
        tabs.update(self._tab_strip())
        content = self.query_one("#dashboard-content", Static)
        content.update(self._tab_content())

    def _tab_strip(self) -> str:
        order = ("summary", "budgets", "layers", "performance", "data")
        labels = []
        for name in order:
            label = name.capitalize()
            if name == self.active_tab:
                labels.append(f"[ {label} ]")
            else:
                labels.append(label)
        return "  |  ".join(labels)

    def _tab_content(self) -> str:
        if self.show_help_overlay:
            return "Keyboard: 1-5 switch tabs, r reload, q quit, ? help"
        mapping = {
            "summary": "Summary",
            "budgets": "Budget Analysis",
            "layers": "Layer Analysis",
            "performance": "Performance",
            "data": "Raw Data",
        }
        heading = mapping.get(self.active_tab, "Summary")
        return f"{heading}\n\nContent will be populated by dedicated view modules."
