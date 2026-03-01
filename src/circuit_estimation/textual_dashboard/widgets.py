"""Reusable widget builders for pane-based dashboard composition."""

from __future__ import annotations

from textual.containers import Horizontal, Vertical
from textual.widget import Widget
from textual.widgets import Label, Sparkline, Static


def panel(title: str, *children: Widget, classes: str | None = None, id: str | None = None) -> Widget:
    """Build a bordered panel with a consistent title style."""

    class_names = "panel"
    if classes:
        class_names = f"{class_names} {classes}"
    return Vertical(Label(title, classes="panel-title"), *children, classes=class_names, id=id)


def metric_card(label: str, value: str, *, emphasis: bool = False, id: str | None = None) -> Widget:
    """Build a compact summary metric card for high-level skimming."""

    class_names = "metric-card"
    if emphasis:
        class_names = f"{class_names} metric-card-primary"
    return Vertical(
        Label(label, classes="metric-label"),
        Static(value, classes="metric-value"),
        classes=class_names,
        id=id,
    )


def metric_row(*cards: Widget, id: str | None = None) -> Widget:
    """Build a horizontal row of metric cards."""

    return Horizontal(*cards, classes="metric-row", id=id)


def sparkline_block(
    title: str,
    data: list[float],
    *,
    note: str,
    id: str | None = None,
    min_color: str = "#67e8f9",
    max_color: str = "#fb7185",
) -> Widget:
    """Build a plot-like panel using Textual's Sparkline widget."""

    series = data if data else [0.0]
    return panel(
        title,
        Sparkline(series, min_color=min_color, max_color=max_color, classes="sparkline"),
        Static(note, classes="sparkline-note"),
        classes="plot-panel",
        id=id,
    )
