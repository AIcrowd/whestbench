"""Layer diagnostics tab view for the Textual dashboard."""

from __future__ import annotations

from rich.table import Table
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from ..plots import build_layer_trend_plot
from ..state import DashboardState
from ..widgets import panel


def render_layers_view(state: DashboardState) -> str:
    """Render the layer diagnostics deep-dive tab."""

    points = ", ".join(f"{value:.6f}" for value in state.derived.layer_mse_mean_by_index)
    return (
        "Layer Analysis\n\n"
        "Layer Diagnostics\n"
        "- mse_mean_by_layer trend\n"
        "Layer Trend Plot\n"
        "- mse_mean trajectory over layer index\n"
        f"- layer means: [{points}]\n"
    )


def build_layers_pane(state: DashboardState) -> Widget:
    """Build the Layers tab with depth-wise diagnostics."""

    values = state.derived.layer_mse_mean_by_index
    trend_chart, trend_legend = build_layer_trend_plot(
        mse_by_layer=values,
        width=86,
        height=14,
    )
    stats = panel(
        "Layer Stats",
        Static(_layer_table(values), classes="table-static"),
        id="layers-stats-panel",
    )
    trend = panel(
        "Layer Trend",
        Static(trend_chart, classes="plot-body"),
        Static(trend_legend, classes="plot-legend"),
        id="layers-trend-panel",
    )
    insight = panel(
        "Layer Insight",
        Static(
            "Look for late-depth spikes to identify estimator drift.\n"
            "Use this with Budget and Performance tabs to determine if runtime limits or model bias dominate error.",
            classes="insight-text",
        ),
        id="layers-insight-panel",
    )
    return VerticalScroll(
        stats,
        trend,
        insight,
        classes="tab-scroll",
        id="layers-pane",
    )


def _layer_table(values: list[float]) -> Table:
    table = Table(box=None, expand=True, pad_edge=False, header_style="bold #9ca3af")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("layer_count", str(len(values)))
    table.add_row("min mse", f"{min(values) if values else 0.0:.6f}")
    table.add_row("max mse", f"{max(values) if values else 0.0:.6f}")
    table.add_row("mean mse", f"{sum(values) / len(values) if values else 0.0:.6f}")
    table.add_row("p95 mse", f"{_percentile(values, 0.95):.6f}")
    return table


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]
