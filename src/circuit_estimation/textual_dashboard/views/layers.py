"""Layer diagnostics tab view for the Textual dashboard."""

from __future__ import annotations

from rich.table import Table
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from ..state import DashboardState
from ..widgets import panel, sparkline_block


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
    plot_row = Horizontal(
        sparkline_block(
            "Layer MSE Trend",
            values,
            note=f"layers={len(values)} | {_range(values)}",
            id="layers-plot-mse",
            min_color="#93c5fd",
            max_color="#38bdf8",
        ),
        sparkline_block(
            "Layer Runtime Pressure",
            _runtime_pressure_proxy(values),
            note="proxy derived from MSE shape for skim diagnostics",
            id="layers-plot-runtime-proxy",
            min_color="#fcd34d",
            max_color="#fb923c",
        ),
        classes="pane-row",
        id="layers-plot-row",
    )
    stats = panel(
        "Layer Stats",
        Static(_layer_table(values), classes="table-static"),
        id="layers-stats-panel",
    )
    insight = panel(
        "Layer Insight",
        Static(
            "Look for late-depth spikes to identify estimator drift.\n"
            "Use this with Budget and Performance tabs to determine if runtime limits or model bias dominate error.",
            classes="insight-text",
        ),
        id="layers-insight",
    )
    return VerticalScroll(
        plot_row,
        stats,
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


def _runtime_pressure_proxy(values: list[float]) -> list[float]:
    if not values:
        return [0.0]
    baseline = sum(values) / len(values)
    if baseline <= 0:
        return values
    return [value / baseline for value in values]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def _range(values: list[float]) -> str:
    if not values:
        return "range: n/a"
    return f"range: {min(values):.6f} -> {max(values):.6f}"
