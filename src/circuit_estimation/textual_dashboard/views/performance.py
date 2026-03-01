"""Performance diagnostics tab view for the Textual dashboard."""

from __future__ import annotations

from statistics import fmean
from typing import Any

from rich.table import Table
from textual.containers import VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from ..plots import build_profile_memory_plot, build_profile_runtime_plot
from ..state import DashboardState
from ..widgets import panel


def render_performance_view(state: DashboardState) -> str:
    """Render the runtime and memory diagnostics tab."""

    if not state.derived.has_profile:
        return "Performance\n\nNo profiling calls available for this run."
    profile_calls = state.raw_report.get("profile_calls", [])
    calls = profile_calls if isinstance(profile_calls, list) else []
    wall = [_as_float(entry.get("wall_time_s", 0.0)) for entry in calls if isinstance(entry, dict)]
    cpu = [_as_float(entry.get("cpu_time_s", 0.0)) for entry in calls if isinstance(entry, dict)]
    rss = [_as_float(entry.get("rss_bytes", 0.0)) for entry in calls if isinstance(entry, dict)]
    peak = [
        _as_float(entry.get("peak_rss_bytes", 0.0)) for entry in calls if isinstance(entry, dict)
    ]
    return (
        "Performance\n\n"
        "Profile\n"
        f"- calls: {len(calls)}\n"
        f"- mean wall_time_s: {fmean(wall) if wall else 0.0:.6f}\n"
        f"- mean cpu_time_s: {fmean(cpu) if cpu else 0.0:.6f}\n"
        f"- mean rss_bytes: {fmean(rss) if rss else 0.0:.0f}\n"
        f"- mean peak_rss_bytes: {fmean(peak) if peak else 0.0:.0f}\n\n"
        "Profile Runtime Plot\n"
        "- wall_time_s and cpu_time_s by call index\n\n"
        "Profile Memory Plot\n"
        "- rss_bytes and peak_rss_bytes by call index\n"
    )


def build_performance_pane(state: DashboardState) -> Widget:
    """Build the Performance tab with runtime and memory plots."""

    if not state.derived.has_profile:
        return VerticalScroll(
            panel(
                "Profile Unavailable",
                Static("No profiling calls available for this run.", classes="insight-text"),
                id="performance-unavailable-panel",
            ),
            classes="tab-scroll",
            id="performance-pane",
        )

    runtime_chart, runtime_legend = build_profile_runtime_plot(
        wall_s=state.derived.profile_wall_s,
        cpu_s=state.derived.profile_cpu_s,
        width=86,
        height=12,
    )
    memory_chart, memory_legend = build_profile_memory_plot(
        rss_mb=state.derived.profile_rss_mb,
        peak_mb=state.derived.profile_peak_rss_mb,
        width=86,
        height=12,
    )

    summary_panel = panel(
        "Profile Summary",
        Static(_profile_table(state), classes="table-static"),
        id="performance-summary-panel",
    )
    runtime_panel = panel(
        "Runtime Plot",
        Static(runtime_chart, classes="plot-body"),
        Static(runtime_legend, classes="plot-legend"),
        id="performance-runtime-plot",
    )
    memory_panel = panel(
        "Memory Plot",
        Static(memory_chart, classes="plot-body"),
        Static(memory_legend, classes="plot-legend"),
        id="performance-memory-plot",
    )
    outlier_panel = panel(
        "Outlier Spotlight",
        Static(_outlier_text(state), classes="insight-text"),
        id="performance-outlier-panel",
    )
    return VerticalScroll(
        summary_panel,
        runtime_panel,
        memory_panel,
        outlier_panel,
        classes="tab-scroll",
        id="performance-pane",
    )


def _profile_table(state: DashboardState) -> Table:
    table = Table(box=None, expand=True, pad_edge=False, header_style="bold #9ca3af")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("calls", str(len(state.derived.profile_wall_s)))
    table.add_row(
        "mean wall(s)",
        f"{sum(state.derived.profile_wall_s) / len(state.derived.profile_wall_s) if state.derived.profile_wall_s else 0.0:.6f}",
    )
    table.add_row(
        "mean cpu(s)",
        f"{sum(state.derived.profile_cpu_s) / len(state.derived.profile_cpu_s) if state.derived.profile_cpu_s else 0.0:.6f}",
    )
    table.add_row(
        "mean rss(MB)",
        f"{sum(state.derived.profile_rss_mb) / len(state.derived.profile_rss_mb) if state.derived.profile_rss_mb else 0.0:.2f}",
    )
    table.add_row(
        "mean peak(MB)",
        f"{sum(state.derived.profile_peak_rss_mb) / len(state.derived.profile_peak_rss_mb) if state.derived.profile_peak_rss_mb else 0.0:.2f}",
    )
    return table


def _range_note(values: list[float], label: str) -> str:
    if not values:
        return f"{label}: n/a"
    return f"{label}: {min(values):.6f} -> {max(values):.6f}"


def _outlier_text(state: DashboardState) -> str:
    wall = state.derived.profile_wall_s
    cpu = state.derived.profile_cpu_s
    peak = state.derived.profile_peak_rss_mb
    if not wall or not cpu or not peak:
        return "Insufficient profile series to compute outliers."

    max_wall_index = max(range(len(wall)), key=wall.__getitem__)
    max_cpu_index = max(range(len(cpu)), key=cpu.__getitem__)
    max_peak_index = max(range(len(peak)), key=peak.__getitem__)

    return (
        f"Highest wall-time call: idx={max_wall_index}, wall={wall[max_wall_index]:.6f}s\n"
        f"Highest cpu-time call: idx={max_cpu_index}, cpu={cpu[max_cpu_index]:.6f}s\n"
        f"Highest peak memory call: idx={max_peak_index}, peak={peak[max_peak_index]:.2f}MB"
    )


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
