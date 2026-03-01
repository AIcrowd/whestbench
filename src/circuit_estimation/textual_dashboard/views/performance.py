"""Performance diagnostics tab view for the Textual dashboard."""

from __future__ import annotations

from statistics import fmean

from rich.table import Table
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from ..state import DashboardState
from ..widgets import panel, sparkline_block


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
                "Performance",
                Static("No profiling calls available for this run.", classes="insight-text"),
                id="performance-empty",
            ),
            classes="tab-scroll",
            id="performance-pane",
        )

    summary = panel(
        "Profile Summary",
        Static(_profile_table(state), classes="table-static"),
        id="performance-summary",
    )
    plots = Horizontal(
        sparkline_block(
            "Wall Time (s)",
            state.derived.profile_wall_s,
            note=_range_note(state.derived.profile_wall_s, "wall"),
            id="performance-plot-wall",
            min_color="#67e8f9",
            max_color="#38bdf8",
        ),
        sparkline_block(
            "CPU Time (s)",
            state.derived.profile_cpu_s,
            note=_range_note(state.derived.profile_cpu_s, "cpu"),
            id="performance-plot-cpu",
            min_color="#a7f3d0",
            max_color="#34d399",
        ),
        sparkline_block(
            "Memory (MB)",
            state.derived.profile_peak_rss_mb,
            note=_range_note(state.derived.profile_peak_rss_mb, "peak rss"),
            id="performance-plot-memory",
            min_color="#fcd34d",
            max_color="#fb923c",
        ),
        classes="pane-row",
        id="performance-plot-row",
    )
    return VerticalScroll(summary, plots, classes="tab-scroll", id="performance-pane")


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


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
