"""Performance diagnostics tab view for the Textual dashboard."""

from __future__ import annotations

from statistics import fmean

from ..state import DashboardState


def render_performance_view(state: DashboardState) -> str:
    """Render the runtime and memory diagnostics tab."""

    if not state.derived.has_profile:
        return "Performance\n\nNo profiling calls available for this run."
    profile_calls = state.raw_report.get("profile_calls", [])
    calls = profile_calls if isinstance(profile_calls, list) else []
    wall = [_as_float(entry.get("wall_time_s", 0.0)) for entry in calls if isinstance(entry, dict)]
    cpu = [_as_float(entry.get("cpu_time_s", 0.0)) for entry in calls if isinstance(entry, dict)]
    rss = [_as_float(entry.get("rss_bytes", 0.0)) for entry in calls if isinstance(entry, dict)]
    peak = [_as_float(entry.get("peak_rss_bytes", 0.0)) for entry in calls if isinstance(entry, dict)]
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


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
