"""Performance diagnostics tab view for the Textual dashboard."""

from __future__ import annotations

from ..state import DashboardState


def render_performance_view(state: DashboardState) -> str:
    """Render the runtime and memory diagnostics tab."""

    if not state.derived.has_profile:
        return "Performance\n\nNo profiling calls available for this run."
    return (
        "Performance\n\n"
        "Profile\n"
        "- wall_time_s\n"
        "- cpu_time_s\n"
        "- rss_bytes\n"
        "- peak_rss_bytes\n"
    )
