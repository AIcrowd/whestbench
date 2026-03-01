"""Summary tab view for the Textual dashboard."""

from __future__ import annotations

from typing import Any

from rich.table import Table
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from ..plots import (
    build_budget_frontier_plot,
    build_budget_runtime_plot,
    build_layer_trend_plot,
    build_profile_memory_plot,
    build_profile_runtime_plot,
)
from ..state import DashboardState
from ..widgets import metric_card, metric_row, panel


def render_summary_view(state: DashboardState) -> str:
    """Render the overview-first summary tab with legacy metric coverage."""

    run_meta = _as_dict(state.raw_report.get("run_meta"))
    run_config = _as_dict(state.raw_report.get("run_config"))
    lines = [
        "Summary",
        "",
        "Executive Overview",
        f"Final Score: {state.derived.final_score:.8f}",
        f"Best Budget Score: {state.derived.best_budget_score:.8f}",
        f"Worst Budget Score: {state.derived.worst_budget_score:.8f}",
        f"Score Spread: {state.derived.score_spread:.8f}",
        "",
        "Interesting Plots",
        "- Budget Frontier Plot",
        "- Layer Trend Plot",
        "- Runtime/Memory Spotlight",
        "",
        "Run Context",
        f"- Started: {run_meta.get('run_started_at_utc', 'n/a')}",
        f"- Finished: {run_meta.get('run_finished_at_utc', 'n/a')}",
        f"- Duration (s): {_as_float(run_meta.get('run_duration_s', 0.0)):.6f}",
        f"- Circuits: {run_config.get('n_circuits', 'n/a')}",
        f"- Samples/Circuit: {run_config.get('n_samples', 'n/a')}",
        f"- Width: {run_config.get('width', 'n/a')}",
        f"- Max Depth: {run_config.get('max_depth', 'n/a')}",
        f"- Budgets: {run_config.get('budgets', [])}",
        "",
        "Readiness Scorecard",
        "- lower score is better",
        "- score quality and runtime are summarized here",
        "",
        "Budget",
        "- per-budget score, mse, and runtime ratio",
        "",
        "Layer Diagnostics",
        "- layer-wise MSE trends and aggregate statistics",
    ]
    if state.derived.has_profile:
        lines.extend(
            [
                "",
                "Profile",
                "- call-level wall/cpu/memory diagnostics are available",
            ]
        )
    lines.extend(
        [
            "",
            "What To Do Next",
            "- Open Budgets for budget tradeoffs.",
            "- Open Layers for depth-specific issues.",
            "- Open Performance for runtime and memory hotspots.",
            "- Open Data for raw payload inspection.",
        ]
    )
    return "\n".join(lines)


def build_summary_pane(state: DashboardState) -> Widget:
    """Build the Summary tab with strict legacy-complete pane matrix."""

    run_meta = _as_dict(state.raw_report.get("run_meta"))
    run_config = _as_dict(state.raw_report.get("run_config"))

    status_strip = metric_row(
        metric_card("Final Score", f"{state.derived.final_score:.8f}", emphasis=True, id="metric-final"),
        metric_card("Best Budget", f"{state.derived.best_budget_score:.8f}"),
        metric_card("Worst Budget", f"{state.derived.worst_budget_score:.8f}"),
        metric_card("Spread", f"{state.derived.score_spread:.8f}"),
        id="summary-status-strip",
    )

    top_row = Horizontal(
        panel(
            "Run Context",
            Static(_context_table(run_meta, run_config), classes="table-static"),
            id="summary-run-context",
        ),
        panel(
            "Readiness Scorecard",
            Static(_score_table(state), classes="table-static"),
            id="summary-readiness",
        ),
        panel(
            "Hardware & Runtime",
            Static(_hardware_table(state), classes="table-static"),
            id="summary-hardware-runtime",
        ),
        classes="pane-row",
        id="summary-row-top",
    )

    frontier_chart, frontier_legend = build_budget_frontier_plot(
        budgets=state.derived.budgets,
        adjusted_mse=state.derived.budget_adjusted_scores,
        mse_mean=state.derived.budget_mse_means,
        width=64,
        height=11,
    )
    runtime_chart, runtime_legend = build_budget_runtime_plot(
        budgets=state.derived.budgets,
        time_ratio=state.derived.budget_time_ratio_means,
        effective_time=state.derived.budget_effective_time_means,
        width=64,
        height=11,
    )
    budget_row = Horizontal(
        panel(
            "Budget Table",
            Static(_budget_table(state), classes="table-static"),
            id="summary-budget-table",
        ),
        panel(
            "Budget Frontier Plot",
            Static(frontier_chart, classes="plot-body"),
            Static(frontier_legend, classes="plot-legend"),
            id="summary-budget-frontier",
        ),
        panel(
            "Budget Runtime Plot",
            Static(runtime_chart, classes="plot-body"),
            Static(runtime_legend, classes="plot-legend"),
            id="summary-budget-runtime",
        ),
        classes="pane-row",
        id="summary-row-budget",
    )

    layer_chart, layer_legend = build_layer_trend_plot(
        mse_by_layer=state.derived.layer_mse_mean_by_index,
        width=64,
        height=11,
    )
    layer_row = Horizontal(
        panel(
            "Layer Diagnostics",
            Static(_layer_table(state), classes="table-static"),
            id="summary-layer-diagnostics",
        ),
        panel(
            "Layer Trend Plot",
            Static(layer_chart, classes="plot-body"),
            Static(layer_legend, classes="plot-legend"),
            id="summary-layer-trend",
        ),
        classes="pane-row",
        id="summary-row-layer",
    )

    children: list[Widget] = [
        status_strip,
        top_row,
        budget_row,
        layer_row,
    ]

    if state.derived.has_profile:
        profile_summary = panel(
            "Profile Summary",
            Static(_profile_table(state), classes="table-static"),
            id="summary-profile-summary",
        )
        runtime_chart, runtime_legend = build_profile_runtime_plot(
            wall_s=state.derived.profile_wall_s,
            cpu_s=state.derived.profile_cpu_s,
            width=64,
            height=10,
        )
        profile_runtime = panel(
            "Profile Runtime Plot",
            Static(runtime_chart, classes="plot-body"),
            Static(runtime_legend, classes="plot-legend"),
            id="summary-profile-runtime",
        )
        memory_chart, memory_legend = build_profile_memory_plot(
            rss_mb=state.derived.profile_rss_mb,
            peak_mb=state.derived.profile_peak_rss_mb,
            width=64,
            height=10,
        )
        profile_memory = panel(
            "Profile Memory Plot",
            Static(memory_chart, classes="plot-body"),
            Static(memory_legend, classes="plot-legend"),
            id="summary-profile-memory",
        )
        children.append(
            Horizontal(
                profile_summary,
                profile_runtime,
                profile_memory,
                classes="pane-row",
                id="summary-row-profile",
            )
        )
    else:
        children.append(
            panel(
                "Profile Unavailable",
                Static("Profiling data unavailable for this run.", classes="insight-text"),
                id="summary-profile-unavailable",
            )
        )

    children.append(
        panel(
            "Next Actions",
            Static(
                "• Budgets: compare score/runtime tradeoffs\n"
                "• Layers: inspect where error grows by depth\n"
                "• Performance: inspect runtime + memory call behavior\n"
                "• Data: inspect full raw payload",
                classes="next-actions",
            ),
            id="summary-next-actions",
        )
    )

    return VerticalScroll(*children, classes="tab-scroll summary-scroll", id="summary-pane")


def _context_table(run_meta: dict[str, Any], run_config: dict[str, Any]) -> Table:
    table = Table(box=None, show_header=False, expand=True, pad_edge=False)
    table.add_column("field", style="bold #9ca3af")
    table.add_column("value", style="#e5e7eb")
    table.add_row("Started", str(run_meta.get("run_started_at_utc", "n/a")))
    table.add_row("Finished", str(run_meta.get("run_finished_at_utc", "n/a")))
    table.add_row("Duration (s)", f"{_as_float(run_meta.get('run_duration_s', 0.0)):.6f}")
    table.add_row("Circuits", str(run_config.get("n_circuits", "n/a")))
    table.add_row("Samples/Circuit", str(run_config.get("n_samples", "n/a")))
    table.add_row("Width", str(run_config.get("width", "n/a")))
    table.add_row("Max Depth", str(run_config.get("max_depth", "n/a")))
    table.add_row("Budgets", str(run_config.get("budgets", [])))
    table.add_row("Time Tolerance", str(run_config.get("time_tolerance", "n/a")))
    return table


def _score_table(state: DashboardState) -> Table:
    table = Table(box=None, show_header=False, expand=True, pad_edge=False)
    table.add_column("metric", style="bold #9ca3af")
    table.add_column("value", style="#f8fafc", justify="right")
    table.add_row("Final Score", f"{state.derived.final_score:.8f}")
    table.add_row("Best Budget", f"{state.derived.best_budget_score:.8f}")
    table.add_row("Worst Budget", f"{state.derived.worst_budget_score:.8f}")
    table.add_row("Spread", f"{state.derived.score_spread:.8f}")
    table.add_row("Interpretation", "lower is better")
    return table


def _budget_table(state: DashboardState) -> Table:
    table = Table(box=None, expand=True, pad_edge=False, header_style="bold #9ca3af")
    table.add_column("budget", justify="right")
    table.add_column("adj_mse", justify="right")
    table.add_column("mse_mean", justify="right")
    table.add_column("time_ratio", justify="right")
    table.add_column("eff_time(s)", justify="right")
    for budget, score, mse, ratio, effective in zip(
        state.derived.budgets,
        state.derived.budget_adjusted_scores,
        state.derived.budget_mse_means,
        state.derived.budget_time_ratio_means,
        state.derived.budget_effective_time_means,
        strict=False,
    ):
        table.add_row(
            str(budget),
            f"{score:.6f}",
            f"{mse:.6f}",
            f"{ratio:.4f}",
            f"{effective:.6f}",
        )
    return table


def _layer_table(state: DashboardState) -> Table:
    table = Table(box=None, expand=True, pad_edge=False, header_style="bold #9ca3af")
    table.add_column("stat")
    table.add_column("value", justify="right")
    table.add_row("layer_count", str(state.derived.layer_count))
    table.add_row("p05", f"{state.derived.layer_mse_p05:.6f}")
    table.add_row("min", f"{state.derived.layer_mse_min:.6f}")
    table.add_row("mean", f"{state.derived.layer_mse_mean:.6f}")
    table.add_row("p95", f"{state.derived.layer_mse_p95:.6f}")
    table.add_row("max", f"{state.derived.layer_mse_max:.6f}")
    return table


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


def _hardware_table(state: DashboardState) -> Table:
    table = Table(box=None, show_header=False, expand=True, pad_edge=False)
    table.add_column("field", style="bold #9ca3af")
    table.add_column("value", style="#e5e7eb")
    table.add_row("Host", state.derived.host_hostname)
    table.add_row("OS", state.derived.host_os)
    table.add_row("Release", state.derived.host_release)
    table.add_row("Platform", state.derived.host_platform)
    table.add_row("Machine", state.derived.host_machine)
    table.add_row("Python", state.derived.host_python_version)
    return table


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
