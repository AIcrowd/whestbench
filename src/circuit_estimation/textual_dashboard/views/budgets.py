"""Budget deep-dive tab view for the Textual dashboard."""

from __future__ import annotations

from rich.table import Table
from textual.containers import Horizontal, VerticalScroll
from textual.widget import Widget
from textual.widgets import Static

from ..plots import build_budget_frontier_plot, build_budget_runtime_plot
from ..state import DashboardState
from ..widgets import panel


def render_budgets_view(state: DashboardState) -> str:
    """Render the budgets deep-dive tab."""

    return (
        "Budget Analysis\n\n"
        "Budget Table\n"
        "- budget\n"
        "- adjusted_mse\n"
        "- mse_mean\n"
        "- call_time_ratio_mean\n"
        "- call_effective_time_s_mean\n\n"
        "Budget Frontier Plot\n"
        "- adjusted_mse vs budget\n\n"
        "Budget Runtime Plot\n"
        "- call_time_ratio_mean vs budget\n\n"
        f"Best score: {state.derived.best_budget_score:.8f}\n"
        f"Worst score: {state.derived.worst_budget_score:.8f}\n"
        f"Spread: {state.derived.score_spread:.8f}\n"
    )


def build_budgets_pane(state: DashboardState) -> Widget:
    """Build the Budgets tab with structured panes and plotted trends."""

    table_panel = panel(
        "Budget Table",
        Static(_budget_table(state), classes="table-static"),
        id="budgets-table-panel",
    )
    frontier_chart, frontier_legend = build_budget_frontier_plot(
        budgets=state.derived.budgets,
        adjusted_mse=state.derived.budget_adjusted_scores,
        mse_mean=state.derived.budget_mse_means,
        width=64,
        height=12,
    )
    runtime_chart, runtime_legend = build_budget_runtime_plot(
        budgets=state.derived.budgets,
        time_ratio=state.derived.budget_time_ratio_means,
        effective_time=state.derived.budget_effective_time_means,
        width=64,
        height=12,
    )
    frontier_panel = panel(
        "Budget Frontier",
        Static(frontier_chart, classes="plot-body"),
        Static(frontier_legend, classes="plot-legend"),
        id="budgets-frontier-panel",
    )
    runtime_panel = panel(
        "Budget Runtime",
        Static(runtime_chart, classes="plot-body"),
        Static(runtime_legend, classes="plot-legend"),
        id="budgets-runtime-panel",
    )
    insight_panel = panel(
        "Budget Insight",
        Static(
            f"Best budget score: {state.derived.best_budget_score:.8f}\n"
            f"Worst budget score: {state.derived.worst_budget_score:.8f}\n"
            f"Spread: {state.derived.score_spread:.8f}\n"
            "Use this tab to pick an operating budget for quality/runtime tradeoff.",
            classes="insight-text",
        ),
        id="budgets-insight-panel",
    )
    return VerticalScroll(
        table_panel,
        Horizontal(
            frontier_panel,
            runtime_panel,
            classes="pane-row",
            id="budgets-plot-row",
        ),
        insight_panel,
        classes="tab-scroll",
        id="budgets-pane",
    )


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
