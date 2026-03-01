"""Budget deep-dive tab view for the Textual dashboard."""

from __future__ import annotations

from ..state import DashboardState


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
        f"Best score: {state.derived.best_budget_score:.8f}\n"
        f"Worst score: {state.derived.worst_budget_score:.8f}\n"
        f"Spread: {state.derived.score_spread:.8f}\n"
    )
