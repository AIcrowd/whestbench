"""Summary tab view for the Textual dashboard."""

from __future__ import annotations

from typing import Any

from ..state import DashboardState


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


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
