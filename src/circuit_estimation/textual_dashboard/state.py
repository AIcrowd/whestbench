"""Shared data model and derived metrics for the Textual dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import fmean
from typing import Any


@dataclass(slots=True)
class DashboardDerivedState:
    """Precomputed metrics shared across dashboard views."""

    final_score: float
    best_budget_score: float
    worst_budget_score: float
    score_spread: float
    has_profile: bool
    layer_mse_mean_by_index: list[float]


@dataclass(slots=True)
class DashboardState:
    """Normalized dashboard state with raw and derived data."""

    raw_report: dict[str, Any]
    derived: DashboardDerivedState


def build_dashboard_state(report: dict[str, Any]) -> DashboardState:
    """Build dashboard state from the scorer report payload."""

    results = report.get("results", {})
    by_budget = results.get("by_budget_raw", [])
    entries = [entry for entry in by_budget if isinstance(entry, dict)]
    scores = [_as_float(entry.get("adjusted_mse", 0.0)) for entry in entries]
    best = min(scores) if scores else 0.0
    worst = max(scores) if scores else 0.0

    series = []
    for entry in entries:
        raw_layers = entry.get("mse_by_layer", [])
        if isinstance(raw_layers, list):
            series.append([_as_float(v) for v in raw_layers])
    max_len = max((len(values) for values in series), default=0)
    layer_mse_mean_by_index = []
    for idx in range(max_len):
        layer_values = [values[idx] for values in series if idx < len(values)]
        layer_mse_mean_by_index.append(fmean(layer_values) if layer_values else 0.0)

    derived = DashboardDerivedState(
        final_score=_as_float(results.get("final_score", 0.0)),
        best_budget_score=best,
        worst_budget_score=worst,
        score_spread=worst - best,
        has_profile=bool(report.get("profile_calls")),
        layer_mse_mean_by_index=layer_mse_mean_by_index,
    )
    return DashboardState(raw_report=report, derived=derived)


def _as_float(value: Any) -> float:
    """Convert arbitrary scalar input to float for dashboard computation."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
