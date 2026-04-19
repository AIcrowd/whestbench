from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

_GAUGE_BAR_WIDTH = 20


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def fmt_flops(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:
        return "n/a"
    if abs(numeric) < 1e6:
        return f"{int(numeric):,}"
    return f"{numeric:.2e}"


def gauge_bar_fragment(utilization: float) -> str:
    if utilization <= 0.0:
        filled = 0
    else:
        filled = min(round(utilization * _GAUGE_BAR_WIDTH), _GAUGE_BAR_WIDTH)
    empty = _GAUGE_BAR_WIDTH - filled
    return "[" + ("#" * filled) + ("-" * empty) + "]"


@dataclass(frozen=True)
class GaugeState:
    state_name: str
    mean_utilization: float
    worst_mlp_pct: Optional[int]
    any_busted: bool
    flop_budget: int
    has_budget: bool


def compute_gauge_state(report: dict[str, Any]) -> GaugeState:
    run_config = report.get("run_config", {}) if isinstance(report, dict) else {}
    results = report.get("results", {}) if isinstance(report, dict) else {}
    per_mlp_raw = results.get("per_mlp", []) if isinstance(results, dict) else []
    per_mlp: list[dict[str, Any]] = [entry for entry in per_mlp_raw if isinstance(entry, dict)]

    try:
        flop_budget = int(run_config.get("flop_budget", 0) or 0)
    except (TypeError, ValueError):
        flop_budget = 0
    has_budget = flop_budget > 0

    flops_used = [as_float(entry.get("flops_used", 0.0)) for entry in per_mlp]
    mean_flops = sum(flops_used) / len(flops_used) if flops_used else 0.0
    mean_utilization = mean_flops / flop_budget if has_budget else 0.0
    any_busted = any(bool(entry.get("budget_exhausted", False)) for entry in per_mlp)

    if any_busted and has_budget:
        worst_flops = max(
            as_float(entry.get("flops_used", 0.0))
            for entry in per_mlp
            if bool(entry.get("budget_exhausted", False))
        )
        worst_mlp_pct: Optional[int] = int(worst_flops * 100 // flop_budget)
    else:
        worst_mlp_pct = None

    if mean_utilization >= 1.0:
        state_name = "catastrophic"
    elif any_busted:
        state_name = "busted"
    elif mean_utilization >= 0.80:
        state_name = "tight"
    else:
        state_name = "healthy"

    return GaugeState(
        state_name=state_name,
        mean_utilization=mean_utilization,
        worst_mlp_pct=worst_mlp_pct,
        any_busted=any_busted,
        flop_budget=flop_budget,
        has_budget=has_budget,
    )


@dataclass(frozen=True)
class OverBudgetRow:
    mlp_index: int
    flops_used: float
    pct_of_budget: Optional[int]


@dataclass(frozen=True)
class OverBudgetSelection:
    rows: list[OverBudgetRow]
    busted_count: int
    n_mlps: int
    is_truncated: bool
    is_all_busted: bool


def select_top_over_budget(
    report: dict[str, Any], *, top_n: int = 5
) -> OverBudgetSelection:
    run_config = report.get("run_config", {}) if isinstance(report, dict) else {}
    results = report.get("results", {}) if isinstance(report, dict) else {}
    per_mlp_raw = results.get("per_mlp", []) if isinstance(results, dict) else []
    per_mlp: list[dict[str, Any]] = [entry for entry in per_mlp_raw if isinstance(entry, dict)]
    n_mlps = len(per_mlp)

    try:
        flop_budget = int(run_config.get("flop_budget", 0) or 0)
    except (TypeError, ValueError):
        flop_budget = 0

    busted = [entry for entry in per_mlp if bool(entry.get("budget_exhausted", False))]

    def _overage(entry: dict[str, Any]) -> float:
        flops = as_float(entry.get("flops_used", 0.0))
        if flop_budget > 0:
            return flops / flop_budget
        return flops

    sorted_busted = sorted(
        busted,
        key=lambda entry: (-_overage(entry), int(entry.get("mlp_index", 0))),
    )
    rows_to_show = sorted_busted if len(busted) <= top_n + 1 else sorted_busted[:top_n]

    rows = [
        OverBudgetRow(
            mlp_index=int(entry.get("mlp_index", 0)),
            flops_used=as_float(entry.get("flops_used", 0.0)),
            pct_of_budget=(
                int(as_float(entry.get("flops_used", 0.0)) * 100 // flop_budget)
                if flop_budget > 0
                else None
            ),
        )
        for entry in rows_to_show
    ]
    busted_count = len(busted)
    return OverBudgetSelection(
        rows=rows,
        busted_count=busted_count,
        n_mlps=n_mlps,
        is_truncated=busted_count > len(rows),
        is_all_busted=busted_count > 0 and busted_count == n_mlps,
    )
