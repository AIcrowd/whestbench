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

    # Mean utilization is computed from effective_compute (C_m), not flops_used (F_m).
    effective_computes = [as_float(entry.get("effective_compute", 0.0)) for entry in per_mlp]
    mean_compute = sum(effective_computes) / len(effective_computes) if effective_computes else 0.0
    mean_utilization = mean_compute / flop_budget if has_budget else 0.0

    flag_keys = (
        "budget_exhausted",
        "time_exhausted",
        "residual_wall_time_exhausted",
        "combined_budget_exhausted",
    )
    any_busted = any(
        bool(entry.get("error_code")) or any(bool(entry.get(k)) for k in flag_keys)
        for entry in per_mlp
    )

    if any_busted and has_budget:
        worst_cm = max(
            as_float(entry.get("effective_compute", 0.0))
            for entry in per_mlp
            if any(bool(entry.get(k)) for k in flag_keys) or bool(entry.get("error_code"))
        )
        worst_mlp_pct: Optional[int] = int(worst_cm * 100 // flop_budget)
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
    reason: str  # "COMBINED" | "BUDGET" | "RESIDUAL" | "TIME" | "ERROR"
    metric_name: str  # "C_m" | "F_m" | "R_m" | "wall" | "error"
    metric_value: str  # e.g. "4.08e+10", "3.40e+10", "4.2s"
    flops_used: float
    pct_of_budget: Optional[int]


@dataclass(frozen=True)
class OverBudgetSelection:
    rows: list[OverBudgetRow]
    busted_count: int
    n_mlps: int
    is_truncated: bool
    is_all_busted: bool
    reason_counts: dict[str, int]


def select_top_over_budget(report: dict[str, Any], *, top_n: int = 5) -> OverBudgetSelection:
    run_config = report.get("run_config", {}) if isinstance(report, dict) else {}
    results = report.get("results", {}) if isinstance(report, dict) else {}
    per_mlp_raw = results.get("per_mlp", []) if isinstance(results, dict) else []
    per_mlp: list[dict[str, Any]] = [entry for entry in per_mlp_raw if isinstance(entry, dict)]
    n_mlps = len(per_mlp)

    try:
        flop_budget = int(run_config.get("flop_budget", 0) or 0)
    except (TypeError, ValueError):
        flop_budget = 0

    def _reason_for(entry: dict[str, Any]) -> Optional[str]:
        if entry.get("combined_budget_exhausted"):
            return "COMBINED"
        if entry.get("budget_exhausted"):
            return "BUDGET"
        if entry.get("residual_wall_time_exhausted"):
            return "RESIDUAL"
        if entry.get("time_exhausted"):
            return "TIME"
        if entry.get("error_code"):
            return "ERROR"
        return None

    def _metric_for(entry: dict[str, Any], reason: str) -> tuple[str, str]:
        if reason in ("COMBINED", "BUDGET"):
            cm = as_float(entry.get("effective_compute", 0.0))
            return ("C_m", fmt_flops(cm))
        if reason == "RESIDUAL":
            rt = as_float(entry.get("residual_wall_time_s", 0.0))
            return ("R_m", f"{rt:.3f}s")
        if reason == "TIME":
            wt = as_float(entry.get("wall_time_s", 0.0))
            return ("wall", f"{wt:.3f}s")
        if reason == "ERROR":
            return ("error", str(entry.get("error_code") or "ERROR"))
        return ("?", "?")

    def _sort_key(entry: dict[str, Any]) -> tuple[float, int]:
        cm = as_float(entry.get("effective_compute", 0.0))
        ratio = (cm / flop_budget) if flop_budget > 0 else 0.0
        return (-ratio, int(entry.get("mlp_index", 0)))

    failed: list[tuple[dict[str, Any], str]] = []
    for entry in per_mlp:
        reason = _reason_for(entry)
        if reason is not None:
            failed.append((entry, reason))

    sorted_failed = sorted(failed, key=lambda pair: _sort_key(pair[0]))
    rows_to_show = sorted_failed if len(sorted_failed) <= top_n + 1 else sorted_failed[:top_n]

    rows = []
    for entry, reason in rows_to_show:
        metric_name, metric_value = _metric_for(entry, reason)
        cm = as_float(entry.get("effective_compute", 0.0))
        rows.append(
            OverBudgetRow(
                mlp_index=int(entry.get("mlp_index", 0)),
                reason=reason,
                metric_name=metric_name,
                metric_value=metric_value,
                flops_used=as_float(entry.get("flops_used", 0.0)),
                pct_of_budget=(int(cm * 100 // flop_budget) if flop_budget > 0 else None),
            )
        )

    reason_counts = {"COMBINED": 0, "BUDGET": 0, "RESIDUAL": 0, "TIME": 0, "ERROR": 0}
    for _, reason in failed:
        reason_counts[reason] += 1

    busted_count = len(failed)
    return OverBudgetSelection(
        rows=rows,
        busted_count=busted_count,
        n_mlps=n_mlps,
        is_truncated=busted_count > len(rows),
        is_all_busted=busted_count > 0 and busted_count == n_mlps,
        reason_counts=reason_counts,
    )
