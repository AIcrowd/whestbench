from __future__ import annotations

import io
import re
from typing import Any, Dict, List, Optional

import pytest
from rich.console import Console

from whestbench.reporting import (
    GaugeState,
    OverBudgetRow,
    OverBudgetSelection,
    _compute_gauge_state,
    _select_top_over_budget,
)


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


def _render_via(console_fn) -> str:
    buffer = io.StringIO()
    console = Console(
        record=True, file=buffer, force_terminal=True, color_system="truecolor", width=120
    )
    console_fn(console)
    return buffer.getvalue()


def _report(
    *,
    flop_budget: int = 100,
    per_mlp: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if per_mlp is None:
        per_mlp = []
    return {
        "run_config": {"n_mlps": len(per_mlp), "flop_budget": flop_budget},
        "results": {"per_mlp": per_mlp},
    }


def _mlp(
    i: int,
    *,
    flops_used: float,
    budget_exhausted: bool = False,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "mlp_index": i,
        "flops_used": flops_used,
        "budget_exhausted": budget_exhausted,
    }
    if error is not None:
        entry["error"] = error
        entry["error_code"] = "ESTIMATOR_RUNTIME_ERROR"
    return entry


# --- _compute_gauge_state ---------------------------------------------------


def test_gauge_state_healthy_when_mean_under_80_and_no_busts() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[_mlp(i, flops_used=30.0) for i in range(3)],
    )
    state = _compute_gauge_state(report)
    assert isinstance(state, GaugeState)
    assert state.state_name == "healthy"
    assert state.mean_utilization == pytest.approx(0.30)
    assert state.worst_mlp_pct is None
    assert state.any_busted is False
    assert state.has_budget is True


def test_gauge_state_tight_when_mean_between_80_and_100_no_busts() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[_mlp(0, flops_used=85.0), _mlp(1, flops_used=95.0)],
    )
    state = _compute_gauge_state(report)
    assert state.state_name == "tight"
    assert state.mean_utilization == pytest.approx(0.90)
    assert state.worst_mlp_pct is None
    assert state.any_busted is False


def test_gauge_state_busted_when_any_mlp_budget_exhausted_but_mean_under_1() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[
            _mlp(0, flops_used=40.0),
            _mlp(1, flops_used=60.0),
            _mlp(2, flops_used=138.0, budget_exhausted=True),
        ],
    )
    state = _compute_gauge_state(report)
    assert state.state_name == "busted"
    assert state.mean_utilization == pytest.approx(0.7933, rel=1e-3)
    assert state.worst_mlp_pct == 138
    assert state.any_busted is True


def test_gauge_state_catastrophic_when_mean_at_least_1() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[
            _mlp(0, flops_used=120.0, budget_exhausted=True),
            _mlp(1, flops_used=212.0, budget_exhausted=True),
        ],
    )
    state = _compute_gauge_state(report)
    assert state.state_name == "catastrophic"
    assert state.mean_utilization == pytest.approx(1.66, rel=1e-3)
    assert state.worst_mlp_pct == 212
    assert state.any_busted is True


def test_gauge_state_handles_n_mlps_zero() -> None:
    report = _report(flop_budget=100, per_mlp=[])
    state = _compute_gauge_state(report)
    assert state.state_name == "healthy"
    assert state.mean_utilization == 0.0
    assert state.worst_mlp_pct is None
    assert state.any_busted is False
    assert state.has_budget is True


def test_gauge_state_handles_flop_budget_zero() -> None:
    report = _report(
        flop_budget=0,
        per_mlp=[_mlp(0, flops_used=42.0)],
    )
    state = _compute_gauge_state(report)
    assert state.has_budget is False
    assert state.mean_utilization == 0.0
    assert state.worst_mlp_pct is None
    assert state.state_name == "healthy"


def test_gauge_state_ignores_errored_entries_for_bust_check() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[
            _mlp(0, flops_used=0.0, error="boom"),
            _mlp(1, flops_used=50.0),
        ],
    )
    state = _compute_gauge_state(report)
    assert state.any_busted is False
    assert state.state_name == "healthy"


# --- _select_top_over_budget ------------------------------------------------


def _busted(i: int, flops: float) -> Dict[str, Any]:
    return _mlp(i, flops_used=flops, budget_exhausted=True)


def test_select_over_budget_empty_when_no_busts() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[_mlp(0, flops_used=10.0), _mlp(1, flops_used=20.0)],
    )
    sel = _select_top_over_budget(report)
    assert isinstance(sel, OverBudgetSelection)
    assert sel.rows == []
    assert sel.busted_count == 0
    assert sel.n_mlps == 2
    assert sel.is_truncated is False
    assert sel.is_all_busted is False


def test_select_over_budget_single_row() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[_mlp(0, flops_used=50.0), _busted(1, 130.0)],
    )
    sel = _select_top_over_budget(report)
    assert [row.mlp_index for row in sel.rows] == [1]
    assert isinstance(sel.rows[0], OverBudgetRow)
    assert sel.rows[0].flops_used == 130.0
    assert sel.rows[0].pct_of_budget == 130
    assert sel.busted_count == 1
    assert sel.is_truncated is False
    assert sel.is_all_busted is False


def test_select_over_budget_sorts_by_overage_desc_then_index_asc() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[
            _busted(5, 150.0),
            _busted(2, 150.0),  # tie with #5; lower index wins tie
            _busted(0, 200.0),
        ],
    )
    sel = _select_top_over_budget(report)
    assert [row.mlp_index for row in sel.rows] == [0, 2, 5]


def test_select_over_budget_shows_top_5_and_truncates_with_7_busted() -> None:
    per_mlp = [_busted(i, 100.0 + (10 - i)) for i in range(7)]
    report = _report(flop_budget=100, per_mlp=per_mlp)
    sel = _select_top_over_budget(report)
    assert len(sel.rows) == 5
    assert sel.busted_count == 7
    assert sel.is_truncated is True
    assert sel.is_all_busted is True  # 7 of 7


def test_select_over_budget_keeps_all_6_when_exactly_6_busted() -> None:
    per_mlp = [_busted(i, 100.0 + (10 - i)) for i in range(6)]
    report = _report(flop_budget=100, per_mlp=per_mlp)
    sel = _select_top_over_budget(report)
    assert len(sel.rows) == 6
    assert sel.busted_count == 6
    assert sel.is_truncated is False


def test_select_over_budget_marks_all_busted_when_n_equals_m() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[_busted(0, 120.0), _busted(1, 130.0)],
    )
    sel = _select_top_over_budget(report)
    assert sel.is_all_busted is True


def test_select_over_budget_excludes_errored_entries() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[
            _mlp(0, flops_used=0.0, error="boom"),
            _busted(1, 120.0),
        ],
    )
    sel = _select_top_over_budget(report)
    assert [row.mlp_index for row in sel.rows] == [1]


def test_select_over_budget_handles_flop_budget_zero() -> None:
    report = _report(
        flop_budget=0,
        per_mlp=[_busted(0, 100.0)],
    )
    sel = _select_top_over_budget(report)
    # If budget is 0, pct_of_budget is None; rows still sort by flops_used desc.
    assert len(sel.rows) == 1
    assert sel.rows[0].pct_of_budget is None
