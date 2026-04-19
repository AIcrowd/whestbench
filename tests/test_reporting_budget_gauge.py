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
    _render_budget_gauge,
    _render_over_budget_panel,
    _select_top_over_budget,
    render_human_report,
    render_human_results,
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


# --- _render_budget_gauge ---------------------------------------------------


def _render_gauge(report: Dict[str, Any]) -> str:
    return _render_via(lambda console: _render_budget_gauge(console, report))


def test_render_gauge_healthy_shows_green_bar_and_percent() -> None:
    report = _report(
        flop_budget=100_000_000,
        per_mlp=[_mlp(i, flops_used=30_000_000.0) for i in range(3)],
    )
    out = _render_gauge(report)
    plain = _strip_ansi(out)
    assert "Estimator FLOPs" in plain
    assert "30%" in plain
    # _fmt_flops uses `{:.2e}` → "1.00e+08" for 100_000_000
    assert "of 1.00e+08" in plain
    # no worst-MLP suffix on healthy runs
    assert "worst MLP" not in plain
    # Rich ProgressBar renders ━-style glyphs (responsive width, not fixed cells)
    assert "━" in plain


def test_render_gauge_tight_shows_91_percent_yellow() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[_mlp(0, flops_used=85.0), _mlp(1, flops_used=95.0)],
    )
    out = _render_gauge(report)
    plain = _strip_ansi(out)
    assert "90%" in plain
    assert "worst MLP" not in plain


def test_render_gauge_busted_shows_worst_mlp_suffix() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[
            _mlp(0, flops_used=40.0),
            _mlp(1, flops_used=60.0),
            _mlp(2, flops_used=138.0, budget_exhausted=True),
        ],
    )
    out = _render_gauge(report)
    plain = _strip_ansi(out)
    assert "worst MLP 138%" in plain
    assert "⚠" in plain


def test_render_gauge_catastrophic_shows_overflow_arrow() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[
            _mlp(0, flops_used=120.0, budget_exhausted=True),
            _mlp(1, flops_used=212.0, budget_exhausted=True),
        ],
    )
    out = _render_gauge(report)
    plain = _strip_ansi(out)
    assert "▶" in plain
    assert "worst MLP 212%" in plain
    # Rich ProgressBar renders ━-style glyphs and clamps at 100%;
    # overflow is signaled by the ▶ above, not by the bar itself.
    assert "━" in plain


def test_render_gauge_with_flop_budget_zero_suppresses_bar() -> None:
    report = _report(
        flop_budget=0,
        per_mlp=[_mlp(0, flops_used=42.0)],
    )
    out = _render_gauge(report)
    plain = _strip_ansi(out)
    assert "--" in plain
    assert "of 0 FLOPs" in plain
    # no bar is rendered when there's no budget
    assert "━" not in plain


# --- _render_over_budget_panel ----------------------------------------------


def _render_panel_out(report: Dict[str, Any]) -> str:
    return _render_via(lambda console: _render_over_budget_panel(console, report))


def test_panel_not_rendered_when_no_busts() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[_mlp(0, flops_used=50.0), _mlp(1, flops_used=60.0)],
    )
    out = _render_panel_out(report)
    assert out.strip() == ""


def test_panel_renders_single_row_with_singular_summary() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[_mlp(0, flops_used=50.0), _busted(1, 138.0)],
    )
    out = _render_panel_out(report)
    plain = _strip_ansi(out)
    assert "Over-Budget MLPs" in plain
    assert "MLP #1" in plain
    assert "138% of budget" in plain
    assert "zeroed" in plain
    # 1 of 2 busted (singular summary not used: is_all_busted=False)
    assert "1 of 2 MLPs exceeded the per-MLP FLOP cap" in plain
    # no truncation footer
    assert "more over budget" not in plain
    assert "--json" not in plain


def test_panel_top5_and_truncation_footer_when_7_busted() -> None:
    per_mlp = [_busted(i, 200.0 - i) for i in range(7)]
    report = _report(flop_budget=100, per_mlp=per_mlp)
    out = _render_panel_out(report)
    plain = _strip_ansi(out)
    # top 5 shown (highest overage first)
    for idx in range(5):
        assert f"MLP #{idx}" in plain
    # remaining 2 excluded
    assert "MLP #5" not in plain
    assert "MLP #6" not in plain
    # truncation footer
    assert "... and 2 more over budget" in plain
    assert "run with --format json for the full list" in plain
    # all-busted reframe (7 of 7)
    assert "All 7 MLPs exceeded the per-MLP FLOP cap — predictions entirely zeroed" in plain


def test_panel_shows_all_6_when_exactly_6_busted_no_truncation() -> None:
    per_mlp = [_busted(i, 200.0 - i) for i in range(6)]
    report = _report(flop_budget=100, per_mlp=per_mlp)
    out = _render_panel_out(report)
    plain = _strip_ansi(out)
    for idx in range(6):
        assert f"MLP #{idx}" in plain
    assert "... and" not in plain
    assert "--json" not in plain


def test_panel_normal_vs_all_busted_summary_variant() -> None:
    # Normal case: 3 of 10
    per_mlp_normal = [_busted(i, 150.0) for i in range(3)] + [
        _mlp(i, flops_used=30.0) for i in range(3, 10)
    ]
    out_normal = _render_panel_out(_report(flop_budget=100, per_mlp=per_mlp_normal))
    plain_normal = _strip_ansi(out_normal)
    assert "3 of 10 MLPs exceeded the per-MLP FLOP cap" in plain_normal
    assert "entirely zeroed" not in plain_normal

    # All-busted: 3 of 3
    per_mlp_all = [_busted(i, 150.0) for i in range(3)]
    out_all = _render_panel_out(_report(flop_budget=100, per_mlp=per_mlp_all))
    plain_all = _strip_ansi(out_all)
    assert "All 3 MLPs exceeded the per-MLP FLOP cap — predictions entirely zeroed" in plain_all


def test_panel_excludes_errored_mlps() -> None:
    report = _report(
        flop_budget=100,
        per_mlp=[
            _mlp(0, flops_used=0.0, error="boom"),
            _busted(1, 120.0),
        ],
    )
    out = _render_panel_out(report)
    plain = _strip_ansi(out)
    assert "MLP #1" in plain
    assert "MLP #0" not in plain


def test_panel_flop_budget_zero_shows_dashes_for_pct() -> None:
    report = _report(
        flop_budget=0,
        per_mlp=[_busted(0, 100.0)],
    )
    out = _render_panel_out(report)
    plain = _strip_ansi(out)
    assert "MLP #0" in plain
    assert "--% of budget" in plain


# --- Wiring into render_human_results / render_human_report -----------------


def _full_report(per_mlp: List[Dict[str, Any]], *, flop_budget: int = 100) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "primary_score": 0.123,
        "secondary_score": 0.456,
        "per_mlp": per_mlp,
    }
    # The gauge and over-budget content now live inside the Estimator Budget
    # Breakdown panel. Provide a minimal breakdown so wiring tests see that
    # panel render (and, transitively, the gauge).
    if per_mlp:
        results["breakdowns"] = {
            "estimator": {
                "flops_used": sum(float(entry.get("flops_used", 0.0) or 0.0) for entry in per_mlp),
                "tracked_time_s": 0.0,
                "untracked_time_s": 0.0,
                "by_namespace": {},
            }
        }
    return {
        "schema_version": "1.0",
        "mode": "human",
        "detail": "raw",
        "run_meta": {
            "run_started_at_utc": "2026-04-18T00:00:00+00:00",
            "run_finished_at_utc": "2026-04-18T00:00:01+00:00",
            "run_duration_s": 1.0,
        },
        "run_config": {
            "n_mlps": len(per_mlp),
            "width": 4,
            "depth": 3,
            "flop_budget": flop_budget,
        },
        "results": results,
        "notes": [],
    }


def test_render_human_results_uses_main_style_estimator_breakdown_without_gauge() -> None:
    report = _full_report([_mlp(i, flops_used=30.0) for i in range(3)], flop_budget=100)
    rendered = render_human_results(report)
    plain = _strip_ansi(rendered)

    assert "Estimator Budget Breakdown" in plain
    assert "Estimator FLOPs" not in plain
    assert "Over-Budget MLPs" not in plain
    assert "Total FLOPs [flops_used]" in plain


def test_render_human_results_omits_over_budget_panel_when_busted() -> None:
    per_mlp = [_mlp(0, flops_used=50.0), _busted(1, 138.0)]
    rendered = render_human_results(_full_report(per_mlp, flop_budget=100))
    plain = _strip_ansi(rendered)
    assert "Estimator Budget Breakdown" in plain
    assert "Over-Budget MLPs" not in plain
    assert "worst MLP" not in plain


def test_render_human_report_omits_over_budget_panel_when_busted() -> None:
    per_mlp = [_mlp(0, flops_used=50.0), _busted(1, 138.0)]
    rendered = render_human_report(_full_report(per_mlp, flop_budget=100))
    plain = _strip_ansi(rendered)
    assert "Over-Budget MLPs" not in plain
    assert "worst MLP" not in plain
    # render_human_report also includes the header — sanity check it's there too
    assert "WhestBench Report" in plain


def test_render_human_results_omits_over_budget_panel_when_clean() -> None:
    rendered = render_human_results(
        _full_report([_mlp(i, flops_used=30.0) for i in range(3)], flop_budget=100)
    )
    plain = _strip_ansi(rendered)
    assert "Over-Budget MLPs" not in plain


def test_gauge_and_over_budget_are_not_embedded_in_settled_human_output() -> None:
    per_mlp = [_busted(0, 120.0), _mlp(1, flops_used=50.0)]
    report = _full_report(per_mlp, flop_budget=100)
    report["results"]["breakdowns"]["sampling"] = {
        "flops_used": 30,
        "tracked_time_s": 0.001,
        "untracked_time_s": 0.0,
        "by_namespace": {},
    }
    rendered = render_human_results(report)
    plain = _strip_ansi(rendered)

    breakdown_idx = plain.index("Estimator Budget Breakdown")
    sampling_idx = plain.index("Sampling Budget Breakdown")
    assert sampling_idx < breakdown_idx
    assert "Estimator FLOPs" not in plain
    assert "Over-Budget MLPs" not in plain


# --- Plain-text fallback ----------------------------------------------------


def test_plain_text_run_output_uses_main_style_estimator_breakdown_without_gauge() -> None:
    from whestbench.cli import _render_plain_text_report

    report = _full_report(
        [_mlp(i, flops_used=30_000_000.0) for i in range(3)],
        flop_budget=100_000_000,
    )
    out = _render_plain_text_report(report)
    assert "Estimator Budget Breakdown" in out
    assert "Estimator FLOPs" not in out
    assert "Over-Budget MLPs" not in out
    assert "Total FLOPs [flops_used]" in out


def test_plain_text_run_output_omits_over_budget_panel_when_busted() -> None:
    from whestbench.cli import _render_plain_text_report

    per_mlp = [_mlp(0, flops_used=50.0), _busted(1, 138.0)]
    out = _render_plain_text_report(_full_report(per_mlp, flop_budget=100))
    assert "Estimator Budget Breakdown" in out
    assert "Over-Budget MLPs" not in out
    assert "worst MLP" not in out


def test_plain_text_over_budget_section_omitted_when_clean() -> None:
    from whestbench.cli import _render_plain_text_report

    out = _render_plain_text_report(
        _full_report([_mlp(i, flops_used=30.0) for i in range(3)], flop_budget=100)
    )
    assert "Over-Budget MLPs" not in out


# --- JSON regression --------------------------------------------------------


def test_render_agent_report_unchanged_when_gauge_inputs_present() -> None:
    """Gauge introduces no changes to the JSON payload shape.

    Reporting helpers must not mutate the report dict. This test asserts that
    feeding a report with ``budget_exhausted`` entries through the JSON path
    produces bytes whose parsed form is identical to the input dict.
    """
    from whestbench.reporting import render_agent_report

    per_mlp = [_mlp(0, flops_used=50.0), _busted(1, 138.0)]
    report = _full_report(per_mlp, flop_budget=100)
    import copy
    import json

    snapshot = copy.deepcopy(report)
    rendered = render_agent_report(report)
    parsed = json.loads(rendered)

    assert parsed == snapshot
    assert report == snapshot  # no mutation side-effect
