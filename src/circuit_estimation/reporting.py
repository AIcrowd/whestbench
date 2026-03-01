"""Render score reports for machine and human consumers."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    import plotext as _plotext  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - optional dependency
    _plotext = None

_SPARK_CHARS = " .:-=+*#%@"


def render_agent_report(report: dict[str, Any]) -> str:
    """Return stable pretty JSON for machine parsing."""
    return f"{json.dumps(report, indent=2)}\n"


def render_human_report(report: dict[str, Any]) -> str:
    """Render a multi-section Rich report for local CLI exploration."""
    buffer = io.StringIO()
    console = Console(record=True, file=buffer)
    console.print(Panel("Circuit Estimation Report", expand=False, border_style="cyan"))
    console.print(
        Panel(
            "Use --agent-mode for JSON output when calling from automated agents or UIs.",
            title="Agent Tip",
            border_style="green",
        )
    )
    _render_run_context(console, report)
    _render_score_summary(console, report)
    _render_budget_breakdown(console, report)
    _render_layer_diagnostics(console, report)
    _render_profile(console, report)
    return buffer.getvalue()


def _render_run_context(console: Console, report: dict[str, Any]) -> None:
    run_meta = report.get("run_meta", {})
    run_config = report.get("run_config", {})
    table = Table(title="Run Context", box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("field", style="bold")
    table.add_column("value")
    table.add_row("run_started_at_utc", str(run_meta.get("run_started_at_utc", "n/a")))
    table.add_row("run_finished_at_utc", str(run_meta.get("run_finished_at_utc", "n/a")))
    table.add_row("run_duration_s", _fmt_float(run_meta.get("run_duration_s", 0.0), 6))
    table.add_row("n_circuits", str(run_config.get("n_circuits", "n/a")))
    table.add_row("n_samples", str(run_config.get("n_samples", "n/a")))
    table.add_row("width", str(run_config.get("width", "n/a")))
    table.add_row("layer_count", str(run_config.get("layer_count", "n/a")))
    table.add_row("budgets", str(run_config.get("budgets", [])))
    table.add_row("time_tolerance", str(run_config.get("time_tolerance", "n/a")))
    console.print(table)


def _render_score_summary(console: Console, report: dict[str, Any]) -> None:
    results = report.get("results", {})
    final_score = _as_float(results.get("final_score", 0.0))
    by_budget = _budget_rows(report)
    budget_scores = [_as_float(entry.get("score", 0.0)) for entry in by_budget]
    summary = Table(title="Score Summary", box=box.SIMPLE_HEAVY, show_header=False)
    summary.add_column("field", style="bold")
    summary.add_column("value")
    summary.add_row("final_score", _fmt_float(final_score, 8))
    summary.add_row("score_direction", str(results.get("score_direction", "lower_is_better")))
    if budget_scores:
        summary.add_row("best_budget_score", _fmt_float(min(budget_scores), 8))
        summary.add_row("worst_budget_score", _fmt_float(max(budget_scores), 8))
    console.print(summary)


def _render_budget_breakdown(console: Console, report: dict[str, Any]) -> None:
    by_budget = _budget_rows(report)
    table = Table(title="Budget Breakdown", box=box.SIMPLE_HEAVY)
    table.add_column("budget", justify="right")
    table.add_column("score", justify="right")
    table.add_column("avg_mse", justify="right")
    table.add_column("avg_time_ratio", justify="right")
    table.add_column("avg_effective_time_s", justify="right")
    for entry in by_budget:
        mse = _to_float_list(entry.get("mse_by_layer", []))
        time_ratio = _to_float_list(entry.get("time_ratio_by_layer", []))
        effective = _to_float_list(entry.get("effective_time_s_by_layer", []))
        table.add_row(
            str(entry.get("budget", "n/a")),
            _fmt_float(entry.get("score", 0.0), 8),
            _fmt_float(fmean(mse) if mse else 0.0, 8),
            _fmt_float(fmean(time_ratio) if time_ratio else 0.0, 4),
            _fmt_float(fmean(effective) if effective else 0.0, 6),
        )
    console.print(table)
    _render_budget_frontier_plot(console, by_budget)


def _render_layer_diagnostics(console: Console, report: dict[str, Any]) -> None:
    console.print(Rule("Layer Diagnostics"))
    by_budget = _budget_rows(report)
    mse_series = [_to_float_list(entry.get("mse_by_layer", [])) for entry in by_budget]
    ratio_series = [_to_float_list(entry.get("time_ratio_by_layer", [])) for entry in by_budget]
    adj_series = [_to_float_list(entry.get("adjusted_mse_by_layer", [])) for entry in by_budget]
    avg_mse = _mean_series(mse_series)
    avg_ratio = _mean_series(ratio_series)
    avg_adj = _mean_series(adj_series)

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("metric", style="bold")
    table.add_column("sparkline")
    table.add_column("min", justify="right")
    table.add_column("max", justify="right")
    table.add_column("mean", justify="right")

    for name, values in (
        ("mse_by_layer", avg_mse),
        ("time_ratio_by_layer", avg_ratio),
        ("adjusted_mse_by_layer", avg_adj),
    ):
        table.add_row(
            name,
            _sparkline(values),
            _fmt_float(min(values) if values else 0.0, 6),
            _fmt_float(max(values) if values else 0.0, 6),
            _fmt_float(fmean(values) if values else 0.0, 6),
        )
    console.print(table)
    _render_layer_trend_plot(console, avg_mse, avg_ratio, avg_adj)


def _render_profile(console: Console, report: dict[str, Any]) -> None:
    profile_calls = report.get("profile_calls")
    if not isinstance(profile_calls, list) or not profile_calls:
        return

    console.print(Rule("Profiling"))
    wall = [
        _as_float(entry.get("wall_time_s", 0.0))
        for entry in profile_calls
        if isinstance(entry, dict)
    ]
    cpu = [
        _as_float(entry.get("cpu_time_s", 0.0))
        for entry in profile_calls
        if isinstance(entry, dict)
    ]
    rss = [
        _as_float(entry.get("rss_bytes", 0.0)) for entry in profile_calls if isinstance(entry, dict)
    ]
    peak = [
        _as_float(entry.get("peak_rss_bytes", 0.0))
        for entry in profile_calls
        if isinstance(entry, dict)
    ]

    summary = Table(box=box.SIMPLE_HEAVY, show_header=False)
    summary.add_column("field", style="bold")
    summary.add_column("value")
    summary.add_row("calls", str(len(profile_calls)))
    summary.add_row("wall_time_s", _fmt_float(fmean(wall) if wall else 0.0, 6))
    summary.add_row("cpu_time_s", _fmt_float(fmean(cpu) if cpu else 0.0, 6))
    summary.add_row("rss_bytes", _fmt_float(fmean(rss) if rss else 0.0, 2))
    summary.add_row("peak_rss_bytes", _fmt_float(max(peak) if peak else 0.0, 2))
    console.print(summary)

    trend = Table(box=box.SIMPLE_HEAVY)
    trend.add_column("metric", style="bold")
    trend.add_column("per_call_trend")
    trend.add_row("wall_time_s", _sparkline(wall))
    trend.add_row("cpu_time_s", _sparkline(cpu))
    trend.add_row("rss_bytes", _sparkline(rss))
    trend.add_row("peak_rss_bytes", _sparkline(peak))
    console.print(trend)
    _render_profile_runtime_plot(console, wall, cpu, rss, peak)


def _render_budget_frontier_plot(console: Console, by_budget: Sequence[dict[str, Any]]) -> None:
    budgets = [_as_float(entry.get("budget", 0.0)) for entry in by_budget]
    scores = [_as_float(entry.get("score", 0.0)) for entry in by_budget]
    mean_mse = [
        fmean(_to_float_list(entry.get("mse_by_layer", [])))
        if _to_float_list(entry.get("mse_by_layer", []))
        else 0.0
        for entry in by_budget
    ]
    mean_ratio = [
        fmean(_to_float_list(entry.get("time_ratio_by_layer", [])))
        if _to_float_list(entry.get("time_ratio_by_layer", []))
        else 0.0
        for entry in by_budget
    ]
    _render_plot_panel(
        console=console,
        title="Budget Frontier Plot",
        x=budgets,
        series=[
            ("score", scores),
            ("avg_mse", mean_mse),
            ("avg_time_ratio", mean_ratio),
        ],
        x_label="budget",
        y_label="value",
    )


def _render_layer_trend_plot(
    console: Console,
    avg_mse: Sequence[float],
    avg_ratio: Sequence[float],
    avg_adj: Sequence[float],
) -> None:
    x = list(range(len(avg_mse)))
    _render_plot_panel(
        console=console,
        title="Layer Trend Plot",
        x=x,
        series=[
            ("mse", avg_mse),
            ("time_ratio", avg_ratio),
            ("adjusted_mse", avg_adj),
        ],
        x_label="layer",
        y_label="value",
    )


def _render_profile_runtime_plot(
    console: Console,
    wall: Sequence[float],
    cpu: Sequence[float],
    rss: Sequence[float],
    peak: Sequence[float],
) -> None:
    x = list(range(len(wall)))
    _render_plot_panel(
        console=console,
        title="Profile Runtime Plot",
        x=x,
        series=[
            ("wall_time_s", wall),
            ("cpu_time_s", cpu),
            ("rss_bytes", rss),
            ("peak_rss_bytes", peak),
        ],
        x_label="call_index",
        y_label="metric_value",
    )


def _render_plot_panel(
    *,
    console: Console,
    title: str,
    x: Sequence[float],
    series: Sequence[tuple[str, Sequence[float]]],
    x_label: str,
    y_label: str,
) -> None:
    chart = _build_plotext_line_chart(x=x, series=series, x_label=x_label, y_label=y_label)
    if chart is None:
        body: str | Text = (
            "Plot rendering unavailable (missing optional dependency `plotext` or unsupported terminal)."
        )
    else:
        body = Text.from_ansi(chart)
    console.print(Panel(body, title=title, box=box.ROUNDED))


def _build_plotext_line_chart(
    *,
    x: Sequence[float],
    series: Sequence[tuple[str, Sequence[float]]],
    x_label: str,
    y_label: str,
) -> str | None:
    if _plotext is None or not x:
        return None

    valid_series: list[tuple[str, Sequence[float]]] = [
        (label, values) for label, values in series if len(values) == len(x) and len(values) > 0
    ]
    if not valid_series:
        return None

    try:
        _plotext.clear_data()
        _plotext.clear_figure()
        _plotext.theme("pro")
        width = max(72, min(128, 24 + len(x) * 2))
        _plotext.plotsize(width, 14)
        for label, values in valid_series:
            _plotext.plot(x, values, label=label)
        _plotext.xlabel(x_label)
        _plotext.ylabel(y_label)
        _plotext.grid(True, True)
        legend_fn = getattr(_plotext, "legend", None)
        if callable(legend_fn):
            legend_fn(True)
        return str(_plotext.build())
    except Exception:  # pragma: no cover - terminal backends vary by environment
        return None
    finally:
        try:
            _plotext.clear_data()
            _plotext.clear_figure()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


def _budget_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    results = report.get("results")
    if not isinstance(results, dict):
        return []
    rows = results.get("by_budget_raw")
    if not isinstance(rows, list):
        return []
    return [entry for entry in rows if isinstance(entry, dict)]


def _mean_series(series_list: Sequence[Sequence[float]]) -> list[float]:
    if not series_list:
        return []
    length = min(len(series) for series in series_list)
    if length == 0:
        return []
    return [fmean(series[i] for series in series_list) for i in range(length)]


def _sparkline(values: Iterable[float], width: int = 48) -> str:
    vals = list(values)
    if not vals:
        return "(no data)"
    if len(vals) > width:
        step = len(vals) / width
        sampled: list[float] = []
        cursor = 0.0
        while int(cursor) < len(vals) and len(sampled) < width:
            sampled.append(vals[int(cursor)])
            cursor += step
        vals = sampled

    low = min(vals)
    high = max(vals)
    if high <= low:
        return _SPARK_CHARS[-1] * len(vals)

    scale = (len(_SPARK_CHARS) - 1) / (high - low)
    return "".join(_SPARK_CHARS[int((v - low) * scale)] for v in vals)


def _to_float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    return [_as_float(item) for item in value]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_float(value: object, decimals: int) -> str:
    return f"{_as_float(value):.{decimals}f}"
