"""Render score reports for machine and human consumers."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Any

from rich import box
from rich.columns import Columns
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
    console = Console(
        record=True,
        file=buffer,
        force_terminal=True,
        color_system="truecolor",
        width=120,
    )
    console.print(
        Panel(
            Text("Circuit Estimation Report", style="bold white"),
            expand=False,
            border_style="bright_cyan",
            subtitle="Rich Dashboard",
            subtitle_align="right",
        )
    )
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
    console.print(Rule("Run Context", style="bright_cyan"))
    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("field", style="bold bright_white")
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
    console.print(Rule("Score Summary", style="bright_cyan"))
    cards = [
        Panel(
            Text(_fmt_float(final_score, 8), justify="center", style="bold bright_green"),
            title="final_score",
            border_style="green",
        ),
        Panel(
            Text(str(results.get("score_direction", "lower_is_better")), justify="center"),
            title="score_direction",
            border_style="blue",
        ),
    ]
    if budget_scores:
        cards.append(
            Panel(
                Text(_fmt_float(min(budget_scores), 8), justify="center", style="bold green"),
                title="best_budget_score",
                border_style="green",
            )
        )
        cards.append(
            Panel(
                Text(_fmt_float(max(budget_scores), 8), justify="center", style="bold yellow"),
                title="worst_budget_score",
                border_style="yellow",
            )
        )
    console.print(Columns(cards, equal=True, expand=True))


def _render_budget_breakdown(console: Console, report: dict[str, Any]) -> None:
    by_budget = _budget_rows(report)
    best_score = min((_as_float(entry.get("score", 0.0)) for entry in by_budget), default=0.0)
    table = Table(title="Budget Breakdown", box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    table.add_column("budget", justify="right")
    table.add_column("score", justify="right")
    table.add_column("avg_mse", justify="right")
    table.add_column("avg_time_ratio", justify="right")
    table.add_column("avg_effective_time_s", justify="right")
    table.row_styles = ["none", "dim"]
    for entry in by_budget:
        mse = _to_float_list(entry.get("mse_by_layer", []))
        time_ratio = _to_float_list(entry.get("time_ratio_by_layer", []))
        effective = _to_float_list(entry.get("effective_time_s_by_layer", []))
        row_style = "bold bright_green" if _as_float(entry.get("score", 0.0)) == best_score else ""
        table.add_row(
            str(entry.get("budget", "n/a")),
            _fmt_float(entry.get("score", 0.0), 8),
            _fmt_float(fmean(mse) if mse else 0.0, 8),
            _fmt_float(fmean(time_ratio) if time_ratio else 0.0, 4),
            _fmt_float(fmean(effective) if effective else 0.0, 6),
            style=row_style,
        )
    console.print(table)
    _render_budget_frontier_plot(console, by_budget)


def _render_layer_diagnostics(console: Console, report: dict[str, Any]) -> None:
    console.print(Rule("Layer Diagnostics", style="bright_cyan"))
    by_budget = _budget_rows(report)
    mse_series = [_to_float_list(entry.get("mse_by_layer", [])) for entry in by_budget]
    ratio_series = [_to_float_list(entry.get("time_ratio_by_layer", [])) for entry in by_budget]
    adj_series = [_to_float_list(entry.get("adjusted_mse_by_layer", [])) for entry in by_budget]
    avg_mse = _mean_series(mse_series)
    avg_ratio = _mean_series(ratio_series)
    avg_adj = _mean_series(adj_series)

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    table.add_column("metric", style="bold white")
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

    console.print(Rule("Profiling", style="bright_cyan"))
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
    summary.add_column("field", style="bold bright_white")
    summary.add_column("value")
    summary.add_row("calls", str(len(profile_calls)))
    summary.add_row("wall_time_s", _fmt_float(fmean(wall) if wall else 0.0, 6))
    summary.add_row("cpu_time_s", _fmt_float(fmean(cpu) if cpu else 0.0, 6))
    summary.add_row("rss_bytes", _fmt_float(fmean(rss) if rss else 0.0, 2))
    summary.add_row("peak_rss_bytes", _fmt_float(max(peak) if peak else 0.0, 2))
    console.print(summary)

    trend = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    trend.add_column("metric", style="bold white")
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
    mean_effective = [
        fmean(_to_float_list(entry.get("effective_time_s_by_layer", [])))
        if _to_float_list(entry.get("effective_time_s_by_layer", []))
        else 0.0
        for entry in by_budget
    ]
    _render_plot_panel(
        console=console,
        title="Budget Frontier Plot",
        x=budgets,
        series=[
            ("score", scores, "green+"),
            ("avg_mse", mean_mse, "cyan+"),
            ("avg_time_ratio", mean_ratio, "yellow+"),
        ],
        x_label="budget",
        y_label="raw value",
        x_scale="log",
    )

    _render_plot_panel(
        console=console,
        title="Budget Frontier Plot (Normalized)",
        x=budgets,
        series=[
            ("score_norm", _normalize(scores), "green+"),
            ("avg_time_ratio_norm", _normalize(mean_ratio), "yellow+"),
            ("avg_effective_time_norm", _normalize(mean_effective), "magenta+"),
        ],
        x_label="budget",
        y_label="normalized [0,1]",
        x_scale="log",
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
            ("mse", avg_mse, "cyan+"),
            ("time_ratio", avg_ratio, "yellow+"),
            ("adjusted_mse", avg_adj, "green+"),
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
            ("wall_time_s", wall, "cyan+"),
            ("cpu_time_s", cpu, "magenta+"),
        ],
        x_label="call_index",
        y_label="seconds",
    )
    _render_plot_panel(
        console=console,
        title="Profile Memory Plot",
        x=x,
        series=[
            ("rss_bytes", rss, "yellow+"),
            ("peak_rss_bytes", peak, "red+"),
        ],
        x_label="call_index",
        y_label="bytes",
    )


def _render_plot_panel(
    *,
    console: Console,
    title: str,
    x: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str]],
    x_label: str,
    y_label: str,
    x_scale: str | None = None,
    y_scale: str | None = None,
) -> None:
    chart = _build_plotext_line_chart(
        x=x,
        series=series,
        x_label=x_label,
        y_label=y_label,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    if chart is None:
        fallback = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
        fallback.add_column("series")
        fallback.add_column("trend")
        fallback.add_column("min", justify="right")
        fallback.add_column("max", justify="right")
        for label, values, _color in series:
            fallback.add_row(
                label,
                _sparkline(values, width=42),
                _fmt_float(min(values) if values else 0.0, 6),
                _fmt_float(max(values) if values else 0.0, 6),
            )
        body: str | Text | Table = fallback
    else:
        body = Text.from_ansi(chart)
    console.print(
        Panel(
            body,
            title=title,
            box=box.ROUNDED,
            border_style="bright_black",
            title_align="left",
            expand=False,
        )
    )


def _build_plotext_line_chart(
    *,
    x: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str]],
    x_label: str,
    y_label: str,
    x_scale: str | None = None,
    y_scale: str | None = None,
) -> str | None:
    if _plotext is None or not x:
        return None

    valid_series: list[tuple[str, Sequence[float], str]] = [
        (label, values, color)
        for label, values, color in series
        if len(values) == len(x) and len(values) > 0
    ]
    if not valid_series:
        return None

    try:
        _plotext.clear_data()
        _plotext.clear_figure()
        _plotext.theme("pro")
        width = max(90, min(140, 26 + len(x) * 3))
        _plotext.plotsize(width, 16)
        _plotext.canvas_color("default")
        _plotext.axes_color("default")
        _plotext.ticks_color("white")
        if x_scale is not None:
            _plotext.xscale(x_scale)
        if y_scale is not None:
            _plotext.yscale(y_scale)
        for label, values, color in valid_series:
            _plotext.plot(x, values, label=label, color=color)
        _plotext.xlabel(x_label)
        _plotext.ylabel(y_label)
        _plotext.grid(True, True)
        _plotext.xticks(x)
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


def _normalize(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


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
