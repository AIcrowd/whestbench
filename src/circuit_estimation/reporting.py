"""Render score reports for machine and human consumers."""

from __future__ import annotations

import io
import json
from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Any

from rich import box
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

try:
    import plotext as _plotext  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - optional dependency
    _plotext = None

_SPARK_CHARS = "▁▂▃▄▅▆▇█"


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

    console.print(Columns([_run_context_panel(report), _score_summary_panel(report)], equal=True))
    _render_budget_section(console, report)
    _render_layer_section(console, report)
    _render_profile_section(console, report)
    return buffer.getvalue()


def _run_context_panel(report: dict[str, Any]) -> Panel:
    run_meta = report.get("run_meta", {})
    run_config = report.get("run_config", {})

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("field")
    table.add_column("value")

    rows = [
        ("run_started_at_utc", str(run_meta.get("run_started_at_utc", "n/a"))),
        ("run_finished_at_utc", str(run_meta.get("run_finished_at_utc", "n/a"))),
        ("run_duration_s", _fmt_float(run_meta.get("run_duration_s", 0.0), 6)),
        ("n_circuits", str(run_config.get("n_circuits", "n/a"))),
        ("n_samples", str(run_config.get("n_samples", "n/a"))),
        ("width", str(run_config.get("width", "n/a"))),
        ("layer_count", str(run_config.get("layer_count", "n/a"))),
        ("budgets", str(run_config.get("budgets", []))),
        ("time_tolerance", str(run_config.get("time_tolerance", "n/a"))),
    ]
    for key, value in rows:
        table.add_row(Text(key, style=_context_key_style(key)), value)

    return Panel(table, title="Run Context", border_style="bright_cyan")


def _score_summary_panel(report: dict[str, Any]) -> Panel:
    results = report.get("results", {})
    final_score = _as_float(results.get("final_score", 0.0))
    by_budget = _budget_rows(report)
    budget_scores = [_as_float(entry.get("score", 0.0)) for entry in by_budget]

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

    return Panel(
        Columns(cards, equal=True, expand=True), title="Score Summary", border_style="bright_cyan"
    )


def _render_budget_section(console: Console, report: dict[str, Any]) -> None:
    console.print(Rule("Budget Breakdown", style="bright_cyan"))
    by_budget = _budget_rows(report)
    best_score = min((_as_float(entry.get("score", 0.0)) for entry in by_budget), default=0.0)

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
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

    raw_plot = _budget_frontier_plot_panel(by_budget)
    norm_plot = _budget_frontier_normalized_plot_panel(by_budget)
    console.print(
        Columns(
            [
                Panel(table, title="Budget Table", border_style="bright_black"),
                Group(raw_plot, norm_plot),
            ],
            equal=False,
            expand=True,
        )
    )


def _render_layer_section(console: Console, report: dict[str, Any]) -> None:
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
            _sparkline(values, width=38),
            _fmt_float(min(values) if values else 0.0, 6),
            _fmt_float(max(values) if values else 0.0, 6),
            _fmt_float(fmean(values) if values else 0.0, 6),
        )

    layer_plot = _layer_trend_plot_panel(avg_mse, avg_ratio, avg_adj)
    console.print(
        Columns(
            [
                Panel(table, title="Layer Metric Table", border_style="bright_black"),
                layer_plot,
            ],
            equal=True,
            expand=True,
        )
    )


def _render_profile_section(console: Console, report: dict[str, Any]) -> None:
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

    trend = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    trend.add_column("metric", style="bold white")
    trend.add_column("per_call_trend")
    trend.add_row("wall_time_s", _sparkline(wall, width=42))
    trend.add_row("cpu_time_s", _sparkline(cpu, width=42))
    trend.add_row("rss_bytes", _sparkline(rss, width=42))
    trend.add_row("peak_rss_bytes", _sparkline(peak, width=42))

    runtime_plot = _profile_runtime_plot_panel(wall, cpu)
    memory_plot = _profile_memory_plot_panel(rss, peak)
    console.print(
        Columns(
            [
                Panel(Group(summary, trend), title="Profile Summary", border_style="bright_black"),
                Group(runtime_plot, memory_plot),
            ],
            equal=False,
            expand=True,
        )
    )


def _budget_frontier_plot_panel(by_budget: Sequence[dict[str, Any]]) -> Panel:
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

    return _make_plot_panel(
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


def _budget_frontier_normalized_plot_panel(by_budget: Sequence[dict[str, Any]]) -> Panel:
    budgets = [_as_float(entry.get("budget", 0.0)) for entry in by_budget]
    scores = [_as_float(entry.get("score", 0.0)) for entry in by_budget]
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

    return _make_plot_panel(
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


def _layer_trend_plot_panel(
    avg_mse: Sequence[float], avg_ratio: Sequence[float], avg_adj: Sequence[float]
) -> Panel:
    x = list(range(len(avg_mse)))
    return _make_plot_panel(
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


def _profile_runtime_plot_panel(wall: Sequence[float], cpu: Sequence[float]) -> Panel:
    x = list(range(len(wall)))
    return _make_plot_panel(
        title="Profile Runtime Plot",
        x=x,
        series=[
            ("wall_time_s", wall, "cyan+"),
            ("cpu_time_s", cpu, "magenta+"),
        ],
        x_label="call_index",
        y_label="seconds",
    )


def _profile_memory_plot_panel(rss: Sequence[float], peak: Sequence[float]) -> Panel:
    x = list(range(len(rss)))
    return _make_plot_panel(
        title="Profile Memory Plot",
        x=x,
        series=[
            ("rss_bytes", rss, "yellow+"),
            ("peak_rss_bytes", peak, "red+"),
        ],
        x_label="call_index",
        y_label="bytes",
    )


def _make_plot_panel(
    *,
    title: str,
    x: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str]],
    x_label: str,
    y_label: str,
    x_scale: str | None = None,
    y_scale: str | None = None,
) -> Panel:
    chart = _build_plotext_line_chart(
        x=x,
        series=series,
        x_label=x_label,
        y_label=y_label,
        x_scale=x_scale,
        y_scale=y_scale,
    )
    legend = _legend_table(series)

    if chart is None:
        fallback = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
        fallback.add_column("series")
        fallback.add_column("trend")
        fallback.add_column("min", justify="right")
        fallback.add_column("max", justify="right")
        for label, values, _color in series:
            fallback.add_row(
                label,
                _sparkline(values, width=32),
                _fmt_float(min(values) if values else 0.0, 6),
                _fmt_float(max(values) if values else 0.0, 6),
            )
        body: Table | Text | Group = Group(fallback, legend)
    else:
        body = Group(Text.from_ansi(chart), legend)

    return Panel(
        body,
        title=title,
        box=box.ROUNDED,
        border_style="bright_black",
        title_align="left",
        expand=False,
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

    valid_series = [
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
        width = max(66, min(96, 28 + len(x) * 2))
        _plotext.plotsize(width, 13)
        _plotext.canvas_color("default")
        _plotext.axes_color("default")
        _plotext.ticks_color("white")

        if x_scale is not None:
            _plotext.xscale(x_scale)
        if y_scale is not None:
            _plotext.yscale(y_scale)

        for _label, values, color in valid_series:
            # Keep legend external (Rich table), so we avoid in-plot overlap.
            _plotext.plot(x, values, color=color)

        _plotext.xlabel(x_label)
        _plotext.ylabel(y_label)
        _plotext.grid(True, True)

        if len(x) <= 8:
            ticks = list(x)
        else:
            step = max(1, len(x) // 6)
            ticks = [x[i] for i in range(0, len(x), step)]
            if ticks[-1] != x[-1]:
                ticks.append(x[-1])
        _plotext.xticks(ticks)
        return str(_plotext.build())
    except Exception:  # pragma: no cover - terminal backends vary by environment
        return None
    finally:
        try:
            _plotext.clear_data()
            _plotext.clear_figure()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


def _legend_table(series: Sequence[tuple[str, Sequence[float], str]]) -> Table:
    legend = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    legend.add_column("key")
    legend.add_column("range")
    for label, values, color in series:
        key = Text("■ ", style=_rich_style_for_plot_color(color))
        key.append(label, style="bold")
        legend.add_row(
            key,
            f"[dim]{_fmt_float(min(values) if values else 0.0, 6)} -> {_fmt_float(max(values) if values else 0.0, 6)}[/dim]",
        )
    return legend


def _context_key_style(key: str) -> str:
    if key.startswith("run_"):
        return "bold bright_cyan"
    if key in {"n_circuits", "n_samples", "width", "layer_count"}:
        return "bold bright_magenta"
    if key == "budgets":
        return "bold bright_yellow"
    if key.endswith("_s"):
        return "bold bright_green"
    if "tolerance" in key:
        return "bold bright_red"
    return "bold bright_white"


def _rich_style_for_plot_color(color: str) -> str:
    mapping = {
        "green+": "bright_green",
        "cyan+": "bright_cyan",
        "yellow+": "bright_yellow",
        "magenta+": "bright_magenta",
        "red+": "bright_red",
    }
    return mapping.get(color, "bright_white")


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
        vals = [vals[int(i * step)] for i in range(width)]

    lower = _percentile(vals, 0.05)
    upper = _percentile(vals, 0.95)
    if upper <= lower:
        lower = min(vals)
        upper = max(vals)
    if upper <= lower:
        return "▅" * len(vals)

    scaled = []
    for value in vals:
        clamped = max(lower, min(upper, value))
        pos = (clamped - lower) / (upper - lower)
        idx = int(round(pos * (len(_SPARK_CHARS) - 1)))
        scaled.append(_SPARK_CHARS[idx])
    return "".join(scaled)


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    frac = pos - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


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
