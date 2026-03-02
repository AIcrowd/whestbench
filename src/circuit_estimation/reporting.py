"""Render score reports for machine and human consumers."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
from collections.abc import Sequence
from datetime import datetime, timezone
from statistics import fmean
from typing import Any

from rich import box
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import plotext as _plotext  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - optional dependency
    _plotext = None


def render_agent_report(report: dict[str, Any]) -> str:
    """Return stable pretty JSON for machine parsing."""
    return f"{json.dumps(report, indent=2)}\n"


def render_human_report(report: dict[str, Any], *, show_diagnostic_plots: bool = False) -> str:
    """Render a multi-section Rich report for local CLI exploration."""
    buffer = io.StringIO()
    width = _dashboard_width()
    console = Console(
        record=True,
        file=buffer,
        force_terminal=True,
        color_system="truecolor",
        _environ=_rich_console_environ(width),
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
        "[dim]Use --json for JSON output when calling from automated agents or UIs.[/dim]"
    )
    console.print("[dim]Use --show-diagnostic-plots to include diagnostic plot panes.[/dim]")
    console.print(
        "[dim]Runtime scoring uses budget-by-depth checks at each streamed predict() row.[/dim]"
    )
    _render_top_row(console, report)
    _render_budget_section(console, report, show_diagnostic_plots=show_diagnostic_plots)
    _render_layer_section(console, report, show_diagnostic_plots=show_diagnostic_plots)
    _render_profile_section(console, report, show_diagnostic_plots=show_diagnostic_plots)
    return buffer.getvalue()


def _render_top_row(console: Console, report: dict[str, Any]) -> None:
    mode = _layout_mode(console.width)
    run_context = _run_context_panel(report)
    readiness = _score_summary_panel(report)
    hardware = _hardware_runtime_panel(report)

    if mode == "three_col":
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(run_context, readiness, hardware)
        console.print(grid)
        return
    if mode == "two_col":
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(run_context, readiness)
        console.print(grid)
        console.print(hardware)
        return
    console.print(run_context)
    console.print(readiness)
    console.print(hardware)


def _dashboard_width() -> int:
    columns = shutil.get_terminal_size((120, 40)).columns
    return max(80, columns)


def _rich_console_environ(width: int) -> dict[str, str]:
    environ = dict(os.environ)
    environ["COLUMNS"] = str(width)
    environ.setdefault("LINES", "40")
    return environ


def _layout_mode(console_width: int) -> str:
    if console_width >= 180:
        return "three_col"
    if console_width >= 110:
        return "two_col"
    return "narrow"


def _run_context_panel(report: dict[str, Any]) -> Panel:
    run_meta = report.get("run_meta", {})
    run_config = report.get("run_config", {})

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("field")
    table.add_column("value")

    rows = [
        (
            "Started [run_started_at_utc]",
            _human_utc(str(run_meta.get("run_started_at_utc", "n/a"))),
        ),
        (
            "Finished [run_finished_at_utc]",
            _human_utc(str(run_meta.get("run_finished_at_utc", "n/a"))),
        ),
        ("Duration(s) [run_duration_s]", _fmt_float(run_meta.get("run_duration_s", 0.0), 6)),
        ("Circuits [n_circuits]", str(run_config.get("n_circuits", "n/a"))),
        ("Samples/Circuit [n_samples]", str(run_config.get("n_samples", "n/a"))),
        ("Width/Wires [width]", str(run_config.get("width", "n/a"))),
        ("Max Depth [max_depth]", str(run_config.get("max_depth", "n/a"))),
        ("Layers [layer_count]", str(run_config.get("layer_count", "n/a"))),
        ("Budgets [budgets]", str(run_config.get("budgets", []))),
        ("Tolerance [time_tolerance]", str(run_config.get("time_tolerance", "n/a"))),
    ]
    for key, value in rows:
        table.add_row(_render_context_label(key), value)

    return Panel(Align.center(table), title="Run Context", border_style="bright_cyan")


def _hardware_runtime_panel(report: dict[str, Any]) -> Panel:
    host = report.get("run_meta", {}).get("host", {})
    host_meta = host if isinstance(host, dict) else {}

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("field")
    table.add_column("value")
    rows = [
        ("Host [host.hostname]", str(host_meta.get("hostname", "n/a"))),
        ("OS [host.os]", str(host_meta.get("os", "n/a"))),
        ("Release [host.os_release]", str(host_meta.get("os_release", "n/a"))),
        ("Platform [host.platform]", str(host_meta.get("platform", "n/a"))),
        ("Arch [host.machine]", str(host_meta.get("machine", "n/a"))),
        ("Python [host.python_version]", str(host_meta.get("python_version", "n/a"))),
    ]
    for label, value in rows:
        table.add_row(_render_context_label(label), value)
    return Panel(Align.center(table), title="Hardware & Runtime", border_style="bright_blue")


def _score_summary_panel(report: dict[str, Any]) -> Panel:
    results = report.get("results", {})
    final_score = _as_float(results.get("final_score", 0.0))
    by_budget = _budget_rows(report)
    budget_scores = [_as_float(entry.get("adjusted_mse", 0.0)) for entry in by_budget]
    mse_means = [_as_float(entry.get("mse_mean", 0.0)) for entry in by_budget]
    summary = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row(
        _label_with_code("Final Score", "final_score", "bold bright_green"),
        f"[bold bright_green]✓ {_fmt_float(final_score, 8)}[/]",
    )
    if mse_means:
        summary.add_row(
            _label_with_code("MSE Mean", "mse_mean", "bold bright_cyan"),
            f"[cyan]{_fmt_float(fmean(mse_means), 8)}[/]",
        )
    if budget_scores:
        summary.add_row(
            _label_with_code("Best Budget Score", "best_budget_score", "bold green"),
            f"[green]{_fmt_float(min(budget_scores), 8)}[/]",
        )
        summary.add_row(
            _label_with_code("Worst Budget Score", "worst_budget_score", "bold yellow"),
            f"[yellow]{_fmt_float(max(budget_scores), 8)}[/]",
        )

    # Keep top-row panes visually balanced with the 10-row Run Context panel.
    # Scorecard has a header + separator, so fewer data rows are needed to align heights.
    while len(summary.rows) < 8:
        summary.add_row("", "")
    return Panel(
        Align.center(summary),
        title="Readiness Scorecard",
        subtitle="lower score is better; final score is adjusted MSE mean",
        subtitle_align="left",
        border_style="bright_cyan",
    )


def _render_budget_section(
    console: Console, report: dict[str, Any], *, show_diagnostic_plots: bool
) -> None:
    console.print(_budget_lane_panel(report, show_diagnostic_plots=show_diagnostic_plots))


def _budget_lane_panel(report: dict[str, Any], *, show_diagnostic_plots: bool = False) -> Panel:
    by_budget = _budget_rows(report)
    best_score = min(
        (_as_float(entry.get("adjusted_mse", 0.0)) for entry in by_budget), default=0.0
    )

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    table.add_column("Budget [budget]", justify="right")
    table.add_column("Score [adjusted_mse]", justify="right")
    table.add_column("MSE Mean [mse_mean]", justify="right")
    table.add_column("Call Time Ratio Mean [call_time_ratio_mean]", justify="right")
    table.add_column("Effective Call Time Mean (s) [call_effective_time_s_mean]", justify="right")
    for entry in by_budget:
        score = _as_float(entry.get("adjusted_mse", 0.0))
        row_style = "bold bright_green" if score == best_score else ""
        table.add_row(
            str(entry.get("budget", "n/a")),
            _fmt_float(score, 8),
            _fmt_float(entry.get("mse_mean", 0.0), 8),
            _fmt_float(entry.get("call_time_ratio_mean", 0.0), 4),
            _fmt_float(entry.get("call_effective_time_s_mean", 0.0), 6),
            style=row_style,
        )

    body: list[Any] = [Align.center(table)]
    if show_diagnostic_plots:
        accuracy_plot = _budget_frontier_plot_panel(by_budget)
        runtime_plot = _budget_runtime_plot_panel(by_budget)
        body.append(Columns([accuracy_plot, runtime_plot], equal=True, expand=True))
    return Panel(Group(*body), title="Budget", border_style="bright_cyan")


def _render_layer_section(
    console: Console, report: dict[str, Any], *, show_diagnostic_plots: bool
) -> None:
    console.print(_layer_lane_panel(report, show_diagnostic_plots=show_diagnostic_plots))


def _layer_lane_panel(report: dict[str, Any], *, show_diagnostic_plots: bool = False) -> Panel:
    by_budget = _budget_rows(report)
    mse_series = [_to_float_list(entry.get("mse_by_layer", [])) for entry in by_budget]
    avg_mse = _mean_series(mse_series)

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    table.add_column("metric", style="bold white")
    table.add_column("p05", justify="right")
    table.add_column("min", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("max", justify="right")
    table.add_column("mean", justify="right")

    table.add_row(
        _label_with_code("MSE by Layer", "mse_by_layer", "bold white"),
        _fmt_float(_percentile(avg_mse, 0.05) if avg_mse else 0.0, 6),
        _fmt_float(min(avg_mse) if avg_mse else 0.0, 6),
        _fmt_float(_percentile(avg_mse, 0.95) if avg_mse else 0.0, 6),
        _fmt_float(max(avg_mse) if avg_mse else 0.0, 6),
        _fmt_float(fmean(avg_mse) if avg_mse else 0.0, 6),
    )

    body: list[Any] = [Align.center(table)]
    if show_diagnostic_plots:
        body.append(_layer_trend_plot_panel(avg_mse))
    return Panel(Group(*body), title="Layer Diagnostics", border_style="bright_magenta")


def _render_profile_section(
    console: Console, report: dict[str, Any], *, show_diagnostic_plots: bool
) -> None:
    profile_calls = report.get("profile_calls")
    if not isinstance(profile_calls, list) or not profile_calls:
        return

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
    summary.add_row(
        _label_with_code("Estimator Calls", "calls", "bold bright_magenta"),
        f"[bright_magenta]{len(profile_calls)}[/]",
    )
    summary.add_row(
        _label_with_code("Mean Wall Time (s)", "wall_time_s", "bold bright_cyan"),
        f"[bright_cyan]{_fmt_float(fmean(wall) if wall else 0.0, 6)}[/]",
    )
    summary.add_row(
        _label_with_code("Mean CPU Time (s)", "cpu_time_s", "bold bright_green"),
        f"[bright_green]{_fmt_float(fmean(cpu) if cpu else 0.0, 6)}[/]",
    )
    summary.add_row(
        _label_with_code("Mean RSS (bytes)", "rss_bytes", "bold bright_blue"),
        f"[bright_blue]{_fmt_float(fmean(rss) if rss else 0.0, 2)}[/]",
    )
    summary.add_row(
        _label_with_code("Peak RSS (bytes)", "peak_rss_bytes", "bold bright_magenta"),
        f"[bright_magenta]{_fmt_float(max(peak) if peak else 0.0, 2)}[/]",
    )

    dist = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    dist.add_column("metric", style="bold white")
    dist.add_column("p05", justify="right")
    dist.add_column("p95", justify="right")
    dist.add_column("min", justify="right")
    dist.add_column("max", justify="right")
    dist.add_row(
        _label_with_code("Wall Time (s)", "wall_time_s", "bold bright_cyan"),
        _fmt_float(_percentile(wall, 0.05) if wall else 0.0, 6),
        _fmt_float(_percentile(wall, 0.95) if wall else 0.0, 6),
        _fmt_float(min(wall) if wall else 0.0, 6),
        _fmt_float(max(wall) if wall else 0.0, 6),
    )
    dist.add_row(
        _label_with_code("CPU Time (s)", "cpu_time_s", "bold bright_green"),
        _fmt_float(_percentile(cpu, 0.05) if cpu else 0.0, 6),
        _fmt_float(_percentile(cpu, 0.95) if cpu else 0.0, 6),
        _fmt_float(min(cpu) if cpu else 0.0, 6),
        _fmt_float(max(cpu) if cpu else 0.0, 6),
    )
    dist.add_row(
        _label_with_code("RSS (bytes)", "rss_bytes", "bold bright_blue"),
        _fmt_float(_percentile(rss, 0.05) if rss else 0.0, 2),
        _fmt_float(_percentile(rss, 0.95) if rss else 0.0, 2),
        _fmt_float(min(rss) if rss else 0.0, 2),
        _fmt_float(max(rss) if rss else 0.0, 2),
    )
    dist.add_row(
        _label_with_code("Peak RSS (bytes)", "peak_rss_bytes", "bold bright_magenta"),
        _fmt_float(_percentile(peak, 0.05) if peak else 0.0, 2),
        _fmt_float(_percentile(peak, 0.95) if peak else 0.0, 2),
        _fmt_float(min(peak) if peak else 0.0, 2),
        _fmt_float(max(peak) if peak else 0.0, 2),
    )

    profile_tables = Columns(
        [
            Panel(Align.center(summary), title="Summary", border_style="bright_blue"),
            Panel(Align.center(dist), title="Distribution", border_style="bright_blue"),
        ],
        align="center",
        equal=True,
        expand=False,
    )
    console.print(Panel(Align.center(profile_tables), title="Profile", border_style="bright_blue"))
    if show_diagnostic_plots:
        runtime_plot = _profile_runtime_plot_panel(wall, cpu)
        memory_plot = _profile_memory_plot_panel(rss, peak)
        console.print(Columns([runtime_plot, memory_plot], equal=True, expand=True))


def _budget_frontier_plot_panel(by_budget: Sequence[dict[str, Any]]) -> Panel:
    budgets = [_as_float(entry.get("budget", 0.0)) for entry in by_budget]
    scores = [_as_float(entry.get("adjusted_mse", 0.0)) for entry in by_budget]
    mean_mse = [_as_float(entry.get("mse_mean", 0.0)) for entry in by_budget]
    return _make_plot_panel(
        title="Budget Frontier Plot",
        x=budgets,
        series=[
            ("adjusted_mse", scores, "green+"),
            ("mse_mean", mean_mse, "cyan+"),
        ],
        x_label="budget",
        y_label="accuracy metrics",
        x_scale="log",
        sparse_style="line",
    )


def _budget_runtime_plot_panel(by_budget: Sequence[dict[str, Any]]) -> Panel:
    budgets = [_as_float(entry.get("budget", 0.0)) for entry in by_budget]
    mean_ratio = [_as_float(entry.get("call_time_ratio_mean", 0.0)) for entry in by_budget]
    mean_effective = [
        _as_float(entry.get("call_effective_time_s_mean", 0.0)) for entry in by_budget
    ]

    return _make_plot_panel(
        title="Budget Runtime Plot",
        x=budgets,
        series=[
            ("call_time_ratio_mean_norm", _normalize(mean_ratio), "yellow+"),
            ("call_effective_time_s_mean_norm", _normalize(mean_effective), "magenta+"),
        ],
        x_label="budget",
        y_label="normalized runtime [0,1]",
        x_scale="log",
        sparse_style="line",
    )


def _layer_trend_plot_panel(avg_mse: Sequence[float]) -> Panel:
    x = list(range(len(avg_mse)))
    return _make_plot_panel(
        title="Layer Trend Plot",
        x=x,
        series=[("mse_by_layer", avg_mse, "cyan+")],
        x_label="layer",
        y_label="mse",
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
    rss_mb = [value / (1024.0 * 1024.0) for value in rss]
    peak_mb = [value / (1024.0 * 1024.0) for value in peak]
    series: list[tuple[str, Sequence[float], str]] = [("rss_mb", rss_mb, "cyan+")]
    if not _series_nearly_equal(rss, peak):
        series.append(("peak_rss_mb", peak_mb, "magenta+"))
    else:
        series[0] = ("rss_mb (same as peak_rss_mb)", rss_mb, "cyan+")
    return _make_plot_panel(
        title="Profile Memory Plot",
        x=x,
        series=series,
        x_label="call_index",
        y_label="Memory Usage (MB)",
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
    sparse_style: str = "scatter",
) -> Panel:
    chart = _build_plotext_line_chart(
        x=x,
        series=series,
        x_label=x_label,
        y_label=y_label,
        x_scale=x_scale,
        y_scale=y_scale,
        sparse_style=sparse_style,
    )
    legend = Align.center(_legend_table(series))

    if chart is None:
        fallback = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
        fallback.add_column("series")
        fallback.add_column("p05", justify="right")
        fallback.add_column("min", justify="right")
        fallback.add_column("mean", justify="right")
        fallback.add_column("max", justify="right")
        fallback.add_column("p95", justify="right")
        for label, values, _color in series:
            fallback.add_row(
                label,
                _fmt_float(_percentile(values, 0.05) if values else 0.0, 6),
                _fmt_float(min(values) if values else 0.0, 6),
                _fmt_float(fmean(values) if values else 0.0, 6),
                _fmt_float(max(values) if values else 0.0, 6),
                _fmt_float(_percentile(values, 0.95) if values else 0.0, 6),
            )
        body: Table | Text | Group = Group(Align.center(fallback), legend)
    else:
        body = Group(Text.from_ansi(chart), legend)

    return Panel(
        body,
        title=title,
        box=box.ROUNDED,
        border_style="bright_white",
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
    sparse_style: str = "scatter",
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
        _plotext.theme("clear")
        width = max(66, min(92, 26 + len(x)))
        _plotext.plotsize(width, 11)
        _plotext.canvas_color("black")
        _plotext.axes_color("white")
        _plotext.ticks_color("white")

        if x_scale is not None:
            _plotext.xscale(x_scale)
        if y_scale is not None:
            _plotext.yscale(y_scale)

        all_values = [value for _label, values, _color in valid_series for value in values]
        if all_values:
            low = min(all_values)
            high = max(all_values)
            if high <= low:
                pad = max(1e-9, abs(low) * 0.05 + 1e-9)
                _plotext.ylim(low - pad, high + pad)
            else:
                pad = (high - low) * 0.08
                _plotext.ylim(low - pad, high + pad)

        scatter_fn = getattr(_plotext, "scatter", None)
        for _label, values, color in valid_series:
            # Keep legend external (Rich table), so we avoid in-plot overlap.
            if len(x) <= 12:
                if sparse_style == "line":
                    # Thin connected line plus explicit points for sparse series.
                    _plotext.plot(x, values, color=color, marker="hd")
                    if callable(scatter_fn):
                        scatter_fn(x, values, color=color, marker="●")
                elif callable(scatter_fn):
                    scatter_fn(x, values, color=color, marker="●")
                else:
                    _plotext.plot(x, values, color=color, marker="●")
            else:
                _plotext.plot(x, values, color=color, marker="hd")

        _plotext.xlabel(x_label)
        _plotext.ylabel(y_label)
        _plotext.grid(True, False)

        if len(x) <= 8:
            ticks = list(x)
        else:
            step = max(1, len(x) // 6)
            ticks = [x[i] for i in range(0, len(x), step)]
            if ticks[-1] != x[-1]:
                ticks.append(x[-1])
        _plotext.xticks(ticks)
        return _sanitize_plotext_ansi(str(_plotext.build()))
    except Exception:  # pragma: no cover - terminal backends vary by environment
        return None
    finally:
        try:
            _plotext.clear_data()
            _plotext.clear_figure()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


def _sanitize_plotext_ansi(chart: str) -> str:
    """Remove plotext background ANSI escapes to prevent light chart panels."""
    # Remove background color and reset-background escapes while preserving foreground styles.
    without_bg = re.sub(r"\x1b\[48;[0-9;]*m", "", chart)
    return re.sub(r"\x1b\[49m", "", without_bg)


def _legend_table(series: Sequence[tuple[str, Sequence[float], str]]) -> Table:
    legend = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    legend.add_column("key")
    legend.add_column("range")
    for label, values, color in series:
        key = Text("■ ", style=_rich_style_for_plot_color(color))
        key.append(label, style="bold")
        legend.add_row(
            key,
            f"[white]{_fmt_float(min(values) if values else 0.0, 6)} -> {_fmt_float(max(values) if values else 0.0, 6)}[/white]",
        )
    return legend


def _human_utc(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return value
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%b %d, %Y %H:%M:%S UTC")


def _context_key_style(key: str) -> str:
    if "[" in key and "]" in key:
        key = key[key.find("[") + 1 : key.rfind("]")]
    if key.startswith("run_"):
        return "bold bright_cyan"
    if key.startswith("host."):
        return "bold bright_blue"
    if key in {"n_circuits", "n_samples", "width", "layer_count"}:
        return "bold bright_magenta"
    if key == "budgets":
        return "bold bright_yellow"
    if key.endswith("_s"):
        return "bold bright_green"
    if "tolerance" in key:
        return "bold bright_red"
    return "bold bright_white"


def _render_context_label(label: str) -> Text:
    if "[" not in label or "]" not in label:
        return Text(label, style=_context_key_style(label))

    start = label.find("[")
    end = label.rfind("]")
    human = label[:start].rstrip()
    code = label[start + 1 : end]
    text = Text(human + " ", style=_context_key_style(code))
    text.append(f"[{code}]", style="bold bright_white")
    return text


def _label_with_code(human: str, code: str, style: str) -> Text:
    text = Text(human + " ", style=style)
    text.append(f"[{code}]", style="bold bright_white")
    return text


def _rich_style_for_plot_color(color: str) -> str:
    mapping = {
        "green+": "bright_green",
        "cyan+": "bright_cyan",
        "yellow+": "bright_yellow",
        "magenta+": "bright_magenta",
        "red+": "bright_red",
        "blue+": "bright_blue",
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


def _series_nearly_equal(
    lhs: Sequence[float], rhs: Sequence[float], *, tolerance: float = 1e-12
) -> bool:
    if len(lhs) != len(rhs):
        return False
    return all(abs(left - right) <= tolerance for left, right in zip(lhs, rhs, strict=True))


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
