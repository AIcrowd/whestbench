"""Render score reports for machine and human consumers."""

from __future__ import annotations

import io
import json
from collections.abc import Sequence
from datetime import datetime, timezone
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
        "[dim]Use --agent-mode for JSON output when calling from automated agents or UIs.[/dim]"
    )
    _render_top_row(console, report)
    _render_budget_section(console, report)
    _render_layer_section(console, report)
    _render_profile_section(console, report)
    return buffer.getvalue()


def _render_top_row(console: Console, report: dict[str, Any]) -> None:
    mode = _layout_mode(console.width)
    run_context = _run_context_panel(report)
    readiness = _score_summary_panel(report)
    hardware = _hardware_runtime_panel(report)

    if mode == "wide":
        console.print(Columns([run_context, readiness, hardware], equal=True, expand=True))
        return
    if mode == "medium":
        console.print(Columns([run_context, readiness], equal=True, expand=True))
        console.print(hardware)
        return
    console.print(run_context)
    console.print(readiness)
    console.print(hardware)


def _layout_mode(console_width: int) -> str:
    if console_width >= 120:
        return "wide"
    if console_width >= 90:
        return "medium"
    return "narrow"


def _run_context_panel(report: dict[str, Any]) -> Panel:
    run_meta = report.get("run_meta", {})
    run_config = report.get("run_config", {})

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("field")
    table.add_column("value")

    rows = [
        (
            "Run Started (UTC) [run_started_at_utc]",
            _human_utc(str(run_meta.get("run_started_at_utc", "n/a"))),
        ),
        (
            "Run Finished (UTC) [run_finished_at_utc]",
            _human_utc(str(run_meta.get("run_finished_at_utc", "n/a"))),
        ),
        ("Run Duration(s) [run_duration_s]", _fmt_float(run_meta.get("run_duration_s", 0.0), 6)),
        ("Number of Circuits [n_circuits]", str(run_config.get("n_circuits", "n/a"))),
        ("Samples per Circuit [n_samples]", str(run_config.get("n_samples", "n/a"))),
        ("Circuit Width / Wire Count [width]", str(run_config.get("width", "n/a"))),
        ("Maximum Depth [max_depth]", str(run_config.get("max_depth", "n/a"))),
        ("Layer Count [layer_count]", str(run_config.get("layer_count", "n/a"))),
        ("Budgets [budgets]", str(run_config.get("budgets", []))),
        ("Time Tolerance [time_tolerance]", str(run_config.get("time_tolerance", "n/a"))),
    ]
    for key, value in rows:
        table.add_row(_render_context_label(key), value)

    return Panel(table, title="Run Context", border_style="bright_cyan")


def _hardware_runtime_panel(report: dict[str, Any]) -> Panel:
    host = report.get("run_meta", {}).get("host", {})
    host_meta = host if isinstance(host, dict) else {}

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("field")
    table.add_column("value")
    rows = [
        ("Host Name [host.hostname]", str(host_meta.get("hostname", "n/a"))),
        ("Operating System [host.os]", str(host_meta.get("os", "n/a"))),
        ("OS Release [host.os_release]", str(host_meta.get("os_release", "n/a"))),
        ("Platform [host.platform]", str(host_meta.get("platform", "n/a"))),
        ("Machine Architecture [host.machine]", str(host_meta.get("machine", "n/a"))),
        ("Python Runtime [host.python_version]", str(host_meta.get("python_version", "n/a"))),
    ]
    for label, value in rows:
        table.add_row(_render_context_label(label), value)
    return Panel(table, title="Hardware & Runtime", border_style="bright_blue")


def _score_summary_panel(report: dict[str, Any]) -> Panel:
    results = report.get("results", {})
    final_score = _as_float(results.get("final_score", 0.0))
    by_budget = _budget_rows(report)
    budget_scores = [_as_float(entry.get("score", 0.0)) for entry in by_budget]
    mse_means = [
        fmean(_to_float_list(entry.get("mse_by_layer", [])))
        for entry in by_budget
        if _to_float_list(entry.get("mse_by_layer", []))
    ]
    summary = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row(
        _label_with_code("Final Score (Adjusted MSE Mean)", "final_score", "bold bright_green"),
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

    footnote = Text("Footnote: lower score is better.", style="dim")
    return Panel(Group(summary, footnote), title="Readiness Scorecard", border_style="bright_cyan")


def _render_budget_section(console: Console, report: dict[str, Any]) -> None:
    console.print(Rule("Budget Breakdown", style="bright_cyan"))
    console.print(_budget_lane_panel(report))


def _budget_lane_panel(report: dict[str, Any]) -> Panel:
    by_budget = _budget_rows(report)
    best_score = min((_as_float(entry.get("score", 0.0)) for entry in by_budget), default=0.0)

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    table.add_column("Budget [budget]", justify="right")
    table.add_column("Score [score]", justify="right")
    table.add_column("MSE Mean [avg_mse]", justify="right")
    table.add_column("Time Ratio Mean [avg_time_ratio]", justify="right")
    table.add_column("Effective Time Mean (s) [avg_effective_time_s]", justify="right")
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

    accuracy_plot = _budget_frontier_plot_panel(by_budget)
    runtime_plot = _budget_runtime_plot_panel(by_budget)
    return Panel(
        Columns(
            [Panel(table, title="Budget Table", border_style="bright_black"), Group(accuracy_plot, runtime_plot)],
            equal=False,
            expand=True,
        ),
        title="Budget Intelligence",
        border_style="bright_black",
    )


def _render_layer_section(console: Console, report: dict[str, Any]) -> None:
    console.print(Rule("Layer Diagnostics", style="bright_cyan"))
    console.print(_layer_lane_panel(report))


def _layer_lane_panel(report: dict[str, Any]) -> Panel:
    by_budget = _budget_rows(report)
    mse_series = [_to_float_list(entry.get("mse_by_layer", [])) for entry in by_budget]
    ratio_series = [_to_float_list(entry.get("time_ratio_by_layer", [])) for entry in by_budget]
    adj_series = [_to_float_list(entry.get("adjusted_mse_by_layer", [])) for entry in by_budget]

    avg_mse = _mean_series(mse_series)
    avg_ratio = _mean_series(ratio_series)
    avg_adj = _mean_series(adj_series)

    table = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    table.add_column("metric", style="bold white")
    table.add_column("p05", justify="right")
    table.add_column("min", justify="right")
    table.add_column("p95", justify="right")
    table.add_column("max", justify="right")
    table.add_column("mean", justify="right")

    for human_name, code_name, values in (
        ("MSE by Layer", "mse_by_layer", avg_mse),
        ("Time Ratio by Layer", "time_ratio_by_layer", avg_ratio),
        ("Adjusted MSE by Layer", "adjusted_mse_by_layer", avg_adj),
    ):
        table.add_row(
            _label_with_code(human_name, code_name, "bold white"),
            _fmt_float(_percentile(values, 0.05) if values else 0.0, 6),
            _fmt_float(min(values) if values else 0.0, 6),
            _fmt_float(_percentile(values, 0.95) if values else 0.0, 6),
            _fmt_float(max(values) if values else 0.0, 6),
            _fmt_float(fmean(values) if values else 0.0, 6),
        )

    accuracy_plot = _layer_trend_plot_panel(avg_mse, avg_adj)
    runtime_plot = _layer_runtime_plot_panel(avg_ratio)
    return Panel(
        Columns(
            [Panel(table, title="Layer Metric Table", border_style="bright_black"), Group(accuracy_plot, runtime_plot)],
            equal=False,
            expand=True,
        ),
        title="Layer Intelligence",
        border_style="bright_black",
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
    summary.add_row(_label_with_code("Estimator Calls", "calls", "bold bright_white"), str(len(profile_calls)))
    summary.add_row(
        _label_with_code("Mean Wall Time (s)", "wall_time_s", "bold bright_white"),
        _fmt_float(fmean(wall) if wall else 0.0, 6),
    )
    summary.add_row(
        _label_with_code("Mean CPU Time (s)", "cpu_time_s", "bold bright_white"),
        _fmt_float(fmean(cpu) if cpu else 0.0, 6),
    )
    summary.add_row(
        _label_with_code("Mean RSS (bytes)", "rss_bytes", "bold bright_white"),
        _fmt_float(fmean(rss) if rss else 0.0, 2),
    )
    summary.add_row(
        _label_with_code("Peak RSS (bytes)", "peak_rss_bytes", "bold bright_white"),
        _fmt_float(max(peak) if peak else 0.0, 2),
    )

    dist = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    dist.add_column("metric", style="bold white")
    dist.add_column("p05", justify="right")
    dist.add_column("p95", justify="right")
    dist.add_column("min", justify="right")
    dist.add_column("max", justify="right")
    dist.add_row(
        _label_with_code("Wall Time (s)", "wall_time_s", "bold white"),
        _fmt_float(_percentile(wall, 0.05) if wall else 0.0, 6),
        _fmt_float(_percentile(wall, 0.95) if wall else 0.0, 6),
        _fmt_float(min(wall) if wall else 0.0, 6),
        _fmt_float(max(wall) if wall else 0.0, 6),
    )
    dist.add_row(
        _label_with_code("CPU Time (s)", "cpu_time_s", "bold white"),
        _fmt_float(_percentile(cpu, 0.05) if cpu else 0.0, 6),
        _fmt_float(_percentile(cpu, 0.95) if cpu else 0.0, 6),
        _fmt_float(min(cpu) if cpu else 0.0, 6),
        _fmt_float(max(cpu) if cpu else 0.0, 6),
    )
    dist.add_row(
        _label_with_code("RSS (bytes)", "rss_bytes", "bold white"),
        _fmt_float(_percentile(rss, 0.05) if rss else 0.0, 2),
        _fmt_float(_percentile(rss, 0.95) if rss else 0.0, 2),
        _fmt_float(min(rss) if rss else 0.0, 2),
        _fmt_float(max(rss) if rss else 0.0, 2),
    )
    dist.add_row(
        _label_with_code("Peak RSS (bytes)", "peak_rss_bytes", "bold white"),
        _fmt_float(_percentile(peak, 0.05) if peak else 0.0, 2),
        _fmt_float(_percentile(peak, 0.95) if peak else 0.0, 2),
        _fmt_float(min(peak) if peak else 0.0, 2),
        _fmt_float(max(peak) if peak else 0.0, 2),
    )

    runtime_plot = _profile_runtime_plot_panel(wall, cpu)
    memory_plot = _profile_memory_plot_panel(rss, peak)
    console.print(
        Columns(
            [
                Panel(Group(summary, dist), title="Profile Summary", border_style="bright_black"),
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
    return _make_plot_panel(
        title="Budget Frontier Plot",
        x=budgets,
        series=[
            ("score", scores, "green+"),
            ("avg_mse", mean_mse, "cyan+"),
        ],
        x_label="budget",
        y_label="accuracy metrics",
        x_scale="log",
    )


def _budget_runtime_plot_panel(by_budget: Sequence[dict[str, Any]]) -> Panel:
    budgets = [_as_float(entry.get("budget", 0.0)) for entry in by_budget]
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
        title="Budget Runtime Plot",
        x=budgets,
        series=[
            ("avg_time_ratio_norm", _normalize(mean_ratio), "yellow+"),
            ("avg_effective_time_norm", _normalize(mean_effective), "magenta+"),
        ],
        x_label="budget",
        y_label="normalized runtime [0,1]",
        x_scale="log",
    )


def _layer_trend_plot_panel(avg_mse: Sequence[float], avg_adj: Sequence[float]) -> Panel:
    x = list(range(len(avg_mse)))
    return _make_plot_panel(
        title="Layer Trend Plot",
        x=x,
        series=[
            ("mse", avg_mse, "cyan+"),
            ("adjusted_mse", avg_adj, "green+"),
        ],
        x_label="layer",
        y_label="accuracy metrics",
    )


def _layer_runtime_plot_panel(avg_ratio: Sequence[float]) -> Panel:
    x = list(range(len(avg_ratio)))
    return _make_plot_panel(
        title="Layer Runtime Plot",
        x=x,
        series=[("time_ratio", avg_ratio, "yellow+")],
        x_label="layer",
        y_label="time ratio",
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
    series: list[tuple[str, Sequence[float], str]] = [("rss_bytes", rss, "yellow+")]
    if not _series_nearly_equal(rss, peak):
        series.append(("peak_rss_bytes", peak, "red+"))
    else:
        series[0] = ("rss_bytes (same as peak_rss_bytes)", rss, "yellow+")
    return _make_plot_panel(
        title="Profile Memory Plot",
        x=x,
        series=series,
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
                if callable(scatter_fn):
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
    text.append(f"[{code}]", style="bold dim")
    return text


def _label_with_code(human: str, code: str, style: str) -> Text:
    text = Text(human + " ", style=style)
    text.append(f"[{code}]", style="bold dim")
    return text


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
