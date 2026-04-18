"""Render score reports for machine and human consumers."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import fmean
from typing import Any, Optional

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


def render_agent_report(report: "dict[str, Any]") -> str:
    """Return stable pretty JSON for machine parsing."""
    return f"{json.dumps(report, indent=2)}\n"


def render_human_header() -> str:
    """Render only the title/header block for append-first human runs."""
    buffer = io.StringIO()
    console = _new_console(buffer)
    console.print(
        Panel(
            Align.center(Text("WhestBench Report", style="bold white")),
            expand=True,
            border_style="bright_cyan",
        )
    )
    return buffer.getvalue()


def render_human_context_panels(report: "dict[str, Any]") -> str:
    """Render context panels shown before scoring starts."""
    buffer = io.StringIO()
    console = _new_console(buffer)
    console.print(build_human_context_renderable(report, console_width=console.width))
    return buffer.getvalue()


def render_human_results(
    report: "dict[str, Any]",
    *,
    show_diagnostic_plots: bool = False,
    debug: bool = False,
) -> str:
    """Render post-run sections for append-only human flows."""
    buffer = io.StringIO()
    console = _new_console(buffer)
    _render_score_row(console, report)
    _render_errors_section(console, report, debug=debug)
    _render_breakdown_sections(console, report)
    _render_profile_section(console, report, show_diagnostic_plots=show_diagnostic_plots)
    return buffer.getvalue()


def render_human_report(
    report: "dict[str, Any]",
    *,
    show_diagnostic_plots: bool = False,
    debug: bool = False,
) -> str:
    """Render a multi-section Rich report for local CLI exploration."""
    buffer = io.StringIO()
    console = _new_console(buffer)

    console.print(
        Panel(
            Align.center(Text("WhestBench Report", style="bold white")),
            expand=True,
            border_style="bright_cyan",
        )
    )
    console.print(
        "[dim]Use --json for JSON output when calling from automated agents or UIs.[/dim]"
    )
    console.print("[dim]Use --show-diagnostic-plots to include diagnostic plot panes.[/dim]")
    _render_top_row(console, report)
    _render_score_row(console, report)
    _render_errors_section(console, report, debug=debug)
    _render_breakdown_sections(console, report)
    _render_profile_section(console, report, show_diagnostic_plots=show_diagnostic_plots)
    return buffer.getvalue()


def _new_console(buffer: io.StringIO) -> Console:
    width = _dashboard_width()
    return Console(
        record=True,
        file=buffer,
        force_terminal=True,
        color_system="truecolor",
        _environ=_rich_console_environ(width),
    )


def render_smoke_test_next_steps() -> str:
    """Render onboarding next-steps panel for ``whest smoke-test``."""
    buffer = io.StringIO()
    width = _dashboard_width()
    console = Console(
        record=True,
        file=buffer,
        force_terminal=True,
        color_system="truecolor",
        _environ=_rich_console_environ(width),
    )

    purpose_lines = _smoke_next_step_lines()
    commands = _smoke_next_step_commands()
    body_items: "list[Text]" = [
        Text("We are all set! Welcome onboard", style="bold bright_green"),
        Text("Run these steps:", style="bold bright_white"),
        Text(),
    ]
    for purpose_line, command in zip(purpose_lines, commands):
        body_items.append(purpose_line)
        body_items.append(Text(command, style="white"))
        body_items.append(Text())

    body_items.append(Text("Optional: run bundled example estimators:", style="bold bright_cyan"))
    for command in _smoke_optional_example_commands():
        body_items.append(Text(command, style="white"))
    body_items.append(Text())

    body_items.append(
        Text(
            "Tip: use --json on validate/run/package for machine-readable output.",
            style="dim",
        )
    )

    body = Group(*body_items)
    console.print(Panel(body, title="Next Steps", border_style="bright_cyan"))
    return buffer.getvalue()


def _smoke_next_step_commands() -> "list[str]":
    return [
        "whest init ./my-estimator",
        "whest validate --estimator ./my-estimator/estimator.py",
        "whest run --estimator ./my-estimator/estimator.py --runner local",
        "whest package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz",
    ]


def _smoke_next_step_lines() -> "list[Text]":
    purposes = [
        ("Create starter files you can edit.", "bold bright_cyan"),
        ("Validate an Estimator implementation.", "bold bright_green"),
        ("Run local evaluation with isolation.", "bold bright_yellow"),
        ("Build submission artifacts for AIcrowd.", "bold bright_magenta"),
    ]
    return [
        Text(f"# {idx}) {purpose}", style=style)
        for idx, (purpose, style) in enumerate(purposes, start=1)
    ]


def _smoke_optional_example_commands() -> "list[str]":
    return [
        "whest run --estimator ./examples/estimators/combined_estimator.py --runner local",
        "whest run --estimator ./examples/estimators/covariance_propagation.py --runner local",
        "whest run --estimator ./examples/estimators/mean_propagation.py --runner local",
        "whest run --estimator ./examples/estimators/random_estimator.py --runner local",
    ]


def _render_top_row(console: Console, report: "dict[str, Any]") -> None:
    mode = _layout_mode(console.width)
    run_context = _run_context_panel(report)
    hardware = _hardware_runtime_panel(report)

    if mode in {"three_col", "two_col"}:
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(run_context, hardware)
        console.print(grid)
        return
    console.print(run_context)
    console.print(hardware)


def _render_score_row(console: Console, report: "dict[str, Any]") -> None:
    score = _score_summary_panel(report)
    console.print(score)


def _render_context_row(console: Console, report: "dict[str, Any]") -> None:
    console.print(build_human_context_renderable(report, console_width=console.width))


def build_human_context_renderable(
    report: "dict[str, Any]", *, console_width: Optional[int] = None
) -> Any:
    width = _dashboard_width() if console_width is None else max(80, int(console_width))
    mode = _layout_mode(width)
    run_context = _run_context_panel(report)
    hardware = _hardware_runtime_panel(report)

    if mode in {"three_col", "two_col"}:
        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(ratio=1)
        grid.add_row(run_context, hardware)
        return grid

    return Group(run_context, hardware)


def _dashboard_width() -> int:
    columns = shutil.get_terminal_size((120, 40)).columns
    return max(80, columns)


def _rich_console_environ(width: int) -> "dict[str, str]":
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


def _run_context_panel(report: "dict[str, Any]") -> Panel:
    run_meta = report.get("run_meta", {})
    run_config = report.get("run_config", {})

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("field")
    table.add_column("value", no_wrap=False)

    rows: "list[tuple[str, Any]]" = []
    estimator_class = run_config.get("estimator_class")
    if estimator_class is not None:
        rows.append(
            (
                "Estimator Class [estimator_class]",
                Text(str(estimator_class), style="bold bright_cyan"),
            )
        )
    estimator_path = run_config.get("estimator_path")
    if estimator_path is not None:
        rows.append(
            (
                "Estimator Path [estimator_path]",
                str(estimator_path),
            )
        )

    rows.extend(
        [
            (
                "Started [run_started_at_utc]",
                _human_utc(str(run_meta.get("run_started_at_utc", "n/a"))),
            ),
            (
                "Finished [run_finished_at_utc]",
                _human_utc(str(run_meta.get("run_finished_at_utc", "n/a"))),
            ),
            ("Duration(s) [run_duration_s]", _fmt_duration(run_meta.get("run_duration_s"))),
            ("MLPs [n_mlps]", str(run_config.get("n_mlps", "n/a"))),
            ("Width [width]", str(run_config.get("width", "n/a"))),
            ("Depth [depth]", str(run_config.get("depth", "n/a"))),
            ("FLOP Budget [flop_budget]", _fmt_flops(run_config.get("flop_budget"))),
        ]
    )
    for key, value in rows:
        table.add_row(_render_context_label(key), value)

    return Panel(Align.center(table), title="Run Context", border_style="bright_cyan")


def _fmt_bytes(value: Optional[int]) -> str:
    """Format byte count as human-readable string (e.g. '32.0 GB')."""
    if value is None:
        return "n/a"
    gb = value / (1024**3)
    return f"{gb:.1f} GB"


def _fmt_flops(value: Any) -> str:
    """Format FLOP counts legibly.

    - Non-numeric / ``None`` -> "n/a".
    - |n| < 1e6 -> comma-grouped integer (e.g. "12,345").
    - Otherwise -> scientific notation with 3 significant figures,
      e.g. "8.46e+11".

    Exact integer values are always available via ``whest run --json``.
    """
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if numeric != numeric:  # NaN
        return "n/a"
    if abs(numeric) < 1e6:
        return f"{int(numeric):,}"
    return f"{numeric:.2e}"


# --- Budget gauge state -----------------------------------------------------


@dataclass(frozen=True)
class GaugeState:
    """Computed inputs for the FLOP budget gauge renderer.

    state_name: one of "healthy", "tight", "busted", "catastrophic".
    mean_utilization: mean(per_mlp.flops_used) / flop_budget. May exceed 1.0.
    worst_mlp_pct: integer percent of the worst MLP; None iff any_busted is False.
    any_busted: any(per_mlp[i].budget_exhausted).
    flop_budget: echoed from run_config for rendering (0 allowed).
    has_budget: False iff flop_budget <= 0 (renderer uses this to suppress the bar).
    """

    state_name: str
    mean_utilization: float
    worst_mlp_pct: Optional[int]
    any_busted: bool
    flop_budget: int
    has_budget: bool


def _compute_gauge_state(report: "dict[str, Any]") -> GaugeState:
    run_config = report.get("run_config", {}) if isinstance(report, dict) else {}
    results = report.get("results", {}) if isinstance(report, dict) else {}
    per_mlp_raw = results.get("per_mlp", []) if isinstance(results, dict) else []
    per_mlp: list[dict[str, Any]] = [e for e in per_mlp_raw if isinstance(e, dict)]

    try:
        flop_budget = int(run_config.get("flop_budget", 0) or 0)
    except (TypeError, ValueError):
        flop_budget = 0
    has_budget = flop_budget > 0

    flops_used = [_as_float(entry.get("flops_used", 0.0)) for entry in per_mlp]
    if flops_used:
        mean_flops = sum(flops_used) / len(flops_used)
    else:
        mean_flops = 0.0
    mean_utilization = mean_flops / flop_budget if has_budget else 0.0

    any_busted = any(bool(entry.get("budget_exhausted", False)) for entry in per_mlp)

    if any_busted and has_budget:
        worst_flops = max(
            _as_float(entry.get("flops_used", 0.0))
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


# --- Over-budget selection --------------------------------------------------


@dataclass(frozen=True)
class OverBudgetRow:
    mlp_index: int
    flops_used: float
    pct_of_budget: Optional[int]  # None iff flop_budget <= 0


@dataclass(frozen=True)
class OverBudgetSelection:
    """Selection of over-budget MLPs to render in the Over-Budget panel.

    rows: up to top_n entries sorted by (overage desc, mlp_index asc).
    busted_count: total number of MLPs with budget_exhausted == True.
    n_mlps: total number of MLP entries in the run.
    is_truncated: busted_count > len(rows) (triggers the ``... and N more`` footer).
    is_all_busted: busted_count == n_mlps and busted_count > 0 (triggers the
        ``All M MLPs`` summary variant).
    """

    rows: "list[OverBudgetRow]"
    busted_count: int
    n_mlps: int
    is_truncated: bool
    is_all_busted: bool


def _select_top_over_budget(report: "dict[str, Any]", *, top_n: int = 5) -> OverBudgetSelection:
    run_config = report.get("run_config", {}) if isinstance(report, dict) else {}
    results = report.get("results", {}) if isinstance(report, dict) else {}
    per_mlp_raw = results.get("per_mlp", []) if isinstance(results, dict) else []
    per_mlp: list[dict[str, Any]] = [e for e in per_mlp_raw if isinstance(e, dict)]
    n_mlps = len(per_mlp)

    try:
        flop_budget = int(run_config.get("flop_budget", 0) or 0)
    except (TypeError, ValueError):
        flop_budget = 0

    busted = [entry for entry in per_mlp if bool(entry.get("budget_exhausted", False))]
    busted_count = len(busted)

    def _overage(entry: "dict[str, Any]") -> float:
        flops = _as_float(entry.get("flops_used", 0.0))
        if flop_budget > 0:
            return flops / flop_budget
        return flops  # fallback when budget is unknown; keeps sort stable

    sorted_busted = sorted(
        busted,
        key=lambda entry: (-_overage(entry), int(entry.get("mlp_index", 0))),
    )

    # Skip-truncation rule: if busted_count is exactly top_n + 1, show all.
    rows_to_show = sorted_busted if busted_count <= top_n + 1 else sorted_busted[:top_n]

    rows: list[OverBudgetRow] = []
    for entry in rows_to_show:
        flops = _as_float(entry.get("flops_used", 0.0))
        pct: Optional[int]
        if flop_budget > 0:
            pct = int(flops * 100 // flop_budget)
        else:
            pct = None
        rows.append(
            OverBudgetRow(
                mlp_index=int(entry.get("mlp_index", 0)),
                flops_used=flops,
                pct_of_budget=pct,
            )
        )

    is_truncated = busted_count > len(rows)
    is_all_busted = busted_count > 0 and busted_count == n_mlps

    return OverBudgetSelection(
        rows=rows,
        busted_count=busted_count,
        n_mlps=n_mlps,
        is_truncated=is_truncated,
        is_all_busted=is_all_busted,
    )


def _hardware_runtime_panel(report: "dict[str, Any]") -> Panel:
    host = report.get("run_meta", {}).get("host", {})
    host_meta = host if isinstance(host, dict) else {}

    table = Table(box=box.SIMPLE_HEAVY, show_header=False)
    table.add_column("field")
    table.add_column("value")

    def _s(val: object) -> str:
        """Stringify a metadata value, treating None as 'n/a'."""
        return "n/a" if val is None else str(val)

    rows = [
        ("Host [host.hostname]", _s(host_meta.get("hostname"))),
        ("OS [host.os]", _s(host_meta.get("os"))),
        ("Release [host.os_release]", _s(host_meta.get("os_release"))),
        ("Platform [host.platform]", _s(host_meta.get("platform"))),
        ("Arch [host.machine]", _s(host_meta.get("machine"))),
        ("CPU [host.cpu_brand]", _s(host_meta.get("cpu_brand"))),
        ("CPU Cores (logical) [host.cpu_count_logical]", _s(host_meta.get("cpu_count_logical"))),
        ("CPU Cores (physical) [host.cpu_count_physical]", _s(host_meta.get("cpu_count_physical"))),
        ("RAM Total [host.ram_total_bytes]", _fmt_bytes(host_meta.get("ram_total_bytes"))),
        ("Python [host.python_version]", _s(host_meta.get("python_version"))),
        ("NumPy [host.numpy_version]", _s(host_meta.get("numpy_version"))),
    ]
    for label, value in rows:
        table.add_row(_render_context_label(label), value)
    return Panel(Align.center(table), title="Hardware & Runtime", border_style="bright_blue")


def _score_summary_panel(report: "dict[str, Any]") -> Panel:
    results = report.get("results", {})
    primary_score = _as_float(results.get("primary_score", 0.0))
    secondary_score = _as_float(results.get("secondary_score", 0.0))
    summary = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    summary.add_column("metric")
    summary.add_column("value", justify="right")
    summary.add_row(
        _label_with_code("Primary Score", "primary_score", "bold bright_green"),
        f"[bold bright_green]{_fmt_float(primary_score, 8)}[/]",
    )
    summary.add_row(
        _label_with_code("Secondary Score", "secondary_score", "bold bright_cyan"),
        f"[cyan]{_fmt_float(secondary_score, 8)}[/]",
    )

    per_mlp = results.get("per_mlp", [])
    if isinstance(per_mlp, list) and per_mlp:
        mlp_primaries = [
            _as_float(entry.get("final_mse", 0.0)) for entry in per_mlp if isinstance(entry, dict)
        ]
        if mlp_primaries:
            summary.add_row(
                _label_with_code("Best MLP Score", "best_mlp_score", "bold green"),
                f"[green]{_fmt_float(min(mlp_primaries), 8)}[/]",
            )
            summary.add_row(
                _label_with_code("Worst MLP Score", "worst_mlp_score", "bold yellow"),
                f"[yellow]{_fmt_float(max(mlp_primaries), 8)}[/]",
            )

    return Panel(
        Align.center(summary),
        title="Final Score",
        subtitle="lower MSE is better; primary score = mean across MLPs of final-layer MSE",
        subtitle_align="left",
        border_style="bright_cyan",
    )


# --- Budget gauge renderer --------------------------------------------------


_GAUGE_BAR_WIDTH = 20
_GAUGE_COLOR = {
    "healthy": "green",
    "tight": "yellow",
    "busted": "red",
    "catastrophic": "red",
}


def _gauge_bar_fragment(utilization: float) -> str:
    """Return the ``[filled░░]`` fragment (without trailing ▶) for utilization.

    Catastrophic overflow is handled by the renderer — it appends ▶ after.
    """
    if utilization <= 0.0:
        filled = 0
    else:
        filled = min(round(utilization * _GAUGE_BAR_WIDTH), _GAUGE_BAR_WIDTH)
    empty = _GAUGE_BAR_WIDTH - filled
    return "[" + ("█" * filled) + ("░" * empty) + "]"


def _gauge_renderable(report: "dict[str, Any]") -> Text:
    """Return the gauge line as a Rich Text object without printing it.

    Used both by the standalone ``_render_budget_gauge`` backward-compat
    wrapper and by ``_breakdown_panel`` when embedding the gauge inside the
    Estimator Budget Breakdown panel.
    """
    state = _compute_gauge_state(report)

    line = Text()
    line.append("Estimator FLOPs", style="bold bright_yellow")
    line.append("  ")

    if not state.has_budget:
        line.append("-- of 0 FLOPs", style="dim")
        return line

    color = _GAUGE_COLOR.get(state.state_name, "white")
    bar = _gauge_bar_fragment(state.mean_utilization)
    pct_int = int(state.mean_utilization * 100)  # truncates toward zero
    budget_label = _fmt_flops(state.flop_budget)

    line.append(bar, style=color)
    if state.state_name == "catastrophic":
        line.append("▶")
    line.append(" ")
    line.append(f"{pct_int}% of {budget_label}", style="bold")

    if state.worst_mlp_pct is not None:
        line.append(" ")
        line.append("·", style="dim")
        line.append(" ")
        line.append(f"worst MLP {state.worst_mlp_pct}%", style="red")
        line.append(" ")
        line.append("⚠", style="red")

    return line


def _over_budget_renderable(report: "dict[str, Any]") -> Optional[Group]:
    """Return a Group of the over-budget rows + truncation footer + summary.

    Returns None when there are no busted MLPs. Does NOT wrap the Group in a
    Panel — callers decide whether to wrap (standalone panel) or embed
    (inside the Estimator Budget Breakdown panel).
    """
    sel = _select_top_over_budget(report)
    if sel.busted_count == 0:
        return None

    lines: list[Text] = []

    for row in sel.rows:
        pct_label = f"{row.pct_of_budget}%" if row.pct_of_budget is not None else "--%"
        line = Text()
        line.append(f"MLP #{row.mlp_index:<4}", style="red")
        line.append("  ")
        line.append(f"{_fmt_flops(row.flops_used):>9} FLOPs", style="red")
        line.append("  ")
        line.append(f"{pct_label:>4} of budget", style="red")
        line.append("  ")
        line.append("zeroed", style="red")
        lines.append(line)

    if sel.is_truncated:
        remainder = sel.busted_count - len(sel.rows)
        lines.append(Text(f"... and {remainder} more over budget", style="dim"))
        lines.append(Text("run with --json for the full list", style="dim"))

    lines.append(Text(""))  # blank line before summary
    if sel.is_all_busted:
        summary = (
            f"All {sel.n_mlps} MLPs exceeded the per-MLP FLOP cap — predictions entirely zeroed"
        )
    else:
        summary = f"{sel.busted_count} of {sel.n_mlps} MLPs exceeded the per-MLP FLOP cap"
    lines.append(Text(summary, style="bold red"))

    return Group(*lines)


def _render_budget_gauge(console: Console, report: "dict[str, Any]") -> None:
    """Backward-compat wrapper that prints the gauge line standalone.

    The gauge is normally embedded inside the Estimator Budget Breakdown
    panel (see ``_breakdown_panel``); this wrapper preserves the original
    standalone call surface for existing unit tests.
    """
    console.print(_gauge_renderable(report))


def _render_over_budget_panel(console: Console, report: "dict[str, Any]") -> None:
    """Backward-compat wrapper that prints the over-budget content in a Panel.

    The over-budget content is normally inlined inside the Estimator Budget
    Breakdown panel; this wrapper preserves the original standalone
    red-bordered Panel for existing unit tests.
    """
    body = _over_budget_renderable(report)
    if body is None:
        return
    console.print(
        Panel(
            body,
            title="Over-Budget MLPs",
            border_style="red",
        )
    )


def _render_errors_section(
    console: Console, report: "dict[str, Any]", *, debug: bool = False
) -> None:
    """Render a red panel listing per-MLP estimator errors when any are present."""
    results = report.get("results") if isinstance(report, dict) else None
    per_mlp = results.get("per_mlp") if isinstance(results, dict) else None
    if not isinstance(per_mlp, list):
        return
    failures = [entry for entry in per_mlp if isinstance(entry, dict) and entry.get("error")]
    if not failures:
        return

    total = len(per_mlp)
    table = Table(box=box.SIMPLE, expand=True, show_lines=False)
    table.add_column("MLP", justify="right", style="bold")
    table.add_column("Code", style="bright_red")
    table.add_column("Message")
    for entry in failures:
        idx = str(entry.get("mlp_index", "?"))
        code = str(entry.get("error_code") or "UNKNOWN")
        message = str(entry.get("error") or "").splitlines()[0] if entry.get("error") else ""
        table.add_row(idx, code, message)

    children: list[Any] = [
        Text(f"{len(failures)} of {total} MLP(s) raised during predict.", style="bold red"),
        table,
    ]
    if debug:
        for entry in failures:
            tb_text = entry.get("traceback")
            if not tb_text:
                continue
            tb_block = Panel(
                Text(str(tb_text).rstrip(), style="dim"),
                title=f"Traceback — MLP {entry.get('mlp_index', '?')}",
                title_align="left",
                border_style="red",
                expand=True,
            )
            children.append(tb_block)
    else:
        children.append(
            Text(
                "Rerun with --debug to include full tracebacks; --fail-fast to stop on first error.",
                style="dim",
            )
        )

    console.print(
        Panel(
            Group(*children),
            title="Estimator Errors",
            border_style="red",
            expand=True,
        )
    )


def _render_breakdown_sections(console: Console, report: "dict[str, Any]") -> None:
    for breakdown_key, title in (
        ("sampling", "Sampling Budget Breakdown"),
        ("estimator", "Estimator Budget Breakdown"),
    ):
        panel = _breakdown_panel(report, breakdown_key=breakdown_key, title=title)
        if panel is not None:
            console.print(panel)


def _breakdown_panel(
    report: "dict[str, Any]", *, breakdown_key: str, title: str
) -> Optional[Panel]:
    results = report.get("results", {})
    if not isinstance(results, dict):
        return None
    breakdowns = results.get("breakdowns")
    if not isinstance(breakdowns, dict):
        return None
    breakdown = breakdowns.get(breakdown_key)
    if not isinstance(breakdown, dict):
        return None
    by_namespace = breakdown.get("by_namespace")
    if not isinstance(by_namespace, dict):
        by_namespace = {}

    run_config = report.get("run_config", {})
    n_mlps = int(run_config.get("n_mlps", 0) or 0) if isinstance(run_config, dict) else 0
    if n_mlps <= 0:
        n_mlps = 1

    total_flops = _as_float(breakdown.get("flops_used", 0.0))
    if total_flops <= 0.0:
        total_flops = sum(
            _as_float(bucket.get("flops_used", 0.0)) for bucket in by_namespace.values()
        )

    summary = Table(box=box.SIMPLE_HEAVY, show_header=False)
    summary.add_column("field", style="bold bright_white")
    summary.add_column("value")
    summary.add_row(
        _label_with_code("Total FLOPs", "flops_used", "bold bright_yellow"),
        _fmt_flops(total_flops),
    )
    summary.add_row(
        _label_with_code("Tracked Time", "tracked_time_s", "bold bright_green"),
        f"{_as_float(breakdown.get('tracked_time_s', 0.0)):.6f}s",
    )
    summary.add_row(
        _label_with_code("Untracked Time", "untracked_time_s", "bold bright_green"),
        f"{_as_float(breakdown.get('untracked_time_s', 0.0)):.6f}s",
    )

    table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold bright_white")
    table.add_column("namespace", style="bold bright_white", no_wrap=False)
    table.add_column("total flops", justify="right")
    table.add_column("% of section flops", justify="right")
    table.add_column("mean flops / MLP", justify="right")
    table.add_column("tracked time", justify="right")

    for namespace, bucket in sorted(
        by_namespace.items(),
        key=lambda item: _as_float(item[1].get("flops_used", 0.0)),
        reverse=True,
    ):
        namespace_label = (
            "(unlabeled)" if namespace in {None, "", "null", "None"} else str(namespace)
        )
        flops_used = _as_float(bucket.get("flops_used", 0.0))
        percent = (flops_used / total_flops * 100.0) if total_flops > 0.0 else 0.0
        mean_flops = flops_used / n_mlps if n_mlps > 0 else 0.0
        tracked_time_s = _as_float(bucket.get("tracked_time_s", 0.0))
        table.add_row(
            namespace_label,
            _fmt_flops(flops_used),
            f"{percent:.1f}%",
            _fmt_flops(mean_flops),
            f"{tracked_time_s:.6f}s",
        )

    body: "list[Any]" = []
    if breakdown_key == "estimator":
        body.append(_gauge_renderable(report))
        over_budget = _over_budget_renderable(report)
        if over_budget is not None:
            body.append(Text(""))
            body.append(Text("Over-Budget MLPs", style="bold red"))
            body.append(over_budget)
        body.append(Text(""))
    body.append(Align.center(summary))
    if by_namespace:
        body.append(Align.center(table))

    if breakdown_key == "estimator":
        any_busted = _compute_gauge_state(report).any_busted
        border_style = "red" if any_busted else "bright_magenta"
    else:
        border_style = "bright_yellow"
    return Panel(
        Group(*body),
        title=title,
        subtitle="aggregated across all evaluated MLPs",
        subtitle_align="left",
        border_style=border_style,
    )


def _render_profile_section(
    console: Console, report: "dict[str, Any]", *, show_diagnostic_plots: bool
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
    profile_body: "list[Any]" = [Align.center(profile_tables)]
    if show_diagnostic_plots:
        runtime_plot = _profile_runtime_plot_panel(wall, cpu)
        memory_plot = _profile_memory_plot_panel(rss, peak)
        profile_body.append(
            Align.center(
                Columns(
                    [runtime_plot, memory_plot],
                    align="center",
                    equal=True,
                    expand=False,
                )
            )
        )
    console.print(Panel(Group(*profile_body), title="Profile", border_style="bright_blue"))


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
    series: "list[tuple[str, Sequence[float], str]]" = [("rss_mb", rss_mb, "cyan+")]
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
    series: Sequence["tuple[str, Sequence[float], str]"],
    x_label: str,
    y_label: str,
    x_scale: Optional[str] = None,
    y_scale: Optional[str] = None,
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
        body: "Any" = Group(Align.center(fallback), legend)
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
    series: Sequence["tuple[str, Sequence[float], str]"],
    x_label: str,
    y_label: str,
    x_scale: Optional[str] = None,
    y_scale: Optional[str] = None,
    sparse_style: str = "scatter",
) -> Optional[str]:
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


def _legend_table(series: Sequence["tuple[str, Sequence[float], str]"]) -> Table:
    legend = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    legend.add_column("key")
    legend.add_column("range")
    for label, values, color in series:
        key = Text("  ", style=_rich_style_for_plot_color(color))
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


def _fmt_duration(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "n/a", "na", "none"}:
            return "n/a"
        try:
            return f"{float(value):.6f}"
        except ValueError:
            return "n/a"
    return "n/a"


def _left_ellipsis(value: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return "." * max_chars
    # Preserve the filename (last path component) and truncate the directory.
    sep_idx = value.rfind("/")
    if sep_idx == -1:
        sep_idx = value.rfind("\\")
    if sep_idx != -1:
        basename = value[sep_idx:]  # includes the leading /
        if len(basename) + 3 <= max_chars:
            # Fit as much of the directory as possible before the basename
            dir_budget = max_chars - len(basename) - 3  # 3 for "..."
            return "..." + value[sep_idx - dir_budget : sep_idx] + basename
    return "..." + value[-(max_chars - 3) :]


def _context_key_style(key: str) -> str:
    if "[" in key and "]" in key:
        key = key[key.find("[") + 1 : key.rfind("]")]
    if key.startswith("run_"):
        return "bold bright_cyan"
    if key.startswith("host."):
        return "bold bright_blue"
    if key in {"n_mlps", "width", "depth"}:
        return "bold bright_magenta"
    if key == "flop_budget":
        return "bold bright_yellow"
    if key.endswith("_s"):
        return "bold bright_green"
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


def _mean_series(series_list: Sequence[Sequence[float]]) -> "list[float]":
    if not series_list:
        return []
    length = min(len(series) for series in series_list)
    if length == 0:
        return []
    return [fmean(series[i] for series in series_list) for i in range(length)]


def _normalize(values: Sequence[float]) -> "list[float]":
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
    return all(abs(left - right) <= tolerance for left, right in zip(lhs, rhs))


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


def _to_float_list(value: object) -> "list[float]":
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
