from __future__ import annotations

import re
from typing import Any

from rich import box
from rich.align import Align
from rich.console import Group, RenderableType
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import (
    BudgetBreakdownSection,
    ChecklistSection,
    ErrorSection,
    KeyValueSection,
    RunErrorsSection,
    StepItem,
    StepsSection,
    TableSection,
    format_error_detail_lines,
)


def make_keyed_label(human: str, code: str, style: str) -> Text:
    # The bracketed [code] tag is metadata subordinate to the human label.
    # `dim` (SGR 2 faint) reduces the terminal foreground colour rather than
    # using a fixed palette slot, so it stays legible against both light and
    # dark backgrounds.
    text = Text()
    text.append(human + " ", style=style)
    text.append(f"[{code}]", style="dim")
    return text


def _score_value_markup(metric: str, value: str) -> str:
    # Only the rows with semantic colour (accuracy / range) carry markup. The
    # efficiency rows (multiplier / utilization / failed) render in terminal
    # default foreground so they stay legible on both light and dark themes —
    # `bright_white` is a fixed palette slot that becomes invisible on light
    # backgrounds.
    value_styles = {
        "Adjusted Final-Layer Score [adjusted_final_layer_score]": "bold bright_green",
        "Raw Final-Layer MSE [final_layer_mse]": "cyan",
        "All-Layers MSE [all_layers_mse]": "cyan",
        "Best MLP [best_mlp_adjusted_final_layer_score]": "green",
        "Worst MLP [worst_mlp_adjusted_final_layer_score]": "yellow",
    }
    style = value_styles.get(metric)
    if style is None:
        return value
    return f"[{style}]{value}[/]"


def build_score_block(section: TableSection) -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold", expand=True)
    table.add_column("metric")
    table.add_column("value", justify="right")
    # Note column carries the "← primary score" annotation on the adjusted row;
    # it's empty on every other row. Kept as a separate left-aligned column so
    # the value column stays uniformly right-aligned across all rows.
    has_note_column = len(section.columns) >= 3
    if has_note_column:
        table.add_column("", justify="left", no_wrap=True)

    metric_labels = {
        "Adjusted Final-Layer Score [adjusted_final_layer_score]": make_keyed_label(
            "Adjusted Final-Layer Score", "adjusted_final_layer_score", "bold bright_green"
        ),
        "Raw Final-Layer MSE [final_layer_mse]": make_keyed_label(
            "Raw Final-Layer MSE", "final_layer_mse", "bold cyan"
        ),
        "All-Layers MSE [all_layers_mse]": make_keyed_label(
            "All-Layers MSE", "all_layers_mse", "bold cyan"
        ),
        "Best MLP [best_mlp_adjusted_final_layer_score]": make_keyed_label(
            "Best MLP", "best_mlp_adjusted_final_layer_score", "bold green"
        ),
        "Worst MLP [worst_mlp_adjusted_final_layer_score]": make_keyed_label(
            "Worst MLP", "worst_mlp_adjusted_final_layer_score", "bold yellow"
        ),
        "Mean Score Multiplier [mean_score_multiplier]": make_keyed_label(
            "Mean Score Multiplier", "mean_score_multiplier", "bold"
        ),
        "Mean Compute Utilization [mean_compute_utilization]": make_keyed_label(
            "Mean Compute Utilization", "mean_compute_utilization", "bold"
        ),
        "Failed MLPs [n_failed_mlps]": make_keyed_label("Failed MLPs", "n_failed_mlps", "bold"),
    }

    # Section dividers: insert visual separators after row 3 (accuracy → range)
    # and after row 5 (range → efficiency). Rows are 1-indexed in this loop.
    DIVIDER_AFTER_ROWS = {3, 5}
    divider_cell = Text("─" * 8, style="dim")
    for idx, row in enumerate(section.rows, start=1):
        metric = row[0]
        value = row[1]
        note = row[2] if has_note_column and len(row) >= 3 else ""
        cells = [metric_labels.get(metric, Text(metric)), _score_value_markup(metric, value)]
        if has_note_column:
            # The annotation is dim italic so the value remains the headline; the
            # arrow draws the eye without competing visually with the score.
            cells.append(Text(note, style="dim italic") if note else Text(""))
        table.add_row(*cells)
        if idx in DIVIDER_AFTER_ROWS and idx < len(section.rows):
            divider_cells = [divider_cell, divider_cell]
            if has_note_column:
                divider_cells.append(Text(""))
            table.add_row(*divider_cells)

    panel_kwargs: dict[str, Any] = {
        "title": escape(section.title),
    }
    if section.subtitle:
        formula_match = re.search(
            r"max\(0\.1,\s*effective_compute/flop_budget\)",
            section.subtitle,
        )
        panel_kwargs["subtitle"] = (
            f"{formula_match.group(0)}\nfinal_layer_mse"
            if formula_match
            else escape(section.subtitle)
        )
        panel_kwargs["subtitle_align"] = "left"
    if section.border_style is not None:
        panel_kwargs["border_style"] = section.border_style
    return Panel(Align.center(table), **panel_kwargs)


def build_budget_breakdown_block(section: BudgetBreakdownSection) -> Panel:
    if not section.available:
        message = section.unavailable_message or "Unavailable."
        return Panel(
            Text(message),
            title=escape(section.title),
            border_style="bright_yellow",
        )

    body: list[RenderableType] = []
    if section.source_note:
        body.append(Text(section.source_note, style="dim"))

    summary = Table(box=box.SIMPLE_HEAVY, show_header=False)
    summary.add_column("field")
    summary.add_column("value")
    if section.total_flops is not None:
        summary.add_row(
            make_keyed_label("Total FLOPs", "flops_used", "bold bright_yellow"),
            Text(section.total_flops),
        )
    if section.effective_compute is not None:
        summary.add_row(
            make_keyed_label("Effective Compute", "effective_compute", "bold bright_yellow"),
            Text(section.effective_compute),
        )
    if section.flopscope_backend_time is not None:
        summary.add_row(
            make_keyed_label("Flopscope Backend", "flopscope_backend_time_s", "bold bright_green"),
            Text(section.flopscope_backend_time),
        )
    if section.flopscope_overhead_time is not None:
        summary.add_row(
            make_keyed_label(
                "Flopscope Overhead", "flopscope_overhead_time_s", "bold bright_yellow"
            ),
            Text(section.flopscope_overhead_time),
        )
    if section.residual_wall_time is not None:
        summary.add_row(
            make_keyed_label("Residual Wall Time", "residual_wall_time_s", "bold bright_green"),
            Text(section.residual_wall_time),
        )
    if summary.row_count:
        body.append(Align.center(summary))

    if section.namespace_rows:
        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        max_namespace_width = max(
            len("namespace"),
            *(len(row.namespace) for row in section.namespace_rows),
        )
        table.add_column(
            "namespace",
            style="bold",
            no_wrap=True,
            min_width=max_namespace_width,
        )
        table.add_column("total flops", justify="right")
        table.add_column("% of section flops", justify="right")
        table.add_column("mean flops / MLP", justify="right")
        table.add_column("tracked time", justify="right")
        table.add_column("flopscope overhead", justify="right")
        for row in section.namespace_rows:
            table.add_row(
                row.namespace,
                row.total_flops,
                row.percent_of_section_flops,
                row.mean_flops_per_mlp,
                row.flopscope_backend_time,
                row.flopscope_overhead_time,
            )
        body.append(Align.center(table))

    if section.over_budget_rows:
        ob_table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold")
        ob_table.add_column("MLP")
        ob_table.add_column("reason")
        ob_table.add_column("value")
        ob_table.add_column("% of B_m", justify="right")
        for row in section.over_budget_rows:
            ob_table.add_row(
                str(row.mlp_index),
                row.reason,
                f"{row.metric_name} = {row.metric_value}",
                row.percent_of_budget or "—",
            )
        body.append(Align.center(ob_table))

    if section.over_budget_summary:
        body.append(Text(section.over_budget_summary))

    if section.over_budget_truncated_remainder:
        body.append(
            Text(
                f"... and {section.over_budget_truncated_remainder} more over budget — "
                "run with --format json for the full list",
                style="dim",
            )
        )

    footer_note = section.footer_note or "aggregated across all evaluated MLPs"
    if footer_note:
        body.append(Text(footer_note))

    border_style = "bright_yellow" if "Ground Truth" in section.title else "bright_magenta"
    return Panel(Group(*body), title=escape(section.title), border_style=border_style)


def _render_step(step: str | StepItem) -> str:
    if isinstance(step, StepItem):
        return f"{step.purpose}\n{step.command}"
    return step


def _checklist_status_style(status: str) -> str:
    if status == "ok":
        return "green"
    if status == "warn":
        return "yellow"
    return "red"


def build_section_renderables(section: object) -> list[RenderableType]:
    if isinstance(section, KeyValueSection):
        table = Table(show_header=False)
        table.add_column("field")
        table.add_column("value", overflow="fold")
        for row in section.rows:
            table.add_row(Text(row.label), Text(row.value))
        return [Panel(table, title=escape(section.title))]

    if isinstance(section, TableSection):
        if section.title == "Final Score" and section.columns[:2] == ["metric", "value"]:
            return [build_score_block(section)]
        table = Table(show_header=True)
        for column in section.columns:
            table.add_column(Text(column))
        for row in section.rows:
            table.add_row(*(Text(cell) for cell in row))
        renderable: RenderableType = Align.center(table) if section.align_center else table
        panel_kwargs: dict[str, Any] = {
            "title": escape(section.title),
            "subtitle": escape(section.subtitle or ""),
            "subtitle_align": "left",
        }
        if section.border_style is not None:
            panel_kwargs["border_style"] = section.border_style
        return [Panel(renderable, **panel_kwargs)]

    if isinstance(section, StepsSection):
        return [
            Panel(
                Text("\n".join(_render_step(step) for step in section.steps)),
                title=escape(section.title),
            )
        ]

    if isinstance(section, ChecklistSection):
        table = Table(show_header=True)
        table.add_column("Status")
        table.add_column("Check")
        table.add_column("Detail")
        for item in section.items:
            table.add_row(
                Text(item.status.upper(), style=_checklist_status_style(item.status)),
                Text(item.label),
                Text(item.detail),
            )
        return [Panel(table, title=escape(section.title))]

    if isinstance(section, ErrorSection):
        detail_lines = [section.message] if section.message else []
        detail_lines.extend(format_error_detail_lines(section.details))
        if section.traceback:
            detail_lines.append("Traceback:")
            detail_lines.extend(section.traceback.rstrip("\n").splitlines())
        if detail_lines:
            return [
                Panel(
                    Text("\n".join(detail_lines)),
                    title=escape(section.title),
                    border_style="red",
                )
            ]
        return []

    if isinstance(section, RunErrorsSection):
        table = Table(show_header=True)
        table.add_column("MLP")
        table.add_column("Code")
        table.add_column("Message")
        for entry in section.entries:
            label = (
                f"{entry.mlp_name} (#{entry.mlp_index})" if entry.mlp_name else str(entry.mlp_index)
            )
            table.add_row(label, entry.code, entry.message)
        children: list[RenderableType] = [Text(section.summary, style="bold red"), table]
        for entry in section.entries:
            entry_label = (
                f"MLP {entry.mlp_name} (#{entry.mlp_index})"
                if entry.mlp_name
                else f"MLP {entry.mlp_index}"
            )
            detail_lines = format_error_detail_lines(entry.details)
            if detail_lines:
                children.append(
                    Panel(
                        Text("\n".join(detail_lines)),
                        title=f"{entry_label} Details",
                        border_style="red",
                    )
                )
            if entry.traceback:
                children.append(
                    Panel(
                        Text(entry.traceback.rstrip("\n"), style="dim"),
                        title=f"Traceback — {entry_label}",
                        border_style="red",
                    )
                )
        if section.footer:
            children.append(Text(section.footer, style="dim"))
        return [Panel(Group(*children), title=escape(section.title), border_style="red")]

    if isinstance(section, BudgetBreakdownSection):
        return [build_budget_breakdown_block(section)]

    return []
