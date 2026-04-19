from __future__ import annotations

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
    text = Text()
    text.append(human + " ", style=style)
    text.append(f"[{code}]", style="bold bright_white")
    return text


def _score_value_markup(metric: str, value: str) -> str:
    value_styles = {
        "Primary Score [primary_score]": "bold bright_green",
        "Secondary Score [secondary_score]": "cyan",
        "Best MLP Score [best_mlp_score]": "green",
        "Worst MLP Score [worst_mlp_score]": "yellow",
    }
    style = value_styles.get(metric)
    if style is None:
        return value
    return f"[{style}]{value}[/]"


def build_score_block(section: TableSection) -> Panel:
    table = Table(box=box.SIMPLE_HEAVY, header_style="bold bright_white")
    table.add_column("metric")
    table.add_column("value", justify="right")

    metric_labels = {
        "Primary Score [primary_score]": make_keyed_label(
            "Primary Score", "primary_score", "bold bright_green"
        ),
        "Secondary Score [secondary_score]": make_keyed_label(
            "Secondary Score", "secondary_score", "bold bright_cyan"
        ),
        "Best MLP Score [best_mlp_score]": make_keyed_label(
            "Best MLP Score", "best_mlp_score", "bold green"
        ),
        "Worst MLP Score [worst_mlp_score]": make_keyed_label(
            "Worst MLP Score", "worst_mlp_score", "bold yellow"
        ),
    }

    for metric, value in section.rows:
        table.add_row(metric_labels.get(metric, Text(metric)), _score_value_markup(metric, value))

    panel_kwargs = {
        "title": escape(section.title),
        "subtitle": escape(section.subtitle or ""),
        "subtitle_align": "left",
    }
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
    if section.tracked_time is not None:
        summary.add_row(
            make_keyed_label("Tracked Time", "tracked_time_s", "bold bright_green"),
            Text(section.tracked_time),
        )
    if section.untracked_time is not None:
        summary.add_row(
            make_keyed_label("Untracked Time", "untracked_time_s", "bold bright_green"),
            Text(section.untracked_time),
        )
    if summary.row_count:
        body.append(Align.center(summary))

    if section.namespace_rows:
        table = Table(box=box.SIMPLE_HEAVY, show_header=True, header_style="bold bright_white")
        max_namespace_width = max(
            len("namespace"),
            *(len(row.namespace) for row in section.namespace_rows),
        )
        table.add_column(
            "namespace",
            style="bold bright_white",
            no_wrap=True,
            min_width=max_namespace_width,
        )
        table.add_column("total flops", justify="right")
        table.add_column("% of section flops", justify="right")
        table.add_column("mean flops / MLP", justify="right")
        table.add_column("tracked time", justify="right")
        for row in section.namespace_rows:
            table.add_row(
                row.namespace,
                row.total_flops,
                row.percent_of_section_flops,
                row.mean_flops_per_mlp,
                row.tracked_time,
            )
        body.append(Align.center(table))

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
        if section.title == "Final Score" and section.columns == ["metric", "value"]:
            return [build_score_block(section)]
        table = Table(show_header=True)
        for column in section.columns:
            table.add_column(Text(column))
        for row in section.rows:
            table.add_row(*(Text(cell) for cell in row))
        renderable: RenderableType = Align.center(table) if section.align_center else table
        panel_kwargs = {
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
            table.add_row(str(entry.mlp_index), entry.code, entry.message)
        children: list[RenderableType] = [Text(section.summary, style="bold red"), table]
        for entry in section.entries:
            detail_lines = format_error_detail_lines(entry.details)
            if detail_lines:
                children.append(
                    Panel(
                        Text("\n".join(detail_lines)),
                        title=f"MLP {entry.mlp_index} Details",
                        border_style="red",
                    )
                )
            if entry.traceback:
                children.append(
                    Panel(
                        Text(entry.traceback.rstrip("\n"), style="dim"),
                        title=f"Traceback — MLP {entry.mlp_index}",
                        border_style="red",
                    )
                )
        if section.footer:
            children.append(Text(section.footer, style="dim"))
        return [Panel(Group(*children), title=escape(section.title), border_style="red")]

    if isinstance(section, BudgetBreakdownSection):
        return [build_budget_breakdown_block(section)]

    return []
