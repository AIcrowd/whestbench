from __future__ import annotations

import io

from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import (
    ChecklistSection,
    CommandPresentation,
    ErrorSection,
    KeyValueSection,
    StepItem,
    StepsSection,
    format_error_detail_lines,
)


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


def _primary_error_section(doc: CommandPresentation) -> ErrorSection | None:
    for section in doc.sections:
        if isinstance(section, ErrorSection):
            return section
    return None


def render_rich_presentation(doc: CommandPresentation) -> str:
    buffer = io.StringIO()
    console = Console(record=True, file=buffer, force_terminal=True, color_system="truecolor")
    body: list[object] = []

    primary_error = _primary_error_section(doc) if doc.status == "error" else None
    if primary_error is None:
        body.append(Text.assemble(("Command: ", "bold"), doc.command))
        body.append(Text.assemble(("Status: ", "bold"), doc.status))
    elif primary_error.message:
        body.append(Text(primary_error.message, style="bold"))

    for section in doc.sections:
        if isinstance(section, KeyValueSection):
            table = Table(show_header=False)
            table.add_column("field")
            table.add_column("value")
            for row in section.rows:
                table.add_row(Text(row.label), Text(row.value))
            body.append(Panel(table, title=escape(section.title)))
        elif isinstance(section, StepsSection):
            body.append(
                Panel(
                    Text("\n".join(_render_step(step) for step in section.steps)),
                    title=escape(section.title),
                )
            )
        elif isinstance(section, ChecklistSection):
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
            body.append(Panel(table, title=escape(section.title)))
        elif isinstance(section, ErrorSection):
            detail_lines = format_error_detail_lines(section.details)
            if section.traceback:
                detail_lines.append("Traceback:")
                detail_lines.extend(section.traceback.rstrip("\n").splitlines())
            if detail_lines:
                body.append(
                    Panel(
                        Text("\n".join(detail_lines)),
                        title=escape(section.title),
                        border_style="red",
                    )
                )

    for message in doc.epilogue_messages:
        if message:
            body.append(Text(message))

    console.print(Panel(Group(*body), title=escape(doc.title), subtitle=escape(doc.subtitle or "")))
    return buffer.getvalue()
