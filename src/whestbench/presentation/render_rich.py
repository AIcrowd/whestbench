from __future__ import annotations

import io

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import (
    ChecklistSection,
    CommandPresentation,
    ErrorSection,
    KeyValueSection,
    MetricsSection,
    StepsSection,
    TableSection,
    TextSection,
)


def render_rich_presentation(doc: CommandPresentation) -> str:
    buffer = io.StringIO()
    console = Console(record=True, file=buffer, force_terminal=True, color_system="truecolor")
    body: list[object] = []

    for section in doc.sections:
        if isinstance(section, KeyValueSection):
            table = Table(show_header=False)
            table.add_column("field")
            table.add_column("value")
            for row in section.rows:
                table.add_row(row.label, row.value)
            body.append(Panel(table, title=section.title))
        elif isinstance(section, MetricsSection):
            table = Table(show_header=False)
            table.add_column("metric")
            table.add_column("value")
            for label, value in section.metrics:
                table.add_row(label, value)
            body.append(Panel(table, title=section.title))
        elif isinstance(section, TableSection):
            table = Table(*section.columns)
            for row in section.rows:
                table.add_row(*row)
            body.append(Panel(table, title=section.title))
        elif isinstance(section, ChecklistSection):
            text = "\n".join(
                f"{item.status.upper()} {item.label}: {item.detail}" for item in section.items
            )
            body.append(Panel(Text(text), title=section.title))
        elif isinstance(section, StepsSection):
            body.append(Panel(Text("\n".join(section.steps)), title=section.title))
        elif isinstance(section, TextSection):
            body.append(Panel(Text(section.body), title=section.title))
        elif isinstance(section, ErrorSection):
            detail_lines = [f"{section.code}: {section.message}"]
            for key, value in section.details.items():
                detail_lines.append(f"{key}: {value}")
            if section.traceback:
                detail_lines.append(section.traceback)
            body.append(
                Panel(Text("\n".join(detail_lines)), title=section.title, border_style="red")
            )

    for message in doc.epilogue_messages:
        if message:
            body.append(Text(message))

    console.print(Panel(Group(*body), title=doc.title, subtitle=doc.subtitle or ""))
    return buffer.getvalue()
