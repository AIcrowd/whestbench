from __future__ import annotations

import io

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table

from .models import CommandPresentation, KeyValueSection, StepsSection


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
        elif isinstance(section, StepsSection):
            body.append(Panel("\n".join(section.steps), title=section.title))

    for message in doc.epilogue_messages:
        if message:
            body.append(message)

    console.print(Panel(Group(*body), title=doc.title, subtitle=doc.subtitle or ""))
    return buffer.getvalue()
