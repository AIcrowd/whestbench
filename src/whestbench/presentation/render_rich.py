from __future__ import annotations

import io

from rich.console import Console, Group
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .models import CommandPresentation, KeyValueSection, StepsSection


def render_rich_presentation(doc: CommandPresentation) -> str:
    buffer = io.StringIO()
    console = Console(record=True, file=buffer, force_terminal=True, color_system="truecolor")
    body: list[object] = []

    body.append(Text.assemble(("Command: ", "bold"), doc.command))
    body.append(Text.assemble(("Status: ", "bold"), doc.status))

    for section in doc.sections:
        if isinstance(section, KeyValueSection):
            table = Table(show_header=False)
            table.add_column("field")
            table.add_column("value")
            for row in section.rows:
                table.add_row(Text(row.label), Text(row.value))
            body.append(Panel(table, title=escape(section.title)))
        elif isinstance(section, StepsSection):
            body.append(Panel(Text("\n".join(section.steps)), title=escape(section.title)))

    for message in doc.epilogue_messages:
        if message:
            body.append(Text(message))

    console.print(Panel(Group(*body), title=escape(doc.title), subtitle=escape(doc.subtitle or "")))
    return buffer.getvalue()
