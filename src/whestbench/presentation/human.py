from __future__ import annotations

import io
from typing import Literal

from rich.console import Console, Group, RenderableType
from rich.markup import escape
from rich.panel import Panel

HumanOutputFormat = Literal["rich", "plain"]


def create_console(
    output_format: HumanOutputFormat,
    *,
    file: io.StringIO | None = None,
    width: int | None = None,
    force_terminal: bool | None = None,
) -> Console:
    if output_format == "plain":
        return Console(
            record=True,
            file=file,
            width=width,
            force_terminal=False,
            color_system=None,
        )

    terminal = True if force_terminal is None else force_terminal
    return Console(
        record=True,
        file=file,
        width=width,
        force_terminal=terminal,
        color_system="truecolor" if terminal else None,
    )


def render_blocks(
    blocks: list[RenderableType],
    *,
    output_format: HumanOutputFormat = "rich",
    title: str | None = None,
    subtitle: str | None = None,
    width: int | None = None,
    force_terminal: bool | None = None,
) -> str:
    buffer = io.StringIO()
    console = create_console(
        output_format,
        file=buffer,
        width=width,
        force_terminal=force_terminal,
    )
    renderable: RenderableType = Group(*blocks)
    if title is not None or subtitle is not None:
        renderable = Panel(
            renderable,
            title=escape(title or ""),
            subtitle=escape(subtitle or ""),
        )
    console.print(renderable)
    if output_format == "plain":
        return console.export_text(styles=False)
    return buffer.getvalue()


def render_document(
    *,
    title: str,
    blocks: list[RenderableType],
    subtitle: str | None = None,
    output_format: HumanOutputFormat = "rich",
    width: int | None = None,
    force_terminal: bool | None = None,
) -> str:
    return render_blocks(
        blocks,
        output_format=output_format,
        title=title,
        subtitle=subtitle,
        width=width,
        force_terminal=force_terminal,
    )
