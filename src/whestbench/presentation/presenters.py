from __future__ import annotations

from rich.console import RenderableType
from rich.text import Text

from .blocks import build_section_renderables
from .human import HumanOutputFormat, render_document
from .models import CommandPresentation, ErrorSection


def _primary_error_section(doc: CommandPresentation) -> ErrorSection | None:
    for section in doc.sections:
        if isinstance(section, ErrorSection):
            return section
    return None


def build_presentation_blocks(
    doc: CommandPresentation,
    *,
    include_doc_meta: bool = True,
    include_epilogues: bool = True,
) -> list[RenderableType]:
    blocks: list[RenderableType] = []
    primary_error = _primary_error_section(doc) if doc.status == "error" else None

    if include_doc_meta:
        if primary_error is None:
            blocks.append(Text.assemble(("Command: ", "bold"), doc.command))
            blocks.append(Text.assemble(("Status: ", "bold"), doc.status))

    for section in doc.sections:
        blocks.extend(build_section_renderables(section))

    if include_epilogues:
        blocks.extend(Text(message) for message in doc.epilogue_messages if message)

    return blocks


def render_command_presentation(
    doc: CommandPresentation,
    *,
    output_format: HumanOutputFormat,
    force_terminal: bool = True,
    include_doc_meta: bool = True,
    include_epilogues: bool = True,
    width: int | None = None,
) -> str:
    primary_error = _primary_error_section(doc) if doc.status == "error" else None
    title = doc.title
    if include_doc_meta and primary_error is not None and primary_error.message:
        title = f"{title}: {primary_error.message}"

    return render_document(
        title=title,
        subtitle=doc.subtitle,
        blocks=build_presentation_blocks(
            doc,
            include_doc_meta=include_doc_meta,
            include_epilogues=include_epilogues,
        ),
        output_format=output_format,
        width=width,
        force_terminal=force_terminal,
    )
