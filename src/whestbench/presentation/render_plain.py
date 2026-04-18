from __future__ import annotations

from .models import (
    ChecklistSection,
    CommandPresentation,
    ErrorSection,
    KeyValueSection,
    StepItem,
    StepsSection,
    TableSection,
    format_error_detail_lines,
)


def _render_step(step: str | StepItem) -> list[str]:
    if isinstance(step, StepItem):
        return [step.purpose, step.command]
    return [step]


def _render_checklist_line(label: str, status: str, detail: str) -> str:
    rendered = f"[{status}] {label}"
    if detail:
        rendered = f"{rendered}: {detail}"
    return rendered


def _primary_error_section(doc: CommandPresentation) -> ErrorSection | None:
    for section in doc.sections:
        if isinstance(section, ErrorSection):
            return section
    return None


def render_plain_presentation(doc: CommandPresentation) -> str:
    primary_error = _primary_error_section(doc) if doc.status == "error" else None
    if primary_error is not None:
        title = doc.title
        if primary_error.message:
            title = f"{title}: {primary_error.message}"
        lines: list[str] = [title]
        if doc.subtitle:
            lines.append(doc.subtitle)
    else:
        lines = [doc.title, f"Command: {doc.command}", f"Status: {doc.status}"]
        if doc.subtitle:
            lines.append(doc.subtitle)

    for section in doc.sections:
        lines.append("")
        lines.append(section.title)
        if isinstance(section, KeyValueSection):
            for row in section.rows:
                lines.append(f"{row.label}: {row.value}")
        elif isinstance(section, TableSection):
            if section.columns:
                lines.append(" | ".join(section.columns))
                lines.append(" | ".join("-" * len(column) for column in section.columns))
            for row in section.rows:
                lines.append(" | ".join(row))
        elif isinstance(section, StepsSection):
            for step in section.steps:
                lines.extend(_render_step(step))
        elif isinstance(section, ChecklistSection):
            for item in section.items:
                lines.append(_render_checklist_line(item.label, item.status, item.detail))
        elif isinstance(section, ErrorSection):
            content = format_error_detail_lines(section.details)
            if section.traceback:
                content.append("Traceback:")
                content.extend(section.traceback.rstrip("\n").splitlines())
            if not content:
                lines.pop()
                lines.pop()
                continue
            lines.extend(content)

    if doc.epilogue_messages:
        lines.append("")
        lines.extend(message for message in doc.epilogue_messages if message)

    return "\n".join(lines) + "\n"
