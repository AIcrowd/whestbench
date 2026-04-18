from __future__ import annotations

from .models import CommandPresentation, KeyValueSection, StepItem, StepsSection


def _render_step(step: str | StepItem) -> list[str]:
    if isinstance(step, StepItem):
        return [step.purpose, step.command]
    return [step]


def render_plain_presentation(doc: CommandPresentation) -> str:
    lines: list[str] = [doc.title, f"Command: {doc.command}", f"Status: {doc.status}"]
    if doc.subtitle:
        lines.append(doc.subtitle)

    for section in doc.sections:
        lines.append("")
        lines.append(section.title)
        if isinstance(section, KeyValueSection):
            for row in section.rows:
                lines.append(f"{row.label}: {row.value}")
        elif isinstance(section, StepsSection):
            for step in section.steps:
                lines.extend(_render_step(step))

    if doc.epilogue_messages:
        lines.append("")
        lines.extend(message for message in doc.epilogue_messages if message)

    return "\n".join(lines) + "\n"
