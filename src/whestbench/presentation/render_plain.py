from __future__ import annotations

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


def render_plain_presentation(doc: CommandPresentation) -> str:
    lines: list[str] = [doc.title]
    if doc.subtitle:
        lines.append(doc.subtitle)

    for section in doc.sections:
        lines.append("")
        lines.append(section.title)
        if isinstance(section, KeyValueSection):
            for row in section.rows:
                lines.append(f"{row.label}: {row.value}")
        elif isinstance(section, MetricsSection):
            for label, value in section.metrics:
                lines.append(f"{label}: {value}")
        elif isinstance(section, TableSection):
            lines.append(" | ".join(section.columns))
            for row in section.rows:
                lines.append(" | ".join(row))
        elif isinstance(section, ChecklistSection):
            for item in section.items:
                lines.append(f"{item.status.upper()} {item.label}: {item.detail}")
        elif isinstance(section, StepsSection):
            for step in section.steps:
                lines.append(step)
        elif isinstance(section, TextSection):
            lines.append(section.body)
        elif isinstance(section, ErrorSection):
            lines.append(f"{section.code}: {section.message}")
            for key, value in section.details.items():
                lines.append(f"{key}: {value}")
            if section.traceback:
                lines.append(section.traceback)

    if doc.epilogue_messages:
        lines.append("")
        lines.extend(message for message in doc.epilogue_messages if message)

    return "\n".join(lines) + "\n"
