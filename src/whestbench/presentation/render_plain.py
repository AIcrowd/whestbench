from __future__ import annotations

from .models import (
    BudgetBreakdownSection,
    ChecklistSection,
    CommandPresentation,
    ErrorSection,
    KeyValueSection,
    RunErrorsSection,
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


def _render_budget_breakdown(section: BudgetBreakdownSection) -> list[str]:
    if not section.available:
        return [section.unavailable_message] if section.unavailable_message else []

    lines: list[str] = []
    if section.source_note:
        lines.append(section.source_note)
    if section.total_flops is not None:
        lines.append(f"Total FLOPs [flops_used]: {section.total_flops}")
    if section.tracked_time is not None:
        lines.append(f"Tracked Time [tracked_time_s]: {section.tracked_time}")
    if section.untracked_time is not None:
        lines.append(f"Untracked Time [untracked_time_s]: {section.untracked_time}")
    if section.namespace_rows:
        lines.append("namespace | total flops | % of section flops | mean flops / MLP | tracked time")
        lines.append(
            "--------- | ----------- | ------------------ | ---------------- | ------------"
        )
    for row in section.namespace_rows:
        lines.append(
            " | ".join(
                [
                    row.namespace,
                    row.total_flops,
                    row.percent_of_section_flops,
                    row.mean_flops_per_mlp,
                    row.tracked_time,
                ]
            )
        )
    footer_note = section.footer_note or "aggregated across all evaluated MLPs"
    if footer_note:
        lines.append(footer_note)
    return lines


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
            if section.subtitle:
                lines.append(section.subtitle)
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
        elif isinstance(section, RunErrorsSection):
            lines.append(section.summary)
            for entry in section.entries:
                lines.append(f"  MLP {entry.mlp_index} [{entry.code}]: {entry.message}")
                for detail_line in format_error_detail_lines(entry.details):
                    lines.append(f"    {detail_line}")
                if entry.traceback:
                    for tb_line in entry.traceback.rstrip("\n").splitlines():
                        lines.append(f"    {tb_line}")
            if section.footer:
                lines.append(section.footer)
        elif isinstance(section, BudgetBreakdownSection):
            lines.extend(_render_budget_breakdown(section))

    if doc.epilogue_messages:
        lines.append("")
        lines.extend(message for message in doc.epilogue_messages if message)

    return "\n".join(lines) + "\n"
