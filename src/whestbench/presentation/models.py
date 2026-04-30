from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

PresentationStatus = Literal["success", "warning", "error"]
ChecklistStatus = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class KeyValueRow:
    label: str
    value: str


@dataclass(frozen=True)
class KeyValueSection:
    title: str
    rows: list[KeyValueRow]


@dataclass(frozen=True)
class TableSection:
    title: str
    columns: list[str]
    rows: list[list[str]]
    subtitle: str | None = None
    align_center: bool = False
    border_style: str | None = None


@dataclass(frozen=True)
class StepItem:
    purpose: str
    command: str


@dataclass(frozen=True)
class StepsSection:
    title: str
    steps: list[str | StepItem]


@dataclass(frozen=True)
class ChecklistItem:
    label: str
    status: ChecklistStatus
    detail: str


@dataclass(frozen=True)
class ChecklistSection:
    title: str
    items: list[ChecklistItem]


@dataclass(frozen=True)
class ErrorSection:
    title: str
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    traceback: str | None = None


def format_error_detail_lines(details: Mapping[str, Any]) -> list[str]:
    if not details:
        return []

    lines: list[str] = []
    handled: set[str] = set()

    def _has_content(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (str, bytes, bytearray)):
            return len(value) > 0
        if isinstance(value, (list, tuple, set, frozenset, dict)):
            return len(value) > 0
        return True

    for key, label in (
        ("expected_shape", "Expected shape"),
        ("got_shape", "Got shape"),
        ("hint", "Hint"),
    ):
        value = details.get(key)
        if _has_content(value):
            lines.append(f"{label}: {value}")
            handled.add(key)

    cause_hints = details.get("cause_hints")
    if isinstance(cause_hints, list):
        cause_lines = [str(item) for item in cause_hints if _has_content(item)]
        if cause_lines:
            lines.append("Possible causes:")
            lines.extend(f"  - {item}" for item in cause_lines)
            handled.add("cause_hints")

    for key in sorted(details):
        if key in handled:
            continue
        value = details[key]
        if _has_content(value):
            lines.append(f"{key}: {value}")

    return lines


@dataclass(frozen=True)
class RunErrorEntry:
    mlp_index: int
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    traceback: str | None = None


@dataclass(frozen=True)
class RunErrorsSection:
    title: str
    summary: str
    entries: list[RunErrorEntry]
    footer: str | None = None


@dataclass(frozen=True)
class BudgetBreakdownNamespaceRow:
    namespace: str
    total_flops: str
    percent_of_section_flops: str
    mean_flops_per_mlp: str
    tracked_time: str


@dataclass(frozen=True)
class BudgetBreakdownGauge:
    label: str
    bar: str
    overflow: bool
    percent_of_budget: str
    budget_label: str
    worst_mlp_percent: str | None = None


@dataclass(frozen=True)
class BudgetBreakdownOverBudgetRow:
    mlp_index: int
    flops_used: str
    percent_of_budget: str | None = None


@dataclass(frozen=True)
class BudgetBreakdownSection:
    title: str
    available: bool
    unavailable_message: str | None = None
    total_flops: str | None = None
    tracked_time: str | None = None
    untracked_time: str | None = None
    namespace_rows: list[BudgetBreakdownNamespaceRow] = field(default_factory=list)
    gauge: BudgetBreakdownGauge | None = None
    over_budget_rows: list[BudgetBreakdownOverBudgetRow] = field(default_factory=list)
    over_budget_summary: str | None = None
    over_budget_truncated_remainder: int | None = None
    source_note: str | None = None
    footer_note: str | None = None


@dataclass(frozen=True)
class CommandPresentation:
    command: str
    status: PresentationStatus
    title: str
    subtitle: str | None = None
    sections: Sequence[
        KeyValueSection
        | TableSection
        | StepsSection
        | ChecklistSection
        | ErrorSection
        | RunErrorsSection
        | BudgetBreakdownSection
    ] = field(default_factory=list)
    epilogue_messages: Sequence[str] = field(default_factory=list)
