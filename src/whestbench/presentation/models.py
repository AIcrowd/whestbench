from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

PresentationStatus = Literal["success", "warning", "error"]


@dataclass(frozen=True)
class KeyValueRow:
    label: str
    value: str


@dataclass(frozen=True)
class KeyValueSection:
    title: str
    rows: list[KeyValueRow]


@dataclass(frozen=True)
class StepItem:
    purpose: str
    command: str


@dataclass(frozen=True)
class StepsSection:
    title: str
    steps: list[str | StepItem]


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
class CommandPresentation:
    command: str
    status: PresentationStatus
    title: str
    subtitle: str | None = None
    sections: list[KeyValueSection | StepsSection | ErrorSection] = field(default_factory=list)
    epilogue_messages: list[str] = field(default_factory=list)
