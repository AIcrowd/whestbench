from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

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
class StepsSection:
    title: str
    steps: list[str]


@dataclass(frozen=True)
class TextSection:
    title: str
    body: str


@dataclass(frozen=True)
class MetricsSection:
    title: str
    metrics: list[tuple[str, str]]


@dataclass(frozen=True)
class TableSection:
    title: str
    columns: list[str]
    rows: list[list[str]]


@dataclass(frozen=True)
class ChecklistItem:
    label: str
    status: Literal["ok", "warn", "fail"]
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
    details: dict[str, object] = field(default_factory=dict)
    traceback: str | None = None


PresentationSection = Union[
    KeyValueSection,
    StepsSection,
    TextSection,
    MetricsSection,
    TableSection,
    ChecklistSection,
    ErrorSection,
]


@dataclass(frozen=True)
class CommandPresentation:
    command: str
    status: PresentationStatus
    title: str
    subtitle: str | None = None
    sections: list[PresentationSection] = field(default_factory=list)
    epilogue_messages: list[str] = field(default_factory=list)
