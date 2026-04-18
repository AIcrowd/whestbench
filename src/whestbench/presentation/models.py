from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

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
class CommandPresentation:
    command: str
    status: PresentationStatus
    title: str
    subtitle: str | None = None
    sections: list[KeyValueSection | StepsSection] = field(default_factory=list)
    epilogue_messages: list[str] = field(default_factory=list)
