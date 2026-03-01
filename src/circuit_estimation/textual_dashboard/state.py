"""Shared data model and derived metrics for the Textual dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class DashboardState:
    """Minimal placeholder state; expanded in subsequent tasks."""

    raw_report: dict[str, Any]


def build_dashboard_state(report: dict[str, Any]) -> DashboardState:
    """Build dashboard state from the scorer report payload."""

    return DashboardState(raw_report=report)
