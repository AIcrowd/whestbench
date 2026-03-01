"""CLI and convenience entrypoints for local scoring runs."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Literal, overload

from .estimators import combined_estimator
from .reporting import render_agent_report, render_human_report
from .scoring import ContestParams, score_estimator_report


@overload
def run_default_score(profile: Literal[False] = False) -> float: ...


@overload
def run_default_score(profile: Literal[True]) -> tuple[float, list[dict[str, Any]]]: ...


def run_default_score(profile: bool = False) -> float | tuple[float, list[dict[str, Any]]]:
    """Run default scenario and return score-only compatibility output.

    When ``profile`` is true, this mirrors legacy behavior by returning
    ``(score, profile_calls)`` instead of just the numeric score.
    """
    report = run_default_report(profile=profile, detail="raw")
    score = float(report["results"]["final_score"])
    if profile:
        return score, list(report.get("profile_calls", []))
    return score


def run_default_report(*, profile: bool = False, detail: str = "raw") -> dict[str, Any]:
    """Run the default local evaluator scenario and return report payload."""
    return score_estimator_report(
        combined_estimator,
        n_circuits=10,
        n_samples=10000,
        contest_params=ContestParams(
            width=100,
            max_depth=30,
            budgets=[10, 100, 1000, 10000],
            time_tolerance=0.1,
        ),
        profile=profile,
        detail=detail,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse CLI flags, run default report, and route to JSON/Textual/static output."""
    parser = argparse.ArgumentParser(description="Run local circuit-estimator scoring.")
    parser.add_argument(
        "--agent-mode",
        action="store_true",
        help="Emit pretty JSON only for machine consumers.",
    )
    args = parser.parse_args(argv)

    mode = "agent" if argsinternal agent config_mode else "human"
    detail = "raw" if mode == "agent" else "full"
    profile = mode == "human"
    report = run_default_report(profile=profile, detail=detail)
    report["mode"] = mode
    if mode == "agent":
        output = render_agent_report(report)
        print(output, end="" if output.endswith("\n") else "\n")
        return 0

    if _supports_textual_dashboard():
        try:
            _launch_textual_dashboard(report)
            return 0
        except Exception as exc:
            print(
                f"Textual UI unavailable, using static dashboard fallback: {exc}",
                file=sys.stderr,
            )
    else:
        print("Textual UI unavailable, using static dashboard fallback.", file=sys.stderr)

    output = render_human_report(report, show_diagnostic_plots=True)
    print(output, end="" if output.endswith("\n") else "\n")
    return 0


def _supports_textual_dashboard() -> bool:
    """Return True when terminal capabilities are suitable for Textual UI."""

    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "").lower()
    if term in {"", "dumb"}:
        return False
    return True


def _launch_textual_dashboard(report: dict[str, Any]) -> None:
    """Launch the full-screen Textual dashboard."""

    from .textual_dashboard import DashboardApp

    app = DashboardApp(report=report)
    app.run()
