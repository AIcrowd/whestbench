"""CLI and convenience entrypoints for local scoring runs."""

from __future__ import annotations

import argparse
from typing import Any, Literal, overload

from .estimators import combined_estimator
from .reporting import render_agent_report, render_human_report
from .scoring import ContestParams, score_estimator_report


@overload
def run_default_score(profile: Literal[False] = False) -> float: ...


@overload
def run_default_score(profile: Literal[True]) -> tuple[float, list[dict[str, Any]]]: ...


def run_default_score(profile: bool = False) -> float | tuple[float, list[dict[str, Any]]]:
    """Backward-compatible numeric entrypoint for starter-kit callers."""
    report = run_default_report(profile=profile, detail="raw")
    score = float(report["results"]["final_score"])
    if profile:
        return score, list(report.get("profile_calls", []))
    return score


def run_default_report(*, profile: bool = False, detail: str = "raw") -> dict[str, Any]:
    """Run the default local score scenario and return structured report data."""
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
    """CLI entrypoint used by ``main.py``."""
    parser = argparse.ArgumentParser(description="Run local circuit-estimator scoring.")
    parser.add_argument(
        "--agent-mode",
        action="store_true",
        help="Emit pretty JSON only for machine consumers. Default output is the human dashboard.",
    )
    parser.add_argument(
        "--detail",
        choices=("raw", "full"),
        default="raw",
        help="Report detail level. Use `full` for extra derived metrics.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit per-call profiling diagnostics (wall, cpu, rss, peak_rss).",
    )
    parser.add_argument(
        "--show-diagnostic-plots",
        action="store_true",
        help="Include plot panes for budget/layer/profile diagnostics in human mode.",
    )
    args = parser.parse_args(argv)

    report = run_default_report(profile=args.profile, detail=args.detail)
    mode = "agent" if argsinternal agent config_mode else "human"
    report["mode"] = mode
    if mode == "agent":
        output = render_agent_report(report)
    else:
        output = render_human_report(report, show_diagnostic_plots=args.show_diagnostic_plots)
    print(output, end="" if output.endswith("\n") else "\n")
    return 0
