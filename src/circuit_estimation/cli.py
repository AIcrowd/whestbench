"""CLI and convenience entrypoints for local scoring runs."""

from __future__ import annotations

import argparse
import json
from typing import Any, Literal, overload

from .estimators import combined_estimator
from .scoring import ContestParams, score_estimator


@overload
def run_default_score(profile: Literal[False] = False) -> float: ...


@overload
def run_default_score(profile: Literal[True]) -> tuple[float, list[dict[str, Any]]]: ...


def run_default_score(profile: bool = False) -> float | tuple[float, list[dict[str, Any]]]:
    """Run the default local score scenario, optionally collecting profile events."""
    events: list[dict[str, Any]] = []
    profiler = events.append if profile else None
    score = score_estimator(
        combined_estimator,
        n_circuits=10,
        n_samples=10000,
        contest_params=ContestParams(
            width=100,
            max_depth=30,
            budgets=[10, 100, 1000, 10000],
            time_tolerance=0.1,
        ),
        profiler=profiler,
    )
    if profile:
        return score, events
    return score


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used by ``main.py``."""
    parser = argparse.ArgumentParser(description="Run local circuit-estimator scoring.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Emit per-layer profiling diagnostics (time, cpu, rss, peak_rss).",
    )
    args = parser.parse_args(argv)

    if args.profile:
        score, events = run_default_score(profile=True)
        print(score)
        print(json.dumps(events))
    else:
        print(run_default_score())
    return 0
