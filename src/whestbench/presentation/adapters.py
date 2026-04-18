from __future__ import annotations

from typing import Any

from .models import CommandPresentation, KeyValueRow, KeyValueSection, StepsSection

_JSON_OUTPUT_TIP = "Use --json for JSON output when calling from automated agents or UIs."
_DIAGNOSTIC_PLOTS_TIP = "Use --show-diagnostic-plots to include diagnostic plot panes."
_SMOKE_TEST_NEXT_STEPS = [
    "whest init ./my-estimator",
    "whest validate --estimator ./my-estimator/estimator.py",
    "whest run --estimator ./my-estimator/estimator.py --runner local",
    "whest package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz",
]


def _display_value(value: Any, *, fallback: str = "n/a") -> str:
    if value in {None, ""}:
        return fallback
    return str(value)


def _status_for_report(report: dict[str, Any]) -> str:
    results = report.get("results")
    if not isinstance(results, dict):
        return "success"
    per_mlp = results.get("per_mlp")
    if not isinstance(per_mlp, list):
        return "success"
    if any(isinstance(entry, dict) and entry.get("error") for entry in per_mlp):
        return "error"
    return "success"


def _base_sections(report: dict[str, Any]) -> list[KeyValueSection]:
    run_config = report.get("run_config")
    if not isinstance(run_config, dict):
        run_config = {}
    run_meta = report.get("run_meta")
    if not isinstance(run_meta, dict):
        run_meta = {}
    results = report.get("results")
    if not isinstance(results, dict):
        results = {}

    return [
        KeyValueSection(
            title="Run Context",
            rows=[
                KeyValueRow("MLPs", _display_value(run_config.get("n_mlps"))),
                KeyValueRow("Width", _display_value(run_config.get("width"))),
                KeyValueRow("Depth", _display_value(run_config.get("depth"))),
                KeyValueRow("FLOP Budget", _display_value(run_config.get("flop_budget"))),
                KeyValueRow("Duration(s)", _display_value(run_meta.get("run_duration_s"))),
                KeyValueRow(
                    "Wall Time Limit",
                    _display_value(run_config.get("wall_time_limit_s"), fallback="unlimited"),
                ),
                KeyValueRow(
                    "Untracked Time Limit",
                    _display_value(run_config.get("untracked_time_limit_s"), fallback="unlimited"),
                ),
            ],
        ),
        KeyValueSection(
            title="Final Score",
            rows=[
                KeyValueRow("Primary Score", _display_value(results.get("primary_score"))),
                KeyValueRow("Secondary Score", _display_value(results.get("secondary_score"))),
            ],
        ),
    ]


def build_run_presentation(report: dict[str, Any], *, debug: bool) -> CommandPresentation:
    del debug
    return CommandPresentation(
        command="run",
        status=_status_for_report(report),
        title="WhestBench Report",
        sections=_base_sections(report),
        epilogue_messages=[_JSON_OUTPUT_TIP, _DIAGNOSTIC_PLOTS_TIP],
    )


def build_smoke_test_presentation(report: dict[str, Any], *, debug: bool) -> CommandPresentation:
    base_doc = build_run_presentation(report, debug=debug)
    return CommandPresentation(
        command="smoke-test",
        status=base_doc.status,
        title=base_doc.title,
        subtitle=base_doc.subtitle,
        sections=[
            *base_doc.sections,
            StepsSection(
                title="Next Steps",
                steps=list(_SMOKE_TEST_NEXT_STEPS),
            ),
        ],
        epilogue_messages=list(base_doc.epilogue_messages),
    )
