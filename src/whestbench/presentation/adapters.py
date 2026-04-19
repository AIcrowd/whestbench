from __future__ import annotations

from typing import Any

from .breakdowns import (
    as_float,
    compute_gauge_state,
    fmt_flops,
    gauge_bar_fragment,
    select_top_over_budget,
)
from .models import (
    BudgetBreakdownGauge,
    BudgetBreakdownNamespaceRow,
    BudgetBreakdownOverBudgetRow,
    BudgetBreakdownSection,
    ChecklistItem,
    ChecklistSection,
    CommandPresentation,
    ErrorSection,
    KeyValueRow,
    KeyValueSection,
    RunErrorEntry,
    RunErrorsSection,
    StepItem,
    StepsSection,
    TableSection,
)

_JSON_OUTPUT_TIP = "Use --format json for JSON output when calling from automated agents or UIs."
_DIAGNOSTIC_PLOTS_TIP = "Use --show-diagnostic-plots to include diagnostic plot panes."
_SMOKE_TEST_NEXT_STEPS = [
    StepItem(
        purpose="Create starter files you can edit.",
        command="whest init ./my-estimator",
    ),
    StepItem(
        purpose="Validate an Estimator implementation.",
        command="whest validate --estimator ./my-estimator/estimator.py",
    ),
    StepItem(
        purpose="Run local evaluation with isolation.",
        command="whest run --estimator ./my-estimator/estimator.py --runner local",
    ),
    StepItem(
        purpose="Build submission artifacts for AIcrowd.",
        command="whest package --estimator ./my-estimator/estimator.py --output ./submission.tar.gz",
    ),
]


def _non_empty_messages(messages: list[str]) -> list[str]:
    return [message for message in messages if message]


def _display_value(value: Any, *, fallback: str = "n/a") -> str:
    if value in {None, ""}:
        return fallback
    return str(value)


def _display_bytes(value: Any, *, fallback: str = "n/a") -> str:
    if value in {None, ""}:
        return fallback
    try:
        return f"{float(value) / (1024**3):.1f} GB"
    except (TypeError, ValueError):
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


def _display_time_seconds(value: Any) -> str:
    return f"{as_float(value):.6f}s"


def _display_metric_value(value: Any, *, decimals: int = 8) -> str:
    if value in {None, ""}:
        return "n/a"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return str(value)


def _run_context_sections(report: dict[str, Any]) -> list[KeyValueSection]:
    run_config = report.get("run_config")
    if not isinstance(run_config, dict):
        run_config = {}
    run_meta = report.get("run_meta")
    if not isinstance(run_meta, dict):
        run_meta = {}
    host_meta = run_meta.get("host")
    if not isinstance(host_meta, dict):
        host_meta = {}
    return [
        KeyValueSection(
            title="Run Context",
            rows=[
                KeyValueRow(
                    "Estimator Class [estimator_class]",
                    _display_value(run_config.get("estimator_class")),
                ),
                KeyValueRow(
                    "Estimator Path [estimator_path]",
                    _display_value(run_config.get("estimator_path")),
                ),
                KeyValueRow(
                    "Started [run_started_at_utc]",
                    _display_value(run_meta.get("run_started_at_utc")),
                ),
                KeyValueRow(
                    "Finished [run_finished_at_utc]",
                    _display_value(run_meta.get("run_finished_at_utc")),
                ),
                KeyValueRow(
                    "Duration(s) [run_duration_s]", _display_value(run_meta.get("run_duration_s"))
                ),
                KeyValueRow("MLPs [n_mlps]", _display_value(run_config.get("n_mlps"))),
                KeyValueRow("Width [width]", _display_value(run_config.get("width"))),
                KeyValueRow("Depth [depth]", _display_value(run_config.get("depth"))),
                KeyValueRow("FLOP Budget [flop_budget]", fmt_flops(run_config.get("flop_budget"))),
                KeyValueRow(
                    "Wall Time Limit [wall_time_limit_s]",
                    _display_value(run_config.get("wall_time_limit_s"), fallback="unlimited"),
                ),
                KeyValueRow(
                    "Untracked Time Limit [untracked_time_limit_s]",
                    _display_value(run_config.get("untracked_time_limit_s"), fallback="unlimited"),
                ),
            ],
        ),
        KeyValueSection(
            title="Hardware & Runtime",
            rows=[
                KeyValueRow("Host [host.hostname]", _display_value(host_meta.get("hostname"))),
                KeyValueRow("OS [host.os]", _display_value(host_meta.get("os"))),
                KeyValueRow(
                    "Release [host.os_release]", _display_value(host_meta.get("os_release"))
                ),
                KeyValueRow("Platform [host.platform]", _display_value(host_meta.get("platform"))),
                KeyValueRow("Arch [host.machine]", _display_value(host_meta.get("machine"))),
                KeyValueRow("CPU [host.cpu_brand]", _display_value(host_meta.get("cpu_brand"))),
                KeyValueRow(
                    "CPU Cores (logical) [host.cpu_count_logical]",
                    _display_value(host_meta.get("cpu_count_logical")),
                ),
                KeyValueRow(
                    "CPU Cores (physical) [host.cpu_count_physical]",
                    _display_value(host_meta.get("cpu_count_physical")),
                ),
                KeyValueRow(
                    "RAM Total [host.ram_total_bytes]",
                    _display_bytes(host_meta.get("ram_total_bytes")),
                ),
                KeyValueRow(
                    "Python [host.python_version]", _display_value(host_meta.get("python_version"))
                ),
                KeyValueRow(
                    "NumPy [host.numpy_version]", _display_value(host_meta.get("numpy_version"))
                ),
            ],
        ),
    ]


def _score_section(report: dict[str, Any]) -> TableSection:
    results = report.get("results")
    if not isinstance(results, dict):
        results = {}
    rows = [
        ["Primary Score [primary_score]", _display_metric_value(results.get("primary_score"))],
        [
            "Secondary Score [secondary_score]",
            _display_metric_value(results.get("secondary_score")),
        ],
    ]
    per_mlp = results.get("per_mlp")
    if isinstance(per_mlp, list) and per_mlp:
        mlp_primaries = [
            as_float(entry.get("final_mse", 0.0)) for entry in per_mlp if isinstance(entry, dict)
        ]
        if mlp_primaries:
            rows.extend(
                [
                    ["Best MLP Score [best_mlp_score]", _display_metric_value(min(mlp_primaries))],
                    [
                        "Worst MLP Score [worst_mlp_score]",
                        _display_metric_value(max(mlp_primaries)),
                    ],
                ]
            )
    return TableSection(
        title="Final Score",
        columns=["metric", "value"],
        rows=rows,
        subtitle="lower MSE is better; primary score = mean across MLPs of final-layer MSE",
        align_center=True,
        border_style="bright_cyan",
    )


def _extract_run_error_entry(entry: dict[str, Any], *, debug: bool) -> RunErrorEntry:
    raw_error = entry.get("error")
    details: dict[str, Any] = {}
    if isinstance(raw_error, dict):
        message = str(raw_error.get("message") or "").strip() or "(no message)"
        raw_details = raw_error.get("details")
        if isinstance(raw_details, dict):
            details = raw_details
    elif raw_error is None:
        message = "(no message)"
    else:
        message = str(raw_error)
    traceback = entry.get("traceback") if debug else None
    return RunErrorEntry(
        mlp_index=int(entry.get("mlp_index", 0)),
        code=str(entry.get("error_code") or "UNKNOWN"),
        message=message,
        details=details,
        traceback=str(traceback) if traceback else None,
    )


def _run_errors_section(report: dict[str, Any], *, debug: bool) -> RunErrorsSection | None:
    results = report.get("results")
    if not isinstance(results, dict):
        return None
    per_mlp = results.get("per_mlp")
    if not isinstance(per_mlp, list):
        return None
    failures = [entry for entry in per_mlp if isinstance(entry, dict) and entry.get("error")]
    if not failures:
        return None
    return RunErrorsSection(
        title="Estimator Errors",
        summary=f"{len(failures)} of {len(per_mlp)} MLP(s) raised during predict.",
        entries=[_extract_run_error_entry(entry, debug=debug) for entry in failures],
        footer=(
            None
            if debug
            else "Rerun with --debug to include full tracebacks; --fail-fast to stop on first error."
        ),
    )


def _breakdown_section(
    report: dict[str, Any],
    *,
    breakdown_key: str,
    title: str,
) -> BudgetBreakdownSection | None:
    results = report.get("results")
    if not isinstance(results, dict):
        return None
    run_config = report.get("run_config")
    if not isinstance(run_config, dict):
        run_config = {}
    breakdowns = results.get("breakdowns")
    if not isinstance(breakdowns, dict):
        breakdowns = {}
    breakdown = breakdowns.get(breakdown_key)

    dataset = run_config.get("dataset")
    dataset_backed = isinstance(dataset, dict)
    if breakdown_key == "sampling" and not isinstance(breakdown, dict) and dataset_backed:
        return BudgetBreakdownSection(
            title=title,
            available=False,
            unavailable_message=(
                "Ground-truth sampling baseline is unavailable for this dataset. "
                "Recreate the dataset with a newer whestbench to compare against sampling."
            ),
        )
    if not isinstance(breakdown, dict):
        return None

    by_namespace = breakdown.get("by_namespace")
    if not isinstance(by_namespace, dict):
        by_namespace = {}
    n_mlps = int(run_config.get("n_mlps", 0) or 0)
    if n_mlps <= 0:
        n_mlps = 1

    total_flops = as_float(breakdown.get("flops_used", 0.0))
    if total_flops <= 0.0:
        total_flops = sum(
            as_float(bucket.get("flops_used", 0.0))
            for bucket in by_namespace.values()
            if isinstance(bucket, dict)
        )

    namespace_rows = []
    for namespace, bucket in sorted(
        (
            (namespace, bucket)
            for namespace, bucket in by_namespace.items()
            if isinstance(bucket, dict)
        ),
        key=lambda item: as_float(item[1].get("flops_used", 0.0)),
        reverse=True,
    ):
        flops_used = as_float(bucket.get("flops_used", 0.0))
        namespace_rows.append(
            BudgetBreakdownNamespaceRow(
                namespace="(unlabeled)"
                if namespace in {None, "", "null", "None"}
                else str(namespace),
                total_flops=fmt_flops(flops_used),
                percent_of_section_flops=(
                    f"{(flops_used / total_flops * 100.0):.1f}%" if total_flops > 0 else "0.0%"
                ),
                mean_flops_per_mlp=fmt_flops(flops_used / n_mlps if n_mlps > 0 else 0.0),
                tracked_time=_display_time_seconds(bucket.get("tracked_time_s", 0.0)),
            )
        )

    gauge = None
    over_budget_rows: list[BudgetBreakdownOverBudgetRow] = []
    over_budget_summary: str | None = None
    over_budget_truncated_remainder: int | None = None
    if breakdown_key == "estimator":
        state = compute_gauge_state(report)
        gauge = BudgetBreakdownGauge(
            label="Estimator FLOPs",
            bar=gauge_bar_fragment(state.mean_utilization),
            overflow=state.state_name == "catastrophic",
            percent_of_budget=f"{int(state.mean_utilization * 100)}%",
            budget_label=fmt_flops(state.flop_budget),
            worst_mlp_percent=(
                f"{state.worst_mlp_pct}%" if state.worst_mlp_pct is not None else None
            ),
        )
        selection = select_top_over_budget(report)
        over_budget_rows = [
            BudgetBreakdownOverBudgetRow(
                mlp_index=row.mlp_index,
                flops_used=fmt_flops(row.flops_used),
                percent_of_budget=(
                    f"{row.pct_of_budget}%" if row.pct_of_budget is not None else None
                ),
            )
            for row in selection.rows
        ]
        if selection.is_truncated:
            over_budget_truncated_remainder = selection.busted_count - len(selection.rows)
        if selection.busted_count > 0:
            over_budget_summary = (
                f"All {selection.n_mlps} MLPs exceeded the per-MLP FLOP cap — predictions entirely zeroed"
                if selection.is_all_busted
                else f"{selection.busted_count} of {selection.n_mlps} MLPs exceeded the per-MLP FLOP cap"
            )

    return BudgetBreakdownSection(
        title=title,
        available=True,
        total_flops=fmt_flops(total_flops),
        tracked_time=_display_time_seconds(breakdown.get("tracked_time_s", 0.0)),
        untracked_time=_display_time_seconds(breakdown.get("untracked_time_s", 0.0)),
        namespace_rows=namespace_rows,
        gauge=gauge,
        over_budget_rows=over_budget_rows,
        over_budget_summary=over_budget_summary,
        over_budget_truncated_remainder=over_budget_truncated_remainder,
        source_note=(
            "restored from dataset metadata for the MLPs used in this run."
            if breakdown_key == "sampling" and dataset_backed
            else None
        ),
        footer_note="aggregated across all evaluated MLPs",
    )


def build_run_presentation(report: dict[str, Any], *, debug: bool) -> CommandPresentation:
    sections: list[Any] = list(_run_context_sections(report))
    errors = _run_errors_section(report, debug=debug)
    if errors is not None:
        sections.append(errors)
    for breakdown_key, title in (
        ("sampling", "Sampling Budget Breakdown (Ground Truth)"),
        ("estimator", "Estimator Budget Breakdown"),
    ):
        section = _breakdown_section(report, breakdown_key=breakdown_key, title=title)
        if section is not None:
            sections.append(section)
    sections.append(_score_section(report))
    return CommandPresentation(
        command="run",
        status=_status_for_report(report),
        title="WhestBench Report",
        sections=sections,
        epilogue_messages=_non_empty_messages([_JSON_OUTPUT_TIP, _DIAGNOSTIC_PLOTS_TIP]),
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
        epilogue_messages=_non_empty_messages(list(base_doc.epilogue_messages)),
    )


def build_validate_presentation(payload: dict[str, Any]) -> CommandPresentation:
    raw_checks = payload.get("checks")
    checks = raw_checks if isinstance(raw_checks, list) else []
    return CommandPresentation(
        command="validate",
        status="success",
        title="Validation",
        sections=[
            ChecklistSection(
                title="Checks",
                items=[
                    ChecklistItem(
                        label=str(item.get("name", "")),
                        status=str(item.get("status", "ok")),
                        detail=str(item.get("detail", "")),
                    )
                    for item in checks
                    if isinstance(item, dict)
                ],
            )
        ],
    )


def build_init_presentation(payload: dict[str, Any]) -> CommandPresentation:
    raw_created = payload.get("created")
    created = [str(path) for path in raw_created] if isinstance(raw_created, list) else []

    if created:
        sections: list[KeyValueSection | StepsSection] = [
            StepsSection(title="Created Files", steps=created)
        ]
    else:
        sections = [
            KeyValueSection(
                title="Status",
                rows=[KeyValueRow("Message", "Starter files already exist; nothing created.")],
            )
        ]

    return CommandPresentation(
        command="init",
        status="success",
        title="Starter Files",
        sections=sections,
    )


def build_create_dataset_presentation(payload: dict[str, Any]) -> CommandPresentation:
    return CommandPresentation(
        command="create-dataset",
        status="success",
        title="Dataset Created",
        sections=[
            KeyValueSection(
                title="Dataset",
                rows=[KeyValueRow("Path", _display_value(payload.get("path")))],
            )
        ],
    )


def build_package_presentation(payload: dict[str, Any]) -> CommandPresentation:
    return CommandPresentation(
        command="package",
        status="success",
        title="Packaged Submission",
        sections=[
            KeyValueSection(
                title="Artifact",
                rows=[KeyValueRow("Path", _display_value(payload.get("artifact_path")))],
            )
        ],
    )


def build_visualizer_ready_presentation(payload: dict[str, Any]) -> CommandPresentation:
    host = str(payload.get("host") or "localhost")
    port = payload.get("port")
    port_text = str(port) if port not in {None, ""} else "5173"
    url = str(payload.get("url") or "")
    if not url:
        browser_host = "localhost" if host == "0.0.0.0" else host
        url = f"http://{browser_host}:{port_text}/"

    return CommandPresentation(
        command="visualizer",
        status="success",
        title="WhestBench Explorer",
        sections=[
            KeyValueSection(
                title="Ready",
                rows=[
                    KeyValueRow("URL", url),
                    KeyValueRow("Host", host),
                    KeyValueRow("Port", port_text),
                ],
            )
        ],
        epilogue_messages=_non_empty_messages(
            [
                "Browser auto-open disabled." if payload.get("no_open") else "",
                (
                    "Dependencies were installed with npm ci before launch."
                    if payload.get("ran_npm_ci")
                    else ""
                ),
            ]
        ),
    )


def build_visualizer_error_presentation(
    title: str,
    code: str,
    message: str,
    *,
    details: str | None = None,
    next_steps: list[str] | None = None,
) -> CommandPresentation:
    sections: list[ErrorSection | StepsSection] = [
        ErrorSection(
            title="Failure Details",
            code=code,
            message=message,
            details={"stderr": details} if details else {},
        )
    ]
    if next_steps:
        sections.append(StepsSection(title="Next Steps", steps=list(next_steps)))

    return CommandPresentation(
        command="visualizer",
        status="error",
        title=title,
        sections=sections,
    )


def build_profile_presentation(payload: dict[str, Any]) -> CommandPresentation:
    hardware = payload.get("hardware")
    correctness = payload.get("correctness")
    timing = payload.get("timing")
    verbose = bool(payload.get("verbose"))

    sections: list[KeyValueSection | TableSection] = []

    hardware_rows: list[KeyValueRow] = []
    if isinstance(hardware, dict):
        for key, label in (
            ("os", "OS"),
            ("machine", "Architecture"),
            ("cpu_count_physical", "Physical Cores"),
            ("cpu_count_logical", "Logical Cores"),
            ("ram_total_bytes", "RAM"),
            ("python_version", "Python"),
            ("numpy_version", "NumPy"),
        ):
            value = hardware.get(key)
            if value is None:
                continue
            if key == "ram_total_bytes":
                value = f"{float(value) / (1024**3):.1f} GB"
            hardware_rows.append(KeyValueRow(label, str(value)))
    if hardware_rows:
        sections.append(KeyValueSection(title="Hardware", rows=hardware_rows))

    correctness_rows: list[KeyValueRow] = []
    passed_any = False
    if isinstance(correctness, list):
        for item in correctness:
            if not isinstance(item, dict):
                continue
            passed = bool(item.get("passed"))
            passed_any = passed_any or passed
            correctness_rows.append(
                KeyValueRow(
                    str(item.get("backend", "")),
                    "PASS" if passed else str(item.get("error") or "FAIL"),
                )
            )
    if correctness_rows:
        sections.append(KeyValueSection(title="Correctness", rows=correctness_rows))

    detail_rows: list[list[str]] = []
    if isinstance(timing, list):
        for row in timing:
            if not isinstance(row, dict):
                continue
            detail_rows.append(
                [
                    str(row.get("backend", "")),
                    str(row.get("dims", "")),
                    str(row.get("run_mlp", "")),
                    str(row.get("sample_layer_statistics", "")),
                ]
            )
    if detail_rows:
        sections.append(
            TableSection(
                title="Detail",
                columns=["Backend", "Dims", "run_mlp", "sample_layer_statistics"],
                rows=detail_rows,
            )
        )

    return CommandPresentation(
        command="profile-simulation",
        status="success" if passed_any else "error",
        title="Simulation Profile",
        sections=sections,
        epilogue_messages=_non_empty_messages(
            [
                (
                    "Use --verbose for full timing tables with raw times"
                    if not verbose and passed_any
                    else ""
                ),
                "Use --verbose for error details." if not verbose and not passed_any else "",
            ]
        ),
    )


def build_error_presentation(
    payload: dict[str, Any],
    *,
    debug: bool,
    show_inprocess_hint: bool,
) -> CommandPresentation:
    error = payload.get("error")
    if not isinstance(error, dict):
        error = {}

    stage = str(error.get("stage") or "scoring")
    code = str(error.get("code") or "SCORING_RUNTIME_ERROR")
    message = str(error.get("message") or "Unknown error")
    details = error.get("details")
    traceback_text = error.get("traceback")

    section = ErrorSection(
        title="Failure Details",
        code=code,
        message=message,
        details=dict(details) if isinstance(details, dict) else {},
        traceback=str(traceback_text) if debug and traceback_text else None,
    )

    return CommandPresentation(
        command=stage,
        status="error",
        title=f"Error [{stage}:{code}]",
        sections=[section],
        epilogue_messages=_non_empty_messages(
            [
                "Use --debug to include a traceback." if not debug else "",
                (
                    "Tip: For estimator-level tracebacks, rerun with --runner local --debug."
                    if show_inprocess_hint
                    else ""
                ),
            ]
        ),
    )
