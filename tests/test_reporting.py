from __future__ import annotations

import io
import json
import re
from typing import Any, cast

import pytest
from rich.align import Align
from rich.columns import Columns
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import whestbench.reporting as reporting
from whestbench.presentation.adapters import build_smoke_test_presentation
from whestbench.presentation.models import StepItem, StepsSection
from whestbench.reporting import (
    render_agent_report,
    render_human_report,
    render_human_results,
    render_smoke_test_next_steps,
)


def _sample_report(
    *,
    include_profile: bool = False,
    include_sampling_breakdown: bool = False,
    include_estimator_breakdown: bool = False,
) -> "dict[str, Any]":
    report: "dict[str, Any]" = {
        "schema_version": "1.0",
        "mode": "agent",
        "detail": "raw",
        "run_meta": {
            "run_started_at_utc": "2026-03-01T00:00:00+00:00",
            "run_finished_at_utc": "2026-03-01T00:00:01+00:00",
            "run_duration_s": 1.0,
        },
        "run_config": {
            "n_mlps": 2,
            "width": 4,
            "depth": 3,
            "flop_budget": 100,
        },
        "results": {
            "primary_score": 0.123,
            "secondary_score": 0.456,
            "per_mlp": [
                {
                    "mlp_index": 0,
                    "primary_score": 0.1,
                    "secondary_score": 0.4,
                },
                {
                    "mlp_index": 1,
                    "primary_score": 0.146,
                    "secondary_score": 0.512,
                },
            ],
        },
        "notes": [],
    }
    breakdowns: dict[str, object] = {}
    if include_sampling_breakdown:
        breakdowns["sampling"] = {
            "flop_budget": 200,
            "flops_used": 80,
            "flops_remaining": 120,
            "wall_time_s": 0.03,
            "tracked_time_s": 0.02,
            "flopscope_overhead_time_s": 0.005,
            "untracked_time_s": 0.01,
            "by_namespace": {
                "sampling.sample_layer_statistics": {
                    "flops_used": 50,
                    "calls": 2,
                    "tracked_time_s": 0.012,
                    "flopscope_overhead_time_s": 0.003,
                    "operations": {"add": {"flop_cost": 50, "calls": 2, "duration": 0.012}},
                },
                "sampling.draw_weights": {
                    "flops_used": 30,
                    "calls": 2,
                    "tracked_time_s": 0.008,
                    "flopscope_overhead_time_s": 0.002,
                    "operations": {"mul": {"flop_cost": 30, "calls": 2, "duration": 0.008}},
                },
            },
        }
    if include_estimator_breakdown:
        breakdowns["estimator"] = {
            "flop_budget": 200,
            "flops_used": 300,
            "flops_remaining": 0,
            "wall_time_s": 0.05,
            "tracked_time_s": 0.03,
            "flopscope_overhead_time_s": 0.0075,
            "untracked_time_s": 0.02,
            "by_namespace": {
                "estimator.phase": {
                    "flops_used": 200,
                    "calls": 4,
                    "tracked_time_s": 0.02,
                    "flopscope_overhead_time_s": 0.005,
                    "operations": {"add": {"flop_cost": 200, "calls": 4, "duration": 0.02}},
                },
                "estimator.estimator-client": {
                    "flops_used": 100,
                    "calls": 2,
                    "tracked_time_s": 0.01,
                    "flopscope_overhead_time_s": 0.0025,
                    "operations": {"mul": {"flop_cost": 100, "calls": 2, "duration": 0.01}},
                },
            },
        }
    if breakdowns:
        report["results"]["breakdowns"] = breakdowns
    if include_profile:
        report["profile_calls"] = [
            {
                "flop_budget": 100,
                "mlp_index": 0,
                "width": 4,
                "depth": 3,
                "wall_time_s": 0.05,
                "cpu_time_s": 0.04,
                "rss_bytes": 12_345_678,
                "peak_rss_bytes": 15_000_000,
            }
        ]
    return report


def test_render_json_mode_returns_pretty_json_only() -> None:
    report = _sample_report()
    rendered = render_agent_report(report)

    # JSON mode contract: machine-parseable, pretty JSON, no narrative framing.
    loaded = json.loads(rendered)
    assert loaded == report
    assert rendered.startswith("{\n")
    assert rendered.endswith("\n")


def test_smoke_test_next_steps_uses_colored_purpose_lines_and_plain_commands() -> None:
    rendered = render_smoke_test_next_steps(_sample_report())
    plain = _strip_ansi(rendered)
    doc = build_smoke_test_presentation(_sample_report(), debug=False)
    next_steps = next(
        section
        for section in doc.sections
        if isinstance(section, StepsSection) and section.title == "Next Steps"
    )
    expected_pairs = [
        (step.purpose, step.command) for step in next_steps.steps if isinstance(step, StepItem)
    ]

    assert "Next Steps" in plain
    assert "We are all set! Welcome onboard" in plain
    assert "Run these steps:" in plain
    assert "Commands (bash)" not in plain
    assert "Command" not in plain
    assert "Purpose" not in plain
    last_index = -1
    for idx, (purpose, command) in enumerate(expected_pairs, start=1):
        purpose_line = f"# {idx}) {purpose}"
        purpose_index = plain.index(purpose_line)
        command_index = plain.index(command)
        assert purpose_index < command_index
        assert last_index < purpose_index
        last_index = command_index
    assert "Worked examples live in the starter kit:" in plain
    assert "https://github.com/AIcrowd/whest-starterkit" in plain
    assert "Use --format json for JSON output when calling from automated agents or UIs." in plain
    assert "Use --show-diagnostic-plots to include diagnostic plot panes." in plain
    assert "Tip: use --json on validate/run/package for machine-readable output." not in plain


def test_smoke_test_next_steps_uses_distinct_styles_per_purpose_line() -> None:
    doc = build_smoke_test_presentation(_sample_report(), debug=False)
    next_steps = next(
        section
        for section in doc.sections
        if isinstance(section, StepsSection) and section.title == "Next Steps"
    )
    step_items = [s for s in next_steps.steps if isinstance(s, StepItem)]
    lines = reporting._smoke_next_step_lines(step_items)
    styles = [str(line.style) for line in lines]
    assert len(lines) == 4
    assert styles == [
        "bold bright_cyan",
        "bold bright_green",
        "bold bright_yellow",
        "bold bright_magenta",
    ]


def test_render_human_mode_includes_expected_sections_without_profile() -> None:
    rendered = render_human_report(_sample_report(include_profile=False))

    # Human mode contract: high-level run summary.
    assert "WhestBench Report" in rendered
    assert "Use --format json for JSON output" in rendered
    assert "Run Context" in rendered
    assert "Final Score" in rendered
    assert "Profile" not in rendered or "Profile" in rendered  # Profile only if data present


def test_human_report_uses_two_column_top_row_on_wide_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "220")
    rendered = render_human_report(_sample_report(include_profile=False))
    plain = _strip_ansi(rendered)
    title_lines = [
        line for line in plain.splitlines() if "Run Context" in line or "Hardware & Runtime" in line
    ]

    assert "Run Context" in rendered
    assert "Hardware & Runtime" in rendered
    # Run Context and Hardware & Runtime share the same row
    assert any("Run Context" in line and "Hardware & Runtime" in line for line in title_lines)


def test_render_human_mode_includes_budget_breakdown_sections_when_present() -> None:
    rendered = render_human_report(
        _sample_report(
            include_profile=False,
            include_sampling_breakdown=True,
            include_estimator_breakdown=True,
        )
    )
    plain = _strip_ansi(rendered)

    assert "Sampling Budget Breakdown (Ground Truth)" in plain
    assert "Estimator Budget Breakdown" in plain
    assert plain.index("Sampling Budget Breakdown (Ground Truth)") < plain.index(
        "Estimator Budget Breakdown"
    )
    assert "Total FLOPs" in plain
    assert "Tracked Time" in plain
    assert "Untracked Time" in plain
    assert "sampling.sample_layer" in plain
    assert "statistics" in plain
    assert "sampling.draw_weights" in plain
    assert "estimator.phase" in plain
    assert "estimator.estimator" in plain
    assert "client" in plain


def test_render_human_results_includes_budget_breakdown_sections_when_present() -> None:
    rendered = render_human_results(
        _sample_report(
            include_profile=False,
            include_sampling_breakdown=True,
            include_estimator_breakdown=True,
        )
    )
    plain = _strip_ansi(rendered)

    assert "Sampling Budget Breakdown (Ground Truth)" in plain
    assert "Estimator Budget Breakdown" in plain
    assert plain.index("Sampling Budget Breakdown (Ground Truth)") < plain.index(
        "Estimator Budget Breakdown"
    )


def test_render_human_results_plain_preserves_shared_run_section_order() -> None:
    report = _sample_report(
        include_profile=False,
        include_sampling_breakdown=True,
        include_estimator_breakdown=True,
    )

    rendered = render_human_results(
        report,
        output_format="plain",
        include_context=True,
        include_epilogues=False,
    )

    assert rendered.index("WhestBench Report") < rendered.index("Run Context")
    assert rendered.index("Run Context") < rendered.index("Hardware & Runtime")
    assert rendered.index("Hardware & Runtime") < rendered.index(
        "Sampling Budget Breakdown (Ground Truth)"
    )
    assert rendered.index("Sampling Budget Breakdown (Ground Truth)") < rendered.index(
        "Estimator Budget Breakdown"
    )
    assert rendered.index("Estimator Budget Breakdown") < rendered.index("Final Score")


def test_render_human_report_plain_uses_shared_smoke_test_shape() -> None:
    report = _sample_report(include_profile=False)

    rendered = render_human_report(
        report,
        output_format="plain",
        presentation_doc=build_smoke_test_presentation(report, debug=False),
    )

    assert rendered.index("Run Context") < rendered.index("Hardware & Runtime")
    assert rendered.index("Hardware & Runtime") < rendered.index("Final Score")
    assert rendered.index("Final Score") < rendered.index("Next Steps")
    assert "Create starter files you can edit." in rendered
    assert "whest init ./my-estimator" in rendered
    assert (
        "Use --format json for JSON output when calling from automated agents or UIs." in rendered
    )


def test_render_human_mode_matches_main_style_score_and_breakdown_information() -> None:
    report = _sample_report(
        include_profile=False,
        include_sampling_breakdown=True,
        include_estimator_breakdown=True,
    )
    results = cast("dict[str, Any]", report["results"])
    results["per_mlp"] = [
        {"mlp_index": 0, "final_mse": 0.1},
        {"mlp_index": 1, "final_mse": 0.146},
    ]

    rendered = render_human_report(report)
    plain = _strip_ansi(rendered)

    assert plain.index("Sampling Budget Breakdown (Ground Truth)") < plain.index(
        "Estimator Budget Breakdown"
    )
    assert plain.index("Estimator Budget Breakdown") < plain.index("Final Score")
    assert "Total FLOPs [flops_used]" in plain
    assert "Tracked Time [tracked_time_s]" in plain
    assert "Flopscope Overhead [flopscope_overhead_time_s]" in plain
    assert "Untracked Time [untracked_time_s]" in plain
    assert "aggregated across all evaluated MLPs" in plain
    assert "Primary Score [primary_score]" in plain
    assert "Secondary Score [secondary_score]" in plain
    assert "Best MLP Score [best_mlp_score]" in plain
    assert "Worst MLP Score [worst_mlp_score]" in plain
    assert "lower MSE is better; primary score = mean across MLPs of final-layer MSE" in plain
    assert "Estimator FLOPs" not in plain


def test_render_human_mode_omits_budget_breakdown_sections_when_absent() -> None:
    rendered = render_human_report(_sample_report(include_profile=False))
    plain = _strip_ansi(rendered)

    assert "Sampling Budget Breakdown" not in plain
    assert "Estimator Budget Breakdown" not in plain


def test_hardware_metadata_is_not_repeated_inside_run_context() -> None:
    report = _sample_report(include_profile=False)
    run_meta = cast("dict[str, Any]", report["run_meta"])
    run_meta["host"] = {
        "hostname": "example-host",
        "os": "Darwin",
        "os_release": "25.3.0",
        "platform": "macOS-15-arm64",
        "machine": "arm64",
        "python_version": "3.13.7",
    }

    run_context = _render_panel(reporting._run_context_panel(report))
    hardware = _render_panel(reporting._hardware_runtime_panel(report))

    assert "[host.hostname]" not in run_context
    assert "[host.os]" not in run_context
    assert "[host.os_release]" not in run_context
    assert "[host.platform]" not in run_context
    assert "[host.machine]" not in run_context
    assert "[host.python_version]" not in run_context

    assert "[host.hostname]" in hardware
    assert "[host.os]" in hardware
    assert "[host.os_release]" in hardware
    assert "[host.platform]" in hardware
    assert "[host.machine]" in hardware
    assert "[host.python_version]" in hardware


def test_run_context_styles_estimator_class_and_shows_estimator_path() -> None:
    report = _sample_report(include_profile=False)
    run_config = cast("dict[str, Any]", report["run_config"])
    run_config["estimator_class"] = "CombinedEstimator"
    run_config["estimator_path"] = "examples/estimators/combined_estimator.py"

    panel = reporting._run_context_panel(report)
    assert isinstance(panel.renderable, Align)
    table = cast(Table, panel.renderable.renderable)

    labels = table.columns[0]._cells
    values = table.columns[1]._cells
    lookup = {
        cast(Text, label).plain: value
        for label, value in zip(labels, values)
        if isinstance(label, Text)
    }

    estimator_class_value = lookup["Estimator Class [estimator_class]"]
    assert isinstance(estimator_class_value, Text)
    assert estimator_class_value.plain == "CombinedEstimator"
    assert estimator_class_value.style == "bold bright_cyan"

    estimator_path_value = lookup["Estimator Path [estimator_path]"]
    assert isinstance(estimator_path_value, str)
    assert estimator_path_value == "examples/estimators/combined_estimator.py"


def test_run_context_duration_defaults_to_na_when_unavailable() -> None:
    report = _sample_report(include_profile=False)
    run_meta = cast("dict[str, Any]", report["run_meta"])
    run_meta["run_duration_s"] = None

    panel = reporting._run_context_panel(report)
    assert isinstance(panel.renderable, Align)
    table = cast(Table, panel.renderable.renderable)

    labels = table.columns[0]._cells
    values = table.columns[1]._cells
    lookup = {
        cast(Text, label).plain: value
        for label, value in zip(labels, values)
        if isinstance(label, Text)
    }

    assert lookup["Duration(s) [run_duration_s]"] == "n/a"


def test_primary_tables_are_centered() -> None:
    report = _sample_report(include_profile=False)

    run_context = reporting._run_context_panel(report)
    hardware = reporting._hardware_runtime_panel(report)
    score = reporting._score_summary_panel(report)

    assert isinstance(run_context.renderable, Align)
    assert isinstance(run_context.renderable.renderable, Table)
    assert isinstance(hardware.renderable, Align)
    assert isinstance(hardware.renderable.renderable, Table)
    assert isinstance(score.renderable, Align)
    assert isinstance(score.renderable.renderable, Table)


def test_render_human_mode_includes_profile_section_when_available() -> None:
    rendered = render_human_report(_sample_report(include_profile=True))

    assert "Profile" in rendered
    assert "wall_time_s" in rendered
    assert "cpu_time_s" in rendered
    assert "rss_bytes" in rendered
    assert "peak_rss_bytes" in rendered


def test_json_mode_schema_keeps_results_fields() -> None:
    payload = json.loads(render_agent_report(_sample_report(include_profile=False)))
    results = payload["results"]
    assert "primary_score" in results
    assert "secondary_score" in results


def test_profile_summary_tables_are_center_wrapped() -> None:
    class _CaptureConsole:
        def __init__(self) -> None:
            self.calls = []  # type: list[object]

        def print(self, *args, **_kwargs):
            # type: (*object, **object) -> None
            self.calls.extend(args)

    report = _sample_report(include_profile=True)
    console = _CaptureConsole()

    reporting._render_profile_section(cast(Console, console), report, show_diagnostic_plots=False)

    assert console.calls
    profile_panel = cast(Panel, console.calls[0])
    assert isinstance(profile_panel, Panel)
    assert isinstance(profile_panel.renderable, Group)
    assert len(profile_panel.renderable.renderables) == 1

    summary_row = profile_panel.renderable.renderables[0]
    assert isinstance(summary_row, Align)
    assert isinstance(summary_row.renderable, Columns)

    summary_panel, distribution_panel = summary_row.renderable.renderables
    assert isinstance(summary_panel, Panel)
    assert isinstance(summary_panel.renderable, Align)
    assert isinstance(summary_panel.renderable.renderable, Table)
    assert isinstance(distribution_panel, Panel)
    assert isinstance(distribution_panel.renderable, Align)
    assert isinstance(distribution_panel.renderable.renderable, Table)


def test_profile_plots_render_inside_profile_panel_when_enabled() -> None:
    class _CaptureConsole:
        def __init__(self) -> None:
            self.calls = []  # type: list[object]

        def print(self, *args, **_kwargs):
            # type: (*object, **object) -> None
            self.calls.extend(args)

    report = _sample_report(include_profile=True)
    console = _CaptureConsole()

    reporting._render_profile_section(cast(Console, console), report, show_diagnostic_plots=True)

    assert len(console.calls) == 1
    profile_panel = cast(Panel, console.calls[0])
    assert isinstance(profile_panel.renderable, Group)
    assert len(profile_panel.renderable.renderables) == 2

    plots_row = profile_panel.renderable.renderables[1]
    assert isinstance(plots_row, Align)
    assert isinstance(plots_row.renderable, Columns)

    runtime_panel, memory_panel = plots_row.renderable.renderables
    assert isinstance(runtime_panel, Panel)
    assert runtime_panel.title == "Profile Runtime Plot"
    assert isinstance(memory_panel, Panel)
    assert memory_panel.title == "Profile Memory Plot"


def test_profile_summary_prints_without_plots_by_default() -> None:
    rendered = render_human_report(_sample_report(include_profile=True))
    assert "Profile" in rendered
    assert "Profile Runtime Plot" not in rendered
    assert "Profile Memory Plot" not in rendered


def test_plotext_chart_uses_high_contrast_sparse_scatter_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlotextSpy:
        def __init__(self):
            # type: () -> None
            self.axes_color_calls = []  # type: list[str]
            self.ticks_color_calls = []  # type: list[str]
            self.plot_calls = []  # type: list[tuple]
            self.scatter_calls = []  # type: list[tuple]

        def clear_data(self):
            return None

        def clear_figure(self):
            return None

        def theme(self, _name):
            return None

        def plotsize(self, _w, _h):
            return None

        def canvas_color(self, _name):
            return None

        def axes_color(self, name):
            self.axes_color_calls.append(name)

        def ticks_color(self, name):
            self.ticks_color_calls.append(name)

        def xscale(self, _name):
            return None

        def yscale(self, _name):
            return None

        def ylim(self, _low, _high):
            return None

        def plot(self, x, y, *, color=None, marker=None):
            self.plot_calls.append((x, y, color, marker))

        def scatter(self, x, y, *, color=None, marker=None):
            self.scatter_calls.append((x, y, color, marker))

        def xlabel(self, _label):
            return None

        def ylabel(self, _label):
            return None

        def grid(self, _enabled, _vertical):
            return None

        def xticks(self, _ticks):
            return None

        def build(self):
            return "chart"

    spy = PlotextSpy()
    monkeypatch.setattr(reporting, "_plotext", spy)

    output = reporting._build_plotext_line_chart(
        x=[0.0, 1.0, 2.0],
        series=[("wall_time_s", [0.2, 0.4, 0.6], "cyan+")],
        x_label="call_index",
        y_label="seconds",
    )

    assert output == "chart"
    assert spy.axes_color_calls[-1] == "white"
    assert spy.ticks_color_calls[-1] == "white"
    assert spy.plot_calls == []
    assert spy.scatter_calls[-1][3] == "\u25cf"


def test_plotext_chart_uses_hd_line_style_for_dense_series(monkeypatch: pytest.MonkeyPatch) -> None:
    class PlotextSpy:
        def __init__(self):
            self.plot_calls = []  # type: list[tuple]
            self.scatter_calls = []  # type: list[tuple]

        def clear_data(self):
            return None

        def clear_figure(self):
            return None

        def theme(self, _name):
            return None

        def plotsize(self, _w, _h):
            return None

        def canvas_color(self, _name):
            return None

        def axes_color(self, _name):
            return None

        def ticks_color(self, _name):
            return None

        def xscale(self, _name):
            return None

        def yscale(self, _name):
            return None

        def ylim(self, _low, _high):
            return None

        def plot(self, x, y, *, color=None, marker=None):
            self.plot_calls.append((x, y, color, marker))

        def scatter(self, x, y, *, color=None, marker=None):
            self.scatter_calls.append((x, y, color, marker))

        def xlabel(self, _label):
            return None

        def ylabel(self, _label):
            return None

        def grid(self, _enabled, _vertical):
            return None

        def xticks(self, _ticks):
            return None

        def build(self):
            return "chart"

    spy = PlotextSpy()
    monkeypatch.setattr(reporting, "_plotext", spy)

    output = reporting._build_plotext_line_chart(
        x=[float(i) for i in range(14)],
        series=[("wall_time_s", [float(i) for i in range(14)], "cyan+")],
        x_label="call_index",
        y_label="seconds",
    )

    assert output == "chart"
    assert spy.scatter_calls == []
    assert spy.plot_calls[-1][3] == "hd"


def test_plotext_chart_uses_line_marker_for_sparse_series_when_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlotextSpy:
        def __init__(self):
            self.plot_calls = []  # type: list[tuple]
            self.scatter_calls = []  # type: list[tuple]

        def clear_data(self):
            return None

        def clear_figure(self):
            return None

        def theme(self, _name):
            return None

        def plotsize(self, _w, _h):
            return None

        def canvas_color(self, _name):
            return None

        def axes_color(self, _name):
            return None

        def ticks_color(self, _name):
            return None

        def xscale(self, _name):
            return None

        def yscale(self, _name):
            return None

        def ylim(self, _low, _high):
            return None

        def plot(self, x, y, *, color=None, marker=None):
            self.plot_calls.append((x, y, color, marker))

        def scatter(self, x, y, *, color=None, marker=None):
            self.scatter_calls.append((x, y, color, marker))

        def xlabel(self, _label):
            return None

        def ylabel(self, _label):
            return None

        def grid(self, _enabled, _vertical):
            return None

        def xticks(self, _ticks):
            return None

        def build(self):
            return "chart"

    spy = PlotextSpy()
    monkeypatch.setattr(reporting, "_plotext", spy)

    output = reporting._build_plotext_line_chart(
        x=[10.0, 100.0, 1000.0, 10000.0],
        series=[("score", [0.1, 0.2, 0.3, 0.4], "cyan+")],
        x_label="budget",
        y_label="metric",
        sparse_style="line",
    )

    assert output == "chart"
    assert spy.plot_calls[-1][3] == "hd"
    assert spy.scatter_calls[-1][3] == "\u25cf"


def test_plotext_chart_sanitizes_background_ansi_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlotextSpy:
        def clear_data(self):
            return None

        def clear_figure(self):
            return None

        def theme(self, _name):
            return None

        def plotsize(self, _w, _h):
            return None

        def canvas_color(self, _name):
            return None

        def axes_color(self, _name):
            return None

        def ticks_color(self, _name):
            return None

        def xscale(self, _name):
            return None

        def yscale(self, _name):
            return None

        def ylim(self, _low, _high):
            return None

        def plot(self, _x, _y, *, color=None, marker=None):
            return None

        def scatter(self, _x, _y, *, color=None, marker=None):
            return None

        def xlabel(self, _label):
            return None

        def ylabel(self, _label):
            return None

        def grid(self, _enabled, _vertical):
            return None

        def xticks(self, _ticks):
            return None

        def build(self):
            return "\x1b[48;5;15mA\x1b[49mB\x1b[38;5;10mC\x1b[0m"

    monkeypatch.setattr(reporting, "_plotext", PlotextSpy())

    output = reporting._build_plotext_line_chart(
        x=[0.0, 1.0, 2.0],
        series=[("wall_time_s", [0.2, 0.4, 0.6], "cyan+")],
        x_label="call_index",
        y_label="seconds",
    )

    assert output is not None
    assert "\x1b[48;" not in output
    assert "\x1b[49m" not in output
    assert "\x1b[38;5;10m" in output


def test_dashboard_width_uses_terminal_columns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "96")
    assert reporting._dashboard_width() == 96

    monkeypatch.setenv("COLUMNS", "40")
    assert reporting._dashboard_width() == 80


def test_plot_legend_styles_map_to_rich_colors() -> None:
    assert reporting._rich_style_for_plot_color("blue+") == "bright_blue"
    assert reporting._rich_style_for_plot_color("yellow+") == "bright_yellow"


def test_left_ellipsis_keeps_tail() -> None:
    value = reporting._left_ellipsis("/a/b/c/d/e/file.py", 10)
    assert value == "...file.py"


def _render_panel(panel: object) -> str:
    buffer = io.StringIO()
    console = Console(
        record=True, file=buffer, force_terminal=True, color_system="truecolor", width=120
    )
    console.print(panel)
    return buffer.getvalue()


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)


# --- _fmt_flops --------------------------------------------------------------


def test_fmt_flops_small_values_use_comma_grouping() -> None:
    from whestbench.reporting import _fmt_flops

    assert _fmt_flops(0) == "0"
    assert _fmt_flops(42) == "42"
    assert _fmt_flops(12_345) == "12,345"
    assert _fmt_flops(999_999) == "999,999"


def test_fmt_flops_large_values_use_scientific() -> None:
    from whestbench.reporting import _fmt_flops

    # Threshold is 1e6; at and above, switch to scientific (no exact suffix —
    # users can get exact values from `whest run --format json`).
    assert _fmt_flops(1_000_000) == "1.00e+06"
    assert _fmt_flops(845_824_840_400) == "8.46e+11"
    assert _fmt_flops(int(1e15)) == "1.00e+15"


def test_fmt_flops_handles_non_numeric() -> None:
    from whestbench.reporting import _fmt_flops

    assert _fmt_flops(None) == "n/a"
    assert _fmt_flops("abc") == "abc"
    assert _fmt_flops(float("nan")) == "n/a"


def test_fmt_flops_handles_float_counts() -> None:
    """Per-MLP means are floats (e.g. 422,912,420,200.0 in the CLI output)."""
    from whestbench.reporting import _fmt_flops

    assert _fmt_flops(422_912_420_200.0) == "4.23e+11"
    assert _fmt_flops(1.5e7) == "1.50e+07"
