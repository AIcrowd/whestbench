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

import network_estimation.reporting as reporting
from network_estimation.reporting import (
    render_agent_report,
    render_human_report,
    render_smoke_test_next_steps,
)


def _sample_report(*, include_profile: bool = False) -> "dict[str, object]":
    report: "dict[str, object]" = {
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
            "estimator_budget": 100,
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
    if include_profile:
        report["profile_calls"] = [
            {
                "estimator_budget": 100,
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
    rendered = render_smoke_test_next_steps()
    plain = _strip_ansi(rendered)

    assert "Next Steps" in plain
    assert "We are all set! Welcome onboard" in plain
    assert "Run these steps:" in plain
    assert "# 1) Create starter files you can edit." in plain
    assert "# 2) Validate an Estimator implementation." in plain
    assert "# 3) Run local evaluation with isolation." in plain
    assert "# 4) Build submission artifacts for AIcrowd." in plain
    assert "Commands (bash)" not in plain
    assert "Command" not in plain
    assert "Purpose" not in plain
    assert "nestim init ./my-estimator" in plain
    assert "nestim validate --estimator ./my-estimator/estimator.py" in plain
    assert "nestim run --estimator ./my-estimator/estimator.py" in plain
    assert "--runner" in plain
    assert "subprocess" in plain
    assert "nestim package --estimator ./my-estimator/estimator.py" in plain
    assert "--output" in plain
    assert "./submission.tar.gz" in plain
    assert "Optional: run bundled example estimators:" in plain
    assert (
        "nestim run --estimator ./examples/estimators/combined_estimator.py --runner subprocess"
        in plain
    )
    assert (
        "nestim run --estimator ./examples/estimators/covariance_propagation.py --runner subprocess"
        in plain
    )
    assert (
        "nestim run --estimator ./examples/estimators/mean_propagation.py --runner subprocess"
        in plain
    )
    assert (
        "nestim run --estimator ./examples/estimators/random_estimator.py --runner subprocess"
        in plain
    )


def test_smoke_test_next_steps_uses_distinct_styles_per_purpose_line() -> None:
    lines = reporting._smoke_next_step_lines()
    styles = [str(line.style) for line in lines]
    assert len(lines) == 4
    assert styles == ["bold bright_cyan", "bold bright_green", "bold bright_yellow", "bold bright_magenta"]


def test_render_human_mode_includes_expected_sections_without_profile() -> None:
    rendered = render_human_report(_sample_report(include_profile=False))

    # Human mode contract: high-level run summary.
    assert "Network Estimation Report" in rendered
    assert "Use --json for JSON output" in rendered
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
        line
        for line in plain.splitlines()
        if "Run Context" in line or "Hardware & Runtime" in line
    ]

    assert "Run Context" in rendered
    assert "Hardware & Runtime" in rendered
    # Run Context and Hardware & Runtime share the same row
    assert any(
        "Run Context" in line and "Hardware & Runtime" in line
        for line in title_lines
    )


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
