from __future__ import annotations

import io
import json
import re
from typing import Any, cast

import pytest
from rich.align import Align
from rich.console import Console, Group
from rich.table import Table

import circuit_estimation.reporting as reporting
from circuit_estimation.reporting import render_agent_report, render_human_report


def _sample_report(*, include_profile: bool = False) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "1.0",
        "mode": "agent",
        "detail": "raw",
        "run_meta": {
            "run_started_at_utc": "2026-03-01T00:00:00+00:00",
            "run_finished_at_utc": "2026-03-01T00:00:01+00:00",
            "run_duration_s": 1.0,
        },
        "run_config": {
            "n_circuits": 2,
            "n_samples": 100,
            "width": 4,
            "max_depth": 3,
            "layer_count": 3,
            "budgets": [10, 100],
            "time_tolerance": 0.1,
            "profile_enabled": include_profile,
        },
        "circuits": [
            {"circuit_index": 0, "wire_count": 4, "layer_count": 3},
            {"circuit_index": 1, "wire_count": 4, "layer_count": 3},
        ],
        "results": {
            "final_score": 0.123,
            "score_direction": "lower_is_better",
            "by_budget_raw": [
                {
                    "budget": 10,
                    "mse_by_layer": [0.1, 0.2, 0.3],
                    "time_budget_by_depth_s": [0.01, 0.02, 0.03],
                    "mse_mean": 0.2,
                    "adjusted_mse": 0.22,
                    "time_ratio_by_depth_mean": [1.1, 1.0, 0.95],
                    "effective_time_s_by_depth_mean": [0.011, 0.020, 0.0285],
                    "timeout_rate_by_depth": [0.0, 0.0, 0.0],
                    "time_floor_rate_by_depth": [0.0, 0.0, 0.0],
                    "call_time_ratio_mean": 1.1,
                    "call_effective_time_s_mean": 0.011,
                    "timeout_rate": 0.0,
                    "time_floor_rate": 0.0,
                },
                {
                    "budget": 100,
                    "mse_by_layer": [0.05, 0.04, 0.03],
                    "time_budget_by_depth_s": [0.02, 0.03, 0.04],
                    "mse_mean": 0.04,
                    "adjusted_mse": 0.036,
                    "time_ratio_by_depth_mean": [0.9, 0.9, 0.9],
                    "effective_time_s_by_depth_mean": [0.018, 0.027, 0.036],
                    "timeout_rate_by_depth": [0.0, 0.0, 0.0],
                    "time_floor_rate_by_depth": [0.0, 0.0, 0.0],
                    "call_time_ratio_mean": 0.9,
                    "call_effective_time_s_mean": 0.018,
                    "timeout_rate": 0.0,
                    "time_floor_rate": 0.0,
                },
            ],
        },
        "notes": [],
    }
    if include_profile:
        report["profile_calls"] = [
            {
                "budget": 10,
                "circuit_index": 0,
                "wire_count": 4,
                "layer_count": 3,
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


def test_render_human_mode_includes_expected_sections_without_profile() -> None:
    rendered = render_human_report(_sample_report(include_profile=False))

    # Human mode contract: high-level run summary plus budget and layer diagnostics.
    assert "Circuit Estimation Report" in rendered
    assert "Use --json for JSON output" in rendered
    assert "budget-by-depth" in rendered.lower()
    assert "Run Context" in rendered
    assert "Readiness Scorecard" in rendered
    assert "Budget" in rendered
    assert "Layer Diagnostics" in rendered
    assert "Budget Breakdown" not in rendered
    assert "Budget Intelligence" not in rendered
    assert "Budget Table" not in rendered
    assert "Layer Intelligence" not in rendered
    assert "Budget Frontier Plot" not in rendered
    assert "Budget Runtime Plot" not in rendered
    assert "Layer Trend Plot" not in rendered
    assert "Layer Runtime Plot" not in rendered
    assert "Profile Summary" not in rendered
    assert "Profile Runtime Plot" not in rendered
    assert "Profile Memory Plot" not in rendered
    assert "Profiling" not in rendered


def test_human_report_uses_three_column_top_row_on_wide_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "220")
    rendered = render_human_report(_sample_report(include_profile=False))
    plain = _strip_ansi(rendered)
    title_lines = [
        line
        for line in plain.splitlines()
        if "Run Context" in line or "Readiness Scorecard" in line or "Hardware & Runtime" in line
    ]

    assert "Run Context" in rendered
    assert "Readiness Scorecard" in rendered
    assert "Hardware & Runtime" in rendered
    assert any(
        "Run Context" in line and "Readiness Scorecard" in line and "Hardware & Runtime" in line
        for line in title_lines
    )
    assert not any(
        "Hardware & Runtime" in line
        and "Run Context" not in line
        and "Readiness Scorecard" not in line
        for line in title_lines
    )


def test_human_report_uses_two_column_plus_stack_layout_on_medium_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "140")
    rendered = render_human_report(_sample_report(include_profile=False))
    plain = _strip_ansi(rendered)
    title_lines = [
        line
        for line in plain.splitlines()
        if "Run Context" in line or "Readiness Scorecard" in line or "Hardware & Runtime" in line
    ]

    assert any(
        "Run Context" in line and "Readiness Scorecard" in line and "Hardware & Runtime" not in line
        for line in title_lines
    )
    assert any(
        "Hardware & Runtime" in line
        and "Run Context" not in line
        and "Readiness Scorecard" not in line
        for line in title_lines
    )


def test_hardware_metadata_is_not_repeated_inside_run_context() -> None:
    report = _sample_report(include_profile=False)
    run_meta = cast(dict[str, Any], report["run_meta"])
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


def test_budget_and_layer_tables_are_centered() -> None:
    report = _sample_report(include_profile=False)

    budget = reporting._budget_lane_panel(report)
    budget_body = cast(Group, budget.renderable)
    budget_table = budget_body.renderables[0]
    assert isinstance(budget_table, Align)
    assert isinstance(budget_table.renderable, Table)

    layer = reporting._layer_lane_panel(report)
    layer_body = cast(Group, layer.renderable)
    layer_table = layer_body.renderables[0]
    assert isinstance(layer_table, Align)
    assert isinstance(layer_table.renderable, Table)


def test_budget_lane_contains_table_and_two_plots() -> None:
    rendered = render_human_report(
        _sample_report(include_profile=False), show_diagnostic_plots=True
    )

    assert "Budget" in rendered
    assert "Budget Frontier Plot" in rendered
    assert "Budget Runtime Plot" in rendered


def test_layer_lane_contains_stats_and_trend_plots() -> None:
    rendered = render_human_report(
        _sample_report(include_profile=False), show_diagnostic_plots=True
    )

    assert "Layer Diagnostics" in rendered
    assert "Layer Trend Plot" in rendered
    assert "Layer Runtime Plot" not in rendered


def test_budget_plots_render_side_by_side_below_full_width_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "220")
    rendered = render_human_report(
        _sample_report(include_profile=False), show_diagnostic_plots=True
    )
    plain = _strip_ansi(rendered)
    lines = plain.splitlines()

    table_line = next(i for i, line in enumerate(lines) if "Budget" in line)
    plot_line = next(
        i
        for i, line in enumerate(lines)
        if "Budget Frontier Plot" in line and "Budget Runtime Plot" in line
    )
    assert table_line < plot_line


def test_layer_plot_renders_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "220")
    rendered = render_human_report(
        _sample_report(include_profile=False), show_diagnostic_plots=True
    )
    plain = _strip_ansi(rendered)
    assert "Layer Trend Plot" in plain
    assert "Layer Runtime Plot" not in plain


def test_layer_diagnostics_are_mse_only() -> None:
    rendered = _strip_ansi(render_human_report(_sample_report(include_profile=False)))
    assert "MSE by Layer [mse_by_layer]" in rendered
    assert "Time Ratio by Layer [time_ratio_by_layer]" not in rendered
    assert "Adjusted MSE by Layer [adjusted_mse_by_layer]" not in rendered


def test_render_human_mode_includes_profile_section_when_available() -> None:
    rendered = render_human_report(_sample_report(include_profile=True))

    assert "Profile" in rendered
    assert "Profiling" not in rendered
    assert "Profile Summary" not in rendered
    assert "Profile Runtime Plot" not in rendered
    assert "Profile Memory Plot" not in rendered
    assert "wall_time_s" in rendered
    assert "cpu_time_s" in rendered
    assert "rss_bytes" in rendered
    assert "peak_rss_bytes" in rendered


def test_json_mode_schema_keeps_stream_runtime_fields() -> None:
    payload = json.loads(render_agent_report(_sample_report(include_profile=False)))
    row = payload["results"]["by_budget_raw"][0]
    assert "time_budget_by_depth_s" in row
    assert "time_ratio_by_depth_mean" in row
    assert "effective_time_s_by_depth_mean" in row


def test_profile_plots_render_side_by_side_and_memory_axis_is_mb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "220")
    rendered = render_human_report(_sample_report(include_profile=True), show_diagnostic_plots=True)
    plain = _strip_ansi(rendered)

    assert any(
        "Profile Runtime Plot" in line and "Profile Memory Plot" in line
        for line in plain.splitlines()
    )
    assert "Memory Usage (MB)" in plain
    assert "rss_mb" in plain


def test_profile_summary_contains_two_structured_side_by_side_tables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COLUMNS", "220")
    rendered = render_human_report(_sample_report(include_profile=True))
    plain = _strip_ansi(rendered)

    assert "Summary" in plain
    assert "Distribution" in plain
    assert any("Summary" in line and "Distribution" in line for line in plain.splitlines())


def test_profile_summary_prints_without_plots_by_default() -> None:
    rendered = render_human_report(_sample_report(include_profile=True))
    assert "Profile" in rendered
    assert "Profile Summary" not in rendered
    assert "Profile Runtime Plot" not in rendered
    assert "Profile Memory Plot" not in rendered


def test_plotext_chart_uses_high_contrast_sparse_scatter_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlotextSpy:
        def __init__(self) -> None:
            self.axes_color_calls: list[str] = []
            self.ticks_color_calls: list[str] = []
            self.plot_calls: list[tuple[list[float], list[float], str | None, str | None]] = []
            self.scatter_calls: list[tuple[list[float], list[float], str | None, str | None]] = []

        def clear_data(self) -> None:
            return None

        def clear_figure(self) -> None:
            return None

        def theme(self, _name: str) -> None:
            return None

        def plotsize(self, _w: int, _h: int) -> None:
            return None

        def canvas_color(self, _name: str) -> None:
            return None

        def axes_color(self, name: str) -> None:
            self.axes_color_calls.append(name)

        def ticks_color(self, name: str) -> None:
            self.ticks_color_calls.append(name)

        def xscale(self, _name: str) -> None:
            return None

        def yscale(self, _name: str) -> None:
            return None

        def ylim(self, _low: float, _high: float) -> None:
            return None

        def plot(
            self,
            x: list[float],
            y: list[float],
            *,
            color: str | None = None,
            marker: str | None = None,
        ) -> None:
            self.plot_calls.append((x, y, color, marker))

        def scatter(
            self,
            x: list[float],
            y: list[float],
            *,
            color: str | None = None,
            marker: str | None = None,
        ) -> None:
            self.scatter_calls.append((x, y, color, marker))

        def xlabel(self, _label: str) -> None:
            return None

        def ylabel(self, _label: str) -> None:
            return None

        def grid(self, _enabled: bool, _vertical: bool) -> None:
            return None

        def xticks(self, _ticks: list[float]) -> None:
            return None

        def build(self) -> str:
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
    assert spy.scatter_calls[-1][3] == "●"


def test_plotext_chart_uses_hd_line_style_for_dense_series(monkeypatch: pytest.MonkeyPatch) -> None:
    class PlotextSpy:
        def __init__(self) -> None:
            self.plot_calls: list[tuple[list[float], list[float], str | None, str | None]] = []
            self.scatter_calls: list[tuple[list[float], list[float], str | None, str | None]] = []

        def clear_data(self) -> None:
            return None

        def clear_figure(self) -> None:
            return None

        def theme(self, _name: str) -> None:
            return None

        def plotsize(self, _w: int, _h: int) -> None:
            return None

        def canvas_color(self, _name: str) -> None:
            return None

        def axes_color(self, _name: str) -> None:
            return None

        def ticks_color(self, _name: str) -> None:
            return None

        def xscale(self, _name: str) -> None:
            return None

        def yscale(self, _name: str) -> None:
            return None

        def ylim(self, _low: float, _high: float) -> None:
            return None

        def plot(
            self,
            x: list[float],
            y: list[float],
            *,
            color: str | None = None,
            marker: str | None = None,
        ) -> None:
            self.plot_calls.append((x, y, color, marker))

        def scatter(
            self,
            x: list[float],
            y: list[float],
            *,
            color: str | None = None,
            marker: str | None = None,
        ) -> None:
            self.scatter_calls.append((x, y, color, marker))

        def xlabel(self, _label: str) -> None:
            return None

        def ylabel(self, _label: str) -> None:
            return None

        def grid(self, _enabled: bool, _vertical: bool) -> None:
            return None

        def xticks(self, _ticks: list[float]) -> None:
            return None

        def build(self) -> str:
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
        def __init__(self) -> None:
            self.plot_calls: list[tuple[list[float], list[float], str | None, str | None]] = []
            self.scatter_calls: list[tuple[list[float], list[float], str | None, str | None]] = []

        def clear_data(self) -> None:
            return None

        def clear_figure(self) -> None:
            return None

        def theme(self, _name: str) -> None:
            return None

        def plotsize(self, _w: int, _h: int) -> None:
            return None

        def canvas_color(self, _name: str) -> None:
            return None

        def axes_color(self, _name: str) -> None:
            return None

        def ticks_color(self, _name: str) -> None:
            return None

        def xscale(self, _name: str) -> None:
            return None

        def yscale(self, _name: str) -> None:
            return None

        def ylim(self, _low: float, _high: float) -> None:
            return None

        def plot(
            self,
            x: list[float],
            y: list[float],
            *,
            color: str | None = None,
            marker: str | None = None,
        ) -> None:
            self.plot_calls.append((x, y, color, marker))

        def scatter(
            self,
            x: list[float],
            y: list[float],
            *,
            color: str | None = None,
            marker: str | None = None,
        ) -> None:
            self.scatter_calls.append((x, y, color, marker))

        def xlabel(self, _label: str) -> None:
            return None

        def ylabel(self, _label: str) -> None:
            return None

        def grid(self, _enabled: bool, _vertical: bool) -> None:
            return None

        def xticks(self, _ticks: list[float]) -> None:
            return None

        def build(self) -> str:
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
    assert spy.scatter_calls[-1][3] == "●"


def test_plotext_chart_sanitizes_background_ansi_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class PlotextSpy:
        def clear_data(self) -> None:
            return None

        def clear_figure(self) -> None:
            return None

        def theme(self, _name: str) -> None:
            return None

        def plotsize(self, _w: int, _h: int) -> None:
            return None

        def canvas_color(self, _name: str) -> None:
            return None

        def axes_color(self, _name: str) -> None:
            return None

        def ticks_color(self, _name: str) -> None:
            return None

        def xscale(self, _name: str) -> None:
            return None

        def yscale(self, _name: str) -> None:
            return None

        def ylim(self, _low: float, _high: float) -> None:
            return None

        def plot(
            self,
            _x: list[float],
            _y: list[float],
            *,
            color: str | None = None,
            marker: str | None = None,
        ) -> None:
            return None

        def scatter(
            self,
            _x: list[float],
            _y: list[float],
            *,
            color: str | None = None,
            marker: str | None = None,
        ) -> None:
            return None

        def xlabel(self, _label: str) -> None:
            return None

        def ylabel(self, _label: str) -> None:
            return None

        def grid(self, _enabled: bool, _vertical: bool) -> None:
            return None

        def xticks(self, _ticks: list[float]) -> None:
            return None

        def build(self) -> str:
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


def _render_panel(panel: object) -> str:
    buffer = io.StringIO()
    console = Console(
        record=True, file=buffer, force_terminal=True, color_system="truecolor", width=120
    )
    console.print(panel)
    return buffer.getvalue()


def _strip_ansi(value: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", value)
