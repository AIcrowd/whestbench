from __future__ import annotations

import json
from typing import Any

import pytest

import circuit_estimation.cli as cli


def _sample_report(*, profile_enabled: bool, detail: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": "agent",
        "detail": detail,
        "run_meta": {
            "run_started_at_utc": "2026-03-01T00:00:00+00:00",
            "run_finished_at_utc": "2026-03-01T00:00:01+00:00",
            "run_duration_s": 1.0,
        },
        "run_config": {
            "n_circuits": 1,
            "n_samples": 10,
            "width": 4,
            "max_depth": 3,
            "layer_count": 3,
            "budgets": [10],
            "time_tolerance": 0.1,
            "profile_enabled": profile_enabled,
        },
        "circuits": [{"circuit_index": 0, "wire_count": 4, "layer_count": 3}],
        "results": {
            "final_score": 0.42,
            "score_direction": "lower_is_better",
            "by_budget_raw": [],
        },
        "notes": [],
    }
    if profile_enabled:
        report["profile_calls"] = [
            {
                "budget": 10,
                "circuit_index": 0,
                "wire_count": 4,
                "layer_count": 3,
                "wall_time_s": 0.01,
                "cpu_time_s": 0.01,
                "rss_bytes": 123,
                "peak_rss_bytes": 456,
            }
        ]
    return report


def test_default_mode_outputs_human_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}
    render_observed: dict[str, Any] = {}

    def fake_score_estimator_report(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["profile"] = kwargs.get("profile")
        observed["detail"] = kwargs.get("detail")
        return _sample_report(profile_enabled=False, detail=str(kwargs.get("detail", "raw")))

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)
    monkeypatch.setattr(
        cli,
        "render_agent_report",
        lambda _report: pytest.fail("agent renderer should not be called"),
    )
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: (
            render_observed.update({"show_diagnostic_plots": show_diagnostic_plots}) or ""
        )
        + (
            "Circuit Estimation Report\n"
            "Readiness Scorecard\n"
            "Run Context\n"
            "Hardware & Runtime\n"
            "Tip: Use --agent-mode\n"
        ),
    )

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Circuit Estimation Report" in captured.out
    assert "Readiness Scorecard" in captured.out
    assert "Run Context" in captured.out
    assert "Hardware & Runtime" in captured.out
    assert "Use --agent-mode" in captured.out
    assert observed == {"profile": False, "detail": "raw"}
    assert render_observed == {"show_diagnostic_plots": False}


def test_agent_mode_stdout_is_json_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}

    def fake_score_estimator_report(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["profile"] = kwargs.get("profile")
        observed["detail"] = kwargs.get("detail")
        return _sample_report(
            profile_enabled=bool(kwargs.get("profile")), detail=str(kwargs.get("detail", "raw"))
        )

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: pytest.fail("human renderer should not be called"),
    )
    monkeypatch.setattr(
        cli,
        "render_agent_report",
        lambda _report: '{\n  "mode": "agent"\n}\n',
    )

    exit_code = cli.main(["--agent-mode", "--profile", "--detail", "full"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == '{\n  "mode": "agent"\n}\n'
    assert json.loads(captured.out) == {"mode": "agent"}
    assert observed == {"profile": True, "detail": "full"}


def test_show_diagnostic_plots_flag_enables_human_plots(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}

    def fake_score_estimator_report(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        return _sample_report(profile_enabled=bool(kwargs.get("profile")), detail=str(kwargs.get("detail", "raw")))

    def fake_render_human_report(_report: dict[str, Any], *, show_diagnostic_plots: bool = False) -> str:
        observed["show_diagnostic_plots"] = show_diagnostic_plots
        return "human\n"

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)
    monkeypatch.setattr(cli, "render_human_report", fake_render_human_report)
    monkeypatch.setattr(
        cli,
        "render_agent_report",
        lambda _report: pytest.fail("agent renderer should not be called"),
    )

    exit_code = cli.main(["--show-diagnostic-plots"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == "human\n"
    assert observed == {"show_diagnostic_plots": True}
