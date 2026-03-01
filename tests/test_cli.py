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


def test_default_mode_launches_textual_dashboard_when_supported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}
    launch_observed: dict[str, Any] = {}

    def fake_score_estimator_report(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["profile"] = kwargs.get("profile")
        observed["detail"] = kwargs.get("detail")
        return _sample_report(profile_enabled=True, detail=str(kwargs.get("detail", "raw")))

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)
    monkeypatch.setattr(cli, "_supports_textual_dashboard", lambda: True)
    monkeypatch.setattr(
        cli,
        "_launch_textual_dashboard",
        lambda report: launch_observed.update({"mode": report.get("mode")}),
    )
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: pytest.fail(
            "static fallback should not render when textual works"
        ),
    )

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == ""
    assert captured.err == ""
    assert observed == {"profile": True, "detail": "full"}
    assert launch_observed == {"mode": "human"}


def test_default_mode_falls_back_to_static_when_textual_unsupported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}

    def fake_score_estimator_report(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["profile"] = kwargs.get("profile")
        observed["detail"] = kwargs.get("detail")
        return _sample_report(profile_enabled=True, detail=str(kwargs.get("detail", "raw")))

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)
    monkeypatch.setattr(cli, "_supports_textual_dashboard", lambda: False)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: "fallback\n",
    )

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Textual UI unavailable" in captured.err
    assert captured.out == "fallback\n"
    assert observed == {"profile": True, "detail": "full"}


def test_default_mode_falls_back_when_textual_launch_raises(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "score_estimator_report",
        lambda *_args, **kwargs: _sample_report(
            profile_enabled=True, detail=str(kwargs.get("detail", "raw"))
        ),
    )
    monkeypatch.setattr(cli, "_supports_textual_dashboard", lambda: True)

    def fake_launch(_report: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_launch_textual_dashboard", fake_launch)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: "fallback\n",
    )

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Textual UI unavailable" in captured.err
    assert "boom" in captured.err
    assert captured.out == "fallback\n"


def test_agent_mode_stdout_is_json_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}

    def fake_score_estimator_report(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["profile"] = kwargs.get("profile")
        observed["detail"] = kwargs.get("detail")
        return _sample_report(profile_enabled=False, detail=str(kwargs.get("detail", "raw")))

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)
    monkeypatch.setattr(
        cli,
        "_launch_textual_dashboard",
        lambda _report: pytest.fail("textual dashboard should not launch in agent mode"),
    )
    monkeypatch.setattr(
        cli,
        "render_agent_report",
        lambda _report: '{\n  "mode": "agent"\n}\n',
    )

    exit_code = cli.main(["--agent-mode"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == '{\n  "mode": "agent"\n}\n'
    assert json.loads(captured.out) == {"mode": "agent"}
    assert observed == {"profile": False, "detail": "raw"}


@pytest.mark.parametrize("removed_flag", ["--detail", "--profile", "--show-diagnostic-plots"])
def test_removed_human_flags_are_rejected(removed_flag: str) -> None:
    with pytest.raises(SystemExit):
        cli.main([removed_flag])
