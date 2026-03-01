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
        "render_human_report",
        lambda _report: pytest.fail("human renderer should not be called"),
    )
    monkeypatch.setattr(cli, "render_agent_report", lambda _report: '{\n  "mode": "agent"\n}\n')

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == '{\n  "mode": "agent"\n}\n'
    assert json.loads(captured.out) == {"mode": "agent"}
    assert observed == {"profile": False, "detail": "raw"}


def test_human_mode_outputs_rich_sections(
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
        "render_agent_report",
        lambda _report: pytest.fail("agent renderer should not be called"),
    )
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report: (
            "Circuit Estimation Report\n"
            "Run Context\n"
            "Score Summary\n"
            "Budget Breakdown\n"
            "Layer Diagnostics\n"
            "Profiling\n"
        ),
    )

    exit_code = cli.main(["--mode", "human", "--profile", "--detail", "full"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "Circuit Estimation Report" in captured.out
    assert "Run Context" in captured.out
    assert "Score Summary" in captured.out
    assert "Budget Breakdown" in captured.out
    assert "Layer Diagnostics" in captured.out
    assert "Profiling" in captured.out
    assert observed == {"profile": True, "detail": "full"}
