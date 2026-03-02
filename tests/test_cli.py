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


def test_default_mode_renders_human_report_by_default(
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
        lambda _report: pytest.fail("agent renderer should not be used in human mode"),
    )
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: (
            render_observed.setdefault("show_diagnostic_plots", show_diagnostic_plots) or "human\n"
        ),
    )

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "human\n"
    assert captured.err == ""
    assert observed == {"profile": False, "detail": "raw"}
    assert render_observed == {"show_diagnostic_plots": False}


def test_show_diagnostic_plots_flag_is_forwarded_to_human_renderer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}
    render_observed: dict[str, Any] = {}

    def fake_score_estimator_report(*_args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["profile"] = kwargs.get("profile")
        observed["detail"] = kwargs.get("detail")
        return _sample_report(profile_enabled=False, detail=str(kwargs.get("detail", "raw")))

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)

    def fake_render_human_report(
        _report: dict[str, Any], *, show_diagnostic_plots: bool = False
    ) -> str:
        render_observed["show_diagnostic_plots"] = show_diagnostic_plots
        return "human\n"

    monkeypatch.setattr(
        cli,
        "render_human_report",
        fake_render_human_report,
    )

    exit_code = cli.main(["--show-diagnostic-plots"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == "human\n"
    assert observed == {"profile": False, "detail": "raw"}
    assert render_observed == {"show_diagnostic_plots": True}


def test_profile_and_detail_flags_are_forwarded(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}

    monkeypatch.setattr(
        cli,
        "score_estimator_report",
        lambda *_args, **kwargs: (
            observed.update({"profile": kwargs.get("profile"), "detail": kwargs.get("detail")})
            or _sample_report(profile_enabled=True, detail=str(kwargs.get("detail", "raw")))
        ),
    )
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: "human\n",
    )

    exit_code = cli.main(["--profile", "--detail", "full"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == "human\n"
    assert observed == {"profile": True, "detail": "full"}


def test_json_flag_stdout_is_json_only(
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
        "render_agent_report",
        lambda _report: '{\n  "mode": "agent"\n}\n',
    )

    exit_code = cli.main(["--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert captured.out == '{\n  "mode": "agent"\n}\n'
    assert json.loads(captured.out) == {"mode": "agent"}
    assert observed == {"profile": False, "detail": "raw"}


def test_legacy_human_flags_are_supported(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, Any] = {}
    monkeypatch.setattr(
        cli,
        "score_estimator_report",
        lambda *_args, **kwargs: (
            observed.update({"profile": kwargs.get("profile"), "detail": kwargs.get("detail")})
            or _sample_report(profile_enabled=True, detail=str(kwargs.get("detail", "raw")))
        ),
    )
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: (
            "plots_on\n" if show_diagnostic_plots else "plots_off\n"
        ),
    )

    exit_code = cli.main(["--profile", "--detail", "full", "--show-diagnostic-plots"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "plots_on\n"
    assert captured.err == ""
    assert observed == {"profile": True, "detail": "full"}


def test_human_mode_surfaces_stream_contract_errors_readably(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_score_estimator_report(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("Estimator emitted more than max_depth rows.")

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda *_args, **_kwargs: pytest.fail("human renderer should not be called on failure"),
    )
    monkeypatch.setattr(
        cli,
        "render_agent_report",
        lambda *_args, **_kwargs: pytest.fail("agent renderer should not be called on failure"),
    )

    exit_code = cli.main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [scoring:ESTIMATOR_STREAM_TOO_MANY_ROWS]" in captured.out
    assert "Estimator emitted more than max_depth rows." in captured.out
    assert "Use --debug to include a traceback." in captured.out
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_json_flag_surfaces_stream_contract_errors_as_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_score_estimator_report(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise ValueError("Estimator row at depth 0 must have shape (4,), got (1,).")

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda *_args, **_kwargs: pytest.fail("human renderer should not be called on failure"),
    )
    monkeypatch.setattr(
        cli,
        "render_agent_report",
        lambda *_args, **_kwargs: pytest.fail("agent renderer should not be called on failure"),
    )

    exit_code = cli.main(["--json"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert payload["error"]["stage"] == "scoring"
    assert payload["error"]["code"] == "ESTIMATOR_STREAM_BAD_ROW_SHAPE"
    assert "shape (4,)" in payload["error"]["message"]


def test_json_flag_debug_includes_traceback_field(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_score_estimator_report(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)

    exit_code = cli.main(["--json", "--debug"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["error"]["code"] == "SCORING_RUNTIME_ERROR"
    assert "RuntimeError: boom" in payload["error"]["traceback"]


def test_agent_mode_flag_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--agent-mode"])
    assert int(exc_info.value.code) == 2
