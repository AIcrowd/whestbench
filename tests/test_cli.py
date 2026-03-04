from __future__ import annotations

from typing import Any

import pytest

import circuit_estimation.cli as cli
from circuit_estimation.runner import RunnerError, RunnerErrorDetail


def _sample_report(*, profile_enabled: bool, detail: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": "1.0",
        "mode": "human",
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


def test_smoke_test_renders_human_report_by_default(
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
        lambda _report: pytest.fail("agent renderer should not be used in smoke-test mode"),
    )
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: (
            render_observed.setdefault("show_diagnostic_plots", show_diagnostic_plots) or "human\n"
        ),
    )

    exit_code = cli.main(["smoke-test"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "human\n" in captured.out
    assert captured.err == ""
    assert observed == {"profile": False, "detail": "raw"}
    assert render_observed == {"show_diagnostic_plots": False}


def test_smoke_test_show_diagnostic_plots_flag_is_forwarded(
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

    exit_code = cli.main(["smoke-test", "--show-diagnostic-plots"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "human\n" in captured.out
    assert observed == {"profile": False, "detail": "raw"}
    assert render_observed == {"show_diagnostic_plots": True}


def test_smoke_test_profile_and_detail_flags_are_forwarded(
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

    exit_code = cli.main(["smoke-test", "--profile", "--detail", "full"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "human\n" in captured.out
    assert observed == {"profile": True, "detail": "full"}


def test_smoke_test_prints_next_steps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "score_estimator_report",
        lambda *_args, **kwargs: _sample_report(
            profile_enabled=False, detail=str(kwargs["detail"])
        ),
    )
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: "human\n",
    )

    exit_code = cli.main(["smoke-test"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Next Steps" in captured.out
    assert "cestim" in captured.out
    assert "init" in captured.out
    assert "./my-estimator" in captured.out
    assert "run" in captured.out
    assert "--estimator" in captured.out


def test_smoke_test_human_mode_surfaces_stream_contract_errors_readably(
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

    exit_code = cli.main(["smoke-test"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [smoke-test:ESTIMATOR_STREAM_TOO_MANY_ROWS]" in captured.out
    assert "Estimator emitted more than max_depth rows." in captured.out
    assert "Use --debug to include a traceback." in captured.out
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_smoke_test_debug_includes_traceback_field(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_score_estimator_report(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "score_estimator_report", fake_score_estimator_report)

    exit_code = cli.main(["smoke-test", "--debug"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [smoke-test:SCORING_RUNTIME_ERROR]: boom" in captured.out
    assert "RuntimeError: boom" in captured.out


def test_run_subprocess_error_includes_inprocess_debug_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_report(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RunnerError(
            "setup",
            RunnerErrorDetail(code="SETUP_ERROR", message="runner failed"),
        )

    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )
    monkeypatch.setattr(cli, "score_estimator_report", fail_report)

    exit_code = cli.main(["run", "--estimator", "estimator.py"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [setup:SETUP_ERROR]: runner failed" in captured.out
    assert "Use --debug to include a traceback." in captured.out
    assert "rerun with --runner inprocess --debug" in captured.out


def test_run_inprocess_error_omits_inprocess_debug_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_report(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RunnerError(
            "setup",
            RunnerErrorDetail(code="SETUP_ERROR", message="runner failed"),
        )

    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )
    monkeypatch.setattr(cli, "score_estimator_report", fail_report)

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "inprocess"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [setup:SETUP_ERROR]: runner failed" in captured.out
    assert "Use --debug to include a traceback." in captured.out
    assert "rerun with --runner inprocess --debug" not in captured.out


def test_smoke_test_json_flag_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["smoke-test", "--json"])
    assert int(exc_info.value.code) == 2


def test_no_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([])
    assert int(exc_info.value.code) == 2


def test_global_json_without_subcommand_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--json"])
    assert int(exc_info.value.code) == 2


def test_agent_mode_flag_is_rejected() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--agent-mode"])
    assert int(exc_info.value.code) == 2
