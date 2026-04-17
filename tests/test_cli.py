from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import pytest

import whestbench.cli as cli
from whestbench.runner import RunnerError, RunnerErrorDetail


def _sample_report(*, profile_enabled: bool, detail: str) -> dict:
    report: dict = {
        "schema_version": "1.0",
        "mode": "human",
        "detail": detail,
        "run_meta": {
            "run_started_at_utc": "2026-03-01T00:00:00+00:00",
            "run_finished_at_utc": "2026-03-01T00:00:01+00:00",
            "run_duration_s": 1.0,
        },
        "run_config": {
            "n_mlps": 1,
            "width": 4,
            "depth": 3,
            "flop_budget": 40000,
            "profile_enabled": profile_enabled,
        },
        "results": {
            "primary_score": 0.42,
            "secondary_score": 0.55,
            "breakdowns": {
                "sampling": {
                    "flops_used": 100.0,
                    "tracked_time_s": 0.02,
                    "untracked_time_s": 0.0,
                    "by_namespace": {
                        "sampling.sample_layer_statistics": {
                            "flops_used": 100.0,
                            "tracked_time_s": 0.02,
                        }
                    },
                },
                "estimator": {
                    "flops_used": 40.0,
                    "tracked_time_s": 0.01,
                    "untracked_time_s": 0.0,
                    "by_namespace": {
                        "estimator.estimator-client": {
                            "flops_used": 40.0,
                            "tracked_time_s": 0.01,
                        }
                    },
                },
            },
            "per_mlp": [],
        },
        "notes": [],
    }
    if profile_enabled:
        report["profile_calls"] = [
            {
                "mlp_index": 0,
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
    observed: dict = {}
    render_observed: dict = {}

    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        observed["profile"] = profile
        observed["detail"] = detail
        return _sample_report(profile_enabled=False, detail=detail)

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)
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
    observed: dict = {}
    render_observed: dict = {}

    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        observed["profile"] = profile
        observed["detail"] = detail
        return _sample_report(profile_enabled=False, detail=detail)

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)

    def fake_render_human_report(_report: dict, *, show_diagnostic_plots: bool = False) -> str:
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
    observed: dict = {}

    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        observed["profile"] = profile
        observed["detail"] = detail
        return _sample_report(profile_enabled=True, detail=detail)

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)
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
    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        return _sample_report(profile_enabled=False, detail=detail)

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda _report, *, show_diagnostic_plots=False: "human\n",
    )

    exit_code = cli.main(["smoke-test"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Next Steps" in captured.out
    assert "whest" in captured.out
    assert "init" in captured.out
    assert "./my-estimator" in captured.out
    assert "run" in captured.out
    assert "--estimator" in captured.out


def test_smoke_test_human_mode_surfaces_validation_errors_readably(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        raise ValueError("Predictions must have shape (2, 4), got (1, 4).")

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)
    monkeypatch.setattr(
        cli,
        "render_human_report",
        lambda *_args, **_kwargs: pytest.fail("human renderer should not be called on failure"),
    )

    exit_code = cli.main(["smoke-test"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [smoke-test:ESTIMATOR_BAD_SHAPE]" in captured.out
    assert "Predictions must have shape" in captured.out
    assert "Use --debug to include a traceback." in captured.out
    assert "Traceback" not in captured.out
    assert captured.err == ""


def test_smoke_test_debug_includes_traceback_field(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)

    exit_code = cli.main(["smoke-test", "--debug"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [smoke-test:SCORING_RUNTIME_ERROR]: boom" in captured.out
    assert "RuntimeError: boom" in captured.out


def test_run_subprocess_error_includes_inprocess_debug_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_run(*_args: Any, **_kwargs: Any) -> dict:
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
    monkeypatch.setattr(cli, "_run_estimator_with_runner", fail_run)

    exit_code = cli.main(["run", "--estimator", "estimator.py"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [setup:SETUP_ERROR]: runner failed" in captured.out
    assert "Use --debug to include a traceback." in captured.out
    assert "rerun with --runner local --debug" in captured.out


def test_run_inprocess_error_omits_inprocess_debug_hint(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_run(*_args: Any, **_kwargs: Any) -> dict:
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
    monkeypatch.setattr(cli, "_run_estimator_with_runner", fail_run)

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "inprocess"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [setup:SETUP_ERROR]: runner failed" in captured.out
    assert "Use --debug to include a traceback." in captured.out
    assert "rerun with --runner local --debug" not in captured.out


def test_run_rich_mode_updates_live_top_pane_with_final_run_meta(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}

    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )
    monkeypatch.setattr(cli, "rich_tqdm", object(), raising=False)

    @contextmanager
    def fake_live_session(
        pre_report: dict, total: int, n_mlps: int, gen_label: str = "Generating MLPs"
    ):
        observed["initial_finished"] = pre_report["run_meta"]["run_finished_at_utc"]
        observed["initial_duration"] = pre_report["run_meta"]["run_duration_s"]
        observed["total"] = total

        class Session:
            def on_progress(self, event: dict) -> None:
                observed["progress_event"] = event

            def update_run_meta(self, run_meta: dict) -> None:
                observed["final_meta"] = run_meta

        yield Session()

    monkeypatch.setattr(cli, "_live_top_pane_session", fake_live_session, raising=False)

    def fake_run_estimator_with_runner(*_args: Any, **kwargs: Any) -> dict:
        progress_cb = kwargs.get("progress")
        assert callable(progress_cb)
        progress_cb({"completed": 1})
        report = _sample_report(profile_enabled=False, detail=str(kwargs.get("detail", "raw")))
        run_meta = report["run_meta"]
        assert isinstance(run_meta, dict)
        run_meta["run_finished_at_utc"] = "2026-03-01T00:00:03+00:00"
        run_meta["run_duration_s"] = 3.0
        return report

    monkeypatch.setattr(cli, "_run_estimator_with_runner", fake_run_estimator_with_runner)
    monkeypatch.setattr(
        cli,
        "render_human_results",
        lambda _report, *, show_diagnostic_plots=False, debug=False: "results\n",
    )

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "inprocess"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "results" in captured.out
    assert observed["initial_finished"] == "n/a"
    assert observed["initial_duration"] is None
    assert observed["total"] == 10
    assert observed["progress_event"] == {"completed": 1}
    assert observed["final_meta"]["run_finished_at_utc"] == "2026-03-01T00:00:03+00:00"
    assert observed["final_meta"]["run_duration_s"] == 3.0


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
