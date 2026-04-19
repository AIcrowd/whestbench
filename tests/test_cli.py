from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Any

import pytest

import whestbench.cli as cli
import whestbench.reporting as reporting
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


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


class _PlotextStub:
    def clear_data(self) -> None:
        return None

    def clear_figure(self) -> None:
        return None

    def theme(self, _name: str) -> None:
        return None

    def plotsize(self, _width: int, _height: int) -> None:
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
        return "chart"


def test_smoke_test_rich_output_includes_dashboard_and_onboarding(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}

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

    exit_code = cli.main(["smoke-test", "--format", "rich"])
    captured = capsys.readouterr()
    plain = _strip_ansi(captured.out)

    assert exit_code == 0
    assert "WhestBench Report" in plain
    assert "Run Context" in plain
    assert "Final Score" in plain
    assert "Primary Score" in plain
    assert "Next Steps" in plain
    assert "whest init ./my-estimator" in plain
    assert "Create starter files you can edit." in plain
    assert "WhestBench Report (Plain Text)" not in plain
    assert (
        plain.count("Use --format json for JSON output when calling from automated agents or UIs.")
        == 1
    )
    assert plain.count("Use --show-diagnostic-plots to include diagnostic plot panes.") == 1
    assert captured.err == ""
    assert observed == {"profile": False, "detail": "raw"}


def test_smoke_test_show_diagnostic_plots_affects_rich_output_when_profile_present(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict = {}

    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        observed["profile"] = profile
        observed["detail"] = detail
        return _sample_report(profile_enabled=False, detail=detail)

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)
    monkeypatch.setattr(reporting, "_plotext", _PlotextStub())

    exit_code = cli.main(["smoke-test", "--format", "rich"])
    default_output = _strip_ansi(capsys.readouterr().out)

    observed.clear()

    def fake_profile_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        observed["profile"] = profile
        observed["detail"] = detail
        return _sample_report(profile_enabled=True, detail=detail)

    monkeypatch.setattr(cli, "run_default_report", fake_profile_report)
    exit_code = cli.main(["smoke-test", "--profile", "--show-diagnostic-plots", "--format", "rich"])
    captured = capsys.readouterr()
    plotted_output = _strip_ansi(captured.out)

    assert exit_code == 0
    assert "Profile Runtime Plot" not in default_output
    assert "Profile Memory Plot" not in default_output
    assert "Profile Runtime Plot" in plotted_output
    assert "Profile Memory Plot" in plotted_output
    assert observed == {"profile": True, "detail": "raw"}


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

    exit_code = cli.main(["smoke-test", "--profile", "--detail", "full", "--format", "rich"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.err == ""
    assert "WhestBench Report" in _strip_ansi(captured.out)
    assert observed == {"profile": True, "detail": "full"}


def test_smoke_test_prints_next_steps(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        return _sample_report(profile_enabled=False, detail=detail)

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)

    exit_code = cli.main(["smoke-test", "--format", "rich"])
    captured = capsys.readouterr()
    plain = _strip_ansi(captured.out)

    assert exit_code == 0
    assert "Next Steps" in plain
    assert "whest init ./my-estimator" in plain
    assert "whest run --estimator ./my-estimator/estimator.py --runner local" in plain


def test_smoke_test_plain_output_uses_shared_report_shape(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "run_default_report",
        lambda **_kwargs: _sample_report(profile_enabled=False, detail="raw"),
    )

    exit_code = cli.main(["smoke-test", "--format", "plain"])
    captured = capsys.readouterr()

    assert exit_code == 0
    plain = _strip_ansi(captured.out)
    assert plain.index("Run Context") < plain.index("Hardware & Runtime")
    assert plain.index("Hardware & Runtime") < plain.index("Final Score")
    assert plain.index("Final Score") < plain.index("Next Steps")
    assert "WhestBench Report (Plain Text)" not in plain


def test_smoke_test_human_mode_surfaces_validation_errors_readably(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress: Any = None
    ) -> dict:
        raise ValueError("Predictions must have shape (2, 4), got (1, 4).")

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)

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


def test_print_error_renders_structured_details_and_unknown_keys(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = {
        "ok": False,
        "error": {
            "stage": "validate",
            "code": "ESTIMATOR_BAD_SHAPE",
            "message": "Predictions must have shape (2, 4), got (4, 2).",
            "details": {
                "expected_shape": [2, 4],
                "got_shape": [4, 2],
                "hint": "Returned predictions appear to be transposed.",
                "extra_note": "Read the estimator contract.",
            },
        },
    }

    cli._print_error(payload, json_output=False, debug=False)
    captured = capsys.readouterr()

    assert "Expected shape: [2, 4]" in captured.out
    assert "Got shape: [4, 2]" in captured.out
    assert "Hint: Returned predictions appear to be transposed." in captured.out
    assert "extra_note: Read the estimator contract." in captured.out


def test_error_payload_keeps_generic_exception_details() -> None:
    exc = ValueError("bad shape")
    exc.details = {"hint": "Return shape (depth, width)."}  # type: ignore[attr-defined]

    payload = cli._error_payload(exc, include_traceback=False, stage="validate")

    assert payload["error"]["details"] == {"hint": "Return shape (depth, width)."}


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

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "server"])
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

    exit_code = cli.main(
        ["run", "--estimator", "estimator.py", "--runner", "inprocess", "--format", "rich"]
    )
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "Error [setup:SETUP_ERROR]: runner failed" in captured.out
    assert "Use --debug to include a traceback." in captured.out
    assert "rerun with --runner local --debug" not in captured.out


def test_run_default_uses_local_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, str] = {}

    def fake_run_estimator_with_runner(*_args: Any, **_kwargs: Any) -> dict:
        observed["runner"] = type(_args[0]).__name__
        return _sample_report(profile_enabled=False, detail="raw")

    monkeypatch.setattr(cli, "_run_estimator_with_runner", fake_run_estimator_with_runner)
    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )

    exit_code = cli.main(["run", "--estimator", "estimator.py"])

    assert exit_code == 0
    assert observed.get("runner") == "LocalRunner"


def test_run_server_alias_is_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, str] = {}

    def fake_run_estimator_with_runner(*_args: Any, **_kwargs: Any) -> dict:
        observed["runner"] = type(_args[0]).__name__
        return _sample_report(profile_enabled=False, detail="raw")

    monkeypatch.setattr(cli, "_run_estimator_with_runner", fake_run_estimator_with_runner)
    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "server"])

    assert exit_code == 0
    assert observed.get("runner") == "SubprocessRunner"


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
        lambda _report, *, show_diagnostic_plots=False, debug=False, output_format="rich", **_kwargs: (
            "results\n"
        ),
    )

    exit_code = cli.main(
        ["run", "--estimator", "estimator.py", "--runner", "inprocess", "--format", "rich"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "results" in captured.out
    assert observed["initial_finished"] == "n/a"
    assert observed["initial_duration"] is None
    assert observed["total"] == 10
    assert observed["progress_event"] == {"completed": 1}
    assert observed["final_meta"]["run_finished_at_utc"] == "2026-03-01T00:00:03+00:00"
    assert observed["final_meta"]["run_duration_s"] == 3.0


def test_smoke_test_json_flag_returns_machine_readable_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "run_default_report",
        lambda **_kwargs: _sample_report(profile_enabled=False, detail="raw"),
    )

    exit_code = cli.main(["smoke-test", "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["mode"] == "agent"
    assert payload["results"]["primary_score"] == 0.42


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


# --- Progress task visibility/start-state (Paul's "both timers tick" report) -


def _find_task(progress: Any, task_id: int) -> Any:
    return next(t for t in progress.tasks if t.id == task_id)


def test_live_top_pane_scoring_task_starts_hidden_and_unstarted() -> None:
    """Before any scoring event, the scoring task's elapsed timer must not tick."""
    session = cli._LiveTopPaneSession({"run_config": {}, "run_meta": {}}, total=5, n_mlps=5)
    scoring = _find_task(session._progress, session._scoring_task_id)
    gen = _find_task(session._progress, session._gen_task_id)

    assert scoring.visible is False
    assert scoring.started is False
    assert gen.visible is True
    assert gen.started is True


def test_live_top_pane_scoring_task_hidden_through_generating_phase() -> None:
    """Emitting generating events must not reveal or start the scoring task."""
    session = cli._LiveTopPaneSession({"run_config": {}, "run_meta": {}}, total=5, n_mlps=5)

    for i in range(1, 6):
        session.on_progress({"phase": "generating", "completed": i, "total": 5})

    scoring = _find_task(session._progress, session._scoring_task_id)
    assert scoring.visible is False
    assert scoring.started is False


def test_live_top_pane_scoring_task_revealed_on_first_scoring_event() -> None:
    session = cli._LiveTopPaneSession({"run_config": {}, "run_meta": {}}, total=5, n_mlps=5)
    # Go through generating first, then first scoring event.
    for i in range(1, 6):
        session.on_progress({"phase": "generating", "completed": i, "total": 5})
    session.on_progress({"phase": "scoring", "completed": 1})

    scoring = _find_task(session._progress, session._scoring_task_id)
    assert scoring.visible is True
    assert scoring.started is True
    assert scoring.completed == 1


def test_progress_callback_scoring_task_starts_hidden_and_unstarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The Rich-based _progress_callback must also keep the scoring task
    dormant until its phase begins."""
    # Force the Rich-tqdm codepath (not classic_tqdm).
    monkeypatch.setattr(cli, "rich_tqdm", object(), raising=False)
    captured: dict[str, Any] = {}

    # Patch Live so we don't actually take over the terminal in the test.
    class _FakeLive:
        def __init__(self, *_a: Any, **_kw: Any) -> None:
            pass

        def start(self) -> None:
            pass

        def stop(self) -> None:
            pass

    monkeypatch.setattr(cli, "Live", _FakeLive)

    # Intercept the Progress object by patching its constructor to record it.
    original_progress_cls = cli.Progress

    def _capture_progress(*args: Any, **kwargs: Any) -> Any:
        progress = original_progress_cls(*args, **kwargs)
        captured["progress"] = progress
        return progress

    monkeypatch.setattr(cli, "Progress", _capture_progress)

    with cli._progress_callback(total=5, n_mlps=5) as on_progress:
        progress = captured["progress"]
        tasks_before = list(progress.tasks)
        assert len(tasks_before) == 2
        gen_task, scoring_task = tasks_before
        assert gen_task.visible is True and gen_task.started is True
        assert scoring_task.visible is False and scoring_task.started is False

        # Emit generating events — scoring task must remain dormant.
        for i in range(1, 6):
            on_progress({"phase": "generating", "completed": i, "total": 5})
        scoring_task = next(t for t in progress.tasks if t.id == scoring_task.id)
        assert scoring_task.visible is False
        assert scoring_task.started is False

        # First scoring event reveals and starts the task.
        on_progress({"phase": "scoring", "completed": 1})
        scoring_task = next(t for t in progress.tasks if t.id == scoring_task.id)
        assert scoring_task.visible is True
        assert scoring_task.started is True
        assert scoring_task.completed == 1
