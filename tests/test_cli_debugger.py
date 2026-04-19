"""Tests for format selection and debugger-aware Rich suppression in cli.py."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Dict

import whestbench.cli as cli


def _sample_report() -> Dict[str, Any]:
    return {
        "mode": "human",
        "run_meta": {"run_duration_s": 1.0},
        "run_config": {
            "n_mlps": 1,
            "width": 4,
            "depth": 3,
            "flop_budget": 1000,
        },
        "results": {
            "primary_score": 0.5,
            "secondary_score": 0.6,
            "per_mlp": [],
            "breakdowns": {},
        },
    }


class _SessionSpy:
    """Raises if constructed — used to assert Rich Live is NOT entered."""

    instantiations = 0

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        type(self).instantiations += 1
        raise AssertionError("_LiveTopPaneSession must not be instantiated in plain-output path")


def _patch_run_to_skip_rich(monkeypatch) -> None:
    """Shared patches so a `run` invocation can complete without real work."""
    monkeypatch.setattr(
        cli,
        "resolve_estimator_class_metadata",
        lambda *_a, **_k: type("Meta", (), {"class_name": "Estimator"})(),
        raising=False,
    )
    monkeypatch.setattr(cli, "_run_estimator_with_runner", lambda *_a, **_k: _sample_report())
    # Any code path that reaches the Rich fallback-when-exception branch would
    # still try to call render_human_results; stub it so failure modes don't
    # hide behind rendering errors.
    monkeypatch.setattr(
        cli,
        "render_human_results",
        lambda *_a, **_k: (
            "WhestBench Report\n" if _k.get("output_format") == "plain" else "RICH OUTPUT"
        ),
        raising=False,
    )


def test_format_plain_skips_live_session_in_run(monkeypatch, capsys) -> None:
    _patch_run_to_skip_rich(monkeypatch)
    _SessionSpy.instantiations = 0
    monkeypatch.setattr(cli, "_LiveTopPaneSession", _SessionSpy, raising=True)
    monkeypatch.setattr(cli, "_live_top_pane_session", _SessionSpy, raising=True)

    exit_code = cli.main(
        ["run", "--estimator", "estimator.py", "--runner", "inprocess", "--format", "plain"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert _SessionSpy.instantiations == 0
    assert "WhestBench Report" in captured.out
    # No Rich-styled tip block in plain-output mode.
    assert "Use --json for JSON output" not in captured.out
    assert "Use [green]--json[/]" not in captured.out


def test_format_plain_skips_live_session_in_smoke_test(monkeypatch, capsys) -> None:
    def fake_run_default_report(
        *, profile: bool = False, detail: str = "raw", progress=None
    ) -> Dict[str, Any]:
        # Drive the progress callback to exercise the plain-text callback path.
        if progress is not None:
            progress({"phase": "ground_truth", "completed": 1, "total": 1})
            progress({"phase": "scoring", "completed": 1, "total": 1})
        return _sample_report()

    monkeypatch.setattr(cli, "run_default_report", fake_run_default_report)

    exit_code = cli.main(["smoke-test", "--format", "plain"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "WhestBench Report" in captured.out
    assert "[smoke-test] ground_truth: 1/1" in captured.err
    assert "[smoke-test] scoring: 1/1" in captured.err


def test_smoke_test_format_rich_forces_plain_when_debugger_detected(monkeypatch, capsys) -> None:
    monkeypatch.setenv("PYTHONBREAKPOINT", "pdb.set_trace")
    monkeypatch.setattr(
        cli,
        "run_default_report",
        lambda **_kwargs: _sample_report(),
    )

    exit_code = cli.main(["smoke-test", "--format", "rich"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "forcing plain output" in captured.err.lower()
    assert "WhestBench Report" in captured.out
    assert "\x1b[" not in captured.out


def test_run_format_rich_forces_plain_when_debugger_detected(monkeypatch, capsys) -> None:
    _patch_run_to_skip_rich(monkeypatch)
    _SessionSpy.instantiations = 0
    monkeypatch.setattr(cli, "_LiveTopPaneSession", _SessionSpy, raising=True)
    monkeypatch.setattr(cli, "_live_top_pane_session", _SessionSpy, raising=True)

    monkeypatch.setenv("PYTHONBREAKPOINT", "pdb.set_trace")

    exit_code = cli.main(
        ["run", "--estimator", "estimator.py", "--runner", "inprocess", "--format", "rich"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert _SessionSpy.instantiations == 0
    assert (
        "Debugger detected (sys.gettrace / PYTHONBREAKPOINT); forcing plain output." in captured.err
    )
    assert "WhestBench Report" in captured.out


def test_pythonbreakpoint_zero_does_not_promote(monkeypatch, capsys) -> None:
    """PYTHONBREAKPOINT=0 is the documented way to disable breakpoint()."""
    _patch_run_to_skip_rich(monkeypatch)
    monkeypatch.setenv("PYTHONBREAKPOINT", "0")
    # Make sure no stale trace function leaks in from pytest internals.
    monkeypatch.setattr(sys, "gettrace", lambda: None)

    # Also block _LiveTopPaneSession but allow _live_top_pane_session so we
    # can verify the non-plain path would try to instantiate it. The run
    # path needs a Live to complete, so we swap in a no-op context manager.
    @contextmanager
    def _fake_live(*_a, **_k):
        class _S:
            def on_progress(self, _event: Dict[str, Any]) -> None:
                return None

            def update_run_meta(self, _rm: Dict[str, Any]) -> None:
                return None

        yield _S()

    monkeypatch.setattr(cli, "_live_top_pane_session", _fake_live, raising=True)

    exit_code = cli.main(["run", "--estimator", "estimator.py", "--runner", "inprocess"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Debugger detected" not in captured.err


def test_sys_gettrace_forces_plain(monkeypatch, capsys) -> None:
    _patch_run_to_skip_rich(monkeypatch)
    _SessionSpy.instantiations = 0
    monkeypatch.setattr(cli, "_LiveTopPaneSession", _SessionSpy, raising=True)
    monkeypatch.setattr(cli, "_live_top_pane_session", _SessionSpy, raising=True)

    # sys.gettrace() returns a non-None value (e.g. under `python -m pdb` or
    # an attached profiler). Patch gettrace rather than installing a real
    # trace function so pytest's own trace state isn't disturbed.
    monkeypatch.setattr(sys, "gettrace", lambda: lambda *a, **k: None)

    exit_code = cli.main(
        ["run", "--estimator", "estimator.py", "--runner", "inprocess", "--format", "rich"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert _SessionSpy.instantiations == 0
    assert "Debugger detected" in captured.err
    assert "WhestBench Report" in captured.out


def test_debugger_active_only_checks_trace_and_pythonbreakpoint(monkeypatch) -> None:
    monkeypatch.delenv("PYTHONBREAKPOINT", raising=False)
    monkeypatch.setattr(sys, "gettrace", lambda: None)
    assert cli._debugger_active() is False

    monkeypatch.setattr(sys, "gettrace", lambda: lambda *a, **k: None)
    assert cli._debugger_active() is True

    monkeypatch.setattr(sys, "gettrace", lambda: None)
    monkeypatch.setenv("PYTHONBREAKPOINT", "pdb.set_trace")
    assert cli._debugger_active() is True

    monkeypatch.setenv("PYTHONBREAKPOINT", "0")
    assert cli._debugger_active() is False

    monkeypatch.setenv("PYTHONBREAKPOINT", "")
    assert cli._debugger_active() is False


def test_breakpointhook_wrapper_stops_live(monkeypatch) -> None:
    """Simulates breakpoint() firing while _LiveTopPaneSession is active."""
    # Record hook calls without actually entering pdb.
    hook_calls: list = []

    def _fake_prev_hook(*args: Any, **kwargs: Any) -> str:
        hook_calls.append((args, kwargs))
        return "hook-return"

    monkeypatch.setattr(sys, "breakpointhook", _fake_prev_hook)

    # Build a session and monkeypatch the expensive pieces so construction
    # doesn't depend on a real terminal. We only need .start() and .stop()
    # semantics plus the hook installation.
    session = cli._LiveTopPaneSession.__new__(cli._LiveTopPaneSession)
    session._pre_report = {"run_meta": {}}  # type: ignore[attr-defined]
    session._progress = None  # type: ignore[attr-defined]
    session._gen_task_id = 0  # type: ignore[attr-defined]
    session._scoring_task_id = 0  # type: ignore[attr-defined]
    session._prev_breakpointhook = None  # type: ignore[attr-defined]

    stop_calls = {"count": 0}

    class _FakeLive:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            stop_calls["count"] += 1

        def update(self, _renderable) -> None:
            return None

    session._live = _FakeLive()  # type: ignore[attr-defined]

    # Exercise the real start/stop methods (which install/restore the hook).
    session.start()
    assert sys.breakpointhook is not _fake_prev_hook, "wrapper should be installed"

    # Fire the wrapper hook (simulating breakpoint() from inside predict()).
    result = sys.breakpointhook("arg1", keyword="val")

    # Wrapper should have stopped Live, restored the prev hook, and
    # delegated to it (passing args/kwargs through).
    assert stop_calls["count"] >= 1
    assert sys.breakpointhook is _fake_prev_hook
    assert hook_calls == [(("arg1",), {"keyword": "val"})]
    assert result == "hook-return"

    # Idempotent stop — calling stop() after the hook already ran is a no-op
    # on the hook slot.
    session.stop()
    assert sys.breakpointhook is _fake_prev_hook


def test_breakpointhook_restored_on_clean_stop(monkeypatch) -> None:
    """When Live ends without any breakpoint firing, the hook is restored."""
    sentinel = lambda *a, **k: None  # noqa: E731
    monkeypatch.setattr(sys, "breakpointhook", sentinel)

    session = cli._LiveTopPaneSession.__new__(cli._LiveTopPaneSession)
    session._prev_breakpointhook = None  # type: ignore[attr-defined]

    class _FakeLive:
        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    session._live = _FakeLive()  # type: ignore[attr-defined]

    session.start()
    assert sys.breakpointhook is not sentinel
    session.stop()
    assert sys.breakpointhook is sentinel
