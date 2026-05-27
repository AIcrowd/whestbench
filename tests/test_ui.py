"""Unit tests for whestbench.ui — shared UX helpers."""

from __future__ import annotations

import io
from unittest.mock import patch

import pytest
from rich.console import Console
from rich.progress import Progress

from whestbench.ui import (
    format_bytes,
    format_duration,
    format_throughput,
    progress_bytes,
    progress_count,
    say,
    status,
)


def _make_console() -> tuple[Console, io.StringIO]:
    """Build a Rich console that writes to a capturable buffer.

    ``force_terminal=True`` keeps the markup logic active even though our buffer
    isn't a real TTY; ``color_system=None`` drops ANSI codes so we can grep
    plain text.
    """
    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        color_system=None,
        width=120,
        highlight=False,
    )
    return console, buf


@pytest.mark.parametrize(
    "n_bytes,expected",
    [
        (0, "0 B"),
        (1, "1 B"),
        (1023, "1023 B"),
        (1024, "1.0 KB"),
        (1536, "1.5 KB"),
        (1_048_576, "1.0 MB"),
        (1_572_864, "1.5 MB"),
        (1_073_741_824, "1.0 GB"),
        (2_118_949_161, "2.0 GB"),
        (1_099_511_627_776, "1.0 TB"),
        # Boundary cases — values just under the next unit must roll over,
        # not render as "1024.0 <prev unit>" (regression test for C1).
        (1024**2 - 1, "1.0 MB"),
        (1024**3 - 1, "1.0 GB"),
        (1024**4 - 1, "1.0 TB"),
        (1024**5 - 1, "1.0 PB"),
    ],
)
def test_format_bytes(n_bytes: int, expected: str) -> None:
    assert format_bytes(n_bytes) == expected


def test_format_bytes_negative_raises() -> None:
    with pytest.raises(ValueError):
        format_bytes(-1)


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0.0, "0ms"),
        (0.05, "50ms"),
        (0.999, "999ms"),
        (1.0, "1.0s"),
        (2.137, "2.1s"),
        (31.7, "31.7s"),
        (59.9, "59.9s"),
        (60.0, "1m 0s"),
        (125.0, "2m 5s"),
        (3599.0, "59m 59s"),
        (3600.0, "1h 0m 0s"),
        (7325.0, "2h 2m 5s"),
    ],
)
def test_format_duration(seconds: float, expected: str) -> None:
    assert format_duration(seconds) == expected


def test_format_duration_negative_raises() -> None:
    with pytest.raises(ValueError):
        format_duration(-0.1)


@pytest.mark.parametrize(
    "n_bytes,seconds,expected",
    [
        (2_000_000_000, 30.0, "63.6 MB/s"),
        (1_048_576, 1.0, "1.0 MB/s"),
        (512, 1.0, "512 B/s"),
        (1_073_741_824, 1.0, "1.0 GB/s"),
    ],
)
def test_format_throughput(n_bytes: int, seconds: float, expected: str) -> None:
    assert format_throughput(n_bytes, seconds) == expected


def test_format_throughput_zero_seconds_returns_dash() -> None:
    assert format_throughput(1024, 0.0) == "— /s"


# ---------------------------------------------------------------------------
# say.* message helpers


def test_say_intent_emits_message() -> None:
    console, buf = _make_console()
    say.intent("Downloading hf://foo — 2.0 GB", console=console)
    assert "Downloading hf://foo — 2.0 GB" in buf.getvalue()


def test_say_step_emits_message() -> None:
    console, buf = _make_console()
    say.step("Resolving revision v1-warmup", console=console)
    assert "Resolving revision v1-warmup" in buf.getvalue()


def test_say_ok_includes_check_and_message() -> None:
    console, buf = _make_console()
    say.ok("Loaded 1,000 MLPs in 2.1s", console=console)
    out = buf.getvalue()
    assert "✓" in out
    assert "Loaded 1,000 MLPs in 2.1s" in out


def test_say_warn_includes_warning_glyph_and_message() -> None:
    console, buf = _make_console()
    say.warn("Streaming — data not cached", console=console)
    out = buf.getvalue()
    assert "⚠" in out
    assert "Streaming — data not cached" in out


def test_say_hint_includes_tip_label_and_message() -> None:
    console, buf = _make_console()
    say.hint("Set HF_TOKEN to lift rate limits", console=console)
    out = buf.getvalue()
    assert "tip" in out.lower()
    assert "HF_TOKEN" in out


@pytest.mark.parametrize(
    "method,arg",
    [
        ("intent", "hello"),
        ("step", "hello"),
        ("ok", "hello"),
        ("warn", "hello"),
        ("hint", "hello"),
    ],
)
def test_say_quiet_suppresses_output(method: str, arg: str) -> None:
    console, buf = _make_console()
    getattr(say, method)(arg, console=console, quiet=True)
    assert buf.getvalue() == ""


def test_say_intent_emits_even_when_hf_progress_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    console, buf = _make_console()
    say.intent("hello", console=console)
    assert "hello" in buf.getvalue()


# ---------------------------------------------------------------------------
# progress_bytes


def test_progress_bytes_advance_updates_bar() -> None:
    console, buf = _make_console()
    with progress_bytes(total=1000, label="dl", console=console) as p:
        p.advance(500)
        p.advance(500)
    out = buf.getvalue()
    # Label should appear in the output and the bar should have rendered.
    assert "dl" in out


def test_progress_bytes_update_completed_sets_absolute() -> None:
    console, buf = _make_console()
    with progress_bytes(total=2000, label="dl", console=console) as p:
        p.update(completed=1024)
    out = buf.getvalue()
    assert "dl" in out


def test_progress_bytes_advance_forwards_to_rich_progress_update() -> None:
    """Spy on ``Progress.update`` to confirm ``advance(N)`` actually advances."""
    console, _buf = _make_console()
    original_update = Progress.update
    with patch.object(Progress, "update", autospec=True) as spy:
        spy.side_effect = original_update
        with progress_bytes(total=1000, label="dl", console=console) as p:
            p.advance(250)
            p.advance(750)
    advance_calls = [c for c in spy.call_args_list if c.kwargs.get("advance") is not None]
    assert [c.kwargs["advance"] for c in advance_calls] == [250, 750]


def test_progress_bytes_update_forwards_to_rich_progress_update() -> None:
    """Spy on ``Progress.update`` to confirm ``update(completed=N)`` propagates."""
    console, _buf = _make_console()
    original_update = Progress.update
    with patch.object(Progress, "update", autospec=True) as spy:
        spy.side_effect = original_update
        with progress_bytes(total=2000, label="dl", console=console) as p:
            p.update(completed=1500)
    completed_calls = [c for c in spy.call_args_list if c.kwargs.get("completed") is not None]
    assert any(c.kwargs["completed"] == 1500 for c in completed_calls)


def test_progress_bytes_quiet_emits_nothing() -> None:
    console, buf = _make_console()
    with progress_bytes(total=1000, label="dl", console=console, quiet=True) as p:
        p.advance(100)
        p.update(completed=500)
    assert buf.getvalue() == ""


@pytest.mark.parametrize("value", ["1", "ON", "true", "YES", "On", "TRUE", "yes"])
def test_progress_bytes_disabled_env_truthy_values_emit_nothing(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # HF Hub convention: only {"1","ON","TRUE","YES"} (case-insensitive) are
    # truthy; whestbench's progress helpers follow the same rule.
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", value)
    console, buf = _make_console()
    with progress_bytes(total=1000, label="dl", console=console) as p:
        p.advance(500)
    assert buf.getvalue() == ""


@pytest.mark.parametrize(
    "value", ["0", "false", "False", "", "no", "off", "FALSE", "NO", "Off", "anything-else"]
)
def test_progress_bytes_disabled_env_non_truthy_values_still_emit(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # Anything outside the HF-truthy allow-list keeps progress on — including
    # "no", "off", "false", empty string, and unrecognised values.
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", value)
    console, buf = _make_console()
    with progress_bytes(total=1000, label="dl", console=console) as p:
        p.advance(500)
    # The label should appear — env is treated as not-set / falsy.
    assert "dl" in buf.getvalue()


# ---------------------------------------------------------------------------
# progress_count


def test_progress_count_advance_updates_bar() -> None:
    console, buf = _make_console()
    with progress_count(total=100, label="Sampling", console=console) as p:
        p.advance(50)
        p.advance(50)
    assert "Sampling" in buf.getvalue()


def test_progress_count_advance_forwards_to_rich_progress_update() -> None:
    """Spy on ``Progress.update`` to confirm ``advance(N)`` actually advances."""
    console, _buf = _make_console()
    original_update = Progress.update
    with patch.object(Progress, "update", autospec=True) as spy:
        spy.side_effect = original_update
        with progress_count(total=100, label="Sampling", console=console) as p:
            p.advance(10)
            p.advance(40)
    advance_calls = [c for c in spy.call_args_list if c.kwargs.get("advance") is not None]
    assert [c.kwargs["advance"] for c in advance_calls] == [10, 40]


def test_progress_count_update_forwards_to_rich_progress_update() -> None:
    """Spy on ``Progress.update`` to confirm ``update(completed=N)`` propagates."""
    console, _buf = _make_console()
    original_update = Progress.update
    with patch.object(Progress, "update", autospec=True) as spy:
        spy.side_effect = original_update
        with progress_count(total=100, label="Sampling", console=console) as p:
            p.update(completed=42)
    completed_calls = [c for c in spy.call_args_list if c.kwargs.get("completed") is not None]
    assert any(c.kwargs["completed"] == 42 for c in completed_calls)


def test_progress_count_quiet_emits_nothing() -> None:
    console, buf = _make_console()
    with progress_count(total=100, label="Sampling", console=console, quiet=True) as p:
        p.advance(10)
    assert buf.getvalue() == ""


def test_progress_count_disabled_env_emits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    console, buf = _make_console()
    with progress_count(total=100, label="Sampling", console=console) as p:
        p.advance(10)
    assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# status spinner


def test_status_context_manager_does_not_raise() -> None:
    console, _buf = _make_console()
    with status("Loading from cache", console=console):
        pass


def test_status_invokes_console_status_with_label() -> None:
    """Spy on ``Console.status`` and assert it was called once with the label."""
    console, _buf = _make_console()
    with patch.object(Console, "status", wraps=console.status) as spy:
        with status("Loading from cache", console=console):
            pass
    assert spy.call_count == 1
    args, _kwargs = spy.call_args
    assert args[0] == "Loading from cache"


def test_status_quiet_does_not_invoke_console_status() -> None:
    """Quiet mode should bypass ``Console.status`` entirely."""
    console, _buf = _make_console()
    with patch.object(Console, "status", wraps=console.status) as spy:
        with status("Loading from cache", console=console, quiet=True):
            pass
    assert spy.call_count == 0


def test_status_quiet_emits_nothing() -> None:
    console, buf = _make_console()
    with status("Loading from cache", console=console, quiet=True):
        pass
    assert buf.getvalue() == ""


def test_status_disabled_env_emits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    console, buf = _make_console()
    with status("Loading from cache", console=console):
        pass
    assert buf.getvalue() == ""
