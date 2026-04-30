# tests/test_profiler.py
"""Tests for the simulation profiler."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import flopscope as flops
import pytest

import whestbench.cli as cli
import whestbench.profiler as profiler
from whestbench.profiler import (
    PRESETS,
    CorrectnessResult,
    PresetConfig,
    TimingResult,
    correctness_check,
    format_compact_output,
    format_dims,
    run_profile,
)


class TestCorrectnessCheck:
    def test_whest_passes(self) -> None:
        result = correctness_check()
        assert result.passed is True
        assert result.error == ""


class TestRunProfile:
    def test_quick_preset_runs(self) -> None:
        terminal_output, _ = run_profile(preset_name="super-quick")
        assert "flopscope" in terminal_output
        assert "Detail" in terminal_output

    def test_terminal_output_contains_compact_summary(self) -> None:
        terminal_output, _ = run_profile(preset_name="super-quick")
        assert "Correctness" in terminal_output
        assert "Detail" in terminal_output
        assert "Use --verbose for full timing tables with raw times" in terminal_output

    def test_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "results.json")
            _, json_data = run_profile(
                preset_name="super-quick",
                output_path=out_path,
            )
            assert json_data is not None
            assert "hardware" in json_data
            assert "timing" in json_data
            assert "correctness" in json_data

            # Verify file was written
            with open(out_path) as f:
                saved = json.load(f)
            assert saved["hardware"]["cpu_count_logical"] is not None

    def test_single_backend_in_output(self) -> None:
        terminal_output, _ = run_profile(preset_name="super-quick")
        assert "flopscope" in terminal_output

    def test_failed_correctness_output_uses_error_detail_hint(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            profiler,
            "correctness_check",
            lambda: CorrectnessResult(backend_name="flopscope", passed=False, error="boom"),
        )

        terminal_output, _ = run_profile(preset_name="super-quick")

        assert "Correctness" in terminal_output
        assert "boom" in terminal_output
        assert "Use --verbose for error details." in terminal_output
        assert "Use --verbose for full timing tables with raw times" not in terminal_output


class TestFormatDims:
    def test_small_number(self) -> None:
        assert format_dims(64, 4, 500) == "64\u00d74\u00d7500"

    def test_thousands(self) -> None:
        assert format_dims(64, 4, 10_000) == "64\u00d74\u00d710k"

    def test_hundreds_of_thousands(self) -> None:
        assert format_dims(256, 8, 100_000) == "256\u00d78\u00d7100k"

    def test_millions(self) -> None:
        assert format_dims(256, 8, 1_000_000) == "256\u00d78\u00d71M"

    def test_large_millions(self) -> None:
        assert format_dims(256, 8, 16_700_000) == "256\u00d78\u00d716.7M"


class TestPresets:
    def test_all_presets_exist(self) -> None:
        assert "quick" in PRESETS
        assert "standard" in PRESETS
        assert "exhaustive" in PRESETS

    def test_quick_is_smallest(self) -> None:
        q = PRESETS["quick"]
        s = PRESETS["standard"]
        assert len(q.widths) <= len(s.widths)
        assert len(q.n_samples_list) <= len(s.n_samples_list)


class TestFormatCompactOutput:
    def _make_results(self):
        """Build minimal test data with single flopscope backend."""
        correctness = [
            CorrectnessResult(backend_name="flopscope", passed=True),
        ]
        timing = [
            TimingResult(
                backend_name="flopscope",
                operation="run_mlp",
                width=64,
                depth=4,
                n_samples=1000,
                times=[0.0001],
                median_time=0.0001,
                speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="flopscope",
                operation="sample_layer_statistics",
                width=64,
                depth=4,
                n_samples=1000,
                times=[0.0009],
                median_time=0.0009,
                speedup_vs_numpy=1.0,
            ),
        ]
        skipped: dict = {}
        hardware = {
            "platform": "macOS-26.3-arm64",
            "machine": "arm64",
            "cpu_count_physical": 16,
            "cpu_count_logical": 16,
            "ram_total_bytes": 64 * 1024**3,
            "python_version": "3.10.17",
            "numpy_version": "2.2.6",
            "flopscope_version": flops.__version__,
            "os": "Darwin",
        }
        return correctness, timing, skipped, hardware

    def test_contains_hardware_context_line(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "arm64" in output
        assert "16 cores" in output
        assert "64.0 GB" in output

    def test_no_skipped_line_when_none_skipped(self) -> None:
        cr, tr, _, hw = self._make_results()
        output = format_compact_output(cr, tr, {}, hardware_info=hw)
        assert "Skipped:" not in output

    def test_contains_detail_table(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "Detail" in output

    def test_contains_verbose_hint(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "--verbose" in output

    def test_single_backend_omits_leaderboard(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "Leaderboard" not in output

    def test_zero_passed_backends(self) -> None:
        cr = [CorrectnessResult(backend_name="flopscope", passed=False, error="boom")]
        output = format_compact_output(cr, [], {})
        assert "No backends passed" in output


class TestRunProfileVerbose:
    def test_default_uses_compact_format(self) -> None:
        """Default (verbose=False) should use compact leaderboard format."""
        output, _ = run_profile(preset_name="super-quick")
        assert "Leaderboard" in output or "Detail" in output
        # Should NOT contain the old-style verbose headers
        assert "Timing Results" not in output

    def test_verbose_includes_both_compact_and_full(self) -> None:
        """verbose=True should show compact output PLUS full tables."""
        output, _ = run_profile(preset_name="super-quick", verbose=True)
        # Compact content present
        assert "Detail" in output
        # Full verbose tables also present
        assert "Timing Results" in output

    def test_plain_verbose_stays_on_shared_plain_output(self) -> None:
        output, _ = run_profile(
            preset_name="super-quick",
            verbose=True,
            output_format="plain",
        )

        assert "Simulation Profile" in output
        assert "Detail" in output
        assert "Timing Results" not in output

    def test_multi_dim_leaderboard_grouping(self) -> None:
        """Multiple dimension combos should produce separate leaderboard groups."""
        PRESETS["_ci"] = PresetConfig(
            widths=[64],
            depths=[4, 8],
            n_samples_list=[1_000],
        )
        try:
            output, _ = run_profile(preset_name="_ci", verbose=False)
        finally:
            del PRESETS["_ci"]
        assert "Detail" in output


class TestLogProgress:
    def test_log_progress_prints_lines(self, capsys) -> None:
        """Non-TTY mode should print one line per benchmark step."""
        run_profile(preset_name="super-quick", log_progress=True)
        captured = capsys.readouterr()
        assert "[correctness]" in captured.out
        assert "[timing]" in captured.out
        assert "flopscope" in captured.out
        assert "[done]" in captured.out

    def test_log_progress_off_by_default(self, capsys) -> None:
        """Default mode should not print log lines."""
        run_profile(preset_name="super-quick")
        captured = capsys.readouterr()
        assert "[correctness]" not in captured.out


class TestCLIFlags:
    def test_verbose_flag_accepted(self) -> None:
        from whestbench.cli import _build_participant_parser

        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--preset", "super-quick", "--verbose"])
        assert args.verbose is True

    def test_verbose_flag_default_false(self) -> None:
        from whestbench.cli import _build_participant_parser

        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--preset", "super-quick"])
        assert args.verbose is False

    def test_log_progress_flag_accepted(self) -> None:
        from whestbench.cli import _build_participant_parser

        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--log-progress"])
        assert args.log_progress is True

    def test_log_progress_flag_default_false(self) -> None:
        from whestbench.cli import _build_participant_parser

        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation"])
        assert args.log_progress is False


def test_profile_simulation_format_json_uses_json_payload(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        profiler,
        "run_profile",
        lambda **_kwargs: (
            "terminal output",
            {
                "hardware": {"cpu_count_logical": 8},
                "timing": [{"backend": "flopscope"}],
                "correctness": [{"backend": "flopscope", "passed": True, "error": ""}],
            },
        ),
        raising=False,
    )

    exit_code = cli.main(["profile-simulation", "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["hardware"]["cpu_count_logical"] == 8
    assert payload["timing"] == [{"backend": "flopscope"}]


def test_profile_simulation_format_plain_uses_plain_human_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        profiler,
        "run_profile",
        lambda **_kwargs: (
            "Simulation Profile\nCommand: profile-simulation\nStatus: success\n\nDetail\n",
            {
                "hardware": {"cpu_count_logical": 8},
                "timing": [{"backend": "flopscope"}],
                "correctness": [{"backend": "flopscope", "passed": True, "error": ""}],
            },
        ),
        raising=False,
    )

    exit_code = cli.main(["profile-simulation", "--format", "plain"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Simulation Profile" in captured.out
    assert "Command: profile-simulation" in captured.out
    assert captured.out.lstrip()[0] != "{"


def test_profile_simulation_plain_uses_shared_presenter_path(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}

    def fake_render_command_presentation(doc, *, output_format, force_terminal, **_kwargs):
        observed["title"] = doc.title
        observed["output_format"] = output_format
        observed["force_terminal"] = force_terminal
        return "shared profile output\n"

    monkeypatch.setattr(profiler, "render_command_presentation", fake_render_command_presentation)

    exit_code = cli.main(["profile-simulation", "--preset", "super-quick", "--format", "plain"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "shared profile output\n"
    assert observed == {
        "title": "Simulation Profile",
        "output_format": "plain",
        "force_terminal": False,
    }


def test_profile_simulation_format_json_keeps_timing_warnings_off_stdout(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        profiler,
        "correctness_check",
        lambda: CorrectnessResult(backend_name="flopscope", passed=True, error=""),
    )

    def fake_run_timing_sweep(*_args, warning_stream=None, **_kwargs):
        if warning_stream is not None:
            print("[warning] timing skipped", file=warning_stream)
        return []

    monkeypatch.setattr(profiler, "run_timing_sweep", fake_run_timing_sweep, raising=False)

    exit_code = cli.main(["profile-simulation", "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["correctness"] == [{"backend": "flopscope", "passed": True, "error": ""}]
    assert "[warning]" not in captured.out
    assert "[warning] timing skipped" in captured.err


def test_profile_simulation_defaults_to_plain_in_non_tty_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    observed: dict[str, object] = {}

    def fake_run_profile(**kwargs):
        observed.update(kwargs)
        return (
            "Simulation Profile\nCommand: profile-simulation\nStatus: success\n",
            {
                "hardware": {"cpu_count_logical": 8},
                "timing": [],
                "correctness": [{"backend": "flopscope", "passed": True, "error": ""}],
            },
        )

    monkeypatch.setattr(profiler, "run_profile", fake_run_profile, raising=False)

    exit_code = cli.main(["profile-simulation"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert observed["output_format"] == "plain"
    assert "Simulation Profile" in captured.out


def test_profile_simulation_format_json_log_progress_keeps_stdout_parseable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        profiler,
        "correctness_check",
        lambda: CorrectnessResult(backend_name="flopscope", passed=True, error=""),
    )
    monkeypatch.setattr(profiler, "run_timing_sweep", lambda *_args, **_kwargs: [], raising=False)

    exit_code = cli.main(["profile-simulation", "--format", "json", "--log-progress"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["correctness"] == [{"backend": "flopscope", "passed": True, "error": ""}]
    assert "[correctness]" not in captured.out
    assert "[timing]" not in captured.out
    assert "[done]" not in captured.out


def test_profile_simulation_format_json_does_not_depend_on_human_renderer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        profiler,
        "correctness_check",
        lambda: CorrectnessResult(backend_name="flopscope", passed=True, error=""),
    )
    monkeypatch.setattr(profiler, "run_timing_sweep", lambda *_args, **_kwargs: [], raising=False)
    monkeypatch.setattr(
        profiler,
        "render_command_presentation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("shared human boom")),
    )

    exit_code = cli.main(["profile-simulation", "--format", "json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["correctness"] == [{"backend": "flopscope", "passed": True, "error": ""}]
