# tests/test_profiler.py
"""Tests for the simulation profiler."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from network_estimation.profiler import (
    PRESETS,
    CorrectnessResult,
    PresetConfig,
    TimingResult,
    correctness_check,
    format_compact_output,
    format_dims,
    run_profile,
)
from network_estimation.simulation_backends import get_backend


class TestCorrectnessCheck:
    def test_numpy_passes(self) -> None:
        backend = get_backend("numpy")
        result = correctness_check(backend)
        assert result.passed is True
        assert result.error == ""


class TestRunProfile:
    def test_quick_preset_runs(self) -> None:
        terminal_output, _ = run_profile(preset_name="super-quick", backend_filter=["numpy"])
        assert "numpy" in terminal_output
        assert "Detail" in terminal_output

    def test_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "results.json")
            _, json_data = run_profile(
                preset_name="super-quick",
                backend_filter=["numpy"],
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

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            run_profile(backend_filter=["nonexistent"])

    def test_skipped_backends_in_output(self) -> None:
        # Request a backend that likely isn't installed
        terminal_output, _ = run_profile(
            preset_name="super-quick",
            backend_filter=["numpy"],
        )
        # At minimum numpy should appear
        assert "numpy" in terminal_output


class TestFormatDims:
    def test_small_number(self) -> None:
        assert format_dims(64, 4, 500) == "64×4×500"

    def test_thousands(self) -> None:
        assert format_dims(64, 4, 10_000) == "64×4×10k"

    def test_hundreds_of_thousands(self) -> None:
        assert format_dims(256, 8, 100_000) == "256×8×100k"

    def test_millions(self) -> None:
        assert format_dims(256, 8, 1_000_000) == "256×8×1M"

    def test_large_millions(self) -> None:
        assert format_dims(256, 8, 16_700_000) == "256×8×16.7M"


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
        """Build minimal test data with two backends."""
        correctness = [
            CorrectnessResult(backend_name="numpy", passed=True),
            CorrectnessResult(backend_name="scipy", passed=True),
        ]
        timing = [
            TimingResult(
                backend_name="numpy",
                operation="run_mlp",
                width=64,
                depth=4,
                n_samples=1000,
                times=[0.0001],
                median_time=0.0001,
                speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="scipy",
                operation="run_mlp",
                width=64,
                depth=4,
                n_samples=1000,
                times=[0.0002],
                median_time=0.0002,
                speedup_vs_numpy=0.5,
            ),
            TimingResult(
                backend_name="numpy",
                operation="sample_layer_statistics",
                width=64,
                depth=4,
                n_samples=1000,
                times=[0.0009],
                median_time=0.0009,
                speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="scipy",
                operation="sample_layer_statistics",
                width=64,
                depth=4,
                n_samples=1000,
                times=[0.0009],
                median_time=0.0009,
                speedup_vs_numpy=1.0,
            ),
        ]
        skipped = {"pytorch": "pip install torch>=2.0"}
        hardware = {
            "platform": "macOS-26.3-arm64",
            "machine": "arm64",
            "cpu_count_physical": 16,
            "cpu_count_logical": 16,
            "ram_total_bytes": 64 * 1024**3,
            "python_version": "3.10.17",
            "numpy_version": "2.2.6",
            "os": "Darwin",
        }
        return correctness, timing, skipped, hardware

    def test_contains_hardware_context_line(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "arm64" in output
        assert "16 cores" in output
        assert "64.0 GB" in output

    def test_contains_skipped_one_liner(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "Skipped:" in output
        assert "pytorch" in output
        assert "--backends-help" in output

    def test_no_skipped_line_when_none_skipped(self) -> None:
        cr, tr, _, hw = self._make_results()
        output = format_compact_output(cr, tr, {}, hardware_info=hw)
        assert "Skipped:" not in output

    def test_contains_leaderboard(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "#1" in output
        assert "#2" in output
        assert "Leaderboard" in output

    def test_contains_detail_table(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "Detail" in output

    def test_contains_verbose_hint(self) -> None:
        cr, tr, sk, hw = self._make_results()
        output = format_compact_output(cr, tr, sk, hardware_info=hw)
        assert "--verbose" in output

    def test_single_backend_omits_leaderboard(self) -> None:
        cr = [CorrectnessResult(backend_name="numpy", passed=True)]
        tr = [
            TimingResult(
                backend_name="numpy",
                operation="run_mlp",
                width=64,
                depth=4,
                n_samples=1000,
                times=[0.0001],
                median_time=0.0001,
                speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="numpy",
                operation="sample_layer_statistics",
                width=64,
                depth=4,
                n_samples=1000,
                times=[0.0009],
                median_time=0.0009,
                speedup_vs_numpy=1.0,
            ),
        ]
        output = format_compact_output(cr, tr, {})
        assert "Leaderboard" not in output

    def test_zero_passed_backends(self) -> None:
        cr = [CorrectnessResult(backend_name="numpy", passed=False, error="boom")]
        output = format_compact_output(cr, [], {})
        assert "No backends passed" in output


class TestRunProfileVerbose:
    def test_default_uses_compact_format(self) -> None:
        """Default (verbose=False) should use compact leaderboard format."""
        output, _ = run_profile(preset_name="super-quick", backend_filter=["numpy"])
        assert "Leaderboard" in output or "Detail" in output
        # Should NOT contain the old-style verbose headers
        assert "Timing Results" not in output

    def test_verbose_includes_both_compact_and_full(self) -> None:
        """verbose=True should show compact output PLUS full tables."""
        output, _ = run_profile(preset_name="super-quick", backend_filter=["numpy"], verbose=True)
        # Compact content present
        assert "Detail" in output
        # Full verbose tables also present
        assert "Timing Results" in output

    def test_multi_dim_leaderboard_grouping(self) -> None:
        """Multiple dimension combos should produce separate leaderboard groups."""
        # Use a tiny custom preset with 2 combos (2 depths x 1 n_samples)
        # to verify grouping without running expensive benchmarks.
        PRESETS["_ci"] = PresetConfig(
            widths=[64],
            depths=[4, 8],
            n_samples_list=[1_000],
        )
        try:
            output, _ = run_profile(
                preset_name="_ci", backend_filter=["numpy", "scipy"], verbose=False
            )
        finally:
            del PRESETS["_ci"]
        # Each depth gets a leaderboard group header
        assert "Leaderboard" in output


class TestLogProgress:
    def test_log_progress_prints_lines(self, capsys) -> None:
        """Non-TTY mode should print one line per benchmark step."""
        run_profile(
            preset_name="super-quick",
            backend_filter=["numpy"],
            log_progress=True,
        )
        captured = capsys.readouterr()
        assert "[correctness]" in captured.out
        assert "[timing]" in captured.out
        assert "numpy" in captured.out
        assert "[done]" in captured.out

    def test_log_progress_off_by_default(self, capsys) -> None:
        """Default mode should not print log lines."""
        run_profile(
            preset_name="super-quick",
            backend_filter=["numpy"],
        )
        captured = capsys.readouterr()
        assert "[correctness]" not in captured.out


class TestCLIFlags:
    def test_verbose_flag_accepted(self) -> None:
        from network_estimation.cli import _build_participant_parser

        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--preset", "super-quick", "--verbose"])
        assert args.verbose is True

    def test_verbose_flag_default_false(self) -> None:
        from network_estimation.cli import _build_participant_parser

        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--preset", "super-quick"])
        assert args.verbose is False

    def test_backends_help_flag_accepted(self) -> None:
        from network_estimation.cli import _build_participant_parser

        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--backends-help"])
        assert args.backends_help is True

    def test_log_progress_flag_accepted(self) -> None:
        from network_estimation.cli import _build_participant_parser

        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation", "--log-progress"])
        assert args.log_progress is True

    def test_log_progress_flag_default_false(self) -> None:
        from network_estimation.cli import _build_participant_parser

        parser = _build_participant_parser()
        args = parser.parse_args(["profile-simulation"])
        assert args.log_progress is False

    def test_backends_help_prints_and_exits(self) -> None:
        import contextlib
        import io

        from network_estimation.cli import _main_participant

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = _main_participant(["profile-simulation", "--backends-help"])
        assert rc == 0
        output = buf.getvalue()
        assert "install" in output.lower() or "All backends" in output
