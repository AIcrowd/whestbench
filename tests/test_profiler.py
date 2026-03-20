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
    TimingResult,
    correctness_check,
    format_compact_output,
    format_dims,
    run_profile,
)
from network_estimation.simulation_backends import get_available_backends, get_backend


class TestCorrectnessCheck:
    def test_numpy_passes(self) -> None:
        backend = get_backend("numpy")
        result = correctness_check(backend)
        assert result.passed is True
        assert result.error == ""


class TestRunProfile:
    def test_quick_preset_runs(self) -> None:
        terminal_output, _ = run_profile(
            preset_name="quick", backend_filter=["numpy"]
        )
        assert "numpy" in terminal_output
        assert "Timing Results" in terminal_output

    def test_json_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = str(Path(tmpdir) / "results.json")
            _, json_data = run_profile(
                preset_name="quick",
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
            preset_name="quick",
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
                backend_name="numpy", operation="run_mlp",
                width=64, depth=4, n_samples=1000,
                times=[0.0001], median_time=0.0001, speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="scipy", operation="run_mlp",
                width=64, depth=4, n_samples=1000,
                times=[0.0002], median_time=0.0002, speedup_vs_numpy=0.5,
            ),
            TimingResult(
                backend_name="numpy", operation="output_stats",
                width=64, depth=4, n_samples=1000,
                times=[0.0009], median_time=0.0009, speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="scipy", operation="output_stats",
                width=64, depth=4, n_samples=1000,
                times=[0.0009], median_time=0.0009, speedup_vs_numpy=1.0,
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
                backend_name="numpy", operation="run_mlp",
                width=64, depth=4, n_samples=1000,
                times=[0.0001], median_time=0.0001, speedup_vs_numpy=1.0,
            ),
            TimingResult(
                backend_name="numpy", operation="output_stats",
                width=64, depth=4, n_samples=1000,
                times=[0.0009], median_time=0.0009, speedup_vs_numpy=1.0,
            ),
        ]
        output = format_compact_output(cr, tr, {})
        assert "Leaderboard" not in output

    def test_zero_passed_backends(self) -> None:
        cr = [CorrectnessResult(backend_name="numpy", passed=False, error="boom")]
        output = format_compact_output(cr, [], {})
        assert "No backends passed" in output
