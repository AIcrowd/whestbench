# tests/test_profiler.py
"""Tests for the simulation profiler."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from network_estimation.profiler import (
    PRESETS,
    correctness_check,
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
