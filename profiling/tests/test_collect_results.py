"""Tests for result collection and aggregation."""

import csv
import json
import io
from profiling.collect_results import build_combined_json, build_summary_csv


SAMPLE_RESULT = {
    "hardware": {"platform": "Linux", "cpu_count_logical": 2},
    "backend_versions": {"numpy": "1.24.0"},
    "correctness": [{"backend": "numpy", "passed": True}],
    "timing": [
        {
            "backend": "numpy",
            "operation": "run_mlp",
            "width": 256,
            "depth": 4,
            "n_samples": 10000,
            "median_time": 0.05,
            "speedup_vs_numpy": 1.0,
        },
        {
            "backend": "scipy",
            "operation": "run_mlp",
            "width": 256,
            "depth": 4,
            "n_samples": 10000,
            "median_time": 0.03,
            "speedup_vs_numpy": 1.67,
        },
    ],
}


def test_build_combined_json():
    config_results = {
        "compute-small": SAMPLE_RESULT,
        "compute-large": SAMPLE_RESULT,
    }
    combined = build_combined_json(
        run_id="2026-03-20-143000-abc1234",
        git_commit="abc1234567890",
        git_dirty=False,
        config_results=config_results,
    )
    assert combined["run_id"] == "2026-03-20-143000-abc1234"
    assert combined["git_commit"] == "abc1234567890"
    assert combined["git_dirty"] is False
    assert "collected_at" in combined
    assert "compute-small" in combined["configs"]
    assert "compute-large" in combined["configs"]


def test_build_combined_json_partial():
    """Should work with partial results."""
    combined = build_combined_json(
        run_id="test",
        git_commit="abc",
        git_dirty=False,
        config_results={"compute-small": SAMPLE_RESULT},
    )
    assert len(combined["configs"]) == 1


def test_build_summary_csv():
    config_results = {"compute-small": SAMPLE_RESULT}
    csv_str = build_summary_csv(config_results)

    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 2  # numpy + scipy timing entries
    assert rows[0]["config_name"] == "compute-small"
    assert rows[0]["backend"] == "numpy"
    assert rows[0]["width"] == "256"
    assert float(rows[0]["speedup_vs_numpy"]) == 1.0


def test_build_summary_csv_empty():
    csv_str = build_summary_csv({})
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 0
