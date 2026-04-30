"""Tests for dashboard generation."""

import json
import os
import tempfile

import pytest

from profiling.generate_dashboard import (
    extract_base_css,
    fetch_cdn_libs,
    generate_html,
    load_data,
    normalize_data,
    parse_args,
    resolve_paths,
)

MULTI_CONFIG_DATA = {
    "run_id": "2026-03-20-test",
    "git_commit": "abc1234",
    "git_dirty": True,
    "collected_at": "2026-03-20T21:00:00+00:00",
    "configs": {
        "compute-small": {
            "hardware": {
                "cpu_count_logical": 2,
                "ram_total_bytes": 2147483648,
                "hostname": "ip-test",
                "platform": "Linux",
            },
            "backend_versions": {"numpy": "1.24.0", "scipy": "1.10.0"},
            "skipped_backends": {"numba": "No numba installed"},
            "correctness": [
                {"backend": "numpy", "passed": True, "error": ""},
                {"backend": "scipy", "passed": False, "error": "max_diff=0.5 exceeds threshold"},
            ],
            "timing": [
                {
                    "backend": "numpy",
                    "operation": "run_mlp",
                    "width": 256,
                    "depth": 4,
                    "n_samples": 10000,
                    "times": [0.05, 0.052, 0.051],
                    "median_time": 0.051,
                    "speedup_vs_numpy": 1.0,
                    "warmup_time": 0.055,
                },
                {
                    "backend": "scipy",
                    "operation": "run_mlp",
                    "width": 256,
                    "depth": 4,
                    "n_samples": 10000,
                    "times": [0.03, 0.031, 0.032],
                    "median_time": 0.031,
                    "speedup_vs_numpy": 1.645,
                    "warmup_time": 0.28,
                },
            ],
        },
        "empty-config": {
            "hardware": {
                "cpu_count_logical": 1,
                "ram_total_bytes": 1073741824,
                "hostname": "ip-empty",
                "platform": "Linux",
            },
            "backend_versions": {"numpy": "1.24.0"},
            "skipped_backends": {},
            "correctness": [],
            "timing": [],
        },
    },
}

SINGLE_CONFIG_DATA = {
    "hardware": {
        "cpu_count_logical": 8,
        "ram_total_bytes": 17179869184,
        "hostname": "my-macbook",
        "platform": "Darwin",
    },
    "backend_versions": {"numpy": "1.24.0"},
    "skipped_backends": {},
    "correctness": [{"backend": "numpy", "passed": True, "error": ""}],
    "timing": [
        {
            "backend": "numpy",
            "operation": "run_mlp",
            "width": 256,
            "depth": 4,
            "n_samples": 10000,
            "times": [0.05, 0.052],
            "median_time": 0.051,
            "speedup_vs_numpy": 1.0,
        },
    ],
}


def test_load_multi_config():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(MULTI_CONFIG_DATA, f)
        path = f.name
    try:
        data = load_data(path)
        assert data["run_id"] == "2026-03-20-test"
        assert "compute-small" in data["configs"]
        assert len(data["configs"]["compute-small"]["timing"]) == 2
    finally:
        os.unlink(path)


def test_normalize_single_config():
    normalized = normalize_data(SINGLE_CONFIG_DATA)
    assert "configs" in normalized
    assert "run_id" in normalized
    config_names = list(normalized["configs"].keys())
    assert len(config_names) == 1
    assert normalized["configs"][config_names[0]]["hardware"]["cpu_count_logical"] == 8


def test_normalize_multi_config_passthrough():
    normalized = normalize_data(MULTI_CONFIG_DATA)
    assert normalized["run_id"] == "2026-03-20-test"
    assert "compute-small" in normalized["configs"]


def test_parse_args_run_id():
    args = parse_args(["--run-id", "2026-03-20-test"])
    assert args.run_id == "2026-03-20-test"
    assert args.input is None


def test_parse_args_input():
    args = parse_args(["--input", "output.json"])
    assert args.input == "output.json"
    assert args.run_id is None


def test_parse_args_output():
    args = parse_args(["--input", "output.json", "--output", "my.html"])
    assert args.output == "my.html"


def test_parse_args_requires_one():
    with pytest.raises(SystemExit):
        parse_args([])


def test_resolve_paths_run_id():
    args = parse_args(["--run-id", "my-run"])
    input_path, output_path = resolve_paths(args)
    assert input_path == os.path.join("profiling", "results", "my-run", "combined.json")
    assert output_path == os.path.join("profiling", "results", "my-run", "dashboard.html")


def test_resolve_paths_input_default_output():
    args = parse_args(["--input", "output.json"])
    input_path, output_path = resolve_paths(args)
    assert input_path == "output.json"
    assert output_path == "dashboard.html"


def test_extract_base_css():
    css = extract_base_css()
    # Must contain CSS variables
    assert "--coral:" in css
    assert "--gray-900:" in css
    assert "--font-sans:" in css
    # Must contain Google Fonts import
    assert "@import url(" in css
    # Must contain .app-header styles
    assert ".app-header {" in css or ".app-header{" in css
    assert ".app-header h1" in css
    # Must NOT contain component-specific class rules
    assert ".sidebar {" not in css


def test_fetch_cdn_libs_returns_dict():
    libs = fetch_cdn_libs()
    assert "react" in libs
    assert "react-dom" in libs
    assert "prop-types" in libs
    assert "recharts" in libs
    for name, source in libs.items():
        assert len(source) > 1000, f"{name} source too small"


def test_generate_html_structure():
    html = generate_html(MULTI_CONFIG_DATA)
    assert "<!DOCTYPE html>" in html
    assert "flopscope Profiling Dashboard" in html
    assert "window.__PROFILING_DATA__" in html
    assert "compute-small" in html
    assert "SpeedupHeatmap" in html or "App" in html
    assert "--coral" in html
    assert "section-help" in html
    assert "warmup_time" in html
    assert "warmup-info" in html


def test_generate_html_single_config():
    normalized = normalize_data(SINGLE_CONFIG_DATA)
    html = generate_html(normalized)
    assert "<!DOCTYPE html>" in html
    assert "window.__PROFILING_DATA__" in html


def test_end_to_end_generates_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fin:
        json.dump(MULTI_CONFIG_DATA, fin)
        input_path = fin.name

    output_path = input_path.replace(".json", ".html")
    try:
        from profiling.generate_dashboard import main

        main(["--input", input_path, "--output", output_path])
        assert os.path.exists(output_path)
        with open(output_path) as f:
            html = f.read()
        assert len(html) > 10000
        assert "compute-small" in html
        assert "<!DOCTYPE html>" in html
    finally:
        os.unlink(input_path)
        if os.path.exists(output_path):
            os.unlink(output_path)
