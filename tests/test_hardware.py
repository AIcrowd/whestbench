from __future__ import annotations

from whestbench.hardware import (
    collect_hardware_fingerprint,
    hardware_matches,
)


def test_collect_hardware_fingerprint_returns_required_keys():
    fp = collect_hardware_fingerprint()
    required = {
        "hostname",
        "os",
        "os_release",
        "platform",
        "machine",
        "python_version",
        "cpu_brand",
        "cpu_count_logical",
        "cpu_count_physical",
        "ram_total_bytes",
        "ram_available_bytes",
        "numpy_version",
        "whest_version",
    }
    assert required <= set(fp.keys())
    assert isinstance(fp["cpu_count_logical"], int)
    assert fp["cpu_count_logical"] > 0


def test_hardware_matches_same_machine():
    fp = collect_hardware_fingerprint()
    assert hardware_matches(fp, fp) is True


def test_hardware_matches_detects_mismatch():
    fp = collect_hardware_fingerprint()
    altered = {**fp, "cpu_brand": "FakeProcessor"}
    assert hardware_matches(fp, altered) is False
