from __future__ import annotations

import builtins
import sys

from whestbench import hardware as hardware_mod
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


def test_collect_hardware_fingerprint_uses_distribution_metadata_without_importing_whest(
    monkeypatch,
):
    previous_whest = sys.modules.pop("whest", None)
    real_import = builtins.__import__

    def _version(name: str) -> str:
        assert name == "whest"
        return "9.9.9"

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "whest":
            raise AssertionError("collect_hardware_fingerprint should not import whest")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(hardware_mod.importlib_metadata, "version", _version)
    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    try:
        fp = collect_hardware_fingerprint()
        assert fp["whest_version"] == "9.9.9"
        assert "whest" not in sys.modules
    finally:
        if previous_whest is not None:
            sys.modules["whest"] = previous_whest


def test_collect_hardware_fingerprint_uses_unknown_when_whest_metadata_missing(
    monkeypatch,
):
    def _missing(name: str) -> str:
        raise hardware_mod.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(hardware_mod.importlib_metadata, "version", _missing)

    fp = collect_hardware_fingerprint()

    assert fp["whest_version"] == "unknown"
