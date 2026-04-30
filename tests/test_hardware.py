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
        "flopscope_version",
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


def test_collect_hardware_fingerprint_uses_distribution_metadata_without_importing_flopscope(
    monkeypatch,
):
    previous_flopscope = sys.modules.pop("flopscope", None)
    real_import = builtins.__import__

    def _version(name: str) -> str:
        assert name == "flopscope"
        return "9.9.9"

    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "flopscope":
            raise AssertionError("collect_hardware_fingerprint should not import flopscope")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(hardware_mod.importlib_metadata, "version", _version)
    monkeypatch.setattr(builtins, "__import__", _guarded_import)
    try:
        fp = collect_hardware_fingerprint()
        assert fp["flopscope_version"] == "9.9.9"
        assert "flopscope" not in sys.modules
    finally:
        if previous_flopscope is not None:
            sys.modules["flopscope"] = previous_flopscope


def test_collect_hardware_fingerprint_uses_unknown_when_flopscope_metadata_missing(
    monkeypatch,
):
    def _missing(name: str) -> str:
        raise hardware_mod.importlib_metadata.PackageNotFoundError(name)

    monkeypatch.setattr(hardware_mod.importlib_metadata, "version", _missing)

    fp = collect_hardware_fingerprint()

    assert fp["flopscope_version"] == "unknown"


def test_collect_hardware_fingerprint_uses_fallback_probes_when_enabled(monkeypatch) -> None:
    monkeypatch.setattr(hardware_mod, "psutil", None)
    monkeypatch.setattr(hardware_mod, "_physical_core_count_fallback", lambda: 12)
    monkeypatch.setattr(hardware_mod, "_ram_total_fallback", lambda: 34)

    fp = collect_hardware_fingerprint(skip_fallback_probes=False)

    assert fp["cpu_count_physical"] == 12
    assert fp["ram_total_bytes"] == 34


def test_collect_hardware_fingerprint_skips_fallback_probes_via_env(monkeypatch) -> None:
    monkeypatch.setenv(hardware_mod._SKIP_FALLBACK_PROBES_ENV, "1")
    monkeypatch.setattr(hardware_mod, "psutil", None)

    def _unexpected_cpu_probe() -> int:
        raise AssertionError("physical core fallback probe should be skipped")

    def _unexpected_ram_probe() -> int:
        raise AssertionError("RAM fallback probe should be skipped")

    monkeypatch.setattr(hardware_mod, "_physical_core_count_fallback", _unexpected_cpu_probe)
    monkeypatch.setattr(hardware_mod, "_ram_total_fallback", _unexpected_ram_probe)

    fp = collect_hardware_fingerprint()

    assert fp["hostname"]
    assert fp["cpu_count_logical"] > 0
    assert fp["cpu_count_physical"] is None
    assert fp["ram_total_bytes"] is None
