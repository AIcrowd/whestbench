from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from whestbench.doctor import (
    Check,
    check_blas,
    check_cwd_writable,
    check_disk,
    check_install_mode,
    check_node,
    check_python,
    check_uv,
)

# --- check_python ------------------------------------------------------------


def test_check_python_ok_when_current_satisfies_requires_python() -> None:
    fake_meta = MagicMock()
    fake_meta.get.return_value = ">=3.10"
    with (
        patch("whestbench.doctor._whestbench_metadata", return_value=fake_meta),
        patch("platform.python_version", return_value="3.12.0"),
    ):
        result = check_python()
    assert result["status"] == "ok"
    assert result["name"] == "python_version"
    assert "3.12.0" in result["detail"]
    assert ">=3.10" in result["detail"]
    assert result["fix_hint"] is None


def test_check_python_fail_when_current_below_requires_python() -> None:
    fake_meta = MagicMock()
    fake_meta.get.return_value = ">=3.10"
    with (
        patch("whestbench.doctor._whestbench_metadata", return_value=fake_meta),
        patch("platform.python_version", return_value="3.9.7"),
    ):
        result = check_python()
    assert result["status"] == "fail"
    assert "3.9.7" in result["detail"]
    assert result["fix_hint"] and "3.10" in result["fix_hint"]


def test_check_python_falls_back_to_hardcoded_floor_when_metadata_missing() -> None:
    fake_meta = MagicMock()
    fake_meta.get.return_value = ""  # no Requires-Python field
    with (
        patch("whestbench.doctor._whestbench_metadata", return_value=fake_meta),
        patch("platform.python_version", return_value="3.11.0"),
    ):
        result = check_python()
    assert result["status"] == "ok"
    assert "(fallback floor)" in result["detail"]


# --- check_uv ----------------------------------------------------------------


def test_check_uv_ok_when_on_path() -> None:
    with patch("shutil.which", return_value="/opt/homebrew/bin/uv"):
        result = check_uv()
    assert result["status"] == "ok"
    assert result["name"] == "uv"
    assert "/opt/homebrew/bin/uv" in result["detail"]
    assert result["fix_hint"] is None


def test_check_uv_fail_when_not_on_path() -> None:
    with patch("shutil.which", return_value=None):
        result = check_uv()
    assert result["status"] == "fail"
    assert result["fix_hint"] and "astral.sh/uv" in result["fix_hint"]


# --- check_install_mode ------------------------------------------------------


def test_check_install_mode_ok_editable() -> None:
    fake_dist = MagicMock()
    fake_dist.version = "0.2.0"
    fake_dist.read_text.return_value = json.dumps(
        {
            "url": "file:///path/to/worktree",
            "dir_info": {"editable": True},
        }
    )
    with patch("whestbench.doctor.Distribution.from_name", return_value=fake_dist):
        result = check_install_mode()
    assert result["status"] == "ok"
    assert "editable" in result["detail"]
    assert "/path/to/worktree" in result["detail"]
    assert result["fix_hint"] is None


def test_check_install_mode_ok_tool_installed_when_direct_url_absent() -> None:
    fake_dist = MagicMock()
    fake_dist.version = "0.2.0"
    fake_dist.read_text.return_value = None  # no direct_url.json
    with patch("whestbench.doctor.Distribution.from_name", return_value=fake_dist):
        result = check_install_mode()
    assert result["status"] == "ok"
    assert "tool-installed" in result["detail"]
    assert "0.2.0" in result["detail"]


def test_check_install_mode_ok_from_source_non_editable() -> None:
    fake_dist = MagicMock()
    fake_dist.version = "0.2.0"
    fake_dist.read_text.return_value = json.dumps(
        {"url": "file:///path/to/sdist", "dir_info": {"editable": False}}
    )
    with patch("whestbench.doctor.Distribution.from_name", return_value=fake_dist):
        result = check_install_mode()
    assert result["status"] == "ok"
    assert "from-source" in result["detail"]
    assert "/path/to/sdist" in result["detail"]


def test_check_install_mode_fail_when_not_installed() -> None:
    from importlib.metadata import PackageNotFoundError

    with patch(
        "whestbench.doctor.Distribution.from_name",
        side_effect=PackageNotFoundError("whestbench"),
    ):
        result = check_install_mode()
    assert result["status"] == "fail"
    assert result["fix_hint"] and "uv sync" in result["fix_hint"]


# --- check_node --------------------------------------------------------------


def test_check_node_ok_when_on_path_and_version_resolvable() -> None:
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stdout = "v20.11.0\n"
    with (
        patch("shutil.which", return_value="/usr/local/bin/node"),
        patch("subprocess.run", return_value=fake_result),
    ):
        result = check_node()
    assert result["status"] == "ok"
    assert result["name"] == "node_js"
    assert "v20.11.0" in result["detail"]
    assert result["fix_hint"] is None


def test_check_node_warn_when_not_on_path() -> None:
    with patch("shutil.which", return_value=None):
        result = check_node()
    assert result["status"] == "warn"
    assert result["fix_hint"] and "nodejs.org" in result["fix_hint"]
    assert "whest visualizer" in result["fix_hint"]


# --- check_blas --------------------------------------------------------------


def test_check_blas_ok_when_pool_detected() -> None:
    fake_pools = [
        {"internal_api": "openblas", "num_threads": 8, "user_api": "blas"},
    ]
    with patch("threadpoolctl.threadpool_info", return_value=fake_pools):
        result = check_blas()
    assert result["status"] == "ok"
    assert result["name"] == "blas_threads"
    assert "openblas" in result["detail"]
    assert "8 threads" in result["detail"]


def test_check_blas_warn_when_no_pool() -> None:
    with patch("threadpoolctl.threadpool_info", return_value=[]):
        result = check_blas()
    assert result["status"] == "warn"
    assert "no blas pool" in result["detail"].lower()


def test_check_blas_fail_when_threadpoolctl_import_fails() -> None:
    import sys

    saved = sys.modules.pop("threadpoolctl", None)
    with patch.dict(sys.modules, {"threadpoolctl": None}):
        result = check_blas()
    if saved is not None:
        sys.modules["threadpoolctl"] = saved
    assert result["status"] == "fail"
    assert result["fix_hint"] and "uv sync" in result["fix_hint"]


# --- check_disk --------------------------------------------------------------


_GIB = 1024 * 1024 * 1024


def test_check_disk_ok_when_at_least_1_gib_free() -> None:
    fake_usage = MagicMock()
    fake_usage.free = 150 * _GIB
    with patch("shutil.disk_usage", return_value=fake_usage):
        result = check_disk()
    assert result["status"] == "ok"
    assert result["name"] == "disk_space"
    assert "150" in result["detail"] or "150.0" in result["detail"]


def test_check_disk_warn_when_less_than_1_gib_free() -> None:
    fake_usage = MagicMock()
    fake_usage.free = 500 * 1024 * 1024  # 500 MiB
    with patch("shutil.disk_usage", return_value=fake_usage):
        result = check_disk()
    assert result["status"] == "warn"
    assert result["fix_hint"] and "free some space" in result["fix_hint"].lower()


def test_check_disk_fail_when_os_error() -> None:
    with patch("shutil.disk_usage", side_effect=OSError("no such file")):
        result = check_disk()
    assert result["status"] == "fail"


# --- check_cwd_writable ------------------------------------------------------


def test_check_cwd_writable_ok(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = check_cwd_writable()
    assert result["status"] == "ok"
    assert result["name"] == "cwd_writable"
    assert str(tmp_path) in result["detail"]


def test_check_cwd_writable_fail_when_permission_error() -> None:
    with patch("tempfile.NamedTemporaryFile", side_effect=PermissionError("denied")):
        result = check_cwd_writable()
    assert result["status"] == "fail"
    assert result["fix_hint"] and "permissions" in result["fix_hint"].lower()


# --- run_all -----------------------------------------------------------------


def test_run_all_returns_seven_checks_in_stable_order() -> None:
    from whestbench.doctor import run_all

    checks = run_all()
    assert len(checks) == 7
    names = [c["name"] for c in checks]
    assert names == [
        "python_version",
        "uv",
        "install_mode",
        "node_js",
        "blas_threads",
        "disk_space",
        "cwd_writable",
    ]
    for c in checks:
        assert set(c.keys()) >= {"name", "label", "status", "detail", "fix_hint"}
        assert c["status"] in {"ok", "warn", "fail"}


def test_run_all_captures_check_crashes_as_fail_by_default() -> None:
    from whestbench.doctor import run_all

    def _boom() -> Check:
        raise RuntimeError("intentional test crash")

    with patch("whestbench.doctor.check_uv", _boom):
        checks = run_all(debug=False)

    uv_check = next(c for c in checks if c["name"] == "uv")
    assert uv_check["status"] == "fail"
    assert "check crashed" in uv_check["detail"].lower()
    assert "RuntimeError" in uv_check["detail"]


def test_run_all_reraises_on_debug() -> None:
    from whestbench.doctor import run_all

    def _boom() -> Check:
        raise RuntimeError("intentional test crash")

    with patch("whestbench.doctor.check_uv", _boom):
        with pytest.raises(RuntimeError, match="intentional test crash"):
            run_all(debug=True)
