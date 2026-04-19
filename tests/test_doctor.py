from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from whestbench.doctor import (
    check_install_mode,
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
