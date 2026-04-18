"""Tests for whestbench.visualizer."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

import whestbench.visualizer as viz_mod
from whestbench import cli
from whestbench.visualizer import (
    ExplorerNotFoundError,
    NodeNotFoundError,
    NodeVersionError,
    check_node_available,
    check_node_version,
    find_explorer_dir,
    is_headless,
    run_visualizer,
)

# --- Prerequisite check tests ---


def test_check_node_available_passes_when_both_found():
    with patch("shutil.which", return_value="/usr/bin/node"):
        check_node_available()


def test_check_node_available_raises_when_node_missing():
    def fake_which(name: str) -> "str | None":
        return None if name == "node" else "/usr/bin/npm"

    with patch("shutil.which", side_effect=fake_which):
        with pytest.raises(NodeNotFoundError, match="node"):
            check_node_available()


def test_check_node_available_raises_when_npm_missing():
    def fake_which(name: str) -> "str | None":
        return "/usr/bin/node" if name == "node" else None

    with patch("shutil.which", side_effect=fake_which):
        with pytest.raises(NodeNotFoundError, match="npm"):
            check_node_available()


def test_check_node_version_passes_for_v18():
    with patch(
        "subprocess.run",
        return_value=type("R", (), {"stdout": "v18.17.0\n", "returncode": 0})(),
    ):
        check_node_version()


def test_check_node_version_passes_for_v22():
    with patch(
        "subprocess.run",
        return_value=type("R", (), {"stdout": "v22.1.0\n", "returncode": 0})(),
    ):
        check_node_version()


def test_check_node_version_raises_for_v16():
    with patch(
        "subprocess.run",
        return_value=type("R", (), {"stdout": "v16.20.0\n", "returncode": 0})(),
    ):
        with pytest.raises(NodeVersionError, match="18"):
            check_node_version()


def test_check_node_version_raises_for_failed_node_command():
    with patch(
        "subprocess.run",
        return_value=type("R", (), {"stdout": "", "returncode": 1})(),
    ):
        with pytest.raises(NodeVersionError):
            check_node_version()


# --- Explorer path resolution tests ---


def test_find_explorer_dir_finds_tools_dir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    explorer = repo / "tools" / "whestbench-explorer"
    explorer.mkdir(parents=True)
    (explorer / "package.json").write_text("{}")

    fake_module_dir = repo / "src" / "whestbench"
    fake_module_dir.mkdir(parents=True)

    result = find_explorer_dir(start=fake_module_dir)
    assert result == explorer


def test_find_explorer_dir_raises_when_no_repo_root(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    with pytest.raises(ExplorerNotFoundError):
        find_explorer_dir(start=deep)


def test_find_explorer_dir_raises_when_no_tools_dir(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    fake_module_dir = repo / "src" / "whestbench"
    fake_module_dir.mkdir(parents=True)
    with pytest.raises(ExplorerNotFoundError):
        find_explorer_dir(start=fake_module_dir)


# --- Headless detection tests ---


def test_is_headless_true_when_ssh_client_set(monkeypatch):
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv("SSH_CLIENT", "1.2.3.4 5678 22")
    assert is_headless() is True


def test_is_headless_true_when_ssh_tty_set(monkeypatch):
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setenv("SSH_TTY", "/dev/pts/0")
    assert is_headless() is True


def test_is_headless_true_when_ssh_connection_set(monkeypatch):
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.setenv("SSH_CONNECTION", "1.2.3.4 5678 1.2.3.5 22")
    assert is_headless() is True


def test_is_headless_false_on_desktop(monkeypatch):
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    if sys.platform == "linux":
        monkeypatch.setenv("DISPLAY", ":0")
    assert is_headless() is False


def test_is_headless_true_on_linux_without_display(monkeypatch):
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(sys, "platform", "linux")
    assert is_headless() is True


def test_is_headless_false_on_linux_with_wayland(monkeypatch):
    monkeypatch.delenv("SSH_CLIENT", raising=False)
    monkeypatch.delenv("SSH_TTY", raising=False)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.setattr(sys, "platform", "linux")
    assert is_headless() is False


# --- Dev server orchestration tests ---


def _make_fake_explorer(tmp_path, *, with_node_modules: bool = True):
    """Create a fake explorer directory and return it."""
    explorer = tmp_path / "whestbench-explorer"
    explorer.mkdir()
    (explorer / "package.json").write_text("{}")
    if with_node_modules:
        (explorer / "node_modules").mkdir()
    return explorer


class FakeProcess:
    """Fake subprocess.Popen for testing."""

    def __init__(self, returncode: int = 0):
        self.stdout = None
        self.returncode = returncode
        self._terminated = False

    def wait(self):
        return self.returncode

    def terminate(self):
        self._terminated = True

    def poll(self):
        return self.returncode


def test_run_visualizer_skips_npm_ci_when_node_modules_exist(tmp_path, monkeypatch):
    explorer = _make_fake_explorer(tmp_path, with_node_modules=True)
    calls: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        calls.append(list(cmd))
        return FakeProcess()

    monkeypatch.setattr(viz_mod, "find_explorer_dir", lambda **kw: explorer)
    monkeypatch.setattr(viz_mod, "check_node_available", lambda: None)
    monkeypatch.setattr(viz_mod, "check_node_version", lambda: None)
    monkeypatch.setattr(viz_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(viz_mod, "is_headless", lambda: True)

    result = run_visualizer(host="localhost", port=5173, no_open=True, debug=False)

    assert result == 0
    assert not any("ci" in " ".join(c) for c in calls)
    assert any("dev" in " ".join(c) for c in calls)


def test_run_visualizer_runs_npm_ci_when_no_node_modules(tmp_path, monkeypatch):
    explorer = _make_fake_explorer(tmp_path, with_node_modules=False)
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def fake_popen(cmd, **kwargs):
        calls.append(list(cmd))
        return FakeProcess()

    monkeypatch.setattr(viz_mod, "find_explorer_dir", lambda **kw: explorer)
    monkeypatch.setattr(viz_mod, "check_node_available", lambda: None)
    monkeypatch.setattr(viz_mod, "check_node_version", lambda: None)
    monkeypatch.setattr(viz_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(viz_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(viz_mod, "is_headless", lambda: True)

    result = run_visualizer(host="localhost", port=5173, no_open=True, debug=False)

    assert result == 0
    cmd_strings = [" ".join(c) for c in calls]
    ci_idx = next(i for i, s in enumerate(cmd_strings) if "ci" in s)
    dev_idx = next(i for i, s in enumerate(cmd_strings) if "dev" in s)
    assert ci_idx < dev_idx


def test_run_visualizer_retries_with_npm_ci_on_dev_server_failure(tmp_path, monkeypatch):
    explorer = _make_fake_explorer(tmp_path, with_node_modules=True)
    calls: list[list[str]] = []
    dev_call_count = 0

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    def fake_popen(cmd, **kwargs):
        nonlocal dev_call_count
        calls.append(list(cmd))
        dev_call_count += 1
        return FakeProcess(returncode=1 if dev_call_count == 1 else 0)

    monkeypatch.setattr(viz_mod, "find_explorer_dir", lambda **kw: explorer)
    monkeypatch.setattr(viz_mod, "check_node_available", lambda: None)
    monkeypatch.setattr(viz_mod, "check_node_version", lambda: None)
    monkeypatch.setattr(viz_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(viz_mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(viz_mod, "is_headless", lambda: True)

    result = run_visualizer(host="localhost", port=5173, no_open=True, debug=False)

    assert result == 0
    cmd_strings = [" ".join(c) for c in calls]
    assert sum(1 for s in cmd_strings if "ci" in s) == 1
    assert sum(1 for s in cmd_strings if "dev" in s) == 2


def test_run_visualizer_prints_ready_state_summary(
    tmp_path, monkeypatch, capsys: pytest.CaptureFixture[str]
):
    explorer = _make_fake_explorer(tmp_path, with_node_modules=True)

    class ReadyProcess(FakeProcess):
        def __init__(self) -> None:
            super().__init__(returncode=0)
            self.stdout = iter(
                [
                    "  VITE v7.3.1  ready in 728 ms\n",
                    "  ➜  Local:   http://127.0.0.1:4173/\n",
                ]
            )

    monkeypatch.setattr(viz_mod, "find_explorer_dir", lambda **kw: explorer)
    monkeypatch.setattr(viz_mod, "check_node_available", lambda: None)
    monkeypatch.setattr(viz_mod, "check_node_version", lambda: None)
    monkeypatch.setattr(viz_mod.subprocess, "Popen", lambda *_a, **_k: ReadyProcess())
    monkeypatch.setattr(viz_mod, "is_headless", lambda: True)

    result = run_visualizer(host="127.0.0.1", port=4173, no_open=True, debug=False)
    captured = capsys.readouterr()

    assert result == 0
    assert "WhestBench Explorer" in captured.out
    assert "http://127.0.0.1:4173/" in captured.out


def test_run_visualizer_returns_1_when_npm_ci_fails(tmp_path, monkeypatch):
    explorer = _make_fake_explorer(tmp_path, with_node_modules=False)

    def fake_run(cmd, **kwargs):
        return type("R", (), {"returncode": 1, "stdout": "", "stderr": "npm ERR!"})()

    monkeypatch.setattr(viz_mod, "find_explorer_dir", lambda **kw: explorer)
    monkeypatch.setattr(viz_mod, "check_node_available", lambda: None)
    monkeypatch.setattr(viz_mod, "check_node_version", lambda: None)
    monkeypatch.setattr(viz_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(viz_mod, "is_headless", lambda: True)

    result = run_visualizer(host="localhost", port=5173, no_open=True, debug=False)
    assert result == 1


def test_run_visualizer_returns_1_when_explorer_not_found(monkeypatch, capsys):
    monkeypatch.setattr(viz_mod, "check_node_available", lambda: None)
    monkeypatch.setattr(viz_mod, "check_node_version", lambda: None)

    def raise_not_found(**kw):
        raise ExplorerNotFoundError()

    monkeypatch.setattr(viz_mod, "find_explorer_dir", raise_not_found)

    result = run_visualizer(host="localhost", port=5173, no_open=True, debug=False)
    captured = capsys.readouterr()
    assert result == 1
    assert "Explorer Not Found" in captured.err
    assert "source checkout of the repository" in captured.err


def test_run_visualizer_returns_1_when_node_missing(monkeypatch, capsys):
    def raise_not_found():
        raise NodeNotFoundError("node")

    monkeypatch.setattr(viz_mod, "check_node_available", raise_not_found)

    result = run_visualizer(host="localhost", port=5173, no_open=True, debug=False)
    captured = capsys.readouterr()
    assert result == 1
    assert "Missing Prerequisite" in captured.err
    assert "node" in captured.err


# --- CLI dispatch tests ---


def test_cli_visualizer_dispatches_to_run_visualizer(monkeypatch):
    """whest visualizer should call run_visualizer with correct args."""
    captured: dict = {}

    def fake_run_visualizer(*, host, port, no_open, debug):
        captured.update(host=host, port=port, no_open=no_open, debug=debug)
        return 0

    monkeypatch.setattr(viz_mod, "run_visualizer", fake_run_visualizer)

    exit_code = cli.main(["visualizer", "--host", "0.0.0.0", "--port", "8080", "--no-open"])
    assert exit_code == 0
    assert captured == {"host": "0.0.0.0", "port": 8080, "no_open": True, "debug": False}


def test_cli_visualizer_defaults(monkeypatch):
    """Default args: host=localhost, port=5173, no_open=False, debug=False."""
    captured: dict = {}

    def fake_run_visualizer(*, host, port, no_open, debug):
        captured.update(host=host, port=port, no_open=no_open, debug=debug)
        return 0

    monkeypatch.setattr(viz_mod, "run_visualizer", fake_run_visualizer)

    exit_code = cli.main(["visualizer"])
    assert exit_code == 0
    assert captured == {"host": "localhost", "port": 5173, "no_open": False, "debug": False}
