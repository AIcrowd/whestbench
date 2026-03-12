# `cestim visualizer` Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `cestim visualizer` subcommand that checks Node.js prerequisites, installs npm dependencies, launches the Vite dev server for the circuit explorer, and auto-opens a browser.

**Architecture:** Single new Python module (`visualizer.py`) handles all logic — prerequisite checks, path resolution, npm ci, subprocess management, headless detection, browser opening. CLI registration in existing `cli.py` via argparse subparser. No changes to the circuit-explorer JavaScript.

**Tech Stack:** Python 3.10+, argparse, subprocess, shutil, webbrowser, threading (for timeout), Rich (panels/console for error display)

**Spec:** `docs/superpowers/specs/2026-03-13-cestim-visualizer-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/circuit_estimation/visualizer.py` | All visualizer logic: prereq checks, path resolution, npm ci, dev server, browser open, headless detection |
| Modify | `src/circuit_estimation/cli.py` | Register `visualizer` subparser, dispatch to `visualizer.py` |
| Create | `tests/test_visualizer.py` | Unit + integration tests for visualizer module |
| Modify | `docs/how-to/use-circuit-explorer.md` | Update to show `cestim visualizer` as primary method |
| Modify | `docs/reference/cli-reference.md` | Add `cestim visualizer` command reference |
| Modify | `README.md` | Add one-liner in quickstart section |

---

## Chunk 1: Core Visualizer Module

### Task 1: Prerequisite check helpers — tests

**Files:**
- Create: `tests/test_visualizer.py`

- [ ] **Step 1: Write failing tests for `check_node_available` and `check_node_version`**

```python
"""Tests for circuit_estimation.visualizer."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from circuit_estimation.visualizer import (
    NodeNotFoundError,
    NodeVersionError,
    check_node_available,
    check_node_version,
)


def test_check_node_available_passes_when_both_found():
    with patch("shutil.which", return_value="/usr/bin/node"):
        check_node_available()  # should not raise


def test_check_node_available_raises_when_node_missing():
    def fake_which(name: str) -> str | None:
        return None if name == "node" else "/usr/bin/npm"

    with patch("shutil.which", side_effect=fake_which):
        with pytest.raises(NodeNotFoundError, match="node"):
            check_node_available()


def test_check_node_available_raises_when_npm_missing():
    def fake_which(name: str) -> str | None:
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_visualizer.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` (module doesn't exist yet)

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_visualizer.py
git commit -m "test: add failing tests for visualizer prereq checks"
```

---

### Task 2: Prerequisite check helpers — implementation

**Files:**
- Create: `src/circuit_estimation/visualizer.py`

- [ ] **Step 1: Implement prereq check functions**

```python
"""Visualizer command: launch the circuit explorer dev server."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

from rich.console import Console
from rich.panel import Panel


class NodeNotFoundError(RuntimeError):
    """Raised when node or npm is not on PATH."""

    def __init__(self, missing: str) -> None:
        self.missing = missing
        super().__init__(
            f"'{missing}' not found on PATH. "
            "Install Node.js from https://nodejs.org/ or via a package manager."
        )


class NodeVersionError(RuntimeError):
    """Raised when the installed Node.js version is too old."""

    def __init__(self, found: str, minimum: int = 18) -> None:
        self.found = found
        self.minimum = minimum
        super().__init__(
            f"Node.js >= {minimum} is required (found {found!r}). "
            "Upgrade from https://nodejs.org/ or via your package manager."
        )


class ExplorerNotFoundError(RuntimeError):
    """Raised when the circuit explorer directory cannot be located."""

    def __init__(self) -> None:
        super().__init__(
            "Circuit explorer not found. This command requires a source checkout "
            "of the repository. Clone the repo and install with `uv tool install -e .`"
        )


class NpmCiError(RuntimeError):
    """npm ci failed."""

    def __init__(self, stderr: str) -> None:
        self.stderr = stderr
        super().__init__("Installing dependencies failed.")


_MIN_NODE_MAJOR = 18


def check_node_available() -> None:
    """Raise NodeNotFoundError if node or npm is not on PATH."""
    for name in ("node", "npm"):
        if shutil.which(name) is None:
            raise NodeNotFoundError(name)


def check_node_version(minimum: int = _MIN_NODE_MAJOR) -> None:
    """Raise NodeVersionError if node major version < minimum."""
    result = subprocess.run(
        ["node", "--version"],
        capture_output=True,
        text=True,
    )
    version_str = result.stdout.strip()  # e.g. "v18.17.0"
    if result.returncode != 0 or not version_str:
        raise NodeVersionError(version_str or "<unknown>", minimum)
    try:
        major = int(version_str.lstrip("v").split(".")[0])
    except (ValueError, IndexError):
        raise NodeVersionError(version_str, minimum)
    if major < minimum:
        raise NodeVersionError(version_str, minimum)


def find_explorer_dir(start: Path | None = None) -> Path:
    """Walk up from *start* (default: this file's directory) to find the repo root,
    then return tools/circuit-explorer/.

    The repo root is identified by the presence of pyproject.toml.
    """
    current = start or Path(__file__).resolve().parent
    while True:
        if (current / "pyproject.toml").is_file():
            explorer = current / "tools" / "circuit-explorer"
            if explorer.is_dir() and (explorer / "package.json").is_file():
                return explorer
            raise ExplorerNotFoundError()
        parent = current.parent
        if parent == current:
            raise ExplorerNotFoundError()
        current = parent


def is_headless() -> bool:
    """Return True if running in a headless/SSH environment."""
    if any(os.environ.get(v) for v in ("SSH_CLIENT", "SSH_TTY", "SSH_CONNECTION")):
        return True
    if sys.platform == "linux" and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        return True
    return False


def _run_npm_ci(explorer_dir: Path, *, debug: bool) -> None:
    """Run npm ci in the explorer directory."""
    console = Console(stderr=True)
    with console.status("[bold cyan]Installing dependencies (npm ci)...[/]"):
        result = subprocess.run(
            ["npm", "ci"],
            cwd=explorer_dir,
            capture_output=True,
            text=True,
        )
    if result.returncode != 0:
        if debug:
            console.print(f"[dim]{result.stderr}[/]")
        raise NpmCiError(result.stderr)


def _open_browser(host: str, port: int) -> None:
    """Open the browser with the correct URL."""
    browser_host = "localhost" if host == "0.0.0.0" else host
    url = f"http://{browser_host}:{port}"
    webbrowser.open(url)


def _start_dev_server(
    explorer_dir: Path,
    *,
    host: str,
    port: int,
    no_open: bool,
    debug: bool,
) -> int:
    """Start the Vite dev server and block until it exits.

    Returns the process exit code (0 for clean Ctrl+C shutdown).
    """
    cmd = ["npm", "run", "dev", "--", "--host", host, "--port", str(port)]
    console = Console()
    process = subprocess.Popen(
        cmd,
        cwd=explorer_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Handle Ctrl+C gracefully
    def _on_sigint(signum: int, frame: object) -> None:
        process.terminate()

    original_handler = signal.signal(signal.SIGINT, _on_sigint)

    browser_opened = False
    should_open = not no_open and not is_headless()

    # Timeout: if Vite ready line not detected within 30s, print URL anyway
    def _timeout_fallback() -> None:
        nonlocal browser_opened
        if not browser_opened:
            browser_host = "localhost" if host == "0.0.0.0" else host
            url = f"http://{browser_host}:{port}"
            console.print(
                f"\n[bold green]Circuit Explorer should be at:[/] [link={url}]{url}[/]\n"
            )
            browser_opened = True

    timer = threading.Timer(30.0, _timeout_fallback)
    timer.daemon = True
    timer.start()

    try:
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
                # Detect Vite ready line and open browser
                if not browser_opened and "Local:" in line and "http" in line:
                    timer.cancel()
                    browser_host = "localhost" if host == "0.0.0.0" else host
                    url = f"http://{browser_host}:{port}"
                    console.print(
                        f"\n[bold green]Circuit Explorer running at:[/] [link={url}]{url}[/]\n"
                    )
                    if should_open:
                        _open_browser(host, port)
                    browser_opened = True
        process.wait()
    finally:
        timer.cancel()
        signal.signal(signal.SIGINT, original_handler)
        if process.poll() is None:
            process.terminate()

    rc = process.returncode
    # Negative return code means killed by signal (e.g. SIGTERM from Ctrl+C) — treat as clean
    if rc is None or rc < 0:
        return 0
    return rc


def run_visualizer(
    *,
    host: str = "localhost",
    port: int = 5173,
    no_open: bool = False,
    debug: bool = False,
) -> int:
    """Main entry point for the visualizer command.

    Returns exit code (0 = success, 1 = error).
    """
    console = Console(stderr=True)

    try:
        check_node_available()
        check_node_version()
    except NodeNotFoundError as exc:
        console.print(
            Panel(
                f"[bold red]{exc.missing}[/] is not installed.\n\n"
                "[bold]Install Node.js:[/]\n"
                "  macOS:         [cyan]brew install node[/]\n"
                "  Ubuntu/Debian: [cyan]sudo apt install nodejs npm[/]\n"
                "  Other:         [link=https://nodejs.org/en/download]https://nodejs.org/en/download[/]",
                title="[bold red]Missing Prerequisite[/]",
                border_style="red",
            )
        )
        return 1
    except NodeVersionError as exc:
        console.print(
            Panel(
                f"Found Node.js [bold]{exc.found}[/] but version "
                f"[bold]>= {exc.minimum}[/] is required.\n\n"
                "[bold]Upgrade Node.js:[/]\n"
                "  macOS:         [cyan]brew upgrade node[/]\n"
                "  Ubuntu/Debian: [cyan]sudo apt install nodejs npm[/] or use [cyan]nvm[/]\n"
                "  Other:         [link=https://nodejs.org/en/download]https://nodejs.org/en/download[/]",
                title="[bold red]Node.js Version Too Old[/]",
                border_style="red",
            )
        )
        return 1

    try:
        explorer_dir = find_explorer_dir()
    except ExplorerNotFoundError as exc:
        console.print(
            Panel(
                str(exc),
                title="[bold red]Explorer Not Found[/]",
                border_style="red",
            )
        )
        return 1

    # npm ci if needed
    ran_npm_ci = False
    if not (explorer_dir / "node_modules").is_dir():
        try:
            _run_npm_ci(explorer_dir, debug=debug)
            ran_npm_ci = True
        except NpmCiError as exc:
            console.print("\n[bold red]Installing dependencies failed.[/]")
            if debug:
                console.print(f"[dim]{exc.stderr}[/]")
            return 1

    # Start dev server
    exit_code = _start_dev_server(
        explorer_dir, host=host, port=port, no_open=no_open, debug=debug
    )

    # Retry once with npm ci if server failed and we didn't already run it
    if exit_code != 0 and not ran_npm_ci:
        console.print("[yellow]Dev server failed. Retrying after reinstalling dependencies...[/]")
        try:
            _run_npm_ci(explorer_dir, debug=debug)
        except NpmCiError as exc:
            console.print("\n[bold red]Installing dependencies failed.[/]")
            if debug:
                console.print(f"[dim]{exc.stderr}[/]")
            return 1
        exit_code = _start_dev_server(
            explorer_dir, host=host, port=port, no_open=no_open, debug=debug
        )

    if exit_code != 0:
        console.print(f"\n[bold red]Dev server exited unexpectedly (code {exit_code}).[/]")
        return 1

    return 0
```

Note: This is the **complete** `visualizer.py` file. Write it as a single file — all imports at the top, all classes and functions in one module. The task breakdown below adds tests incrementally, but the implementation is presented here in full to avoid import ordering issues across incremental additions.

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_visualizer.py -v`
Expected: All 7 tests PASS

- [ ] **Step 3: Commit**

```bash
git add src/circuit_estimation/visualizer.py
git commit -m "feat: add visualizer module with prereq checks"
```

---

### Task 3: Explorer path resolution — tests and verification

**Files:**
- Modify: `tests/test_visualizer.py`

- [ ] **Step 1: Add tests for `find_explorer_dir`**

Append to `tests/test_visualizer.py`:

```python
from circuit_estimation.visualizer import ExplorerNotFoundError, find_explorer_dir


def test_find_explorer_dir_finds_tools_dir(tmp_path):
    """Simulates a repo checkout with pyproject.toml and tools/circuit-explorer/."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    explorer = repo / "tools" / "circuit-explorer"
    explorer.mkdir(parents=True)
    (explorer / "package.json").write_text("{}")

    fake_module_dir = repo / "src" / "circuit_estimation"
    fake_module_dir.mkdir(parents=True)

    result = find_explorer_dir(start=fake_module_dir)
    assert result == explorer


def test_find_explorer_dir_raises_when_no_repo_root(tmp_path):
    """No pyproject.toml anywhere above start."""
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    with pytest.raises(ExplorerNotFoundError):
        find_explorer_dir(start=deep)


def test_find_explorer_dir_raises_when_no_tools_dir(tmp_path):
    """pyproject.toml exists but tools/circuit-explorer/ does not."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "pyproject.toml").write_text("[project]\nname = 'test'\n")
    fake_module_dir = repo / "src" / "circuit_estimation"
    fake_module_dir.mkdir(parents=True)
    with pytest.raises(ExplorerNotFoundError):
        find_explorer_dir(start=fake_module_dir)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_visualizer.py -v`
Expected: All 10 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_visualizer.py
git commit -m "test: add explorer path resolution tests"
```

---

### Task 4: Headless detection — tests and verification

**Files:**
- Modify: `tests/test_visualizer.py`

- [ ] **Step 1: Add tests for `is_headless`**

Append to `tests/test_visualizer.py`:

```python
import sys
from circuit_estimation.visualizer import is_headless


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
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_visualizer.py -v`
Expected: All 16 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_visualizer.py
git commit -m "test: add headless detection tests"
```

---

### Task 5: Dev server orchestration — tests

**Files:**
- Modify: `tests/test_visualizer.py`

- [ ] **Step 1: Add tests for `run_visualizer` orchestration**

Append to `tests/test_visualizer.py`:

```python
import circuit_estimation.visualizer as viz_mod
from circuit_estimation.visualizer import run_visualizer


def _make_fake_explorer(tmp_path, *, with_node_modules: bool = True):
    """Create a fake explorer directory and return it."""
    explorer = tmp_path / "circuit-explorer"
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
    """When node_modules/ exists, npm ci should not be called."""
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
    """When node_modules/ is missing, npm ci should be called before dev server."""
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
    """When dev server fails and npm ci wasn't run, retry after npm ci."""
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
        # First dev server call fails, second succeeds
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
    # Should have: dev (fail), npm ci, dev (success)
    assert sum(1 for s in cmd_strings if "ci" in s) == 1
    assert sum(1 for s in cmd_strings if "dev" in s) == 2


def test_run_visualizer_returns_1_when_npm_ci_fails(tmp_path, monkeypatch):
    """When npm ci fails, return 1."""
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


def test_run_visualizer_returns_1_when_explorer_not_found(monkeypatch):
    """When explorer dir not found, return 1."""
    monkeypatch.setattr(viz_mod, "check_node_available", lambda: None)
    monkeypatch.setattr(viz_mod, "check_node_version", lambda: None)
    monkeypatch.setattr(
        viz_mod, "find_explorer_dir",
        lambda **kw: (_ for _ in ()).throw(viz_mod.ExplorerNotFoundError()),
    )

    result = run_visualizer(host="localhost", port=5173, no_open=True, debug=False)
    assert result == 1


def test_run_visualizer_returns_1_when_node_missing(monkeypatch):
    """When node is not found, return 1."""
    monkeypatch.setattr(
        viz_mod, "check_node_available",
        lambda: (_ for _ in ()).throw(viz_mod.NodeNotFoundError("node")),
    )

    result = run_visualizer(host="localhost", port=5173, no_open=True, debug=False)
    assert result == 1
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_visualizer.py -v`
Expected: All 22 tests PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_visualizer.py
git commit -m "test: add dev server orchestration and error path tests"
```

---

## Chunk 2: CLI Integration & Documentation

### Task 6: CLI registration — tests

**Files:**
- Modify: `tests/test_visualizer.py`

- [ ] **Step 1: Add tests for CLI dispatch**

Append to `tests/test_visualizer.py`:

```python
from circuit_estimation import cli


def test_cli_visualizer_dispatches_to_run_visualizer(monkeypatch):
    """cestim visualizer should call run_visualizer with correct args."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --group dev pytest tests/test_visualizer.py::test_cli_visualizer_dispatches_to_run_visualizer -v`
Expected: FAIL (argparse rejects unknown "visualizer" subcommand)

- [ ] **Step 3: Commit**

```bash
git add tests/test_visualizer.py
git commit -m "test: add failing CLI visualizer dispatch tests"
```

---

### Task 7: CLI registration — implementation

**Files:**
- Modify: `src/circuit_estimation/cli.py:565-567` (inside `_build_participant_parser`, before `return parser`)
- Modify: `src/circuit_estimation/cli.py:897-899` (inside `_main_participant`, before `raise ValueError`)

- [ ] **Step 1: Register visualizer subparser in `_build_participant_parser`**

In `cli.py`, inside `_build_participant_parser()`, add before the `return parser` line (line 567):

```python
    visualizer_parser = subparsers.add_parser(
        "visualizer",
        help="Launch the interactive Circuit Explorer in a browser.",
    )
    visualizer_parser.add_argument(
        "--host", default="localhost", help="Bind address (default: localhost)."
    )
    visualizer_parser.add_argument(
        "--port", type=int, default=5173, help="Port number (default: 5173)."
    )
    visualizer_parser.add_argument(
        "--no-open", action="store_true", help="Don't auto-open browser."
    )
    visualizer_parser.add_argument("--debug", action="store_true")
```

- [ ] **Step 2: Add dispatch in `_main_participant`**

In `_main_participant()`, before the `raise ValueError(f"Unsupported command: {command}")` line (line 899), add:

```python
        if command == "visualizer":
            from .visualizer import run_visualizer

            return run_visualizer(
                host=str(args.host),
                port=int(args.port),
                no_open=bool(args.no_open),
                debug=bool(args.debug),
            )
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `uv run --group dev pytest tests/test_visualizer.py -v`
Expected: All 24 tests PASS

- [ ] **Step 4: Run full test suite to ensure no regressions**

Run: `uv run --group dev pytest -m "not exhaustive" -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/circuit_estimation/cli.py
git commit -m "feat: register cestim visualizer subcommand"
```

---

### Task 8: Update docs/how-to/use-circuit-explorer.md

**Files:**
- Modify: `docs/how-to/use-circuit-explorer.md`

- [ ] **Step 1: Rewrite to make `cestim visualizer` the primary method**

Replace the full file content with:

```markdown
# Use Circuit Explorer

![Circuit Explorer – a small circuit with 4 wires and 5 layers, after running Ground Truth estimation](../../assets/circuit-explorer-visualization.svg)

## When to use this page

Use this page when you want visual intuition about circuit behavior and estimator error patterns.

Circuit Explorer is optional and is not the submission interface.

## Do this now

```bash
cestim visualizer
```

This checks for Node.js, installs dependencies if needed, and opens the explorer in your browser.

### Options

```bash
cestim visualizer --host 0.0.0.0 --port 8080   # bind to all interfaces on port 8080
cestim visualizer --no-open                       # don't auto-open browser
```

On SSH/headless environments, the browser won't auto-open — just follow the printed URL.

### Manual setup (fallback)

If `cestim visualizer` doesn't work for your environment:

```bash
cd tools/circuit-explorer
npm ci
npm run dev
```

Open `http://localhost:5173`.

## ✅ Expected outcome

You can interactively inspect circuit structure, layer behavior, and estimator comparisons.

## Suggested workflow

1. Start with small width/depth.
2. Vary seed to inspect structural changes.
3. Compare estimator behavior across layers.
4. Locate where errors concentrate.
5. Convert observations into Python estimator heuristics.

Official score semantics still come from:

```bash
cestim run --estimator <path> --runner subprocess
```

## 🛠 Common first failure

Symptom: app does not start due to missing Node dependencies.

Fix: `cestim visualizer` handles this automatically. For manual setup, run `npm ci` in `tools/circuit-explorer` and retry `npm run dev`.

## ➡️ Next step

- [Validate, Run, and Package](./validate-run-package.md)
- [Problem Setup](../concepts/problem-setup.md)
```

- [ ] **Step 2: Commit**

```bash
git add docs/how-to/use-circuit-explorer.md
git commit -m "docs: update circuit explorer guide to use cestim visualizer"
```

---

### Task 9: Update docs/reference/cli-reference.md

**Files:**
- Modify: `docs/reference/cli-reference.md`

- [ ] **Step 1: Add `cestim visualizer` to entry commands list**

In the "Entry commands" section (around line 17), add `- \`cestim visualizer\`` to the bullet list.

- [ ] **Step 2: Add reference section**

After the `## \`cestim package\`` section (after line 116), before the `## ➡️ Next step` section, add:

```markdown
## `cestim visualizer`

Launch the interactive Circuit Explorer in a browser.

```bash
cestim visualizer [--host HOST] [--port PORT] [--no-open] [--debug]
```

Checks for Node.js (>= 18), installs dependencies if needed, starts the Vite dev server, and auto-opens the browser.

Key options:

- `--host <address>` (default: `localhost`) — bind address, use `0.0.0.0` for remote access
- `--port <number>` (default: `5173`) — port number
- `--no-open` — suppress auto-open browser
- `--debug` — show full npm/Vite output on errors

On SSH/headless environments, browser auto-open is skipped automatically.
```

- [ ] **Step 3: Commit**

```bash
git add docs/reference/cli-reference.md
git commit -m "docs: add cestim visualizer to CLI reference"
```

---

### Task 10: Update README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add visualizer one-liner to quickstart**

In the quickstart section, after the `cestim smoke-test` block (line 61) and before the "Run your first full loop" comment (line 63), add:

```markdown
Explore circuits visually (requires Node.js):

```bash
cestim visualizer
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add cestim visualizer to README quickstart"
```

---

### Task 11: Lint, type-check, and full test pass

**Files:** None (verification only)

- [ ] **Step 1: Run linter**

Run: `uv run --group dev ruff check .`
Expected: No errors

- [ ] **Step 2: Run formatter check**

Run: `uv run --group dev ruff format --check .`
Expected: No reformatting needed (fix if needed with `ruff format .`)

- [ ] **Step 3: Run type checker**

Run: `uv run --group dev pyright`
Expected: No errors in visualizer.py or cli.py changes

- [ ] **Step 4: Run full test suite**

Run: `uv run --group dev pytest -m "not exhaustive" -v`
Expected: All tests pass

- [ ] **Step 5: Commit any fixes**

Only if Steps 1-4 required changes:

```bash
git add -u
git commit -m "fix: address lint/type-check issues in visualizer"
```
