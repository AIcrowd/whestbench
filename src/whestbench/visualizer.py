"""Visualizer command: launch the WhestBench Explorer dev server."""

from __future__ import annotations

import os
import re
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
    """Raised when the WhestBench Explorer directory cannot be located."""

    def __init__(self) -> None:
        super().__init__(
            "WhestBench Explorer not found. This command requires a source checkout "
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
    version_str = result.stdout.strip()
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
    then return tools/whestbench-explorer/.

    The repo root is identified by the presence of pyproject.toml.
    """
    current = start or Path(__file__).resolve().parent
    while True:
        if (current / "pyproject.toml").is_file():
            explorer = current / "tools" / "whestbench-explorer"
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

    def _on_sigint(signum: int, frame: object) -> None:
        process.terminate()

    original_handler = signal.signal(signal.SIGINT, _on_sigint)

    browser_opened = False
    should_open = not no_open and not is_headless()

    def _timeout_fallback() -> None:
        nonlocal browser_opened
        if not browser_opened:
            browser_host = "localhost" if host == "0.0.0.0" else host
            url = f"http://{browser_host}:{port}"
            console.print(
                f"\n[bold green]WhestBench Explorer should be at:[/] [link={url}]{url}[/]\n"
            )
            browser_opened = True

    timer = threading.Timer(30.0, _timeout_fallback)
    timer.daemon = True
    timer.start()

    try:
        if process.stdout is not None:
            for line in process.stdout:
                print(line, end="")
                if not browser_opened and "Local:" in line and "http" in line:
                    timer.cancel()
                    # Parse actual URL from Vite output (port may differ if requested was busy)
                    _m = re.search(r"(https?://\S+)", line)
                    browser_host = "localhost" if host == "0.0.0.0" else host
                    url = _m.group(1).rstrip("/") if _m else f"http://{browser_host}:{port}"
                    _pm = re.search(r":(\d+)", url)
                    actual_port = int(_pm.group(1)) if _pm else port
                    console.print(
                        f"\n[bold green]WhestBench Explorer running at:[/] [link={url}]{url}[/]\n"
                    )
                    if should_open:
                        _open_browser(host, actual_port)
                    browser_opened = True
        process.wait()
    finally:
        timer.cancel()
        signal.signal(signal.SIGINT, original_handler)
        if process.poll() is None:
            process.terminate()

    rc = process.returncode
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

    exit_code = _start_dev_server(explorer_dir, host=host, port=port, no_open=no_open, debug=debug)

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
