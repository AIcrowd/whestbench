"""Install / environment health checks for ``whest doctor``.

Each check returns a :class:`Check` dict. ``run_all()`` runs the
full suite in stable order. Rendering lives in ``reporting.py``.
"""

from __future__ import annotations

import json as _json
import platform
from importlib.metadata import Distribution, PackageMetadata, PackageNotFoundError, metadata
from typing import Literal, Optional, TypedDict

try:
    from packaging.specifiers import SpecifierSet  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # pragma: no cover - packaging is transitively installed
    SpecifierSet = None  # type: ignore[assignment]

_FALLBACK_PYTHON_FLOOR = ">=3.10"


Status = Literal["ok", "warn", "fail"]


class Check(TypedDict):
    """A single health-check result.

    - ``name``: stable machine-readable id (e.g. ``"python_version"``).
    - ``label``: human-readable display label.
    - ``status``: ``"ok"`` / ``"warn"`` / ``"fail"``.
    - ``detail``: one-line summary of what was observed.
    - ``fix_hint``: one-line remediation. Non-null iff ``status != "ok"``.
    """

    name: str
    label: str
    status: Status
    detail: str
    fix_hint: Optional[str]


def _whestbench_metadata() -> PackageMetadata:
    """Indirection so tests can patch it."""
    return metadata("whestbench")


# --- check_python ------------------------------------------------------------


def check_python() -> Check:
    try:
        meta = _whestbench_metadata()
        requires = meta.get("Requires-Python") or ""  # pyright: ignore[reportAttributeAccessIssue]
    except Exception:
        requires = ""

    fallback_used = False
    if not requires:
        requires = _FALLBACK_PYTHON_FLOOR
        fallback_used = True

    current = platform.python_version()
    if SpecifierSet is not None:
        spec = SpecifierSet(requires)
        ok = spec.contains(current, prereleases=True)
    else:  # pragma: no cover
        # Very coarse fallback: extract "X.Y" from ">=X.Y" and compare.
        import re

        m = re.search(r">=\s*(\d+)\.(\d+)", requires)
        if m:
            required = (int(m.group(1)), int(m.group(2)))
            actual = tuple(int(p) for p in current.split(".")[:2])
            ok = actual >= required
        else:
            ok = True

    suffix = " (fallback floor)" if fallback_used else ""
    detail = (
        f"{current} satisfies {requires}{suffix}"
        if ok
        else f"{current} does not satisfy {requires}{suffix}"
    )
    fix_hint = None if ok else f"Install Python {requires} (e.g., via uv: uv python install 3.12)"

    return Check(
        name="python_version",
        label="Python version",
        status="ok" if ok else "fail",
        detail=detail,
        fix_hint=fix_hint,
    )


# --- check_uv ----------------------------------------------------------------


def check_uv() -> Check:
    import shutil

    path = shutil.which("uv")
    if path:
        return Check(
            name="uv",
            label="uv on PATH",
            status="ok",
            detail=path,
            fix_hint=None,
        )
    return Check(
        name="uv",
        label="uv on PATH",
        status="warn",
        detail="not on PATH",
        fix_hint=(
            "uv is recommended for the quickstart commands; install via "
            "'curl -LsSf https://astral.sh/uv/install.sh | sh'. "
            "Safe to ignore if you installed via pip."
        ),
    )


# --- check_install_mode ------------------------------------------------------


def check_install_mode() -> Check:
    try:
        dist = Distribution.from_name("whestbench")
    except PackageNotFoundError:
        return Check(
            name="install_mode",
            label="whest install mode",
            status="fail",
            detail="whestbench package not found",
            fix_hint="Run 'uv sync' in the repo directory.",
        )

    version = dist.version
    direct_url_raw = dist.read_text("direct_url.json")

    if direct_url_raw:
        try:
            direct_url = _json.loads(direct_url_raw)
        except _json.JSONDecodeError:
            direct_url = {}
        dir_info = direct_url.get("dir_info") or {}
        editable = bool(dir_info.get("editable"))
        url = direct_url.get("url", "")
        path = url.replace("file://", "") if url.startswith("file://") else url
        if editable and path:
            detail = f"editable · {path}"
        elif path:
            detail = f"from-source · {path}"
        else:
            detail = f"tool-installed · {version}"
    else:
        detail = f"tool-installed · {version}"

    return Check(
        name="install_mode",
        label="whest install mode",
        status="ok",
        detail=detail,
        fix_hint=None,
    )


# --- check_node --------------------------------------------------------------


def check_node() -> Check:
    import shutil
    import subprocess

    path = shutil.which("node")
    if not path:
        return Check(
            name="node_js",
            label="Node.js on PATH",
            status="warn",
            detail="not found on PATH",
            fix_hint="Install Node.js 20+ from https://nodejs.org — required only for 'whest visualizer'.",
        )

    try:
        result = subprocess.run(
            [path, "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        version = result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        version = ""

    if version:
        return Check(
            name="node_js",
            label="Node.js on PATH",
            status="ok",
            detail=version,
            fix_hint=None,
        )
    return Check(
        name="node_js",
        label="Node.js on PATH",
        status="warn",
        detail="installed but 'node --version' failed",
        fix_hint="Check your Node.js installation.",
    )


# --- check_blas --------------------------------------------------------------


def check_blas() -> Check:
    try:
        import threadpoolctl  # pyright: ignore[reportMissingModuleSource]
    except ImportError:
        return Check(
            name="blas_threads",
            label="BLAS thread pool",
            status="fail",
            detail="threadpoolctl import failed",
            fix_hint="Run 'uv sync' in the repo directory.",
        )

    pools = threadpoolctl.threadpool_info()
    if not pools:
        return Check(
            name="blas_threads",
            label="BLAS thread pool",
            status="warn",
            detail="no BLAS pool detected",
            fix_hint="threadpoolctl detected no BLAS pool; numpy may be using a fallback. Usually harmless.",
        )

    parts = [
        f"{pool.get('internal_api', '?')} · {pool.get('num_threads', '?')} threads"
        for pool in pools
    ]
    return Check(
        name="blas_threads",
        label="BLAS thread pool",
        status="ok",
        detail=" · ".join(parts),
        fix_hint=None,
    )


# --- check_disk --------------------------------------------------------------


_MIN_FREE_GIB = 1.0


def check_disk() -> Check:
    import os
    import shutil

    cwd = os.getcwd()
    try:
        usage = shutil.disk_usage(cwd)
    except OSError as exc:
        return Check(
            name="disk_space",
            label="Free disk in CWD",
            status="fail",
            detail=f"could not read disk usage: {exc}",
            fix_hint="Check that the path exists and is readable.",
        )

    free_gib = usage.free / (1024**3)
    detail = f"{free_gib:.1f} GiB free"
    if free_gib >= _MIN_FREE_GIB:
        return Check(
            name="disk_space",
            label="Free disk in CWD",
            status="ok",
            detail=detail,
            fix_hint=None,
        )
    return Check(
        name="disk_space",
        label="Free disk in CWD",
        status="warn",
        detail=detail,
        fix_hint=(
            "Less than 1 GiB free in CWD. Datasets generated by "
            "'whest create-dataset' can reach hundreds of MB; free some space "
            "or move to a larger volume."
        ),
    )


# --- check_cwd_writable ------------------------------------------------------


def check_cwd_writable() -> Check:
    import os
    import tempfile

    cwd = os.getcwd()
    try:
        with tempfile.NamedTemporaryFile(dir=cwd, delete=True):
            pass
    except (OSError, PermissionError) as exc:
        return Check(
            name="cwd_writable",
            label="CWD writable",
            status="fail",
            detail=f"cannot write to {cwd}: {exc}",
            fix_hint=(
                "Cannot write to current working directory. "
                "Check permissions or cd to a writable location."
            ),
        )
    return Check(
        name="cwd_writable",
        label="CWD writable",
        status="ok",
        detail=cwd,
        fix_hint=None,
    )


# --- run_all -----------------------------------------------------------------


_CHECKS = (
    ("python_version", check_python),
    ("uv", check_uv),
    ("install_mode", check_install_mode),
    ("node_js", check_node),
    ("blas_threads", check_blas),
    ("disk_space", check_disk),
    ("cwd_writable", check_cwd_writable),
)


def _crashed_check(name: str, exc: BaseException) -> Check:
    return Check(
        name=name,
        label=name.replace("_", " ").title(),
        status="fail",
        detail=f"check crashed: {type(exc).__name__}: {exc}",
        fix_hint="Please file a bug at https://github.com/AIcrowd/whestbench/issues.",
    )


def run_all(*, debug: bool = False) -> "list[Check]":
    """Run every check in stable order.

    ``debug=True`` re-raises exceptions from crashing checks; the default
    catches them and surfaces as ``status="fail"``.
    """
    results: "list[Check]" = []
    for name, fn in _CHECKS:
        try:
            # Look up through the module so tests can ``patch`` individual checks.
            import whestbench.doctor as _self  # noqa: PLW0406

            impl = getattr(_self, fn.__name__)
            results.append(impl())
        except BaseException as exc:
            if debug:
                raise
            results.append(_crashed_check(name, exc))
    return results
