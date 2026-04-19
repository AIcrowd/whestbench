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
        status="fail",
        detail="not on PATH",
        fix_hint="Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh",
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
