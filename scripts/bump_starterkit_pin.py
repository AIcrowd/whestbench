#!/usr/bin/env python
"""Bump the whest-starterkit federation pin to the latest upstream ``main``.

Resolves the current tip of AIcrowd/whest-starterkit ``main``, shows which
``docs/`` files changed since the current pin, and updates
``website/starterkit.lock.json``. Keeps federation reproducible and gated: the
deployed site only changes when this bump is reviewed and committed.

Also checks the other half of the relationship, which the pin does not cover: the
starter kit's *dependency* pins against this repo's own. The kit is the published
spec of the round — participants profile against whatever flopscope and whestbench
it resolves — so a kit that lags this repo hands out FLOP counts that submissions
are not actually scored on.

Usage
-----
    python scripts/bump_starterkit_pin.py              # bump the pin to main's tip
    python scripts/bump_starterkit_pin.py --dry-run    # show the change; do not write
    python scripts/bump_starterkit_pin.py --check-deps # compare dependency pins; exit 1 on drift
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import toml

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "website" / "starterkit.lock.json"
REPO_URL = "https://github.com/AIcrowd/whest-starterkit"
WHESTBENCH_PYPROJECT = ROOT / "pyproject.toml"
KIT_PYPROJECT_RAW = (
    "https://raw.githubusercontent.com/AIcrowd/whest-starterkit/{sha}/pyproject.toml"
)

# Dependencies both repos declare and must resolve identically: the kit's local
# harness has to meter with the same code the grader does.
SHARED_PINS = ("flopscope",)
# Fields of a [tool.uv.sources] entry that decide which code is resolved.
SOURCE_FIELDS = ("git", "url", "path", "branch", "rev", "tag", "subdirectory")

_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$")


def load_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def resolve_main_sha(ref: str = "main") -> str:
    out = subprocess.run(
        ["git", "ls-remote", REPO_URL, ref],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    sha = out.split("\t", 1)[0].strip() if out else ""
    if not sha:
        raise RuntimeError(f"Could not resolve {REPO_URL} {ref}")
    return sha


def changed_docs(old: str, new: str) -> list[str] | None:
    """Best-effort list of changed ``docs/`` files via the gh CLI; None if gh is unavailable."""
    try:
        out = subprocess.run(
            [
                "gh",
                "api",
                f"repos/AIcrowd/whest-starterkit/compare/{old}...{new}",
                "--jq",
                ".files[].filename",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [f for f in out.splitlines() if f.startswith("docs/")]


def _canonical(name: str) -> str:
    """PEP 503 name normalization, so ``Flop_Scope`` and ``flop-scope`` compare equal."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_specs(pyproject: dict) -> dict[str, str]:
    """Map canonical distribution name -> version specifier for ``[project] dependencies``."""
    specs: dict[str, str] = {}
    for raw in pyproject.get("project", {}).get("dependencies", []):
        requirement = str(raw).split(";", 1)[0].strip()
        match = _REQUIREMENT_RE.match(requirement)
        if match:
            specs[_canonical(match.group(1))] = match.group(2).replace(" ", "")
    return specs


def uv_source(pyproject: dict, name: str) -> dict:
    """The ``[tool.uv.sources]`` redirect for ``name``, reduced to the fields that
    decide which code is resolved. Empty dict means "resolved from the registry"."""
    sources = pyproject.get("tool", {}).get("uv", {}).get("sources", {})
    entry = sources.get(name)
    if not isinstance(entry, dict):
        return {}
    return {k: v for k, v in entry.items() if k in SOURCE_FIELDS}


def lower_bound(spec: str) -> str | None:
    """The ``>=`` floor of a version specifier, or None if it has none."""
    match = re.search(r">=\s*([^,\s]+)", spec)
    return match.group(1) if match else None


def compare_pins(kit_pyproject: str, whestbench_pyproject: str) -> list[str]:
    """Report every disagreement between the starter kit's dependency pins and this
    repo's own. An empty list means the kit meters with the code the grader runs.

    Two things are checked, because they fail independently:

    * the kit's ``whestbench`` floor must be this repo's released version — a kit a
      minor behind documents an older harness's behaviour;
    * the shared pins (``SHARED_PINS``) must match exactly, both the version range and
      any ``[tool.uv.sources]`` redirect. While this repo tracks a flopscope branch,
      a kit resolving flopscope from the registry meters against different accounting
      even though both declare the same range.
    """
    kit = toml.loads(kit_pyproject)
    bench = toml.loads(whestbench_pyproject)
    kit_specs = requirement_specs(kit)
    bench_specs = requirement_specs(bench)
    bench_version = str(bench.get("project", {}).get("version", ""))
    problems: list[str] = []

    kit_whestbench = kit_specs.get("whestbench")
    if kit_whestbench is None:
        problems.append("whestbench: kit declares no whestbench dependency")
    elif lower_bound(kit_whestbench) != bench_version:
        floor = lower_bound(kit_whestbench) or "unset"
        problems.append(
            f"whestbench: kit pins '{kit_whestbench}' (floor {floor}); "
            f"this repo is version {bench_version}"
        )

    for name in SHARED_PINS:
        bench_spec = bench_specs.get(_canonical(name))
        if bench_spec is None:
            continue
        kit_spec = kit_specs.get(_canonical(name))
        if kit_spec is None:
            problems.append(f"{name}: kit declares no dependency; this repo pins '{bench_spec}'")
        elif kit_spec != bench_spec:
            problems.append(f"{name}: kit pins '{kit_spec}'; this repo pins '{bench_spec}'")
        kit_source = uv_source(kit, name)
        bench_source = uv_source(bench, name)
        if kit_source != bench_source:
            problems.append(
                f"{name}: kit resolves from {kit_source or 'the registry'}; "
                f"this repo resolves from {bench_source or 'the registry'}"
            )

    return problems


def fetch_kit_pyproject(sha: str) -> str:
    """The starter kit's ``pyproject.toml`` at ``sha``, read from raw.githubusercontent."""
    with urllib.request.urlopen(KIT_PYPROJECT_RAW.format(sha=sha), timeout=30) as response:
        return response.read().decode("utf-8")


def check_dependency_pins(ref: str = "main") -> int:
    """Compare the tip of the kit's ``ref`` against this repo's pins. 0 = agreement."""
    sha = resolve_main_sha(ref)
    problems = compare_pins(
        fetch_kit_pyproject(sha),
        WHESTBENCH_PYPROJECT.read_text(encoding="utf-8"),
    )
    print(f"Starter-kit dependency pins @ {ref} ({sha[:7]}) vs this repo:")
    if not problems:
        print("  OK: the kit resolves the same grader dependencies this repo does.")
        return 0
    for problem in problems:
        print(f"  MISMATCH: {problem}")
    print()
    print("The kit is the published spec of the round: participants profile against the")
    print("dependencies it resolves, so a kit behind this repo reports FLOP counts that")
    print("submissions are not scored on. Fix upstream, in the kit's pyproject.toml:")
    print(f"  {REPO_URL}/blob/{ref}/pyproject.toml")
    return 1


def update_lock(lock: dict, new_sha: str) -> dict:
    updated = dict(lock)
    updated["sha"] = new_sha
    updated["synced_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return updated


def write_lock(lock: dict) -> None:
    LOCK.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the pending change without writing the lock file",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help=(
            "Compare the starter kit's dependency pins against this repo's own "
            "and exit 1 on drift. Never writes the lock file."
        ),
    )
    args = parser.parse_args(argv)

    lock = load_lock()
    ref = lock.get("ref", "main")

    if args.check_deps:
        return check_dependency_pins(ref)

    old = lock["sha"]
    new = resolve_main_sha(ref)

    if old == new:
        print(f"Already up to date: pin == {ref} ({new[:7]})")
        return 0

    print(f"Starter-kit pin: {old[:7]} -> {new[:7]}")
    print(f"  Compare: {REPO_URL}/compare/{old}...{new}")
    files = changed_docs(old, new)
    if files is None:
        print("  (install the gh CLI to list changed files; see the compare URL above)")
    elif files:
        print("  Changed docs/ files:")
        for f in files:
            print(f"    {f}")
    else:
        print("  No files under docs/ changed (other paths may have).")

    if args.dry_run:
        print("\n--dry-run: lock file NOT modified.")
        return 0

    write_lock(update_lock(lock, new))
    try:
        display = LOCK.relative_to(ROOT)
    except ValueError:
        display = LOCK
    print(f"\nUpdated {display} to {new[:7]}.")
    print("Next: review the diff, run `make docs-build` to validate, then commit the lock bump.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
