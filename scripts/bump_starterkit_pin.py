#!/usr/bin/env python
"""Bump the whest-starterkit federation pin to the latest upstream ``main``.

Resolves the current tip of AIcrowd/whest-starterkit ``main``, shows which
``docs/`` files changed since the current pin, and updates
``website/starterkit.lock.json``. Keeps federation reproducible and gated: the
deployed site only changes when this bump is reviewed and committed.

Usage
-----
    python scripts/bump_starterkit_pin.py            # bump the pin to main's tip
    python scripts/bump_starterkit_pin.py --dry-run  # show the change; do not write
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = ROOT / "website" / "starterkit.lock.json"
REPO_URL = "https://github.com/AIcrowd/whest-starterkit"


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
    args = parser.parse_args(argv)

    lock = load_lock()
    old = lock["sha"]
    ref = lock.get("ref", "main")
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
