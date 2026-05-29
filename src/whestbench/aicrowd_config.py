"""AIcrowd credential store, interoperable with aicrowd-cli.

aicrowd-cli stores its credentials in a TOML file at
``click.get_app_dir("aicrowd-cli")/config.toml`` under the key
``aicrowd_api_key``. We replicate that exact location and format (without a
hard dependency on click) so a participant logged in via either CLI is
recognized by the other.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import toml

_APP_NAME = "aicrowd-cli"
_KEY = "aicrowd_api_key"  # LoginConstants.CONFIG_KEY in aicrowd-cli


class NotLoggedIn(RuntimeError):
    """No API key found via arg, env, or config file."""


def _app_dir() -> Path:
    """Replicate click.get_app_dir('aicrowd-cli') for the current platform."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP_NAME
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / _APP_NAME
    # Unix: honor XDG_CONFIG_HOME, default ~/.config
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / _APP_NAME


def config_path() -> Path:
    """Absolute path to the aicrowd-cli config.toml."""
    return _app_dir() / "config.toml"


def _load_all() -> dict:
    p = config_path()
    if not p.is_file():
        return {}
    try:
        return toml.load(p)
    except toml.TomlDecodeError:
        return {}


def load_api_key() -> Optional[str]:
    """Return the stored AIcrowd API key, or None."""
    return _load_all().get(_KEY) or None


def save_api_key(api_key: str) -> Path:
    """Write the API key to config.toml, preserving any other keys."""
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _load_all()
    data[_KEY] = api_key
    with open(p, "w") as fh:
        toml.dump(data, fh)
    return p


def resolve_api_key(explicit: Optional[str]) -> str:
    """Resolve the API key by precedence: arg > AICROWD_API_KEY env > config file.

    Raises NotLoggedIn if none is found.
    """
    key = explicit or os.environ.get("AICROWD_API_KEY") or load_api_key()
    if not key:
        raise NotLoggedIn("No AIcrowd API key found. Run `whest login` (or set AICROWD_API_KEY).")
    return key
