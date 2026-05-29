"""Credential store interoperable with aicrowd-cli's config.toml."""

from __future__ import annotations

import pytest
import toml

import whestbench.aicrowd_config as cfg


def test_config_path_matches_aicrowd_cli_app_dir(monkeypatch):
    # On macOS aicrowd-cli uses ~/Library/Application Support/aicrowd-cli;
    # on Linux ~/.config/aicrowd-cli. We assert the leaf is config.toml under
    # an 'aicrowd-cli' app dir, however the platform resolves it.
    p = cfg.config_path()
    assert p.name == "config.toml"
    assert p.parent.name == "aicrowd-cli"


def test_save_then_load_round_trips(tmp_path, monkeypatch):
    target = tmp_path / "aicrowd-cli" / "config.toml"
    monkeypatch.setattr(cfg, "config_path", lambda: target)
    cfg.save_api_key("KEY-123")
    assert cfg.load_api_key() == "KEY-123"
    # Stored under the exact aicrowd-cli key name.
    assert toml.load(target)["aicrowd_api_key"] == "KEY-123"


def test_save_preserves_unknown_keys(tmp_path, monkeypatch):
    target = tmp_path / "aicrowd-cli" / "config.toml"
    target.parent.mkdir(parents=True)
    target.write_text(toml.dumps({"gitlab": {"username": "x"}, "aicrowd_api_key": "old"}))
    monkeypatch.setattr(cfg, "config_path", lambda: target)
    cfg.save_api_key("new")
    data = toml.load(target)
    assert data["aicrowd_api_key"] == "new"
    assert data["gitlab"] == {"username": "x"}  # untouched


def test_load_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "config_path", lambda: tmp_path / "nope" / "config.toml")
    assert cfg.load_api_key() is None


def test_resolve_api_key_precedence(tmp_path, monkeypatch):
    target = tmp_path / "aicrowd-cli" / "config.toml"
    monkeypatch.setattr(cfg, "config_path", lambda: target)
    cfg.save_api_key("from-file")
    monkeypatch.setenv("AICROWD_API_KEY", "from-env")
    # explicit arg wins
    assert cfg.resolve_api_key("from-arg") == "from-arg"
    # then env
    assert cfg.resolve_api_key(None) == "from-env"
    # then file
    monkeypatch.delenv("AICROWD_API_KEY")
    assert cfg.resolve_api_key(None) == "from-file"


def test_resolve_api_key_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "config_path", lambda: tmp_path / "nope" / "config.toml")
    monkeypatch.delenv("AICROWD_API_KEY", raising=False)
    with pytest.raises(cfg.NotLoggedIn):
        cfg.resolve_api_key(None)
