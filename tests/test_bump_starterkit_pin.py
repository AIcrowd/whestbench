import json

import scripts.bump_starterkit_pin as bp


def test_update_lock_sets_sha_and_synced_at():
    lock = {
        "repo": "AIcrowd/whest-starterkit",
        "ref": "main",
        "sha": "old",
        "synced_at": "2020-01-01T00:00:00Z",
    }
    out = bp.update_lock(lock, "newsha")
    assert out["sha"] == "newsha"
    assert out["synced_at"] != "2020-01-01T00:00:00Z"
    assert out["synced_at"].endswith("Z")
    assert out["repo"] == "AIcrowd/whest-starterkit"


def test_main_no_change_does_not_write(tmp_path, monkeypatch):
    lock = {"repo": "AIcrowd/whest-starterkit", "ref": "main", "sha": "abc", "synced_at": "x"}
    p = tmp_path / "starterkit.lock.json"
    p.write_text(json.dumps(lock))
    monkeypatch.setattr(bp, "LOCK", p)
    monkeypatch.setattr(bp, "resolve_main_sha", lambda ref="main": "abc")
    assert bp.main([]) == 0
    assert json.loads(p.read_text())["sha"] == "abc"


def test_main_dry_run_does_not_write(tmp_path, monkeypatch):
    lock = {"repo": "AIcrowd/whest-starterkit", "ref": "main", "sha": "abc", "synced_at": "x"}
    p = tmp_path / "starterkit.lock.json"
    p.write_text(json.dumps(lock))
    monkeypatch.setattr(bp, "LOCK", p)
    monkeypatch.setattr(bp, "resolve_main_sha", lambda ref="main": "def456")
    monkeypatch.setattr(bp, "changed_docs", lambda old, new: ["docs/concepts/x.md"])
    assert bp.main(["--dry-run"]) == 0
    assert json.loads(p.read_text())["sha"] == "abc"


def test_main_writes_new_sha(tmp_path, monkeypatch):
    lock = {"repo": "AIcrowd/whest-starterkit", "ref": "main", "sha": "abc", "synced_at": "x"}
    p = tmp_path / "starterkit.lock.json"
    p.write_text(json.dumps(lock))
    monkeypatch.setattr(bp, "LOCK", p)
    monkeypatch.setattr(bp, "resolve_main_sha", lambda ref="main": "def456")
    monkeypatch.setattr(bp, "changed_docs", lambda old, new: [])
    assert bp.main([]) == 0
    assert json.loads(p.read_text())["sha"] == "def456"
