import json

import scripts.bump_starterkit_pin as bp

WHESTBENCH_PYPROJECT = """
[project]
name = "whestbench"
version = "0.15.0"
dependencies = [
    "flopscope>=0.11.0,<0.12.0",
    "httpx>=0.27",
]

[tool.uv.sources]
flopscope = { git = "https://github.com/AIcrowd/flopscope.git", branch = "main" }
"""

KIT_PYPROJECT = """
[project]
name = "whest-starterkit"
version = "0.1.0"
dependencies = [
    "flopscope>=0.11.0,<0.12.0",
    "whestbench>=0.15.0,<0.16.0",
]

[tool.uv.sources]
flopscope = { git = "https://github.com/AIcrowd/flopscope.git", branch = "main" }
"""


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


def test_requirement_specs_normalizes_names_and_drops_markers():
    specs = bp.requirement_specs(
        {
            "project": {
                "dependencies": [
                    "Flop_Scope >= 0.11.0, <0.12.0",
                    "torch>=2.1; extra == 'gpu'",
                    "rich",
                ]
            }
        }
    )
    assert specs["flop-scope"] == ">=0.11.0,<0.12.0"
    assert specs["torch"] == ">=2.1"
    assert specs["rich"] == ""


def test_compare_pins_agree_when_kit_matches_this_repo():
    assert bp.compare_pins(KIT_PYPROJECT, WHESTBENCH_PYPROJECT) == []


def test_compare_pins_flags_stale_whestbench_floor():
    kit = KIT_PYPROJECT.replace("whestbench>=0.15.0,<0.16.0", "whestbench>=0.14.0,<0.15.0")
    problems = bp.compare_pins(kit, WHESTBENCH_PYPROJECT)
    assert len(problems) == 1
    assert "whestbench" in problems[0]
    assert "0.14.0" in problems[0] and "0.15.0" in problems[0]


def test_compare_pins_flags_missing_whestbench_dependency():
    kit = KIT_PYPROJECT.replace('    "whestbench>=0.15.0,<0.16.0",\n', "")
    problems = bp.compare_pins(kit, WHESTBENCH_PYPROJECT)
    assert problems == ["whestbench: kit declares no whestbench dependency"]


def test_compare_pins_flags_flopscope_range_mismatch():
    kit = KIT_PYPROJECT.replace("flopscope>=0.11.0,<0.12.0", "flopscope>=0.10.0,<0.11.0")
    problems = bp.compare_pins(kit, WHESTBENCH_PYPROJECT)
    assert len(problems) == 1
    assert problems[0].startswith("flopscope: kit pins '>=0.10.0,<0.11.0'")


def test_compare_pins_flags_registry_resolution_while_this_repo_tracks_a_branch():
    # Same declared range, different code: the kit would meter against the last
    # release while the grader meters against the branch tip.
    kit = KIT_PYPROJECT.split("[tool.uv.sources]")[0]
    problems = bp.compare_pins(kit, WHESTBENCH_PYPROJECT)
    assert len(problems) == 1
    assert "the registry" in problems[0]


def test_compare_pins_flags_branch_mismatch():
    kit = KIT_PYPROJECT.replace('branch = "main"', 'branch = "release-0.11"')
    problems = bp.compare_pins(kit, WHESTBENCH_PYPROJECT)
    assert len(problems) == 1
    assert "release-0.11" in problems[0]


def _stub_check_deps(tmp_path, monkeypatch, kit_pyproject):
    lock = {"repo": "AIcrowd/whest-starterkit", "ref": "main", "sha": "abc", "synced_at": "x"}
    lock_path = tmp_path / "starterkit.lock.json"
    lock_path.write_text(json.dumps(lock))
    bench_path = tmp_path / "pyproject.toml"
    bench_path.write_text(WHESTBENCH_PYPROJECT)
    monkeypatch.setattr(bp, "LOCK", lock_path)
    monkeypatch.setattr(bp, "WHESTBENCH_PYPROJECT", bench_path)
    monkeypatch.setattr(bp, "resolve_main_sha", lambda ref="main": "def456")
    monkeypatch.setattr(bp, "fetch_kit_pyproject", lambda sha: kit_pyproject)
    return lock_path


def test_main_check_deps_reports_agreement(tmp_path, monkeypatch, capsys):
    _stub_check_deps(tmp_path, monkeypatch, KIT_PYPROJECT)
    assert bp.main(["--check-deps"]) == 0
    assert "OK:" in capsys.readouterr().out


def test_main_check_deps_exits_nonzero_on_drift(tmp_path, monkeypatch, capsys):
    kit = KIT_PYPROJECT.replace("whestbench>=0.15.0,<0.16.0", "whestbench>=0.14.0,<0.15.0")
    _stub_check_deps(tmp_path, monkeypatch, kit)
    assert bp.main(["--check-deps"]) == 1
    assert "MISMATCH" in capsys.readouterr().out


def test_main_check_deps_never_writes_the_lock(tmp_path, monkeypatch):
    lock_path = _stub_check_deps(tmp_path, monkeypatch, KIT_PYPROJECT)
    assert bp.main(["--check-deps"]) == 0
    assert json.loads(lock_path.read_text())["sha"] == "abc"
