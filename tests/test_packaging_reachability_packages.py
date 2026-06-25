"""summarize_submission must not mislabel a real subpackage as dead code.

Regression for whestbench#107: a participant shipping a helper subpackage via the
supported folder path saw whest warn that arc_tools/__init__.py etc. were
"unreachable (likely unused)", because reachability keyed modules by filename stem
and never matched the package import name. That eroded trust in `whest package` and
pushed people to hand-roll a manifest (which crashed the grader)."""

from __future__ import annotations

from pathlib import Path

from whestbench.packaging import summarize_submission

_EST_IMPORTS_PKG = (
    "from whestbench import BaseEstimator\n"
    "from arc_tools import helper\n"
    "class Estimator(BaseEstimator):\n"
    "    def predict(self, mlp, budget):\n"
    "        import flopscope.numpy as fnp\n"
    "        return fnp.zeros((mlp.depth, mlp.width))\n"
)


def _subpackage_folder(tmp_path: Path) -> None:
    (tmp_path / "estimator.py").write_text(_EST_IMPORTS_PKG, encoding="utf-8")
    pkg = tmp_path / "arc_tools"
    (pkg / "polynomial").mkdir(parents=True)
    (pkg / "__init__.py").write_text("from ._arc_mlp import helper\n", encoding="utf-8")
    (pkg / "_arc_mlp.py").write_text("def helper():\n    return 0\n", encoding="utf-8")
    (pkg / "polynomial" / "__init__.py").write_text("\n", encoding="utf-8")


def test_imported_subpackage_files_not_unreachable(tmp_path: Path) -> None:
    _subpackage_folder(tmp_path)
    s = summarize_submission(tmp_path)
    assert s.unreachable_py == []


def test_unused_toplevel_module_still_flagged(tmp_path: Path) -> None:
    # The reachability hint must still catch genuinely-unused top-level modules.
    _subpackage_folder(tmp_path)
    (tmp_path / "orphan.py").write_text("# never imported\n", encoding="utf-8")
    s = summarize_submission(tmp_path)
    assert s.unreachable_py == ["orphan.py"]


def test_unimported_subpackage_is_flagged_as_a_unit(tmp_path: Path) -> None:
    # A subpackage that nothing imports is reported (its __init__.py stands in for it),
    # but real, imported packages never are.
    _subpackage_folder(tmp_path)
    dead = tmp_path / "unused_pkg"
    dead.mkdir()
    (dead / "__init__.py").write_text("\n", encoding="utf-8")
    (dead / "thing.py").write_text("x = 1\n", encoding="utf-8")
    s = summarize_submission(tmp_path)
    assert "unused_pkg/__init__.py" in s.unreachable_py
    assert "unused_pkg/thing.py" in s.unreachable_py
    assert "arc_tools/__init__.py" not in s.unreachable_py
