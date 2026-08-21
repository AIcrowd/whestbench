"""File mode names what it leaves behind, and refuses the provably-broken case.

A FILE arg ships only that file (see test_packaging_file_vs_folder.py). That is
deliberate — it keeps secrets and stray weights out of a submission by default —
but it used to be announced with boilerplate that never said *what* was dropped,
and it packaged a submission that could not possibly import at grade time.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from whestbench.packaging import package_submission, summarize_submission

_SELF_CONTAINED = (
    "from whestbench import BaseEstimator\n"
    "class Estimator(BaseEstimator):\n"
    "    def predict(self, mlp, budget):\n"
    "        import flopscope.numpy as fnp\n"
    "        return fnp.zeros((mlp.depth, mlp.width))\n"
)

_IMPORTS_HELPER = (
    "import helper\n"
    "from whestbench import BaseEstimator\n"
    "class Estimator(BaseEstimator):\n"
    "    def predict(self, mlp, budget):\n"
    "        import flopscope.numpy as fnp\n"
    "        return fnp.zeros((mlp.depth, mlp.width)) + helper.f()\n"
)


def test_file_mode_summary_names_orphaned_siblings(tmp_path: Path) -> None:
    (tmp_path / "estimator.py").write_text(_SELF_CONTAINED, encoding="utf-8")
    (tmp_path / "weights.npz").write_bytes(b"\x00" * 16)
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "table.npy").write_bytes(b"\x00" * 16)

    summary = summarize_submission(tmp_path / "estimator.py")

    assert summary.orphaned_siblings == ["assets/", "weights.npz"]


def test_file_mode_summary_omits_ignored_siblings(tmp_path: Path) -> None:
    # The whole point of naming orphans is that the list is worth reading. Listing
    # .venv/ and __pycache__/ would make it noise again (cf. whestbench#96).
    (tmp_path / "estimator.py").write_text(_SELF_CONTAINED, encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "estimator.cpython-312.pyc").write_bytes(b"\x00")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "thing.py").write_text("\n", encoding="utf-8")

    summary = summarize_submission(tmp_path / "estimator.py")

    assert summary.orphaned_siblings == []


def test_folder_mode_reports_no_orphaned_siblings(tmp_path: Path) -> None:
    (tmp_path / "estimator.py").write_text(_SELF_CONTAINED, encoding="utf-8")
    (tmp_path / "weights.npz").write_bytes(b"\x00" * 16)

    summary = summarize_submission(tmp_path)

    assert summary.orphaned_siblings == []


def test_file_mode_errors_when_entry_imports_a_sibling_module(tmp_path: Path) -> None:
    (tmp_path / "estimator.py").write_text(_IMPORTS_HELPER, encoding="utf-8")
    (tmp_path / "helper.py").write_text("def f():\n    return 0\n", encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        package_submission(tmp_path / "estimator.py", output_path=tmp_path.parent / "x.tar.gz")

    message = str(excinfo.value)
    assert "helper" in message
    # The message has to carry the fix, not just the diagnosis.
    assert "--estimator" in message


def test_file_mode_errors_when_entry_imports_a_sibling_package(tmp_path: Path) -> None:
    entry = (
        "import arc_tools\n"
        "from whestbench import BaseEstimator\n"
        "class Estimator(BaseEstimator):\n"
        "    def predict(self, mlp, budget):\n"
        "        import flopscope.numpy as fnp\n"
        "        return fnp.zeros((mlp.depth, mlp.width))\n"
    )
    (tmp_path / "estimator.py").write_text(entry, encoding="utf-8")
    pkg = tmp_path / "arc_tools"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("\n", encoding="utf-8")

    with pytest.raises(ValueError, match="arc_tools"):
        package_submission(tmp_path / "estimator.py", output_path=tmp_path.parent / "x.tar.gz")


def test_file_mode_allows_third_party_imports(tmp_path: Path) -> None:
    # `import flopscope` resolves from the grading environment, not from a sibling.
    # Only names that match a sibling file are provably missing.
    (tmp_path / "estimator.py").write_text(_SELF_CONTAINED, encoding="utf-8")
    out = tmp_path.parent / "ok.tar.gz"

    package_submission(tmp_path / "estimator.py", output_path=out)

    assert out.is_file()


def test_file_mode_data_siblings_do_not_block_packaging(tmp_path: Path) -> None:
    # A dropped assets/ dir is a warning, not an error: whether setup() reads it is
    # not statically decidable, and erroring here would train people to reach for an
    # opt-out flag on day one.
    (tmp_path / "estimator.py").write_text(_SELF_CONTAINED, encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "table.npy").write_bytes(b"\x00" * 16)
    out = tmp_path.parent / "ok.tar.gz"

    package_submission(tmp_path / "estimator.py", output_path=out)

    assert out.is_file()


def test_file_mode_ignores_an_unimported_sibling_module(tmp_path: Path) -> None:
    # helper.py exists but is never imported — dropping it is exactly the documented
    # behaviour, so it must not become an error.
    (tmp_path / "estimator.py").write_text(_SELF_CONTAINED, encoding="utf-8")
    (tmp_path / "helper.py").write_text("def f():\n    return 0\n", encoding="utf-8")
    out = tmp_path.parent / "ok.tar.gz"

    package_submission(tmp_path / "estimator.py", output_path=out)

    assert out.is_file()


def test_file_mode_summary_reports_missing_imports(tmp_path: Path) -> None:
    (tmp_path / "estimator.py").write_text(_IMPORTS_HELPER, encoding="utf-8")
    (tmp_path / "helper.py").write_text("def f():\n    return 0\n", encoding="utf-8")

    summary = summarize_submission(tmp_path / "estimator.py")

    assert summary.missing_imports == ["helper"]
