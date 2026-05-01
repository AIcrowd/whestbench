from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from textwrap import dedent
from typing import Dict

import pytest

from whestbench.packaging import package_submission


def _write_estimator_module(tmp_path: Path, class_name: str = "Estimator") -> Path:
    module_path = tmp_path / "estimator.py"
    module_path.write_text(
        dedent(
            f"""
            import numpy as np
            from numpy.typing import NDArray
            from whestbench import BaseEstimator, MLP

            class {class_name}(BaseEstimator):
                def predict(self, mlp: MLP, budget: int) -> NDArray[np.float32]:
                    return np.zeros((mlp.depth, mlp.width), dtype=np.float32)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return module_path


def _read_tar_json(archive_path: Path, member: str) -> Dict[str, object]:
    with tarfile.open(archive_path, "r:gz") as archive:
        file_obj = archive.extractfile(member)
        assert file_obj is not None
        return json.loads(file_obj.read().decode("utf-8"))


def test_package_includes_generated_manifest_json_and_estimator_file(tmp_path: Path) -> None:
    estimator = _write_estimator_module(tmp_path)
    artifact = package_submission(estimator, output_path=tmp_path / "submission.tar.gz")

    with tarfile.open(artifact, "r:gz") as archive:
        members = archive.getnames()
    assert "estimator.py" in members
    assert "manifest.json" in members


def test_manifest_records_resolved_entrypoint_and_sha256_hashes(tmp_path: Path) -> None:
    estimator = _write_estimator_module(tmp_path, class_name="MyEstimator")
    artifact = package_submission(
        estimator,
        class_name="MyEstimator",
        output_path=tmp_path / "submission.tar.gz",
    )
    manifest = _read_tar_json(artifact, "manifest.json")

    assert manifest["entrypoint"] == {"module": "estimator", "class": "MyEstimator"}
    files = manifest["files"]
    assert isinstance(files, list)
    estimator_entry = next(file for file in files if file["name"] == "estimator.py")
    expected_hash = hashlib.sha256(estimator.read_bytes()).hexdigest()
    assert estimator_entry["sha256"] == expected_hash


def test_optional_submission_yaml_and_approach_md_are_included_when_present(tmp_path: Path) -> None:
    estimator = _write_estimator_module(tmp_path)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy\\n", encoding="utf-8")
    submission_yaml = tmp_path / "submission.yaml"
    submission_yaml.write_text("title: test\\n", encoding="utf-8")
    approach_md = tmp_path / "APPROACH.md"
    approach_md.write_text("# Approach\\n", encoding="utf-8")

    artifact = package_submission(
        estimator,
        requirements_path=requirements,
        submission_yaml_path=submission_yaml,
        approach_md_path=approach_md,
        output_path=tmp_path / "submission.tar.gz",
    )

    with tarfile.open(artifact, "r:gz") as archive:
        members = set(archive.getnames())
    assert {"requirements.txt", "submission.yaml", "APPROACH.md"} <= members


def test_built_wheel_includes_estimator_template(tmp_path: Path) -> None:
    # Editable installs have estimator.py.tmpl on disk regardless of
    # package-data config, so they cannot catch the failure mode where
    # the template is missing from the shipped wheel. Build the wheel
    # from this checkout and assert the template lands inside it.
    uv = shutil.which("uv")
    if uv is None:
        pytest.skip("uv not available — required to build the wheel")
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [uv, "build", "--wheel", "--out-dir", str(tmp_path), str(repo_root)],
        check=True,
        capture_output=True,
    )
    wheels = list(tmp_path.glob("whestbench-*.whl"))
    assert wheels, f"no wheel produced under {tmp_path}"
    with zipfile.ZipFile(wheels[0]) as zf:
        names = zf.namelist()
    assert any(n.endswith("whestbench/templates/estimator.py.tmpl") for n in names), (
        "estimator.py.tmpl missing from wheel; contents:\n  " + "\n  ".join(sorted(names))
    )
