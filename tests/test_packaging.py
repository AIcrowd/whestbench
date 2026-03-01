from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from textwrap import dedent

from circuit_estimation.packaging import package_submission


def _write_estimator_module(tmp_path: Path, class_name: str = "Estimator") -> Path:
    module_path = tmp_path / "estimator.py"
    module_path.write_text(
        dedent(
            f"""
            import numpy as np
            from circuit_estimation import BaseEstimator

            class {class_name}(BaseEstimator):
                def predict(self, circuit, budget: int):
                    return np.zeros((circuit.d, circuit.n), dtype=np.float32)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    return module_path


def _read_tar_json(archive_path: Path, member: str) -> dict[str, object]:
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
