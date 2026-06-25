"""`whest validate-package <tarball>` exits 0 for a valid archive, non-zero otherwise."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import whestbench.cli as cli
from whestbench.packaging import package_submission

_EST = (
    "from whestbench import BaseEstimator\n"
    "class Estimator(BaseEstimator):\n"
    "    def predict(self, mlp, budget):\n"
    "        import flopscope.numpy as fnp\n"
    "        return fnp.zeros((mlp.depth, mlp.width))\n"
)


def test_validate_package_ok(tmp_path: Path) -> None:
    (tmp_path / "estimator.py").write_text(_EST, encoding="utf-8")
    out = tmp_path / "submission.tar.gz"
    package_submission(tmp_path / "estimator.py", output_path=out)
    assert cli.main(["validate-package", str(out)]) == 0


def test_validate_package_directory_entry_fails(tmp_path: Path, capsys) -> None:
    body = _EST.encode("utf-8")
    import hashlib

    manifest = {
        "schema_version": "1.0",
        "api_version": "2.0",
        "entrypoint": {"module": "estimator", "class": "Estimator"},
        "files": [
            {"name": "estimator.py", "sha256": hashlib.sha256(body).hexdigest()},
            {"name": "arc_tools/", "sha256": "b8c5ca68"},
        ],
    }
    p = tmp_path / "bad.tar.gz"
    with tarfile.open(p, "w:gz") as tf:
        info = tarfile.TarInfo("estimator.py")
        info.size = len(body)
        tf.addfile(info, io.BytesIO(body))
        blob = json.dumps(manifest, indent=2).encode("utf-8")
        mi = tarfile.TarInfo("manifest.json")
        mi.size = len(blob)
        tf.addfile(mi, io.BytesIO(blob))

    rc = cli.main(["validate-package", str(p), "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert any(i["code"] == "directory_entry" for i in payload["issues"])
