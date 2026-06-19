"""Credential/secret files must never be bundled into a submission."""

from __future__ import annotations

from pathlib import Path

from whestbench.packaging import collect_submission_files, find_secret_files


def _estimator(tmp_path: Path) -> None:
    (tmp_path / "estimator.py").write_text("x = 1\n", encoding="utf-8")


def test_collect_excludes_credential_files(tmp_path: Path) -> None:
    _estimator(tmp_path)
    (tmp_path / ".env").write_text("AICROWD_API_KEY=secret\n", encoding="utf-8")
    (tmp_path / "deploy.pem").write_text("-----BEGIN PRIVATE KEY-----\n", encoding="utf-8")
    (tmp_path / "server.key").write_text("k\n", encoding="utf-8")
    (tmp_path / "id_rsa").write_text("k\n", encoding="utf-8")

    names = {p.name for p in collect_submission_files(tmp_path)}

    assert names == {"estimator.py"}


def test_collect_excludes_dotenv_variants_and_secret_dirs(tmp_path: Path) -> None:
    _estimator(tmp_path)
    (tmp_path / ".env.local").write_text("x\n", encoding="utf-8")
    aws = tmp_path / ".aws"
    aws.mkdir()
    (aws / "credentials").write_text("x\n", encoding="utf-8")

    rels = {str(p.relative_to(tmp_path)) for p in collect_submission_files(tmp_path)}

    assert rels == {"estimator.py"}


def test_find_secret_files_reports_excluded_credentials(tmp_path: Path) -> None:
    _estimator(tmp_path)
    (tmp_path / ".env").write_text("AICROWD_API_KEY=secret\n", encoding="utf-8")
    (tmp_path / "key.pem").write_text("x\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")

    secret_names = {p.name for p in find_secret_files(tmp_path)}

    assert secret_names == {".env", "key.pem"}


def test_find_secret_files_empty_when_clean(tmp_path: Path) -> None:
    _estimator(tmp_path)
    (tmp_path / "requirements.txt").write_text("numpy\n", encoding="utf-8")

    assert find_secret_files(tmp_path) == []


def test_collect_excludes_credentials_case_insensitively(tmp_path: Path) -> None:
    # Uppercase / mixed-case credential files (Windows-origin exports, renames)
    # must be excluded too — fnmatch is case-sensitive on POSIX by default.
    _estimator(tmp_path)
    (tmp_path / "deploy.PEM").write_text("-----BEGIN PRIVATE KEY-----\n", encoding="utf-8")
    (tmp_path / "Server.KEY").write_text("k\n", encoding="utf-8")
    (tmp_path / "ID_RSA").write_text("k\n", encoding="utf-8")

    names = {p.name for p in collect_submission_files(tmp_path)}

    assert names == {"estimator.py"}
    assert {p.name for p in find_secret_files(tmp_path)} == {"deploy.PEM", "Server.KEY", "ID_RSA"}


def test_collect_excludes_bare_credentials_file(tmp_path: Path) -> None:
    # `credentials` (no extension) is the canonical AWS / gcloud credential filename.
    _estimator(tmp_path)
    (tmp_path / "credentials").write_text("[default]\naws_secret_access_key=x\n", encoding="utf-8")

    names = {p.name for p in collect_submission_files(tmp_path)}

    assert "credentials" not in names
    assert {p.name for p in find_secret_files(tmp_path)} == {"credentials"}


def test_collect_skips_symlinks(tmp_path: Path) -> None:
    # Symlinks bake an absolute, out-of-root target into the tar and make the
    # manifest hash read through the link — skip them entirely.
    _estimator(tmp_path)
    outside = tmp_path.parent / "outside_secret.txt"
    outside.write_text("host-only data\n", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)

    names = {p.name for p in collect_submission_files(tmp_path)}

    assert names == {"estimator.py"}
