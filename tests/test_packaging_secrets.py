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


def test_find_secret_files_skips_builtin_ignored_dirs(tmp_path: Path) -> None:
    # A virtualenv is in the built-in ignore set, so nothing under it can ever
    # ship. Reporting certifi's CA bundle (`cacert.pem` — not a secret at all) as
    # an excluded credential is misleading noise that trains participants to skim
    # past the security warnings that DO matter. See whestbench#96.
    _estimator(tmp_path)
    certifi = tmp_path / ".venv" / "lib" / "python3.10" / "site-packages" / "certifi"
    certifi.mkdir(parents=True)
    (certifi / "cacert.pem").write_text("-----BEGIN CERTIFICATE-----\n", encoding="utf-8")

    assert find_secret_files(tmp_path) == []
    assert {p.name for p in collect_submission_files(tmp_path)} == {"estimator.py"}


def test_find_secret_files_skips_pycache(tmp_path: Path) -> None:
    _estimator(tmp_path)
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "id_rsa").write_text("k\n", encoding="utf-8")

    assert find_secret_files(tmp_path) == []
    assert {p.name for p in collect_submission_files(tmp_path)} == {"estimator.py"}


def test_find_secret_files_still_reports_in_scope_secrets(tmp_path: Path) -> None:
    # Regression guard for the two tests above: filtering out ignored directories
    # must NOT stop us reporting a secret that really was in scope and really was
    # dropped — that warning is the whole point of the preview.
    _estimator(tmp_path)
    (tmp_path / ".env").write_text("AICROWD_API_KEY=secret\n", encoding="utf-8")
    nested = tmp_path / "config"
    nested.mkdir()
    (nested / "deploy.pem").write_text("-----BEGIN PRIVATE KEY-----\n", encoding="utf-8")
    venv = tmp_path / ".venv"
    venv.mkdir()
    (venv / "cacert.pem").write_text("cert\n", encoding="utf-8")

    rels = {str(p.relative_to(tmp_path)) for p in find_secret_files(tmp_path)}

    assert rels == {".env", "config/deploy.pem"}
    assert {p.name for p in collect_submission_files(tmp_path)} == {"estimator.py"}


def test_find_secret_files_honours_user_ignore_patterns(tmp_path: Path) -> None:
    # Anything the participant already excluded via .gitignore / .whestignore is
    # out of scope too, so it must not be reported as "excluded for security".
    _estimator(tmp_path)
    (tmp_path / ".gitignore").write_text("vendor/\n", encoding="utf-8")
    (tmp_path / ".whestignore").write_text("scratch/\n", encoding="utf-8")
    for name in ("vendor", "scratch"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "bundle.pem").write_text("cert\n", encoding="utf-8")
    (tmp_path / "server.key").write_text("k\n", encoding="utf-8")

    rels = {str(p.relative_to(tmp_path)) for p in find_secret_files(tmp_path)}

    assert rels == {"server.key"}
