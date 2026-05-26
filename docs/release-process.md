# Release process

This document is the authoritative reference for cutting a new release
of `whestbench` to PyPI. It covers the steady-state flow, the one-time
setup that must happen outside the repo, and a few troubleshooting
notes.

## TL;DR (steady-state)

```bash
git checkout main && git pull origin main
uv run cz bump --dry-run                  # preview the next version + CHANGELOG entry
uv run cz bump                            # writes pyproject version + CHANGELOG.md + creates v<x.y.z> tag
git push --follow-tags                    # tag push triggers the publish workflow
# … open GitHub Actions → approve the `publish-pypi` job → wait ~30s →
# package on PyPI + GitHub Release created
```

Pre-release tags: `uv run cz bump --prerelease alpha` produces tags
like `v0.5.0a0`.

## What happens after `git push --follow-tags`

The tag push fires
[`.github/workflows/pypi-publish.yml`](../.github/workflows/pypi-publish.yml),
which:

1. Builds the sdist + wheel with `uv build`.
2. Pauses for approval in the `pypi` GitHub environment (manual gate).
3. Publishes to PyPI via Trusted Publishing (OIDC; no API token stored
   in repo secrets).
4. Creates a GitHub Release whose body is the matching CHANGELOG
   section for the tag.

End result: `uv add whestbench` / `pip install whestbench` works ~2
minutes after a maintainer clicks "approve" on the `publish-pypi` job.

## One-time setup (per maintainer, per repo)

Before the first release will succeed, two things must be configured
outside the repo.

### 1. PyPI Trusted Publisher

On [pypi.org](https://pypi.org), as an account with `Owner` or
`Maintainer` rights on the `whestbench` project (or as the user
creating it, if not yet published):

1. "Your projects" → `whestbench` → "Publishing" → "Add a pending
   publisher" (or "Add a publisher" if the project already exists).
2. Fill in:
   - PyPI project name: `whestbench`
   - Owner: `AIcrowd`
   - Repository name: `whestbench-public`
   - Workflow filename: `pypi-publish.yml`
   - Environment name: `pypi`

PyPI's "pending publisher" feature allows trusted publishing to
succeed on the very first publish of a brand-new project name.

### 2. GitHub `pypi` environment

In the whestbench-public repo on GitHub:

1. Settings → Environments → "New environment" → name: `pypi`.
2. Enable "Required reviewers".
3. Add yourself (and any other release maintainers) as reviewers.
4. Save.

Without this, publishes proceed without a human approval gate. The
Trusted Publishing OIDC handshake will still work — there is just no
gate to abort a bad tag.

## How CHANGELOG entries get into the GitHub Release

The publish workflow extracts the body of the matching `## v<version>`
section in `CHANGELOG.md` using an awk script and uses it as the
GitHub Release notes. Commitizen writes section headers in the
`## v<version> (<date>)` form, which the workflow expects.

When promoting an existing `## Unreleased` section to a versioned
release manually (rather than via `cz bump`), use the same header
format: `## v0.4.0 (2026-05-26)`.

If no matching section is found, the workflow falls back to a default
body: `Release v<x.y.z>\n\nSee CHANGELOG.md for details.`

## Troubleshooting

### Publish job fails with "Trusted publisher not configured"

PyPI side is not configured. Re-check step 1 of "One-time setup". The
workflow filename and environment name must match exactly
(`pypi-publish.yml`, `pypi`).

### Publish job fails with "File already exists on PyPI"

A version was previously uploaded and yanked. PyPI does not allow
re-uploading the same version, even after a yank. Resolution: delete
the tag locally and on the remote, bump to the next version, retag:

```bash
git tag -d v0.5.0
git push origin :refs/tags/v0.5.0
uv run cz bump   # bumps to v0.5.1
git push --follow-tags
```

### GitHub Release step fails after PyPI succeeded

The package is on PyPI; only the GitHub Release is missing. Re-run
the workflow on the same tag from the GitHub Actions UI. The
`github-release` job's `gh release create` is the only remaining side
effect and is idempotent against the existing tag (will fail if a
release already exists, succeed if not).

### `cz bump --dry-run` previews an unexpected version

The previewed version is computed from conventional-commits types in
the commit range since the last tag. `feat` → minor bump (under v1.x
behaviour: still minor while `major_version_zero = true` in
`[tool.commitizen]`), `fix` → patch, `feat!` or `BREAKING CHANGE` →
minor while `major_version_zero = true`, else major. To bump to a
specific version explicitly, use `cz bump --increment PATCH|MINOR|MAJOR`.

### Pin updates for flopscope

Whestbench pins `flopscope>=0.4.1` and `flopscope-server>=0.4.1`.
When flopscope ships a new minor or major version, bump these floors
in `pyproject.toml` and re-run `uv lock` before cutting the next
whestbench release. (Out of scope for an automated workflow; flag if
Dependabot becomes worth the noise.)
