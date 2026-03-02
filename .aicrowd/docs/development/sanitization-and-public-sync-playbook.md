# Sanitization and Public Sync Playbook

Last updated: 2026-03-03

## Purpose

This playbook documents:

1. How the sanitized public repo was created.
2. How to keep internal and public repos in sync going forward.
3. What failure modes we already hit and how to avoid repeating them.

Primary intent: allow development to continue in the internal repo while safely publishing to a sanitized public repo, including ingesting direct collaborator changes made on the public side.

## Repositories and Roles

Internal source-of-truth repo (active development):

- `origin`: `git@github.com:spMohanty/circuit-estimation-mvp.git`

Sanitized public repo (distribution/collaboration surface):

- `public`: `git@github.com:AIcrowd/circuit-estimation-challenge-internal.git`

Rule: do not assume commit hashes can match between internal and public. History has been rewritten/sanitized and will remain logically separate.

## Sensitive Scope

The following paths are treated as sensitive and must not exist in public history:

- `.aicrowd/`
- `.agent/`
- `.agents/`
- `.codex/`
- `CHALLENGE-CONTEXT.md`
- `docs/context/`
- `docs/plans/`
- `docs/development/worktrees-and-cli.md`
- `tools/circuit-explorer/STYLE_GUIDE.md`

Canonical manifests/scripts:

- `.aicrowd/sanitization/sensitive-paths.txt`
- `.aicrowd/sanitization/blob-replacements.txt`
- `.aicrowd/sanitization/message-replacements.txt`
- `.aicrowd/sanitization/public-sync-excludes.txt`
- `.aicrowd/scripts/sanitize-public-history.sh`
- `.aicrowd/scripts/public-sync.sh`

## What We Already Did (Historical Context)

The current public repo lineage was produced by sanitization/rewrite work (March 2, 2026).

Key facts:

1. We moved sensitive planning/context/internal workflow docs toward `.aicrowd` in internal workflow.
2. We performed sanitized-history publishing from a rewrite clone to public.
3. We force-pushed sanitized history to:
   - `https://github.com/AIcrowd/circuit-estimation-challenge-internal`
4. Current known sanitized public tip after force-push:
   - `5a4c2b6` (as of 2026-03-03 local time)

## Failure Modes We Hit and Lessons

1. Non-fast-forward push failure to public.
- Cause: internal and public histories diverged significantly.
- Fix: avoid direct push from internal branch history to public branch history.

2. Cherry-pick conflicts while trying to bridge histories.
- Cause: public branch had deleted files that internal commits modified (for example `.aicrowd/docs/plans/*`).
- Fix: avoid ad hoc cherry-picking between divergent roots unless explicitly handling conflicts with intent.

3. `git filter-repo` side effects.
- Observed: remotes/upstreams can be reset or become stale; force-with-lease can fail with stale ref info.
- Fix: always fetch and verify remote refs after rewrite; treat rewritten clone as isolated publish artifact.

4. Stale path references (`docs/plans`, `docs/context`) caused future-agent confusion.
- Fix: normalize references to `.aicrowd/...` in local agent instructions and enforce grep checks.

## Operating Model (Going Forward)

Internal repo remains source of truth for development.
Public repo is updated by sanitized mirror publish workflow.

Two explicit sync directions:

1. Public -> Internal (ingest collaborator updates).
2. Internal -> Public (sanitized publish).

Never skip ingest checks before publishing.

## Local Sync State

Local state file (ignored from git):

- `.aicrowd/sanitization/public-sync-state.local.env`

Template:

- `.aicrowd/sanitization/public-sync-state.example.env`

Important key:

- `LAST_INGESTED_PUBLIC_SHA`

Meaning: last public commit known to be ingested into internal workflow. Publish is blocked by default if public has commits newer than this marker.

## Command Reference

### 1) Check status

```bash
bash .aicrowd/scripts/public-sync.sh status
```

Shows:

- internal/public branch SHAs
- divergence
- ingestion marker state

### 2) Set or refresh ingestion marker

```bash
bash .aicrowd/scripts/public-sync.sh mark
```

Sets `LAST_INGESTED_PUBLIC_SHA` to current `public/main`.

### 3) Intake public collaborator commits into internal

Dry-run (list commits that need intake):

```bash
bash .aicrowd/scripts/public-sync.sh intake --dry-run
```

Create intake branch + cherry-pick:

```bash
bash .aicrowd/scripts/public-sync.sh intake --base-branch main --mark-ingested
```

Recommended flow:

1. Run intake on a dedicated sync branch.
2. Resolve conflicts if any.
3. Run tests.
4. Merge into internal `main`.

### 4) Publish internal changes to sanitized public repo

Dry-run:

```bash
bash .aicrowd/scripts/public-sync.sh publish --dry-run --allow-uningested
```

Real publish (explicit destination required):

```bash
bash .aicrowd/scripts/public-sync.sh publish \
  --source-branch main \
  --remote-url https://github.com/AIcrowd/circuit-estimation-challenge-internal.git \
  --message "feat: <describe actual visible change>"
```

Notes:

1. `publish` requires explicit `--remote-url` to prevent accidental pushes to the wrong repo.
2. By default, publish is blocked if marker is stale (public has un-ingested commits).
3. Publish copies internal tree into a temporary workspace, applies excludes and scrubs, verifies leaks, commits on public history, then pushes.

### 5) Commit message policy for publish commits

Use a descriptive commit message that reflects the actual change in the sanitized public tree for normal product/feature/fix updates.

Examples:

- `feat(cli): rename --agent-mode flag to --json`
- `docs: update CLI quickstart commands`
- `fix(scoring): guard depth-row shape validation`

Use a neutral sync/sanitization message only when the publish commit is primarily about sanitization/sync mechanics or could reveal sensitive internal workflow/context details via a descriptive title.

Always avoid sensitive terms/path references in public-facing commit messages (for example: `.aicrowd`, `.agent`, `sanitize`, `filter-repo`, or direct sensitive file paths).

## Standard Daily Workflow

### A) Start of sync cycle

```bash
bash .aicrowd/scripts/public-sync.sh status
```

If marker out-of-date:

```bash
bash .aicrowd/scripts/public-sync.sh intake --dry-run
bash .aicrowd/scripts/public-sync.sh intake --base-branch main --mark-ingested
```

Then test and merge intake branch into internal main.

### B) Publish cycle

After internal changes are merged and tested:

```bash
bash .aicrowd/scripts/public-sync.sh publish \
  --source-branch main \
  --remote-url https://github.com/AIcrowd/circuit-estimation-challenge-internal.git \
  --message "feat: <describe actual visible change>"
```

## Emergency Leak Response

Use full history rewrite only when required (for accidental sensitive history exposure).

Dry-run rewrite:

```bash
bash .aicrowd/scripts/sanitize-public-history.sh --keep-workdir
```

Publish rewritten history:

```bash
bash .aicrowd/scripts/sanitize-public-history.sh \
  --push \
  --target-remote https://github.com/AIcrowd/circuit-estimation-challenge-internal.git
```

After any forced rewritten push:

1. Notify collaborators that history changed.
2. Re-baseline local marker:

```bash
bash .aicrowd/scripts/public-sync.sh mark
```

3. Resume normal intake/publish workflow.

## Guardrails for Future Agents

1. Always ask the human to confirm target push destination before a real push.
2. Prefer `public-sync.sh publish` for normal sync; avoid manual cross-history cherry-picks when possible.
3. Do not push internal branch directly to public remote.
4. Keep sensitive manifests in sync when adding new internal workflow surfaces.
5. Run status and leak checks before and after publish.
6. Default rule: use descriptive commit messages for normal feature/fix/docs updates published to public.
7. Exception: use neutral commit messages only for sync/sanitization-adjacent changes that might expose sensitive internal workflow/context details.
8. In all cases, avoid explicit sensitive terms/path references like `sanitize`, `filter-repo`, `.aicrowd`, `.agent`, or direct sensitive path names in public-facing history.

## Validation Checklist

Before publish:

1. `public-sync.sh status` marker is current (or intentionally overridden).
2. Internal branch is tested.
3. Destination URL is explicitly confirmed.

After publish:

1. Confirm pushed public SHA.
2. Re-run `public-sync.sh status`.
3. If needed, `git fetch public` and inspect latest public commit/log.

## If Something Looks Wrong

1. Stop and do not keep pushing.
2. Run:

```bash
git fetch origin public --prune
bash .aicrowd/scripts/public-sync.sh status
```

3. If divergence is unexpected, branch off and inspect logs/diffs before any destructive action.
4. If leak suspected, run full sanitization dry-run and inspect report.
