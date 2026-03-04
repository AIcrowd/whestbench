---
description: Sanitize and sync internal repo to the public repo (intake collaborator changes, publish sanitized updates)
---

> [!IMPORTANT]
> This workflow mirrors `.aicrowd/docs/development/sanitization-and-public-sync-playbook.md`.
> Any updates here must be replicated there, and vice-versa.

# Sanitization and Public Sync

Keeps the internal repo (`origin`) and sanitized public repo (`public`) in sync.
Two directions: **Intake** (public → internal) and **Publish** (internal → public).

> [!CAUTION]
> Never push the internal branch directly to the public remote.
> Always use `public-sync.sh publish` for normal sync.
> Always ask the human to confirm the target push destination before a real push.

## Repositories

- **Internal (origin):** `git@github.com:spMohanty/circuit-estimation-mvp.git`
- **Public:** `git@github.com:AIcrowd/circuit-estimation-challenge-internal.git`
  - Canonical URL: `https://github.com/AIcrowd/circuit-estimation-challenge-internal.git`

## Sensitive paths (must never appear in public history)

See `.aicrowd/sanitization/sensitive-paths.txt` for the full list. Key entries:
`.aicrowd/`, `.agent/`, `.agents/`, `.codex/`, `CHALLENGE-CONTEXT.md`, `docs/context/`, `docs/plans/`, `tools/circuit-explorer/STYLE_GUIDE.md`

---

## 0) Push local state to origin

Before any sync work, ensure all local commits are pushed to the internal remote.

```bash
git push origin main
```

---

## A) Intake Cycle (Public → Internal)

### 1. Check sync status
// turbo
```bash
bash .aicrowd/scripts/public-sync.sh status
```

### 2. Dry-run intake (list new public commits)
// turbo
```bash
bash .aicrowd/scripts/public-sync.sh intake --dry-run
```

### 3. If there are commits to ingest, run intake
```bash
bash .aicrowd/scripts/public-sync.sh intake --base-branch main --mark-ingested
```

### 4. Resolve any conflicts, run tests, then merge the intake branch into internal `main`.

---

## B) Publish Cycle (Internal → Public)

### 5. Dry-run publish (verify what will be pushed)
// turbo
```bash
bash .aicrowd/scripts/public-sync.sh publish --dry-run --allow-uningested
```

### 6. Publish (requires explicit remote URL and human confirmation)
```bash
bash .aicrowd/scripts/public-sync.sh publish \
  --source-branch main \
  --remote-url https://github.com/AIcrowd/circuit-estimation-challenge-internal.git \
  --message "feat: <describe actual visible change>"
```

> [!IMPORTANT]
> **Commit message policy:**
> - Use descriptive messages for normal feature/fix/docs updates (e.g. `feat(cli): rename --agent-mode flag to --json`).
> - Use neutral messages only for sync/sanitization-adjacent changes.
> - Never use sensitive terms like `sanitize`, `filter-repo`, `.aicrowd`, `.agent`, or sensitive file paths in public commit messages.

### 7. Post-publish validation
// turbo
```bash
bash .aicrowd/scripts/public-sync.sh status
```

Confirm pushed public SHA and inspect the latest public commit log if needed.

---

## C) Emergency Leak Response

Only use when sensitive data was accidentally exposed in public history.

### 8. Dry-run history rewrite
```bash
bash .aicrowd/scripts/sanitize-public-history.sh --keep-workdir
```

### 9. Push rewritten history (requires human confirmation)
```bash
bash .aicrowd/scripts/sanitize-public-history.sh \
  --push \
  --target-remote https://github.com/AIcrowd/circuit-estimation-challenge-internal.git
```

### 10. After force-push: notify collaborators, then re-baseline marker
```bash
bash .aicrowd/scripts/public-sync.sh mark
```

Resume normal intake/publish workflow.

---

## If Something Looks Wrong

1. **Stop.** Do not keep pushing.
2. Fetch and check status:
```bash
git fetch origin public --prune
bash .aicrowd/scripts/public-sync.sh status
```
3. If divergence is unexpected, branch off and inspect logs/diffs before any destructive action.
4. If a leak is suspected, run the full sanitization dry-run and inspect the report.
