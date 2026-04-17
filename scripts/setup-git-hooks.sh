#!/usr/bin/env bash
# Configure local git hook paths for every registered worktree.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
HOOKS_DIR=".githooks"

if [ ! -d "$REPO_ROOT/$HOOKS_DIR" ]; then
  echo "[setup-git-hooks] Missing $HOOKS_DIR in repository root: $REPO_ROOT" >&2
  exit 1
fi

while IFS= read -r worktree; do
  if [ -d "$worktree" ]; then
    git -C "$worktree" config --local core.hooksPath "$HOOKS_DIR"
  fi
done < <(git -C "$REPO_ROOT" worktree list --porcelain | awk '/^worktree / { print $2 }')

echo "[setup-git-hooks] Configured core.hooksPath=.githooks for all tracked worktrees."
