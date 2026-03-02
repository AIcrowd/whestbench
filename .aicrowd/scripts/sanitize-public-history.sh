#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(git rev-parse --show-toplevel)"
MANIFEST_DEFAULT="$ROOT_DIR/.aicrowd/sanitization/sensitive-paths.txt"
MESSAGE_REPLACEMENTS_DEFAULT="$ROOT_DIR/.aicrowd/sanitization/message-replacements.txt"
BLOB_REPLACEMENTS_DEFAULT="$ROOT_DIR/.aicrowd/sanitization/blob-replacements.txt"

SOURCE_BRANCH="main"
TARGET_BRANCH="main"
TARGET_REMOTE=""
WORKDIR=""
PUSH=0
NO_MESSAGE_SCRUB=0
KEEP_WORKDIR=0

usage() {
  cat <<'EOF'
Usage:
  .aicrowd/scripts/sanitize-public-history.sh [options]

Options:
  --source-branch <name>     Branch to sanitize (default: main)
  --target-remote <url>      Push URL for sanitized output (optional)
  --target-branch <name>     Target branch when pushing (default: main)
  --workdir <path>           Use a specific temporary work directory
  --push                     Push sanitized branch after verification
  --no-message-scrub         Skip commit message replacement pass
  --keep-workdir             Keep temp directory after completion
  -h, --help                 Show help

Examples:
  .aicrowd/scripts/sanitize-public-history.sh
  .aicrowd/scripts/sanitize-public-history.sh \
    --target-remote git@github.com:AIcrowd/circuit-estimation-challenge-internal.git \
    --push
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-branch)
      SOURCE_BRANCH="$2"
      shift 2
      ;;
    --target-remote)
      TARGET_REMOTE="$2"
      shift 2
      ;;
    --target-branch)
      TARGET_BRANCH="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --push)
      PUSH=1
      shift
      ;;
    --no-message-scrub)
      NO_MESSAGE_SCRUB=1
      shift
      ;;
    --keep-workdir)
      KEEP_WORKDIR=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$MANIFEST_DEFAULT" ]]; then
  echo "Missing manifest: $MANIFEST_DEFAULT" >&2
  exit 1
fi

if ! command -v git-filter-repo >/dev/null 2>&1; then
  echo "git-filter-repo is required but not installed." >&2
  exit 1
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Working tree is not clean. Commit/stash first for reproducible sanitization." >&2
  exit 1
fi

if ! git show-ref --verify --quiet "refs/heads/$SOURCE_BRANCH"; then
  echo "Source branch does not exist: $SOURCE_BRANCH" >&2
  exit 1
fi

if [[ "$PUSH" -eq 1 && -z "$TARGET_REMOTE" ]]; then
  echo "--push requires --target-remote" >&2
  exit 1
fi

if [[ -z "$WORKDIR" ]]; then
  WORKDIR="$(mktemp -d /tmp/cestim-sanitize-XXXXXX)"
else
  mkdir -p "$WORKDIR"
fi

if [[ "$KEEP_WORKDIR" -ne 1 ]]; then
  trap 'rm -rf "$WORKDIR"' EXIT
fi

WORK_REPO="$WORKDIR/rewrite"

echo "[1/6] Cloning $SOURCE_BRANCH into temporary workspace"
git clone --no-local --single-branch --branch "$SOURCE_BRANCH" "$ROOT_DIR" "$WORK_REPO" >/dev/null

cd "$WORK_REPO"

FILTER_ARGS=(--force --invert-paths)
while IFS= read -r line; do
  path="$(printf '%s' "$line" | sed -E 's/[[:space:]]*#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//')"
  [[ -z "$path" ]] && continue
  FILTER_ARGS+=(--path "$path")
done < "$MANIFEST_DEFAULT"

if [[ "$NO_MESSAGE_SCRUB" -ne 1 && -f "$MESSAGE_REPLACEMENTS_DEFAULT" ]]; then
  FILTER_ARGS+=(--replace-message "$MESSAGE_REPLACEMENTS_DEFAULT")
fi
if [[ -f "$BLOB_REPLACEMENTS_DEFAULT" ]]; then
  FILTER_ARGS+=(--replace-text "$BLOB_REPLACEMENTS_DEFAULT")
fi

echo "[2/6] Rewriting history with git-filter-repo"
git filter-repo "${FILTER_ARGS[@]}" >/dev/null

echo "[3/6] Verifying banned paths are absent from rewritten history"
banned_parts=()
while IFS= read -r line; do
  path="$(printf '%s' "$line" | sed -E 's/[[:space:]]*#.*$//; s/^[[:space:]]+//; s/[[:space:]]+$//')"
  [[ -z "$path" ]] && continue
  escaped="$(printf '%s' "$path" | sed -E 's/[][(){}.^$+*?|\\]/\\&/g')"
  if [[ "$path" == */ ]]; then
    banned_parts+=("^${escaped}")
  else
    banned_parts+=("^${escaped}$")
  fi
done < "$MANIFEST_DEFAULT"
banned_regex="$(IFS='|'; echo "${banned_parts[*]}")"

if git rev-list --objects --all | cut -d' ' -f2- | rg -n -e "$banned_regex" >/tmp/cestim-sanitize-path-leaks.txt; then
  echo "Banned path leak detected in rewritten history:" >&2
  cat /tmp/cestim-sanitize-path-leaks.txt >&2
  exit 1
fi
rm -f /tmp/cestim-sanitize-path-leaks.txt

echo "[4/6] Verifying commit messages do not contain sensitive path references"
message_regex='(\.aicrowd/|\.agent/|\.agents/|\.codex/|docs/plans|docs/context|CHALLENGE-CONTEXT\.md|worktrees-and-cli\.md|STYLE_GUIDE\.md)'
if git log --all --format='%H%x09%s%n%b' | rg -n -i -e "$message_regex" >/tmp/cestim-sanitize-message-leaks.txt; then
  echo "Sensitive message reference detected after rewrite:" >&2
  cat /tmp/cestim-sanitize-message-leaks.txt >&2
  exit 1
fi
rm -f /tmp/cestim-sanitize-message-leaks.txt

echo "[5/6] Verifying content has no sensitive location references across rewritten history"
content_regex='(\.aicrowd/|\.agent/|\.agents/|\.codex/|docs/plans|docs/context|CHALLENGE-CONTEXT\.md|worktrees-and-cli\.md|STYLE_GUIDE\.md)'
all_revs="$(git rev-list --all)"
# Intentional word splitting: git grep expects revisions as separate args.
# shellcheck disable=SC2086
if [[ -n "$all_revs" ]] && git grep -n -I -E "$content_regex" $all_revs >/tmp/cestim-sanitize-content-leaks.txt; then
  echo "Sensitive content reference detected in sanitized history:" >&2
  cat /tmp/cestim-sanitize-content-leaks.txt >&2
  exit 1
fi
rm -f /tmp/cestim-sanitize-content-leaks.txt

echo "[6/6] Completed sanitized rewrite"
echo "Sanitized repository: $WORK_REPO"
echo "Sanitized HEAD: $(git rev-parse --short HEAD)"
echo "Sanitized commit count: $(git rev-list --count HEAD)"

if [[ "$PUSH" -eq 1 ]]; then
  echo "Pushing sanitized branch to: $TARGET_REMOTE ($TARGET_BRANCH)"
  if git remote get-url publish >/dev/null 2>&1; then
    git remote set-url publish "$TARGET_REMOTE"
  else
    git remote add publish "$TARGET_REMOTE"
  fi
  git push publish "HEAD:$TARGET_BRANCH" --force
  echo "Push complete."
else
  cat <<EOF

Dry-run only. To publish:
  cd "$WORK_REPO"
  git remote add publish <target-remote-url>
  git push publish HEAD:$TARGET_BRANCH --force
EOF
fi

if [[ "$KEEP_WORKDIR" -eq 1 ]]; then
  echo "Kept workspace at: $WORKDIR"
fi
