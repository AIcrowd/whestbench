#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"

run_quick() {
  echo "[harness] Running quick suite (non-exhaustive)..."
  uv run --group dev pytest -m "not exhaustive"
}

run_exhaustive() {
  echo "[harness] Running exhaustive suite..."
  uv run --group dev pytest -m "exhaustive"
}

case "$MODE" in
  quick)
    run_quick
    ;;
  full)
    run_quick
    run_exhaustive
    ;;
  exhaustive)
    run_exhaustive
    ;;
  *)
    echo "Usage: scripts/run-test-harness.sh [quick|full|exhaustive]" >&2
    exit 1
    ;;
esac
