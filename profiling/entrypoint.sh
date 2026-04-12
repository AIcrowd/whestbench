#!/usr/bin/env bash
# Container entrypoint for cloud profiling tasks.
# Runs whest profiler and uploads results to S3.
#
# Required env vars: RUN_ID, CONFIG_NAME, S3_BUCKET
# Optional env vars: PRESET, BACKENDS, MAX_THREADS, VERBOSE, TIMEOUT_MINUTES
set -euo pipefail

: "${RUN_ID:?RUN_ID is required}"
: "${CONFIG_NAME:?CONFIG_NAME is required}"
: "${S3_BUCKET:?S3_BUCKET is required}"

PRESET="${PRESET:-exhaustive}"
TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-45}"
OUTPUT_FILE="/tmp/results.json"
S3_KEY="s3://${S3_BUCKET}/${RUN_ID}/${CONFIG_NAME}.json"

# --- Pin thread counts ---
# nproc inside Fargate often reports 1 regardless of allocation.
# The orchestrator passes MAX_THREADS derived from Fargate CPU units.
# Fall back to nproc only if MAX_THREADS is not set.
CPUS_DETECTED=$(nproc 2>/dev/null || echo 1)
if [[ -n "${MAX_THREADS:-}" ]]; then
    THREADS_SOURCE="env"
else
    MAX_THREADS="$CPUS_DETECTED"
    THREADS_SOURCE="nproc"
fi
export OMP_NUM_THREADS="$MAX_THREADS"
export MKL_NUM_THREADS="$MAX_THREADS"
export OPENBLAS_NUM_THREADS="$MAX_THREADS"
export NUMBA_NUM_THREADS="$MAX_THREADS"
export NUMEXPR_NUM_THREADS="$MAX_THREADS"
export VECLIB_MAXIMUM_THREADS="$MAX_THREADS"
# Note: XLA_FLAGS thread options removed — they cause fatal crashes in newer JAX.
# Thread counts are controlled via OMP/MKL/OPENBLAS env vars above.

# Build the whest command
CMD=(whest profile-simulation --preset "$PRESET" --output "$OUTPUT_FILE" --log-progress)

if [[ -n "${BACKENDS:-}" ]]; then
    CMD+=(--backends "$BACKENDS")
fi

CMD+=(--max-threads "$MAX_THREADS")

if [[ "${VERBOSE:-}" == "1" ]]; then
    CMD+=(--verbose)
fi

echo "=== Cloud Profiling Task ==="
echo "Run ID:      $RUN_ID"
echo "Config:      $CONFIG_NAME"
echo "Preset:      $PRESET"
echo "S3 target:   $S3_KEY"
echo "Backends:    ${BACKENDS:-all}"
echo "Threads:     $MAX_THREADS (source: $THREADS_SOURCE, nproc=$CPUS_DETECTED)"
echo "Timeout:     ${TIMEOUT_MINUTES}m"
echo "==========================="
echo ""

# Run the profiler with a timeout
TIMEOUT_SECS=$((TIMEOUT_MINUTES * 60))
timeout "$TIMEOUT_SECS" "${CMD[@]}" && exit_code=0 || exit_code=$?
if [[ $exit_code -ne 0 ]]; then
    if [[ $exit_code -eq 124 ]]; then
        echo "ERROR: Profiler timed out after ${TIMEOUT_MINUTES} minutes" >&2
    elif [[ $exit_code -eq 137 ]]; then
        echo "ERROR: Profiler was killed (OOM or SIGKILL), exit code $exit_code" >&2
    else
        echo "ERROR: Profiler exited with code $exit_code" >&2
    fi
    exit $exit_code
fi

# Upload with retry (3 attempts, exponential backoff)
for attempt in 1 2 3; do
    if aws s3 cp "$OUTPUT_FILE" "$S3_KEY"; then
        echo ""
        echo "Results uploaded to $S3_KEY"
        exit 0
    fi
    if [[ $attempt -lt 3 ]]; then
        sleep_time=$((2 ** attempt))
        echo "S3 upload attempt $attempt failed, retrying in ${sleep_time}s..."
        sleep "$sleep_time"
    fi
done

echo "ERROR: Failed to upload results after 3 attempts" >&2
exit 1
