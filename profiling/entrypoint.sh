#!/usr/bin/env bash
# Container entrypoint for cloud profiling tasks.
# Runs nestim profiler and uploads results to S3.
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

# --- Pin thread counts to available CPUs ---
# Fargate exposes the correct CPU count via nproc.
# Set all threading env vars explicitly to avoid backend auto-detection issues.
CPUS=$(nproc 2>/dev/null || echo 1)
if [[ -z "${MAX_THREADS:-}" ]]; then
    MAX_THREADS="$CPUS"
fi
export OMP_NUM_THREADS="$MAX_THREADS"
export MKL_NUM_THREADS="$MAX_THREADS"
export OPENBLAS_NUM_THREADS="$MAX_THREADS"
export NUMBA_NUM_THREADS="$MAX_THREADS"
export NUMEXPR_NUM_THREADS="$MAX_THREADS"
export VECLIB_MAXIMUM_THREADS="$MAX_THREADS"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_intra_op_parallelism_threads=$MAX_THREADS"

# Build the nestim command
CMD=(nestim profile-simulation --preset "$PRESET" --output "$OUTPUT_FILE" --log-progress)

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
echo "Max threads: $MAX_THREADS"
echo "CPUs avail:  $CPUS"
echo "Timeout:     ${TIMEOUT_MINUTES}m"
echo "==========================="
echo ""

# Run the profiler with a timeout
TIMEOUT_SECS=$((TIMEOUT_MINUTES * 60))
if ! timeout "$TIMEOUT_SECS" "${CMD[@]}"; then
    exit_code=$?
    if [[ $exit_code -eq 124 ]]; then
        echo "ERROR: Profiler timed out after ${TIMEOUT_MINUTES} minutes" >&2
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
