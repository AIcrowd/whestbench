#!/usr/bin/env bash
# Container entrypoint for cloud profiling tasks.
# Runs nestim profiler and uploads results to S3.
#
# Required env vars: RUN_ID, CONFIG_NAME, S3_BUCKET
# Optional env vars: PRESET (default: exhaustive), BACKENDS, MAX_THREADS, VERBOSE
set -euo pipefail

: "${RUN_ID:?RUN_ID is required}"
: "${CONFIG_NAME:?CONFIG_NAME is required}"
: "${S3_BUCKET:?S3_BUCKET is required}"

PRESET="${PRESET:-exhaustive}"
OUTPUT_FILE="/tmp/results.json"
S3_KEY="s3://${S3_BUCKET}/${RUN_ID}/${CONFIG_NAME}.json"

# Build the nestim command
CMD=(nestim profile-simulation --preset "$PRESET" --output "$OUTPUT_FILE")

if [[ -n "${BACKENDS:-}" ]]; then
    CMD+=(--backends "$BACKENDS")
fi

if [[ -n "${MAX_THREADS:-}" ]]; then
    CMD+=(--max-threads "$MAX_THREADS")
fi

if [[ "${VERBOSE:-}" == "1" ]]; then
    CMD+=(--verbose)
fi

echo "=== Cloud Profiling Task ==="
echo "Run ID:      $RUN_ID"
echo "Config:      $CONFIG_NAME"
echo "Preset:      $PRESET"
echo "S3 target:   $S3_KEY"
echo "Backends:    ${BACKENDS:-all}"
echo "Max threads: ${MAX_THREADS:-unlimited}"
echo "==========================="
echo ""

# Run the profiler
"${CMD[@]}"

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
