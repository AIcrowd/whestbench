#!/usr/bin/env bash
# Build the profiler Docker image and push to ECR.
# Must be run from the repository root (or the script cd's there).
#
# Reads ECR repo URI from .infra-config.json (created by setup_infra.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.infra-config.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: ${CONFIG_FILE} not found. Run setup_infra.sh first." >&2
    exit 1
fi

REGION=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['region'])")
ECR_REPO_URI=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['ecr_repo_uri'])")
ACCOUNT_ID=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['account_id'])")

IMAGE_TAG="latest"
FULL_URI="${ECR_REPO_URI}:${IMAGE_TAG}"

echo "=== Building nestim Profiler Image ==="
echo "Repo root:  ${REPO_ROOT}"
echo "ECR target: ${FULL_URI}"
echo ""

# Build from repo root so Dockerfile can access src/, pyproject.toml, etc.
echo "Building Docker image..."
docker build --platform linux/amd64 -f "${SCRIPT_DIR}/Dockerfile" -t "nestim-profiler:${IMAGE_TAG}" "${REPO_ROOT}"
echo ""

# Authenticate to ECR
echo "Authenticating to ECR..."
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
echo ""

# Tag and push
echo "Pushing image..."
docker tag "nestim-profiler:${IMAGE_TAG}" "$FULL_URI"
docker push "$FULL_URI"
echo ""

# Update config with image URI
python3 -c "
import json
with open('${CONFIG_FILE}') as f:
    config = json.load(f)
config['image_uri'] = '${FULL_URI}'
with open('${CONFIG_FILE}', 'w') as f:
    json.dump(config, f, indent=4)
"

echo "=== Build & Push Complete ==="
echo "Image URI: ${FULL_URI}"
echo "Config updated: ${CONFIG_FILE}"
