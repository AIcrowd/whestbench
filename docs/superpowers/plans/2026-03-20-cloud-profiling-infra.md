# Cloud Profiling Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `nestim profile-simulation` across a matrix of AWS Fargate CPU/memory configs in parallel, collect results to S3, and aggregate locally.

**Architecture:** A self-contained `profiling/` directory with shell scripts for AWS infra lifecycle, a Docker image pre-baked with all 6 CPU backends, a Python orchestrator that launches/monitors parallel Fargate tasks, and a collector that pulls results from S3 into aggregated JSON + CSV.

**Tech Stack:** Python 3.10+, AWS CLI, Docker, ECS Fargate, S3, ECR, CloudWatch

**Spec:** `docs/superpowers/specs/2026-03-20-cloud-profiling-infra-design.md`

---

## Chunk 1: Foundation (directory, matrix, entrypoint, Dockerfile)

### Task 1: Directory scaffolding and gitignore

**Files:**
- Create: `profiling/` (directory)
- Create: `profiling/results/.gitkeep`
- Modify: `.gitignore` (append at end)

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p profiling/results
touch profiling/results/.gitkeep
```

- [ ] **Step 2: Add gitignore entries**

Append to `.gitignore`:
```
# Cloud profiling infrastructure
profiling/results/*
!profiling/results/.gitkeep
profiling/.infra-config.json
```

- [ ] **Step 3: Commit**

```bash
git add profiling/results/.gitkeep .gitignore
git commit -m "chore: scaffold profiling directory and gitignore entries"
```

---

### Task 2: Instance matrix

**Files:**
- Create: `profiling/instance_matrix.py`
- Create: `profiling/tests/test_instance_matrix.py`

- [ ] **Step 1: Write the failing test**

Create `profiling/tests/__init__.py` (empty) and `profiling/tests/test_instance_matrix.py`:

```python
"""Tests for the Fargate instance matrix."""

from profiling.instance_matrix import INSTANCE_MATRIX, get_configs


def test_matrix_has_compute_and_general_configs():
    names = [c["name"] for c in INSTANCE_MATRIX]
    compute = [n for n in names if n.startswith("compute-")]
    general = [n for n in names if n.startswith("general-")]
    assert len(compute) >= 3
    assert len(general) >= 3


def test_all_configs_have_required_keys():
    required = {"name", "cpu", "memory", "label"}
    for config in INSTANCE_MATRIX:
        assert required.issubset(config.keys()), f"Missing keys in {config['name']}"


def test_cpu_memory_are_valid_fargate_combos():
    """Fargate valid CPU values: 256, 512, 1024, 2048, 4096, 8192, 16384."""
    valid_cpu = {256, 512, 1024, 2048, 4096, 8192, 16384}
    for config in INSTANCE_MATRIX:
        assert config["cpu"] in valid_cpu, f"Invalid CPU {config['cpu']} for {config['name']}"
        assert config["memory"] >= config["cpu"], (
            f"Memory must be >= CPU for {config['name']}"
        )


def test_get_configs_returns_all_by_default():
    configs = get_configs()
    assert len(configs) == len(INSTANCE_MATRIX)


def test_get_configs_filters_by_name():
    configs = get_configs(names=["compute-small", "general-small"])
    assert len(configs) == 2
    assert {c["name"] for c in configs} == {"compute-small", "general-small"}


def test_get_configs_raises_on_unknown_name():
    try:
        get_configs(names=["nonexistent"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "nonexistent" in str(e)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/profile-debug && python -m pytest profiling/tests/test_instance_matrix.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write the implementation**

Create `profiling/__init__.py` (empty) and `profiling/instance_matrix.py`:

```python
"""Fargate task configuration matrix for cloud profiling.

Defines CPU/memory combinations that simulate compute-optimized (c-series)
and general-purpose (m-series) EC2 instance characteristics on Fargate.

To customize, add or remove entries from INSTANCE_MATRIX. Each entry needs:
- name: unique identifier (used as S3 key prefix and in reports)
- cpu: Fargate CPU units (1024 = 1 vCPU)
- memory: Fargate memory in MiB
- label: human-readable description
"""

from typing import Dict, List, Optional, Any

Config = Dict[str, Any]

INSTANCE_MATRIX: List[Config] = [
    # Compute-optimized (c-series equivalents) — low memory-to-CPU ratio
    {"name": "compute-small",   "cpu": 1024,  "memory": 2048,  "label": "1 vCPU / 2 GB"},
    {"name": "compute-medium",  "cpu": 2048,  "memory": 4096,  "label": "2 vCPU / 4 GB"},
    {"name": "compute-large",   "cpu": 4096,  "memory": 8192,  "label": "4 vCPU / 8 GB"},
    {"name": "compute-xlarge",  "cpu": 8192,  "memory": 16384, "label": "8 vCPU / 16 GB"},
    {"name": "compute-2xlarge", "cpu": 16384, "memory": 32768, "label": "16 vCPU / 32 GB"},
    # General-purpose (m-series equivalents) — higher memory-to-CPU ratio
    {"name": "general-small",   "cpu": 1024,  "memory": 4096,  "label": "1 vCPU / 4 GB"},
    {"name": "general-medium",  "cpu": 2048,  "memory": 8192,  "label": "2 vCPU / 8 GB"},
    {"name": "general-large",   "cpu": 4096,  "memory": 16384, "label": "4 vCPU / 16 GB"},
    {"name": "general-xlarge",  "cpu": 8192,  "memory": 32768, "label": "8 vCPU / 32 GB"},
]


def get_configs(names: Optional[List[str]] = None) -> List[Config]:
    """Return configs filtered by name. Returns all if names is None."""
    if names is None:
        return list(INSTANCE_MATRIX)
    known = {c["name"] for c in INSTANCE_MATRIX}
    unknown = set(names) - known
    if unknown:
        raise ValueError(
            f"Unknown config(s): {', '.join(sorted(unknown))}. "
            f"Available: {', '.join(sorted(known))}"
        )
    return [c for c in INSTANCE_MATRIX if c["name"] in names]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/profile-debug && python -m pytest profiling/tests/test_instance_matrix.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add profiling/__init__.py profiling/instance_matrix.py profiling/tests/__init__.py profiling/tests/test_instance_matrix.py
git commit -m "feat(profiling): add Fargate instance matrix with 9 CPU/memory configs"
```

---

### Task 3: Container entrypoint script

**Files:**
- Create: `profiling/entrypoint.sh`

- [ ] **Step 1: Write the entrypoint**

Create `profiling/entrypoint.sh`:

```bash
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
```

- [ ] **Step 2: Make executable**

```bash
chmod +x profiling/entrypoint.sh
```

- [ ] **Step 3: Commit**

```bash
git add profiling/entrypoint.sh
git commit -m "feat(profiling): add container entrypoint with S3 upload retry"
```

---

### Task 4: Dockerfile

**Files:**
- Create: `profiling/Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

Create `profiling/Dockerfile`:

```dockerfile
# Cloud profiling image — all 6 CPU simulation backends pre-installed.
# Build from repo root: docker build -f profiling/Dockerfile .
FROM python:3.10-slim

# System deps for Cython compilation and BLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas-dev gcc g++ make curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2 for S3 upload in entrypoint
RUN curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscli.zip /tmp/aws

WORKDIR /app

# Copy dependency files first for layer caching
COPY pyproject.toml setup_cython.py ./
COPY src/ src/

# Install project and base deps
RUN pip install --no-cache-dir -e .

# Install all CPU backends explicitly (dependency-groups, not extras)
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        "numba>=0.58" \
        "jax[cpu]>=0.4" \
        "cython>=3.0"

# Build Cython extension (source already in src/ from earlier COPY)
RUN python setup_cython.py build_ext --inplace

# Pre-warm JIT caches to avoid cold-start overhead during profiling
RUN python -c "from network_estimation.simulation_numba import NumbaBackend; NumbaBackend()" 2>/dev/null || true
RUN python -c "import jax; jax.numpy.ones(1)" 2>/dev/null || true

# Copy entrypoint
COPY profiling/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

- [ ] **Step 2: Verify Dockerfile builds locally (optional smoke test)**

```bash
cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/profile-debug
docker build -f profiling/Dockerfile -t nestim-profiler:local .
```

This may take several minutes. The build is optional at this stage — it validates the
Dockerfile syntax and layer ordering but the full build requires all backends.

- [ ] **Step 3: Commit**

```bash
git add profiling/Dockerfile
git commit -m "feat(profiling): add Dockerfile with all 6 CPU backends"
```

---

## Chunk 2: AWS infrastructure scripts

### Task 5: Infrastructure setup script

**Files:**
- Create: `profiling/setup_infra.sh`

- [ ] **Step 1: Write the setup script**

Create `profiling/setup_infra.sh`:

```bash
#!/usr/bin/env bash
# Create AWS infrastructure for cloud profiling.
# Idempotent — safe to re-run without duplicating resources.
#
# Creates: S3 bucket, ECR repo, ECS cluster, IAM roles, CloudWatch log group.
# Writes resource ARNs to .infra-config.json for use by other scripts.
#
# Requires: AWS CLI configured with valid credentials.
# Optional: AWS_REGION env var (defaults to us-east-1).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.infra-config.json"
REGION="${AWS_REGION:-us-east-1}"

echo "=== nestim Cloud Profiling Infrastructure Setup ==="
echo "Region: ${REGION}"
echo ""

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account: ${ACCOUNT_ID}"
echo ""

BUCKET_NAME="nestim-profiling-${ACCOUNT_ID}"
ECR_REPO="nestim-profiler"
CLUSTER_NAME="nestim-profiling"
EXEC_ROLE_NAME="nestim-profiler-execution"
TASK_ROLE_NAME="nestim-profiler-task"
LOG_GROUP="/ecs/nestim-profiling"

# --- S3 Bucket ---
echo "Creating S3 bucket: ${BUCKET_NAME}..."
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "  Bucket already exists, skipping."
else
    if [[ "$REGION" == "us-east-1" ]]; then
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION"
    else
        aws s3api create-bucket --bucket "$BUCKET_NAME" --region "$REGION" \
            --create-bucket-configuration LocationConstraint="$REGION"
    fi
    echo "  Created."
fi

# S3 hardening: block public access + encryption
aws s3api put-public-access-block --bucket "$BUCKET_NAME" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"
aws s3api put-bucket-encryption --bucket "$BUCKET_NAME" \
    --server-side-encryption-configuration \
    '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
echo "  Public access blocked, SSE-S3 encryption enabled."

# Optional: lifecycle rule to expire results after 90 days
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET_NAME" \
    --lifecycle-configuration '{
        "Rules": [{
            "ID": "expire-profiling-results",
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Expiration": {"Days": 90}
        }]
    }'
echo "  Lifecycle rule: auto-expire after 90 days."
echo ""

# --- ECR Repository ---
echo "Creating ECR repository: ${ECR_REPO}..."
ECR_REPO_URI=""
if aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" >/dev/null 2>&1; then
    echo "  Repository already exists, skipping."
    ECR_REPO_URI=$(aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" \
        --query "repositories[0].repositoryUri" --output text)
else
    ECR_REPO_URI=$(aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION" \
        --image-scanning-configuration scanOnPush=true \
        --query "repository.repositoryUri" --output text)
    echo "  Created with scan-on-push enabled."
fi
echo "  URI: ${ECR_REPO_URI}"
echo ""

# --- ECS Cluster ---
echo "Creating ECS cluster: ${CLUSTER_NAME}..."
if aws ecs describe-clusters --clusters "$CLUSTER_NAME" --region "$REGION" \
    --query "clusters[?status=='ACTIVE'].clusterName" --output text | grep -q "$CLUSTER_NAME"; then
    echo "  Cluster already exists, skipping."
else
    aws ecs create-cluster --cluster-name "$CLUSTER_NAME" --region "$REGION" >/dev/null
    echo "  Created."
fi
CLUSTER_ARN=$(aws ecs describe-clusters --clusters "$CLUSTER_NAME" --region "$REGION" \
    --query "clusters[0].clusterArn" --output text)
echo ""

# --- IAM Execution Role ---
echo "Creating IAM execution role: ${EXEC_ROLE_NAME}..."
EXEC_ROLE_ARN=""
if aws iam get-role --role-name "$EXEC_ROLE_NAME" >/dev/null 2>&1; then
    echo "  Role already exists, skipping."
    EXEC_ROLE_ARN=$(aws iam get-role --role-name "$EXEC_ROLE_NAME" \
        --query "Role.Arn" --output text)
else
    EXEC_ROLE_ARN=$(aws iam create-role --role-name "$EXEC_ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' --query "Role.Arn" --output text)
    aws iam attach-role-policy --role-name "$EXEC_ROLE_NAME" \
        --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
    echo "  Created with AmazonECSTaskExecutionRolePolicy."
fi
echo "  ARN: ${EXEC_ROLE_ARN}"
echo ""

# --- IAM Task Role (S3 PutObject only) ---
echo "Creating IAM task role: ${TASK_ROLE_NAME}..."
TASK_ROLE_ARN=""
if aws iam get-role --role-name "$TASK_ROLE_NAME" >/dev/null 2>&1; then
    echo "  Role already exists, skipping."
    TASK_ROLE_ARN=$(aws iam get-role --role-name "$TASK_ROLE_NAME" \
        --query "Role.Arn" --output text)
else
    TASK_ROLE_ARN=$(aws iam create-role --role-name "$TASK_ROLE_NAME" \
        --assume-role-policy-document '{
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "ecs-tasks.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }' --query "Role.Arn" --output text)
    aws iam put-role-policy --role-name "$TASK_ROLE_NAME" \
        --policy-name "nestim-profiler-s3-upload" \
        --policy-document "{
            \"Version\": \"2012-10-17\",
            \"Statement\": [{
                \"Effect\": \"Allow\",
                \"Action\": \"s3:PutObject\",
                \"Resource\": \"arn:aws:s3:::${BUCKET_NAME}/*\"
            }]
        }"
    echo "  Created with S3 PutObject scoped to ${BUCKET_NAME}."
fi
echo "  ARN: ${TASK_ROLE_ARN}"
echo ""

# --- CloudWatch Log Group ---
echo "Creating CloudWatch log group: ${LOG_GROUP}..."
if aws logs describe-log-groups --log-group-name-prefix "$LOG_GROUP" --region "$REGION" \
    --query "logGroups[?logGroupName=='${LOG_GROUP}'].logGroupName" --output text | grep -q "$LOG_GROUP"; then
    echo "  Log group already exists, skipping."
else
    aws logs create-log-group --log-group-name "$LOG_GROUP" --region "$REGION"
    echo "  Created."
fi
echo ""

# --- Write config file ---
cat > "$CONFIG_FILE" <<CONF
{
    "region": "${REGION}",
    "account_id": "${ACCOUNT_ID}",
    "s3_bucket": "${BUCKET_NAME}",
    "ecr_repo_uri": "${ECR_REPO_URI}",
    "cluster_name": "${CLUSTER_NAME}",
    "cluster_arn": "${CLUSTER_ARN}",
    "execution_role_arn": "${EXEC_ROLE_ARN}",
    "task_role_arn": "${TASK_ROLE_ARN}",
    "log_group": "${LOG_GROUP}"
}
CONF
chmod 600 "$CONFIG_FILE"

echo "=== Setup Complete ==="
echo "Config written to: ${CONFIG_FILE}"
echo ""
echo "Next steps:"
echo "  1. Build and push the Docker image: bash profiling/build_and_push.sh"
echo "  2. Run benchmarks: python profiling/run_benchmarks.py"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x profiling/setup_infra.sh
```

- [ ] **Step 3: Commit**

```bash
git add profiling/setup_infra.sh
git commit -m "feat(profiling): add idempotent AWS infrastructure setup script"
```

---

### Task 6: Infrastructure teardown script

**Files:**
- Create: `profiling/teardown_infra.sh`

- [ ] **Step 1: Write the teardown script**

Create `profiling/teardown_infra.sh`:

```bash
#!/usr/bin/env bash
# Tear down all AWS infrastructure created by setup_infra.sh.
# Reads resource identifiers from .infra-config.json.
# Prompts for confirmation before proceeding.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/.infra-config.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: ${CONFIG_FILE} not found. Nothing to tear down." >&2
    exit 1
fi

# Parse config
REGION=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['region'])")
BUCKET=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['s3_bucket'])")
ECR_URI=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['ecr_repo_uri'])")
CLUSTER=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['cluster_name'])")
EXEC_ROLE=$(python3 -c "import json; d=json.load(open('${CONFIG_FILE}')); print(d['execution_role_arn'].split('/')[-1])")
TASK_ROLE=$(python3 -c "import json; d=json.load(open('${CONFIG_FILE}')); print(d['task_role_arn'].split('/')[-1])")
LOG_GROUP=$(python3 -c "import json; print(json.load(open('${CONFIG_FILE}'))['log_group'])")
ECR_REPO=$(echo "$ECR_URI" | cut -d'/' -f2)

echo "=== nestim Cloud Profiling Infrastructure Teardown ==="
echo ""
echo "This will DELETE the following resources in ${REGION}:"
echo "  - S3 bucket:        ${BUCKET} (and all contents)"
echo "  - ECR repository:   ${ECR_REPO} (and all images)"
echo "  - ECS cluster:      ${CLUSTER}"
echo "  - IAM role:         ${EXEC_ROLE}"
echo "  - IAM role:         ${TASK_ROLE}"
echo "  - CloudWatch group: ${LOG_GROUP}"
echo ""
read -rp "Are you sure? Type 'yes' to confirm: " confirm
if [[ "$confirm" != "yes" ]]; then
    echo "Aborted."
    exit 0
fi
echo ""

# Delete in reverse dependency order

# 1. Deregister all task definitions for this family prefix
echo "Deregistering task definitions..."
for family in $(aws ecs list-task-definition-families --family-prefix "nestim-profiling" \
    --region "$REGION" --status ACTIVE --query "families[]" --output text 2>/dev/null); do
    for td in $(aws ecs list-task-definitions --family-prefix "$family" --region "$REGION" \
        --query "taskDefinitionArns[]" --output text 2>/dev/null); do
        aws ecs deregister-task-definition --task-definition "$td" --region "$REGION" >/dev/null 2>&1 || true
    done
done
echo "  Done."

# 2. Delete ECS cluster
echo "Deleting ECS cluster: ${CLUSTER}..."
aws ecs delete-cluster --cluster "$CLUSTER" --region "$REGION" >/dev/null 2>&1 || true
echo "  Done."

# 3. Delete CloudWatch log group
echo "Deleting CloudWatch log group: ${LOG_GROUP}..."
aws logs delete-log-group --log-group-name "$LOG_GROUP" --region "$REGION" 2>/dev/null || true
echo "  Done."

# 4. Delete IAM roles (detach policies first)
echo "Deleting IAM role: ${EXEC_ROLE}..."
aws iam detach-role-policy --role-name "$EXEC_ROLE" \
    --policy-arn "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy" 2>/dev/null || true
aws iam delete-role --role-name "$EXEC_ROLE" 2>/dev/null || true
echo "  Done."

echo "Deleting IAM role: ${TASK_ROLE}..."
aws iam delete-role-policy --role-name "$TASK_ROLE" \
    --policy-name "nestim-profiler-s3-upload" 2>/dev/null || true
aws iam delete-role --role-name "$TASK_ROLE" 2>/dev/null || true
echo "  Done."

# 5. Delete ECR repository (force deletes all images)
echo "Deleting ECR repository: ${ECR_REPO}..."
aws ecr delete-repository --repository-name "$ECR_REPO" --region "$REGION" --force >/dev/null 2>&1 || true
echo "  Done."

# 6. Empty and delete S3 bucket
echo "Emptying and deleting S3 bucket: ${BUCKET}..."
aws s3 rb "s3://${BUCKET}" --force 2>/dev/null || true
echo "  Done."

# Remove config file
rm -f "$CONFIG_FILE"

echo ""
echo "=== Teardown Complete ==="
```

- [ ] **Step 2: Make executable**

```bash
chmod +x profiling/teardown_infra.sh
```

- [ ] **Step 3: Commit**

```bash
git add profiling/teardown_infra.sh
git commit -m "feat(profiling): add infrastructure teardown script with confirmation"
```

---

### Task 7: Build and push script

**Files:**
- Create: `profiling/build_and_push.sh`

- [ ] **Step 1: Write the build script**

Create `profiling/build_and_push.sh`:

```bash
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
docker build -f "${SCRIPT_DIR}/Dockerfile" -t "nestim-profiler:${IMAGE_TAG}" "${REPO_ROOT}"
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
```

- [ ] **Step 2: Make executable**

```bash
chmod +x profiling/build_and_push.sh
```

- [ ] **Step 3: Commit**

```bash
git add profiling/build_and_push.sh
git commit -m "feat(profiling): add Docker build and ECR push script"
```

---

## Chunk 3: Orchestrator (run_benchmarks.py)

### Task 8: Run ID generation and config loading helpers

**Files:**
- Create: `profiling/run_helpers.py`
- Create: `profiling/tests/test_run_helpers.py`

- [ ] **Step 1: Write the failing tests**

Create `profiling/tests/test_run_helpers.py`:

```python
"""Tests for run_benchmarks helper functions."""

import json
import os
import re
from unittest.mock import patch

from profiling.run_helpers import generate_run_id, load_infra_config


def test_run_id_format_matches_pattern():
    """Run ID should be YYYY-MM-DD-HHMMSS-<7char-hash>."""
    with patch("profiling.run_helpers._git_short_hash", return_value="abc1234"):
        with patch("profiling.run_helpers._git_is_dirty", return_value=False):
            run_id = generate_run_id()
    assert re.match(r"\d{4}-\d{2}-\d{2}-\d{6}-[a-f0-9]{7}$", run_id)


def test_run_id_dirty_suffix():
    with patch("profiling.run_helpers._git_short_hash", return_value="abc1234"):
        with patch("profiling.run_helpers._git_is_dirty", return_value=True):
            run_id = generate_run_id()
    assert run_id.endswith("-dirty")


def test_run_id_override():
    run_id = generate_run_id(override="my-custom-id")
    assert run_id == "my-custom-id"


def test_load_infra_config(tmp_path):
    config = {"region": "us-east-1", "s3_bucket": "test-bucket"}
    config_path = tmp_path / ".infra-config.json"
    config_path.write_text(json.dumps(config))
    loaded = load_infra_config(str(config_path))
    assert loaded["region"] == "us-east-1"
    assert loaded["s3_bucket"] == "test-bucket"


def test_load_infra_config_missing_file():
    try:
        load_infra_config("/nonexistent/.infra-config.json")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/profile-debug && python -m pytest profiling/tests/test_run_helpers.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write the implementation**

Create `profiling/run_helpers.py`:

```python
"""Shared helpers for the profiling orchestrator and collector."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


def _git_short_hash() -> str:
    """Return first 7 chars of HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _git_full_hash() -> str:
    """Return full HEAD commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _git_is_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    return bool(result.stdout.strip())


def generate_run_id(override: Optional[str] = None) -> str:
    """Generate a run ID in the format YYYY-MM-DD-HHMMSS-<git-hash>[-dirty].

    Args:
        override: If provided, return this value directly.

    Returns:
        A unique run identifier string.
    """
    if override:
        return override
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H%M%S")
    git_hash = _git_short_hash()
    run_id = f"{now}-{git_hash}"
    if _git_is_dirty():
        run_id += "-dirty"
    return run_id


def git_metadata() -> Dict[str, Any]:
    """Return git metadata for embedding in run results."""
    return {
        "commit": _git_full_hash(),
        "commit_short": _git_short_hash(),
        "dirty": _git_is_dirty(),
    }


def load_infra_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load .infra-config.json created by setup_infra.sh.

    Args:
        config_path: Override path. Defaults to profiling/.infra-config.json.

    Returns:
        Dict with keys: region, account_id, s3_bucket, ecr_repo_uri,
        cluster_name, cluster_arn, execution_role_arn, task_role_arn,
        log_group, image_uri.

    Raises:
        FileNotFoundError: If config file doesn't exist.
    """
    if config_path is None:
        config_path = str(Path(__file__).parent / ".infra-config.json")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. Run setup_infra.sh first."
        )
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/profile-debug && python -m pytest profiling/tests/test_run_helpers.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add profiling/run_helpers.py profiling/tests/test_run_helpers.py
git commit -m "feat(profiling): add run ID generation and config loading helpers"
```

---

### Task 9: Orchestrator script

**Files:**
- Create: `profiling/run_benchmarks.py`

- [ ] **Step 1: Write the orchestrator**

Create `profiling/run_benchmarks.py`:

```python
#!/usr/bin/env python3
"""Orchestrator for running profiling tasks across Fargate configs.

Launches parallel ECS Fargate tasks, one per instance matrix config,
monitors their progress, and reports status.

Usage:
    python profiling/run_benchmarks.py
    python profiling/run_benchmarks.py --preset super-quick --configs compute-small
    python profiling/run_benchmarks.py --dry-run
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from profiling.instance_matrix import get_configs
from profiling.run_helpers import generate_run_id, git_metadata, load_infra_config


def aws_cli(args: List[str], region: str) -> Dict[str, Any]:
    """Run an AWS CLI command and return parsed JSON output."""
    cmd = ["aws"] + args + ["--region", region, "--output", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"AWS CLI error: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"AWS CLI failed: {' '.join(cmd)}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def register_task_definition(
    config: Dict[str, Any],
    infra: Dict[str, Any],
    preset: str,
    run_id: str,
    backends: Optional[str],
    max_threads: Optional[int],
    verbose: bool = False,
) -> str:
    """Register a Fargate task definition for a given config. Returns task def ARN."""
    family = f"nestim-profiling-{config['name']}"

    env_vars = [
        {"name": "RUN_ID", "value": run_id},
        {"name": "CONFIG_NAME", "value": config["name"]},
        {"name": "S3_BUCKET", "value": infra["s3_bucket"]},
        {"name": "PRESET", "value": preset},
    ]
    if backends:
        env_vars.append({"name": "BACKENDS", "value": backends})
    if max_threads is not None:
        env_vars.append({"name": "MAX_THREADS", "value": str(max_threads)})
    if verbose:
        env_vars.append({"name": "VERBOSE", "value": "1"})

    task_def = {
        "family": family,
        "networkMode": "awsvpc",
        "requiresCompatibilities": ["FARGATE"],
        "cpu": str(config["cpu"]),
        "memory": str(config["memory"]),
        "executionRoleArn": infra["execution_role_arn"],
        "taskRoleArn": infra["task_role_arn"],
        "containerDefinitions": [
            {
                "name": "profiler",
                "image": infra["image_uri"],
                "essential": True,
                "environment": env_vars,
                "logConfiguration": {
                    "logDriver": "awslogs",
                    "options": {
                        "awslogs-group": infra["log_group"],
                        "awslogs-region": infra["region"],
                        "awslogs-stream-prefix": run_id,
                    },
                },
            }
        ],
    }

    # Write task def to temp file for aws cli
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(task_def, f)
        tmp_path = f.name

    result = aws_cli(
        ["ecs", "register-task-definition", "--cli-input-json", f"file://{tmp_path}"],
        infra["region"],
    )
    import os
    os.unlink(tmp_path)

    return result["taskDefinition"]["taskDefinitionArn"]


def launch_task(
    task_def_arn: str,
    infra: Dict[str, Any],
) -> str:
    """Launch a Fargate task. Returns task ARN."""
    # Use default VPC and subnets
    # Get default subnets
    subnets_result = aws_cli(
        ["ec2", "describe-subnets",
         "--filters", "Name=default-for-az,Values=true",
         "--query", "Subnets[].SubnetId"],
        infra["region"],
    )
    subnets = subnets_result if isinstance(subnets_result, list) else []
    if not subnets:
        raise RuntimeError("No default subnets found. Ensure a default VPC exists.")

    network_config = {
        "awsvpcConfiguration": {
            "subnets": subnets[:3],  # Use up to 3 subnets
            "assignPublicIp": "ENABLED",  # Needed for ECR pull and S3 upload
        }
    }

    result = aws_cli(
        ["ecs", "run-task",
         "--cluster", infra["cluster_name"],
         "--task-definition", task_def_arn,
         "--launch-type", "FARGATE",
         "--network-configuration", json.dumps(network_config)],
        infra["region"],
    )

    tasks = result.get("tasks", [])
    if not tasks:
        failures = result.get("failures", [])
        reason = failures[0]["reason"] if failures else "unknown"
        raise RuntimeError(f"Failed to launch task: {reason}")

    return tasks[0]["taskArn"]


def monitor_tasks(
    task_arns: Dict[str, str],
    infra: Dict[str, Any],
    timeout_minutes: int,
) -> Dict[str, str]:
    """Poll task status until all stopped or timeout. Returns final statuses."""
    start = time.time()
    timeout_secs = timeout_minutes * 60
    cluster = infra["cluster_name"]
    region = infra["region"]

    # Track per-task start time for duration display
    task_start_times = {name: start for name in task_arns}
    final_statuses: Dict[str, str] = {}

    while True:
        elapsed = time.time() - start
        if elapsed > timeout_secs:
            print(f"\nTimeout after {timeout_minutes} minutes. Stopping remaining tasks...")
            for name, arn in task_arns.items():
                if name not in final_statuses:
                    subprocess.run(
                        ["aws", "ecs", "stop-task", "--cluster", cluster,
                         "--task", arn, "--region", region],
                        capture_output=True,
                    )
                    final_statuses[name] = "TIMEOUT"
            break

        # Batch describe tasks (max 100 per call)
        pending_arns = {n: a for n, a in task_arns.items() if n not in final_statuses}
        if not pending_arns:
            break

        arn_list = list(pending_arns.values())
        result = aws_cli(
            ["ecs", "describe-tasks", "--cluster", cluster,
             "--tasks"] + arn_list,
            region,
        )

        # Build arn-to-name lookup
        arn_to_name = {a: n for n, a in pending_arns.items()}

        # Print status table
        now = time.time()
        print(f"\r\033[{len(task_arns) + 1}A", end="")  # Move cursor up
        print(f"{'Config':<25} {'Status':<12} {'Duration':<12}")
        for name in task_arns:
            if name in final_statuses:
                status = final_statuses[name]
                duration = ""
                marker = " ✓" if status == "STOPPED_OK" else " ✗"
            else:
                task_info = next(
                    (t for t in result.get("tasks", [])
                     if arn_to_name.get(t["taskArn"]) == name),
                    None,
                )
                if task_info:
                    status = task_info.get("lastStatus", "UNKNOWN")
                    dur_secs = int(now - task_start_times[name])
                    minutes, secs = divmod(dur_secs, 60)
                    duration = f"{minutes}m {secs:02d}s"

                    if status == "STOPPED":
                        # Check exit code
                        containers = task_info.get("containers", [])
                        exit_code = containers[0].get("exitCode", -1) if containers else -1
                        if exit_code == 0:
                            final_statuses[name] = "STOPPED_OK"
                            marker = " ✓"
                        else:
                            final_statuses[name] = f"FAILED(exit={exit_code})"
                            marker = " ✗"
                    else:
                        marker = ""
                else:
                    status = "UNKNOWN"
                    duration = ""
                    marker = ""
            print(f"  {name:<23} {status:<12} {duration}{marker}")

        time.sleep(10)

    return final_statuses


def print_dry_run(configs, preset, run_id, backends, max_threads, infra):
    """Print what would be launched without actually launching."""
    print("=== DRY RUN ===")
    print(f"Run ID:    {run_id}")
    print(f"Preset:    {preset}")
    print(f"Backends:  {backends or 'all'}")
    print(f"Threads:   {max_threads or 'unlimited'}")
    print(f"Cluster:   {infra['cluster_name']}")
    print(f"Image:     {infra.get('image_uri', 'NOT SET')}")
    print(f"S3 bucket: {infra['s3_bucket']}")
    print("")
    print(f"Would launch {len(configs)} Fargate tasks:")
    for c in configs:
        print(f"  {c['name']:<25} {c['label']}")
    print("")
    print("S3 output keys:")
    for c in configs:
        print(f"  s3://{infra['s3_bucket']}/{run_id}/{c['name']}.json")


def main():
    parser = argparse.ArgumentParser(
        description="Run nestim profiler across Fargate instance configs.",
    )
    parser.add_argument(
        "--preset", default="exhaustive",
        choices=["super-quick", "quick", "standard", "exhaustive"],
        help="Profiler preset (default: exhaustive)",
    )
    parser.add_argument(
        "--configs",
        help="Comma-separated config names to run (default: all)",
    )
    parser.add_argument(
        "--run-id",
        help="Custom run ID (default: auto-generated from timestamp + git hash)",
    )
    parser.add_argument(
        "--backends",
        help="Comma-separated backend filter (passed to nestim)",
    )
    parser.add_argument(
        "--max-threads", type=int,
        help="Thread cap (passed to nestim --max-threads)",
    )
    parser.add_argument(
        "--timeout", type=int, default=60,
        help="Timeout in minutes for all tasks (default: 60)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Pass --verbose to nestim profiler (more detailed output in logs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be launched without launching",
    )

    args = parser.parse_args()

    # Load configs
    config_names = args.configs.split(",") if args.configs else None
    configs = get_configs(names=config_names)

    # Generate run ID
    run_id = generate_run_id(override=args.run_id)
    git_info = git_metadata()

    if git_info["dirty"] and not args.run_id:
        print(f"WARNING: Working tree is dirty. Run ID: {run_id}", file=sys.stderr)

    # Load infra config
    infra = load_infra_config()

    if args.dry_run:
        print_dry_run(configs, args.preset, run_id, args.backends, args.max_threads, infra)
        return

    print(f"=== Launching Cloud Profiling Run ===")
    print(f"Run ID:    {run_id}")
    print(f"Git:       {git_info['commit_short']}{' (dirty)' if git_info['dirty'] else ''}")
    print(f"Preset:    {args.preset}")
    print(f"Configs:   {len(configs)}")
    print(f"Timeout:   {args.timeout} minutes")
    print("")

    # Register task definitions and launch tasks
    task_arns: Dict[str, str] = {}
    for i, config in enumerate(configs):
        print(f"Registering + launching: {config['name']} ({config['label']})...")
        task_def_arn = register_task_definition(
            config, infra, args.preset, run_id, args.backends, args.max_threads,
            verbose=args.verbose,
        )
        task_arn = launch_task(task_def_arn, infra)
        task_arns[config["name"]] = task_arn

        # Stagger launches to avoid API throttling
        if i < len(configs) - 1:
            time.sleep(1)

    print(f"\nAll {len(task_arns)} tasks launched. Monitoring...\n")
    # Print blank lines for status table
    for _ in range(len(task_arns) + 1):
        print()

    final = monitor_tasks(task_arns, infra, args.timeout)

    # Summary
    print("\n=== Run Complete ===")
    ok = sum(1 for s in final.values() if s == "STOPPED_OK")
    failed = len(final) - ok
    print(f"Succeeded: {ok}/{len(final)}")
    if failed:
        print(f"Failed:    {failed}")
        for name, status in final.items():
            if status != "STOPPED_OK":
                print(f"  {name}: {status}")
        # Tail CloudWatch logs for failed tasks
        print("\n--- CloudWatch logs for failed tasks ---")
        for name, status in final.items():
            if status != "STOPPED_OK":
                print(f"\n[{name}]:")
                subprocess.run(
                    ["aws", "logs", "tail", infra["log_group"],
                     "--log-stream-name-prefix", f"{run_id}",
                     "--since", "2h", "--region", infra["region"]],
                    timeout=10,
                )
                print()

    print(f"\nCollect results with:")
    print(f"  python profiling/collect_results.py --run-id {run_id}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add profiling/run_benchmarks.py
git commit -m "feat(profiling): add Fargate orchestrator with parallel launch and monitoring"
```

---

## Chunk 4: Result collection and documentation

### Task 10: Result collector

**Files:**
- Create: `profiling/collect_results.py`
- Create: `profiling/tests/test_collect_results.py`

- [ ] **Step 1: Write the failing tests**

Create `profiling/tests/test_collect_results.py`:

```python
"""Tests for result collection and aggregation."""

import csv
import json
import io
from profiling.collect_results import build_combined_json, build_summary_csv


SAMPLE_RESULT = {
    "hardware": {"platform": "Linux", "cpu_count_logical": 2},
    "backend_versions": {"numpy": "1.24.0"},
    "correctness": [{"backend": "numpy", "passed": True}],
    "timing": [
        {
            "backend": "numpy",
            "operation": "run_mlp",
            "width": 256,
            "depth": 4,
            "n_samples": 10000,
            "median_time": 0.05,
            "speedup_vs_numpy": 1.0,
        },
        {
            "backend": "scipy",
            "operation": "run_mlp",
            "width": 256,
            "depth": 4,
            "n_samples": 10000,
            "median_time": 0.03,
            "speedup_vs_numpy": 1.67,
        },
    ],
}


def test_build_combined_json():
    config_results = {
        "compute-small": SAMPLE_RESULT,
        "compute-large": SAMPLE_RESULT,
    }
    combined = build_combined_json(
        run_id="2026-03-20-143000-abc1234",
        git_commit="abc1234567890",
        git_dirty=False,
        config_results=config_results,
    )
    assert combined["run_id"] == "2026-03-20-143000-abc1234"
    assert combined["git_commit"] == "abc1234567890"
    assert combined["git_dirty"] is False
    assert "collected_at" in combined
    assert "compute-small" in combined["configs"]
    assert "compute-large" in combined["configs"]


def test_build_combined_json_partial():
    """Should work with partial results."""
    combined = build_combined_json(
        run_id="test",
        git_commit="abc",
        git_dirty=False,
        config_results={"compute-small": SAMPLE_RESULT},
    )
    assert len(combined["configs"]) == 1


def test_build_summary_csv():
    from profiling.instance_matrix import INSTANCE_MATRIX

    config_results = {"compute-small": SAMPLE_RESULT}
    csv_str = build_summary_csv(config_results)

    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 2  # numpy + scipy timing entries
    assert rows[0]["config_name"] == "compute-small"
    assert rows[0]["backend"] == "numpy"
    assert rows[0]["width"] == "256"
    assert float(rows[0]["speedup_vs_numpy"]) == 1.0


def test_build_summary_csv_empty():
    csv_str = build_summary_csv({})
    reader = csv.DictReader(io.StringIO(csv_str))
    rows = list(reader)
    assert len(rows) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/profile-debug && python -m pytest profiling/tests/test_collect_results.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write the implementation**

Create `profiling/collect_results.py`:

```python
#!/usr/bin/env python3
"""Pull profiling results from S3 and aggregate into combined reports.

Usage:
    python profiling/collect_results.py --run-id 2026-03-20-143000-abc1234
    python profiling/collect_results.py --run-id 2026-03-20-v1 --format json
    python profiling/collect_results.py --run-id 2026-03-20-v1 --output ./my-results/
"""

import argparse
import csv
import io
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from profiling.instance_matrix import INSTANCE_MATRIX, get_configs
from profiling.run_helpers import load_infra_config


def build_combined_json(
    run_id: str,
    git_commit: str,
    git_dirty: bool,
    config_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the combined JSON structure from per-config results."""
    return {
        "run_id": run_id,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "configs": config_results,
    }


def build_summary_csv(config_results: Dict[str, Any]) -> str:
    """Build a flattened CSV from per-config timing results.

    Columns: config_name, cpu, memory, backend, width, depth, n_samples,
             run_mlp_time, output_stats_time, speedup_vs_numpy
    """
    # Build config lookup for CPU/memory
    config_lookup = {c["name"]: c for c in INSTANCE_MATRIX}

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "config_name", "cpu", "memory", "backend", "operation",
        "width", "depth", "n_samples", "median_time", "speedup_vs_numpy",
    ])
    writer.writeheader()

    for config_name, result in sorted(config_results.items()):
        config_meta = config_lookup.get(config_name, {})
        for timing in result.get("timing", []):
            writer.writerow({
                "config_name": config_name,
                "cpu": config_meta.get("cpu", ""),
                "memory": config_meta.get("memory", ""),
                "backend": timing.get("backend", ""),
                "operation": timing.get("operation", ""),
                "width": timing.get("width", ""),
                "depth": timing.get("depth", ""),
                "n_samples": timing.get("n_samples", ""),
                "median_time": timing.get("median_time", ""),
                "speedup_vs_numpy": timing.get("speedup_vs_numpy", ""),
            })

    return output.getvalue()


def s3_list_objects(bucket: str, prefix: str, region: str) -> List[str]:
    """List S3 object keys under a prefix."""
    result = subprocess.run(
        ["aws", "s3api", "list-objects-v2",
         "--bucket", bucket, "--prefix", prefix,
         "--query", "Contents[].Key", "--output", "json",
         "--region", region],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return []
    keys = json.loads(result.stdout) if result.stdout.strip() else []
    return keys or []


def s3_download(bucket: str, key: str, dest: str, region: str) -> bool:
    """Download a single S3 object. Returns True on success."""
    result = subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", dest, "--region", region],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Collect and aggregate profiling results from S3.",
    )
    parser.add_argument(
        "--run-id", required=True,
        help="Run ID to collect results for",
    )
    parser.add_argument(
        "--output", default="profiling/results",
        help="Output directory (default: profiling/results/)",
    )
    parser.add_argument(
        "--format", default="json,csv",
        help="Output formats: json, csv, or both (default: json,csv)",
    )

    args = parser.parse_args()
    formats = [f.strip() for f in args.format.split(",")]

    infra = load_infra_config()
    bucket = infra["s3_bucket"]
    region = infra["region"]
    prefix = f"{args.run_id}/"

    # Create output directory
    output_dir = Path(args.output) / args.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # List available results
    print(f"Listing results for run: {args.run_id}")
    keys = s3_list_objects(bucket, prefix, region)

    if not keys:
        print(f"No results found in s3://{bucket}/{prefix}")
        sys.exit(1)

    # Download individual results
    config_results: Dict[str, Any] = {}
    expected_configs = {c["name"] for c in INSTANCE_MATRIX}
    found_configs = set()

    for key in keys:
        config_name = key.split("/")[-1].replace(".json", "")
        dest = str(output_dir / f"{config_name}.json")
        print(f"  Downloading: {config_name}...")
        if s3_download(bucket, key, dest, region):
            with open(dest) as f:
                config_results[config_name] = json.load(f)
            found_configs.add(config_name)
        else:
            print(f"  WARNING: Failed to download {key}")

    # Report missing configs
    missing = expected_configs - found_configs
    if missing:
        print(f"\nWARNING: Missing results for: {', '.join(sorted(missing))}")

    print(f"\nCollected {len(config_results)} / {len(expected_configs)} configs")

    # Extract git info from run ID if it matches auto-generated pattern
    # Format: YYYY-MM-DD-HHMMSS-<7char-hash>[-dirty]
    import re
    git_commit = ""
    git_dirty = args.run_id.endswith("-dirty")
    clean_id = args.run_id.removesuffix("-dirty")
    match = re.match(r"\d{4}-\d{2}-\d{2}-\d{6}-([a-f0-9]{7})$", clean_id)
    if match:
        git_commit = match.group(1)

    # Build combined outputs
    if "json" in formats:
        combined = build_combined_json(
            run_id=args.run_id,
            git_commit=git_commit,
            git_dirty=git_dirty,
            config_results=config_results,
        )
        combined_path = output_dir / "combined.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2)
        print(f"Combined JSON: {combined_path}")

    if "csv" in formats:
        csv_str = build_summary_csv(config_results)
        csv_path = output_dir / "summary.csv"
        with open(csv_path, "w") as f:
            f.write(csv_str)
        print(f"Summary CSV:   {csv_path}")

    # Terminal summary
    print(f"\n{'Config':<25} {'Backends':<10} {'Fastest':<12}")
    print("-" * 47)
    for config_name in sorted(config_results):
        result = config_results[config_name]
        backends = {t["backend"] for t in result.get("timing", [])}
        # Find fastest backend for run_mlp
        run_mlp_times = [
            t for t in result.get("timing", [])
            if t.get("operation") == "run_mlp"
        ]
        if run_mlp_times:
            fastest = min(run_mlp_times, key=lambda t: t["median_time"])
            fastest_name = fastest["backend"]
        else:
            fastest_name = "—"
        print(f"  {config_name:<23} {len(backends):<10} {fastest_name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/profile-debug && python -m pytest profiling/tests/test_collect_results.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add profiling/collect_results.py profiling/tests/test_collect_results.py
git commit -m "feat(profiling): add S3 result collector with JSON + CSV aggregation"
```

---

### Task 11: README documentation

**Files:**
- Create: `profiling/README.md`

- [ ] **Step 1: Write the README**

Create `profiling/README.md`:

```markdown
# Cloud Profiling Infrastructure

Run `nestim profile-simulation` benchmarks across a matrix of AWS Fargate
CPU/memory configurations, collect results to S3, and aggregate locally.

## Prerequisites

- **AWS CLI v2** — configured with valid credentials (`aws configure`)
- **Docker** — for building the profiler image
- **Python 3.10+** — for the orchestrator and collector scripts
- **Fargate quota** — default is 6 vCPU on-demand per region. Request an increase
  via AWS Service Quotas if running the full matrix (which needs ~60 concurrent vCPUs).

## Quick Start

```bash
# 1. Create AWS infrastructure (S3, ECR, ECS, IAM)
bash profiling/setup_infra.sh

# 2. Build and push the Docker image
bash profiling/build_and_push.sh

# 3. Run benchmarks (all 9 configs, exhaustive preset)
python profiling/run_benchmarks.py

# 4. Collect and aggregate results
python profiling/collect_results.py --run-id <run-id-from-step-3>
```

## Infrastructure

### Setup

```bash
# Defaults to us-east-1. Override with AWS_REGION:
AWS_REGION=eu-west-1 bash profiling/setup_infra.sh
```

Creates these resources:

| Resource | Name | Purpose |
|----------|------|---------|
| S3 bucket | `nestim-profiling-{account-id}` | Result storage |
| ECR repository | `nestim-profiler` | Docker image registry |
| ECS cluster | `nestim-profiling` | Fargate task execution |
| IAM execution role | `nestim-profiler-execution` | ECR pull + CloudWatch |
| IAM task role | `nestim-profiler-task` | S3 upload (scoped) |
| CloudWatch log group | `/ecs/nestim-profiling` | Task logs |

Resource ARNs are written to `profiling/.infra-config.json` (gitignored).

### Teardown

```bash
bash profiling/teardown_infra.sh
```

Prompts for confirmation. Deletes all resources including S3 contents.

## Building the Docker Image

```bash
bash profiling/build_and_push.sh
```

Rebuild after any code changes to `src/` or backend dependencies. The image includes
all 6 CPU backends (NumPy, SciPy, PyTorch CPU, Numba, JAX CPU, Cython) with pre-warmed
JIT caches.

## Running Benchmarks

### Full run (default)

```bash
python profiling/run_benchmarks.py
```

Launches all 9 Fargate configs with `--preset exhaustive`. Shows a live status table.

### Debug run

```bash
python profiling/run_benchmarks.py --preset super-quick --configs compute-small
```

Fast iteration: single small config, minimal profiling.

### All options

```
python profiling/run_benchmarks.py \
    --preset exhaustive              # super-quick|quick|standard|exhaustive
    --configs compute-small,general-large  # filter to specific configs
    --run-id my-custom-run           # override auto-generated ID
    --backends numpy,pytorch         # only profile specific backends
    --max-threads 4                  # cap CPU threads
    --timeout 90                     # minutes before aborting (default: 60)
    --dry-run                        # show plan without launching
```

### Run IDs

Auto-generated format: `YYYY-MM-DD-HHMMSS-<git-hash>[-dirty]`
Example: `2026-03-20-143000-bc385ad`

The git hash ties results to a specific code version.

## Collecting Results

```bash
python profiling/collect_results.py --run-id 2026-03-20-143000-bc385ad
```

Downloads individual JSONs from S3 and produces:

- `profiling/results/{run-id}/combined.json` — all configs in one file
- `profiling/results/{run-id}/summary.csv` — flattened for pandas/notebooks

### Options

```
python profiling/collect_results.py \
    --run-id <id>                    # required
    --output ./my-results            # override output directory
    --format json,csv                # or just json, or just csv
```

## Instance Matrix

Defined in `profiling/instance_matrix.py`. Default configs:

### Compute-Optimized (c-series)

| Name | vCPUs | Memory |
|------|-------|--------|
| compute-small | 1 | 2 GB |
| compute-medium | 2 | 4 GB |
| compute-large | 4 | 8 GB |
| compute-xlarge | 8 | 16 GB |
| compute-2xlarge | 16 | 32 GB |

### General-Purpose (m-series)

| Name | vCPUs | Memory |
|------|-------|--------|
| general-small | 1 | 4 GB |
| general-medium | 2 | 8 GB |
| general-large | 4 | 16 GB |
| general-xlarge | 8 | 32 GB |

To add a custom config, edit `INSTANCE_MATRIX` in `instance_matrix.py`.

## Troubleshooting

### "No default subnets found"

The orchestrator uses the default VPC. If your account doesn't have one:
```bash
aws ec2 create-default-vpc
```

### Tasks stuck in PENDING

Check Fargate vCPU quota:
```bash
aws service-quotas get-service-quota \
    --service-code fargate \
    --quota-code L-3032A538
```

### Task fails immediately

Check CloudWatch logs:
```bash
aws logs tail /ecs/nestim-profiling --since 1h
```

### S3 upload fails in container

Ensure the task role has `s3:PutObject` permission and the bucket name matches.

### Docker build fails on Cython

Ensure `_cython_kernels.pyx` is present in `src/network_estimation/`.

## Cost Estimates

Rough per-run costs for the default 9-config matrix with `exhaustive` preset:

- **Fargate compute:** ~$2-5 per full run (depends on task duration)
- **S3 storage:** negligible (~$0.001 per run)
- **ECR storage:** ~$0.10/month for the image
- **CloudWatch:** ~$0.50/GB ingested

Total: **~$3-6 per full matrix run**

Use `--preset super-quick --configs compute-small` for near-free debug runs.
```

- [ ] **Step 2: Commit**

```bash
git add profiling/README.md
git commit -m "docs(profiling): add comprehensive README with setup, usage, and troubleshooting"
```

---

### Task 12: Final integration check

- [ ] **Step 1: Run all profiling tests**

```bash
cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/profile-debug
python -m pytest profiling/tests/ -v
```

Expected: All tests pass (instance matrix + run helpers + collector tests).

- [ ] **Step 2: Verify dry-run works (without AWS credentials)**

```bash
cd /Users/mohanty/conductor/workspaces/circuit-estimation-mvp/profile-debug
python profiling/run_benchmarks.py --help
```

Expected: Help text prints with all options documented.

- [ ] **Step 3: Verify directory structure**

```bash
ls -la profiling/
```

Expected:
```
Dockerfile
README.md
__init__.py
build_and_push.sh
collect_results.py
entrypoint.sh
instance_matrix.py
results/
run_benchmarks.py
run_helpers.py
setup_infra.sh
teardown_infra.sh
tests/
```

- [ ] **Step 4: Final commit (if any cleanup needed)**

```bash
git add -A profiling/
git status
# Only commit if there are changes
git commit -m "chore(profiling): integration cleanup"
```
