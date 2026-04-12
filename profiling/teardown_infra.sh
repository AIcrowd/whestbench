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

echo "=== whest Cloud Profiling Infrastructure Teardown ==="
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
for family in $(aws ecs list-task-definition-families --family-prefix "whest-profiling" \
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
    --policy-name "whest-profiler-s3-upload" 2>/dev/null || true
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
