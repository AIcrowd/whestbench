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

echo "=== whest Cloud Profiling Infrastructure Setup ==="
echo "Region: ${REGION}"
echo ""

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "Account: ${ACCOUNT_ID}"
echo ""

BUCKET_NAME="whest-profiling-${ACCOUNT_ID}"
ECR_REPO="whest-profiler"
CLUSTER_NAME="whest-profiling"
EXEC_ROLE_NAME="whest-profiler-execution"
TASK_ROLE_NAME="whest-profiler-task"
LOG_GROUP="/ecs/whest-profiling"

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
        --policy-name "whest-profiler-s3-upload" \
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
