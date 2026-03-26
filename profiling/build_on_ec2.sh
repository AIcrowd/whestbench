#!/usr/bin/env bash
# Build Docker image on an EC2 spot instance (needed because local machine is ARM).
#
# Launches a c5.4xlarge spot instance (16 vCPU, 32 GB, ~$0.15/hr),
# builds the MKL profiler image, pushes to ECR, and terminates.
#
# The image uses Miniconda with MKL-linked NumPy/SciPy because our target
# hardware is AWS Fargate x86_64 (Intel Xeon Platinum, Cascade Lake/Skylake).
# Benchmarks showed MKL is ~26% faster than OpenBLAS on this hardware for
# our sgemm workload. See profiling/Dockerfile header for full rationale.
#
# Prerequisites: AWS CLI configured, ECR repo exists, SSH key available.
#
# Usage:
#   bash profiling/build_on_ec2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$SCRIPT_DIR/.infra-config.json"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "ERROR: Run profiling/setup_infra.sh first" >&2
    exit 1
fi

REGION=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['region'])")
ACCOUNT_ID=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['account_id'])")
ECR_REPO=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['ecr_repo_uri'])")

echo "=== EC2 Builder ==="
echo "Region:   $REGION"
echo "ECR repo: $ECR_REPO"
echo ""

# Create a temporary key pair for SSH
KEY_NAME="nestim-builder-$(date +%s)"
KEY_FILE="/tmp/${KEY_NAME}.pem"
aws ec2 create-key-pair --key-name "$KEY_NAME" --query 'KeyMaterial' --output text --region "$REGION" > "$KEY_FILE"
chmod 600 "$KEY_FILE"

# Find the latest Amazon Linux 2023 AMI
AMI_ID=$(aws ec2 describe-images \
    --owners amazon \
    --filters "Name=name,Values=al2023-ami-2023*-x86_64" "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text --region "$REGION")

echo "AMI: $AMI_ID"

# Create a security group allowing SSH
SG_NAME="nestim-builder-sg"
VPC_ID=$(aws ec2 describe-vpcs --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text --region "$REGION")
SG_ID=$(aws ec2 describe-security-groups --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" --query 'SecurityGroups[0].GroupId' --output text --region "$REGION" 2>/dev/null || echo "None")

if [[ "$SG_ID" == "None" ]] || [[ -z "$SG_ID" ]]; then
    SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "Temp builder SG" --vpc-id "$VPC_ID" --output text --query 'GroupId' --region "$REGION")
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 --region "$REGION"
fi

# Create IAM instance profile for ECR access (reuse if exists)
INSTANCE_PROFILE="nestim-builder-profile"
aws iam get-instance-profile --instance-profile-name "$INSTANCE_PROFILE" 2>/dev/null || {
    ROLE_NAME="nestim-builder-role"
    aws iam create-role --role-name "$ROLE_NAME" \
        --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' \
        --region "$REGION" || true
    aws iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess || true
    aws iam create-instance-profile --instance-profile-name "$INSTANCE_PROFILE" || true
    aws iam add-role-to-instance-profile --instance-profile-name "$INSTANCE_PROFILE" --role-name "$ROLE_NAME" || true
    sleep 10  # Wait for IAM propagation
}

# Launch spot instance
echo "Launching c5.4xlarge spot instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type c5.4xlarge \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --iam-instance-profile Name="$INSTANCE_PROFILE" \
    --instance-market-options '{"MarketType":"spot","SpotOptions":{"SpotInstanceType":"one-time"}}' \
    --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=nestim-torch-builder}]" \
    --query 'Instances[0].InstanceId' --output text --region "$REGION")

echo "Instance: $INSTANCE_ID"
echo "Waiting for instance to be running..."
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"

PUBLIC_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text --region "$REGION")
echo "Public IP: $PUBLIC_IP"

# Wait for SSH to be ready
echo "Waiting for SSH..."
for i in $(seq 1 30); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" "echo ready" 2>/dev/null; then
        break
    fi
    sleep 5
done

# Upload repo to instance
echo "Uploading source code..."
cd "$REPO_ROOT"
tar czf /tmp/nestim-src.tar.gz --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='profiling/results' .
scp -o StrictHostKeyChecking=no -i "$KEY_FILE" /tmp/nestim-src.tar.gz ec2-user@"$PUBLIC_IP":/tmp/

# Run the build remotely
echo "Starting remote build..."
ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ec2-user@"$PUBLIC_IP" << REMOTE_SCRIPT
set -euo pipefail

# Install Docker
sudo dnf install -y docker
sudo systemctl start docker
sudo usermod -aG docker ec2-user
# Use sudo for docker commands since group change needs re-login

# Extract source
mkdir -p ~/nestim && cd ~/nestim
tar xzf /tmp/nestim-src.tar.gz

# Authenticate to ECR
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin $ECR_REPO

# Build profiler image (Miniconda + MKL NumPy/SciPy + pip torch MKL)
echo ""
echo "=== Building profiler image (MKL BLAS) ==="
sudo docker build -f profiling/Dockerfile -t nestim-profiler:latest .
sudo docker tag nestim-profiler:latest ${ECR_REPO}:latest
sudo docker tag nestim-profiler:latest ${ECR_REPO}:mkl
sudo docker push ${ECR_REPO}:latest
sudo docker push ${ECR_REPO}:mkl
echo "Image pushed."

echo ""
echo "=== Build complete ==="
REMOTE_SCRIPT

BUILD_EXIT=$?

# Terminate instance
echo ""
echo "Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" --region "$REGION" > /dev/null

# Clean up key pair
aws ec2 delete-key-pair --key-name "$KEY_NAME" --region "$REGION"
rm -f "$KEY_FILE" /tmp/nestim-src.tar.gz

if [[ $BUILD_EXIT -eq 0 ]]; then
    echo ""
    echo "=== Build Complete ==="
    echo "  ${ECR_REPO}:latest  — Miniconda + MKL NumPy/SciPy + pip torch (MKL)"
    echo "  ${ECR_REPO}:mkl     — same as :latest"
    echo ""
    echo "Run profiling:"
    echo "  PYTHONPATH=. python profiling/run_benchmarks.py"
else
    echo "ERROR: Build failed with exit code $BUILD_EXIT" >&2
    exit 1
fi
