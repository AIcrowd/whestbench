#!/usr/bin/env bash
# Build PyTorch from source with OpenBLAS on a Fargate 16vCPU task.
#
# WHY: The pip PyTorch CPU wheel statically links Intel MKL for BLAS.
# On Fargate (Intel Xeon Platinum 8259CL, Cascade Lake), OpenBLAS 0.3.29
# with DYNAMIC_ARCH detects SkylakeX and uses AVX-512 sgemm kernels that
# outperform MKL's bundled kernels for the matrix shapes in our workload
# (4096×256 @ 256×256, 128 layers). This script builds a PyTorch wheel
# linked against system OpenBLAS so we can do a head-to-head comparison.
#
# The wheel is uploaded to S3 for use in Dockerfile.openblas.
#
# Usage (runs inside Fargate via build_torch_openblas_task.py):
#   This script is NOT meant to be run locally (ARM Mac can't build x86_64).
set -euo pipefail

: "${S3_BUCKET:?S3_BUCKET is required}"
TORCH_VERSION="${TORCH_VERSION:-v2.5.1}"

echo "=== Building PyTorch ${TORCH_VERSION} with OpenBLAS ==="
echo "CPUs: $(nproc)"
echo "RAM: $(free -h | awk '/^Mem:/{print $2}')"
echo ""

# Install build dependencies
apt-get update && apt-get install -y --no-install-recommends \
    git cmake ninja-build libopenblas-dev \
    gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

pip install --no-cache-dir setuptools wheel pyyaml typing_extensions sympy filelock jinja2 networkx

# Clone PyTorch
cd /tmp
git clone --depth 1 --branch "${TORCH_VERSION}" --recursive https://github.com/pytorch/pytorch.git
cd pytorch

# Configure: OpenBLAS, no CUDA, no MKL
export USE_CUDA=0
export USE_CUDNN=0
export USE_MKL=0
export USE_MKLDNN=0
export BLAS=OpenBLAS
export USE_DISTRIBUTED=0
export USE_NCCL=0
export BUILD_TEST=0
export MAX_JOBS=$(nproc)

echo "Build config: USE_MKL=$USE_MKL BLAS=$BLAS MAX_JOBS=$MAX_JOBS"
echo ""

# Build wheel
python3 setup.py bdist_wheel 2>&1 | tail -50

# Find and upload wheel
WHEEL=$(ls dist/torch-*.whl | head -1)
echo ""
echo "Built wheel: $WHEEL"
echo "Size: $(du -h "$WHEEL" | cut -f1)"

S3_KEY="s3://${S3_BUCKET}/torch-wheels/$(basename "$WHEEL")"
for attempt in 1 2 3; do
    if aws s3 cp "$WHEEL" "$S3_KEY"; then
        echo "Uploaded to $S3_KEY"
        exit 0
    fi
    sleep $((2 ** attempt))
done

echo "ERROR: Failed to upload wheel" >&2
exit 1
