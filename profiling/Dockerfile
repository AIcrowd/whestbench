# Cloud profiling image — all 6 CPU simulation backends pre-installed.
#
# BLAS CHOICE: Intel MKL for all backends (NumPy, SciPy, JAX, PyTorch).
#
# Target hardware: AWS Fargate x86_64, which runs exclusively on Intel Xeon
# Platinum (Cascade Lake / Skylake) with full AVX-512 support. Benchmarks on
# this hardware (c5.4xlarge, Xeon Platinum 8124M @ 3.0GHz) showed:
#
#   NumPy + MKL:      482 µs  (1104 GFLOPS) — single 4096×256 @ 256×256 sgemm
#   NumPy + OpenBLAS:  607 µs  ( 898 GFLOPS) — same workload, ~26% slower
#   PyTorch + MKL:    467 µs  (~1150 GFLOPS) — matches NumPy+MKL
#
# MKL also saturates at 8 threads while OpenBLAS regresses at 16 threads.
# Since our profiling targets are Intel-only Fargate instances, MKL is the
# optimal BLAS for all backends. We use Miniconda to get MKL-linked NumPy/SciPy
# (pip numpy bundles OpenBLAS by default, and there is no pip package for MKL
# numpy).
#
# If targeting AMD hardware in the future, re-evaluate — OpenBLAS or BLIS may
# be faster on EPYC processors.
#
# Build from repo root: docker build -f profiling/Dockerfile .
FROM --platform=linux/amd64 continuumio/miniconda3:latest

# System deps for Cython compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make curl unzip \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2 for S3 upload in entrypoint
RUN curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscli.zip /tmp/aws

WORKDIR /app

# Performance env vars for all CPU backends
ENV JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

# Install NumPy, SciPy, and JAX with MKL BLAS.
# - NumPy/SciPy from defaults channel with blas=mkl constraint
# - JAX/jaxlib from conda-forge: links against conda's libblas (→ MKL)
#   (pip jax[cpu] bundles Eigen which ignores system BLAS entirely)
RUN conda install -y -n base python=3.10 \
        "numpy>=1.24" "scipy>=1.10" "blas=*=mkl" mkl pip \
    && conda install -y -n base -c conda-forge jax jaxlib \
    && conda clean -afy

# Copy dependency files first for layer caching
COPY pyproject.toml setup_cython.py ./
COPY src/ src/

# Install project deps (--no-deps to avoid overwriting conda numpy/scipy with pip versions)
RUN pip install --no-cache-dir --no-deps -e . \
    && pip install --no-cache-dir plotext rich "threadpoolctl>=3.0" tqdm

# Install remaining CPU backends via pip.
# - PyTorch CPU wheel: statically links MKL, matching our NumPy setup.
# - JAX is already installed from conda (MKL-linked), NOT from pip.
# - Numba uses SciPy's BLAS bindings at JIT time → also gets MKL.
RUN pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir \
        "numba>=0.58" \
        "cython>=3.0"

# Build Cython extension with optimization flags
RUN CFLAGS="-O3 -ffast-math" python setup_cython.py build_ext --inplace

# Pre-warm JIT caches — actually call compiled functions with realistic shapes
# so Numba/JAX compile and cache the generated code.
RUN python -c "\
import numpy as np; \
from network_estimation.generation import sample_mlp; \
mlp = sample_mlp(64, 4, np.random.default_rng(42)); \
inputs = np.random.randn(100, 64).astype(np.float32); \
from network_estimation.simulation_numba import NumbaBackend; \
nb = NumbaBackend(); \
nb.run_mlp(mlp, inputs); \
nb.run_mlp_matmul_only(mlp, inputs); \
nb.sample_layer_statistics(mlp, 100); \
print('Numba JIT warm-up complete') \
" || echo 'WARN: Numba warm-up failed (non-fatal)'

RUN python -c "\
import numpy as np; \
from network_estimation.generation import sample_mlp; \
mlp = sample_mlp(64, 4, np.random.default_rng(42)); \
inputs = np.random.randn(100, 64).astype(np.float32); \
from network_estimation.simulation_jax import JAXBackend; \
jb = JAXBackend(); \
jb.run_mlp(mlp, inputs); \
jb.run_mlp_matmul_only(mlp, inputs); \
jb.sample_layer_statistics(mlp, 100); \
print('JAX JIT warm-up complete') \
" || echo 'WARN: JAX warm-up failed (non-fatal)'

# Warm up SciPy and Cython backends (verify import + basic call)
RUN python -c "\
import numpy as np; \
from network_estimation.generation import sample_mlp; \
mlp = sample_mlp(64, 4, np.random.default_rng(42)); \
inputs = np.random.randn(100, 64).astype(np.float32); \
from network_estimation.simulation_scipy import SciPyBackend; \
sb = SciPyBackend(); \
sb.run_mlp(mlp, inputs); \
sb.run_mlp_matmul_only(mlp, inputs); \
sb.sample_layer_statistics(mlp, 100); \
print('SciPy warm-up complete') \
" || echo 'WARN: SciPy warm-up failed (non-fatal)'

RUN python -c "\
import numpy as np; \
from network_estimation.generation import sample_mlp; \
mlp = sample_mlp(64, 4, np.random.default_rng(42)); \
inputs = np.random.randn(100, 64).astype(np.float32); \
from network_estimation.simulation_cython import CythonBackend; \
cb = CythonBackend(); \
cb.run_mlp(mlp, inputs); \
cb.run_mlp_matmul_only(mlp, inputs); \
cb.sample_layer_statistics(mlp, 100); \
print('Cython warm-up complete') \
" || echo 'WARN: Cython warm-up failed (non-fatal)'

# Verify NumPy is using MKL (fail the build if not).
# Must import numpy first so MKL shared libs get loaded before threadpoolctl scans.
COPY profiling/verify_mkl.py /tmp/verify_mkl.py
RUN python /tmp/verify_mkl.py

# Copy entrypoint
COPY profiling/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
