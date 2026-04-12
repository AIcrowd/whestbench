# Cloud profiling image — all 6 CPU simulation backends pre-installed.
# Build from repo root: docker build -f profiling/Dockerfile .
FROM --platform=linux/amd64 python:3.10-slim

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

# Performance env vars for all CPU backends
ENV JAX_PLATFORMS=cpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1

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

# Build Cython extension with optimization flags
RUN CFLAGS="-O3 -ffast-math" python setup_cython.py build_ext --inplace

# Pre-warm JIT caches — actually call compiled functions with realistic shapes
# so Numba/JAX compile and cache the generated code.
RUN python -c "\
import numpy as np; \
from whestbench.generation import sample_mlp; \
mlp = sample_mlp(64, 4, np.random.default_rng(42)); \
inputs = np.random.randn(100, 64).astype(np.float32); \
from whestbench.simulation_numba import NumbaBackend; \
nb = NumbaBackend(); \
nb.run_mlp(mlp, inputs); \
nb.run_mlp_matmul_only(mlp, inputs); \
nb.sample_layer_statistics(mlp, 100); \
print('Numba JIT warm-up complete') \
" || echo 'WARN: Numba warm-up failed (non-fatal)'

RUN python -c "\
import numpy as np; \
from whestbench.generation import sample_mlp; \
mlp = sample_mlp(64, 4, np.random.default_rng(42)); \
inputs = np.random.randn(100, 64).astype(np.float32); \
from whestbench.simulation_jax import JAXBackend; \
jb = JAXBackend(); \
jb.run_mlp(mlp, inputs); \
jb.run_mlp_matmul_only(mlp, inputs); \
jb.sample_layer_statistics(mlp, 100); \
print('JAX JIT warm-up complete') \
" || echo 'WARN: JAX warm-up failed (non-fatal)'

# Warm up SciPy and Cython backends (verify import + basic call)
RUN python -c "\
import numpy as np; \
from whestbench.generation import sample_mlp; \
mlp = sample_mlp(64, 4, np.random.default_rng(42)); \
inputs = np.random.randn(100, 64).astype(np.float32); \
from whestbench.simulation_scipy import SciPyBackend; \
sb = SciPyBackend(); \
sb.run_mlp(mlp, inputs); \
sb.run_mlp_matmul_only(mlp, inputs); \
sb.sample_layer_statistics(mlp, 100); \
print('SciPy warm-up complete') \
" || echo 'WARN: SciPy warm-up failed (non-fatal)'

RUN python -c "\
import numpy as np; \
from whestbench.generation import sample_mlp; \
mlp = sample_mlp(64, 4, np.random.default_rng(42)); \
inputs = np.random.randn(100, 64).astype(np.float32); \
from whestbench.simulation_cython import CythonBackend; \
cb = CythonBackend(); \
cb.run_mlp(mlp, inputs); \
cb.run_mlp_matmul_only(mlp, inputs); \
cb.sample_layer_statistics(mlp, 100); \
print('Cython warm-up complete') \
" || echo 'WARN: Cython warm-up failed (non-fatal)'

# Copy entrypoint
COPY profiling/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
