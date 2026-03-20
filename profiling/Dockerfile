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
