#!/usr/bin/env python3
"""Benchmark NumPy matmul: MKL vs OpenBLAS, plus PyTorch comparison."""
import time
import os
import numpy as np

# Thread config
threads = int(os.environ.get("BENCH_THREADS", "16"))
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

print("=" * 60)
print("NumPy %s" % np.__version__)
print("Threads: %d" % threads)

# Detect BLAS
try:
    from threadpoolctl import threadpool_info
    for lib in threadpool_info():
        api = lib.get("internal_api", "")
        prefix = lib.get("prefix", "")
        ver = lib.get("version", "?")
        nt = lib.get("num_threads", "?")
        if "blas" in lib.get("user_api", ""):
            print("BLAS: %s (%s) v%s threads=%s" % (prefix, api, ver, nt))
except ImportError:
    print("(threadpoolctl not installed)")

try:
    import torch
    print("PyTorch %s" % torch.__version__)
    torch.set_num_threads(threads)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

print("=" * 60)

rng = np.random.default_rng(42)

# === Test 1: Single large matmul ===
print("\n--- Single matmul (4096x256 @ 256x256) ---")
A = rng.standard_normal((4096, 256), dtype=np.float32)
B = rng.standard_normal((256, 256), dtype=np.float32)

# Warmup
for _ in range(20):
    _ = A @ B

N = 1000
t0 = time.perf_counter()
for _ in range(N):
    _ = A @ B
np_t = (time.perf_counter() - t0) / N
print("  numpy:   %.1f us" % (np_t * 1e6))

if HAS_TORCH:
    At = torch.from_numpy(A)
    Bt = torch.from_numpy(B)
    for _ in range(20):
        _ = At @ Bt
    t0 = time.perf_counter()
    for _ in range(N):
        _ = At @ Bt
    pt_t = (time.perf_counter() - t0) / N
    print("  pytorch: %.1f us" % (pt_t * 1e6))
    print("  ratio:   %.2fx (>1 = pytorch faster)" % (np_t / pt_t))

# === Test 2: 128 sequential matmuls (MLP-like) ===
print("\n--- 128 sequential matmuls (4096x256 @ 256x256) ---")
weights = [rng.standard_normal((256, 256), dtype=np.float32) for _ in range(128)]

# Warmup
for _ in range(3):
    x = A.copy()
    for w in weights:
        x = x @ w

N = 20
t0 = time.perf_counter()
for _ in range(N):
    x = A.copy()
    for w in weights:
        x = x @ w
np_chain = (time.perf_counter() - t0) / N
print("  numpy:   %.1f ms" % (np_chain * 1e3))

if HAS_TORCH:
    weights_t = [torch.from_numpy(w) for w in weights]
    At = torch.from_numpy(A)
    for _ in range(3):
        x = At.clone()
        for w in weights_t:
            x = x @ w

    t0 = time.perf_counter()
    for _ in range(N):
        x = torch.from_numpy(A)
        for w in weights_t:
            x = x @ w
        _ = x.numpy()
    pt_chain = (time.perf_counter() - t0) / N
    print("  pytorch: %.1f ms" % (pt_chain * 1e3))
    print("  ratio:   %.2fx (>1 = pytorch faster)" % (np_chain / pt_chain))

# === Test 3: Various matrix sizes ===
print("\n--- Matrix size sweep (single matmul, avg of 500 calls) ---")
sizes = [
    (4096, 64, 64),
    (4096, 128, 128),
    (4096, 256, 256),
    (4096, 512, 512),
    (4096, 1024, 1024),
]
for m, k, n in sizes:
    A = rng.standard_normal((m, k), dtype=np.float32)
    B = rng.standard_normal((k, n), dtype=np.float32)
    # warmup
    for _ in range(10):
        _ = A @ B
    iters = 500
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = A @ B
    t = (time.perf_counter() - t0) / iters
    gflops = 2 * m * k * n / t / 1e9
    print("  %dx%d @ %dx%d: %8.1f us  (%.1f GFLOPS)" % (m, k, k, n, t * 1e6, gflops))

# === Test 4: Thread scaling ===
print("\n--- Thread scaling (4096x256 @ 256x256, single matmul) ---")
A = rng.standard_normal((4096, 256), dtype=np.float32)
B = rng.standard_normal((256, 256), dtype=np.float32)

try:
    from threadpoolctl import threadpool_limits
    for t in [1, 2, 4, 8, 16]:
        with threadpool_limits(limits=t):
            # warmup
            for _ in range(10):
                _ = A @ B
            iters = 500
            t0 = time.perf_counter()
            for _ in range(iters):
                _ = A @ B
            elapsed = (time.perf_counter() - t0) / iters
            print("  %2d threads: %8.1f us" % (t, elapsed * 1e6))
except ImportError:
    print("  (threadpoolctl not installed, skipping)")

print("\nDone.")
