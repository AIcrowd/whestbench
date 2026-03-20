# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython kernels for MLP forward pass.

Uses numpy matmul (backed by the system BLAS — Accelerate on macOS) for
matrix multiplication and numpy's SIMD-optimized np.maximum for ReLU.
The Cython layer provides compiled loop dispatch and buffer management
with no Python interpreter overhead per layer iteration.
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()


def forward_pass(float[:, :] inputs, list weights):
    """Forward pass returning final-layer activations.

    Uses alternating x/out buffers to avoid allocations in the inner loop.
    """
    cdef int n = inputs.shape[0]
    cdef int width = inputs.shape[1]

    x = np.array(inputs, dtype=np.float32, order='C', copy=True)
    out = np.empty((n, width), dtype=np.float32, order='C')

    cdef int i
    cdef int n_layers = len(weights)
    for i in range(n_layers):
        np.matmul(x, weights[i], out=out)
        np.maximum(out, 0.0, out=out)
        x, out = out, x

    return x


def forward_pass_all_layers(float[:, :] inputs, list weights):
    """Forward pass collecting activations after every layer."""
    cdef int n = inputs.shape[0]
    cdef int width = inputs.shape[1]

    x = np.array(inputs, dtype=np.float32, order='C', copy=True)
    out = np.empty((n, width), dtype=np.float32, order='C')
    layers = []

    cdef int i
    cdef int n_layers = len(weights)
    for i in range(n_layers):
        np.matmul(x, weights[i], out=out)
        np.maximum(out, 0.0, out=out)
        layers.append(out.copy())
        x, out = out, x

    return layers
