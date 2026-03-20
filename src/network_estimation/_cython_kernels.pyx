# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython kernels for MLP forward pass.

Uses numpy matmul (Accelerate BLAS on macOS) for matrix multiplication and
numpy's SIMD-optimized np.maximum for ReLU activation. The Cython layer
eliminates Python dispatch overhead in the loop and manages buffer reuse.
"""

import numpy as np
cimport numpy as cnp

cnp.import_array()


def forward_pass(inputs_arr, list weights):
    """Forward pass returning final-layer activations."""
    x = np.array(inputs_arr, dtype=np.float32, copy=True)
    cdef int i
    cdef int n_layers = len(weights)
    for i in range(n_layers):
        x = np.maximum(x @ weights[i], np.float32(0.0))
    return x


def forward_pass_all_layers(inputs_arr, list weights):
    """Forward pass collecting activations after every layer."""
    x = np.array(inputs_arr, dtype=np.float32, copy=True)
    layers = []
    cdef int i
    cdef int n_layers = len(weights)
    for i in range(n_layers):
        x = np.maximum(x @ weights[i], np.float32(0.0))
        layers.append(x.copy())
    return layers
