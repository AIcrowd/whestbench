# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython kernels for MLP forward pass using BLAS sgemm."""

import numpy as np
cimport numpy as cnp
from scipy.linalg.cython_blas cimport sgemm

cnp.import_array()


def forward_pass(float[:, :] inputs, list weights):
    cdef int n = inputs.shape[0]
    cdef int width = inputs.shape[1]
    cdef float alpha = 1.0
    cdef float beta = 0.0
    cdef int i, j

    x = np.array(inputs, dtype=np.float32, order='C', copy=True)
    cdef float[:, :] x_view = x
    out = np.empty((n, width), dtype=np.float32, order='C')
    cdef float[:, :] out_view = out

    for w_np in weights:
        w_view = np.ascontiguousarray(w_np, dtype=np.float32)
        cdef float[:, :] wv = w_view
        sgemm(b"N", b"N", &width, &n, &width, &alpha, &wv[0, 0], &width, &x_view[0, 0], &width, &beta, &out_view[0, 0], &width)
        for i in range(n):
            for j in range(width):
                if out_view[i, j] < 0.0:
                    out_view[i, j] = 0.0
        x_view, out_view = out_view, x_view
        x, out = out, x

    return np.asarray(x_view)


def forward_pass_all_layers(float[:, :] inputs, list weights):
    cdef int n = inputs.shape[0]
    cdef int width = inputs.shape[1]
    cdef float alpha = 1.0
    cdef float beta = 0.0
    cdef int i, j

    x = np.array(inputs, dtype=np.float32, order='C', copy=True)
    cdef float[:, :] x_view = x
    out = np.empty((n, width), dtype=np.float32, order='C')
    cdef float[:, :] out_view = out
    layers = []

    for w_np in weights:
        w_view = np.ascontiguousarray(w_np, dtype=np.float32)
        cdef float[:, :] wv = w_view
        sgemm(b"N", b"N", &width, &n, &width, &alpha, &wv[0, 0], &width, &x_view[0, 0], &width, &beta, &out_view[0, 0], &width)
        for i in range(n):
            for j in range(width):
                if out_view[i, j] < 0.0:
                    out_view[i, j] = 0.0
        layers.append(np.asarray(out_view).copy())
        x_view, out_view = out_view, x_view
        x, out = out, x

    return layers
