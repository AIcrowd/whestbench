"""Build script for Cython extensions."""

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

extensions = [
    Extension(
        "network_estimation._cython_kernels",
        ["src/network_estimation/_cython_kernels.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(ext_modules=cythonize(extensions, language_level=3))
