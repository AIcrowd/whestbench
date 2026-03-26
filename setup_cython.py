"""Build script for Cython extensions."""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "network_estimation._cython_kernels",
        ["src/network_estimation/_cython_kernels.pyx"],
        include_dirs=[np.get_include()],
    ),
]

setup(ext_modules=cythonize(extensions, language_level=3))
