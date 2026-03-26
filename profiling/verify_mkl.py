"""Verify that NumPy is linked against Intel MKL. Fails the Docker build if not."""
import numpy as np

print('NumPy %s' % np.__version__)

from threadpoolctl import threadpool_info

all_libs = threadpool_info()
print('Detected BLAS libs:')
for lib in all_libs:
    print('  %s  api=%s  internal=%s' % (
        lib.get('prefix', '?'), lib.get('user_api', '?'), lib.get('internal_api', '?')))

has_mkl = any(
    'mkl' in lib.get('internal_api', '').lower() or 'mkl' in lib.get('prefix', '').lower()
    for lib in all_libs
)
assert has_mkl, 'Expected MKL but not found. Libs: %s' % [
    (lib.get('prefix'), lib.get('internal_api')) for lib in all_libs
]
print('BLAS verification PASSED: MKL detected')
