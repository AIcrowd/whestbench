import gc
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

# dill (used by datasets.fingerprint with recurse=True) calls dis.dis() which
# in Python 3.14 calls repr() on co_consts entries — this can recurse deeply
# through pyarrow/datasets class hierarchies. The default limit of 1000 is
# insufficient for a full test suite run; 5000 gives enough headroom.
sys.setrecursionlimit(5000)


@pytest.fixture(autouse=True)
def _isolate_import_state():
    """Restore sys.path / sys.modules after each test and GC stale classes.

    The estimator loader inserts a submission dir onto sys.path and imports
    participant modules (estimator.py and its siblings, e.g. 'helper') under
    their plain names. Across tests that leaks:
    - A stale 'helper' module shadows the next test's version.
    - Dynamically-created BaseEstimator subclasses stay alive in
      BaseEstimator.__subclasses__() even after the test ends. dill (used by
      datasets.fingerprint.generate_fingerprint with recurse=True) walks
      __subclasses__() recursively; as the list grows over hundreds of tests the
      traversal exceeds sys.getrecursionlimit() with a RecursionError.
    Removing the test modules from sys.modules and running gc.collect() lets
    Python drop the class objects, clearing them from __subclasses__().
    One estimator per process in production, so this is a test-only hazard."""
    saved_path = list(sys.path)
    saved_modules = set(sys.modules)
    try:
        yield
    finally:
        for name in set(sys.modules) - saved_modules:
            del sys.modules[name]
        sys.path[:] = saved_path
        gc.collect()
