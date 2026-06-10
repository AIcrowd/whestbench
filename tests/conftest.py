import gc
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))


@pytest.fixture(autouse=True)
def _isolate_import_state():
    """Restore sys.path and sys.modules to their pre-test state after each test.

    Two test-only import hazards this guards against:

    1. The estimator loader inserts a submission dir onto sys.path and imports
       participant modules (estimator.py and its siblings, e.g. 'helper') under
       their plain names. Left in place, a stale 'helper' shadows the next
       test's version and a leaked sys.path entry resolves the wrong file.
    2. A test may evict or replace a module to probe import behaviour
       (test_limits pops 'datasets'/'flopscope' to assert whestbench.limits is
       import-light). A half-reimported 'datasets' leaves dill walking the
       self-referential 'datasets.table.array_cast' decorator without hitting
       its dedup guard, which then recurses past the limit inside a later
       create_dataset() fingerprint.

    Snapshotting module *objects* (not just names) lets us both drop modules
    imported during the test and restore any the test evicted or swapped.
    """
    saved_path = list(sys.path)
    saved_modules = dict(sys.modules)
    try:
        yield
    finally:
        for name in set(sys.modules) - set(saved_modules):
            del sys.modules[name]
        for name, module in saved_modules.items():
            if sys.modules.get(name) is not module:
                sys.modules[name] = module
        sys.path[:] = saved_path
        gc.collect()
