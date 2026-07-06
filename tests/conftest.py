import gc
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root))

# Packages with process-global one-init state that a re-import after eviction
# trips over: numpy 2.x raises "ImportError: cannot load module more than once
# per process" when its C extension re-initialises; torch's extension registry
# is similarly single-init; `datasets` registers pyarrow extension types
# (e.g. Array2DExtensionType) in Arrow's process-global registry at import, so
# a re-import raises ArrowKeyError "already defined". Once a test pulls one of
# these in, it must stay in sys.modules for the life of the process — evicting
# it poisons every later import. Full-suite runs used to mask this only
# because some test module imported them at collection time, putting them in
# every snapshot; a single-file run (e.g. `pytest tests/test_cli_dataset.py`)
# had no such luck.
_ONE_INIT_PACKAGES = ("numpy", "torch", "pyarrow", "datasets")


def _is_one_init_module(name: str) -> bool:
    top = name.split(".", 1)[0]
    return top in _ONE_INIT_PACKAGES


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
    One-init-per-process C extensions (see ``_ONE_INIT_PACKAGES``) are exempt
    from the drop: re-importing them after eviction is a hard error, so they
    stay put once loaded.
    """
    saved_path = list(sys.path)
    saved_modules = dict(sys.modules)
    try:
        yield
    finally:
        for name in set(sys.modules) - set(saved_modules):
            if _is_one_init_module(name):
                continue
            del sys.modules[name]
        for name, module in saved_modules.items():
            if sys.modules.get(name) is not module:
                sys.modules[name] = module
        sys.path[:] = saved_path
        gc.collect()
