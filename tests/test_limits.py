"""Submission caps: single source of truth, shared with the evaluator (Plan C)."""

import importlib
import sys

from whestbench import limits


def test_caps_values():
    assert limits.MAX_SUBMISSION_BYTES == 50 * 1024 * 1024
    assert limits.MAX_SUBMISSION_FILES == 50


def test_limits_module_is_import_light():
    # Must not pull flopscope/datasets so the evaluator's constrained venv can import it.
    evicted = {name: sys.modules.get(name) for name in ("flopscope", "datasets")}
    try:
        for name in evicted:
            sys.modules.pop(name, None)
        importlib.reload(limits)
        assert "flopscope" not in sys.modules
        assert "datasets" not in sys.modules
    finally:
        # Restore exactly what we evicted. A half-reimported 'datasets' corrupts
        # dill fingerprinting (table.array_cast self-reference) for later tests.
        for name, module in evicted.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
