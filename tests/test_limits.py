"""Submission caps: single source of truth, shared with the evaluator (Plan C)."""

import importlib
import sys

from whestbench import limits


def test_caps_values():
    assert limits.MAX_SUBMISSION_BYTES == 50 * 1024 * 1024
    assert limits.MAX_SUBMISSION_FILES == 50


def test_limits_module_is_import_light():
    # Must not pull flopscope/datasets so the evaluator's constrained venv can import it.
    for mod in ("flopscope", "datasets"):
        sys.modules.pop(mod, None)
    importlib.reload(limits)
    assert "flopscope" not in sys.modules
