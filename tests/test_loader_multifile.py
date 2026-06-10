"""A multi-file estimator (estimator.py imports a sibling module) must load."""

import textwrap
from pathlib import Path

from whestbench.loader import load_estimator_from_path


def test_estimator_can_import_sibling_module(tmp_path: Path):
    (tmp_path / "helper.py").write_text(
        "import flopscope.numpy as fnp\n"
        "def zeros(mlp):\n    return fnp.zeros((mlp.depth, mlp.width))\n"
    )
    (tmp_path / "estimator.py").write_text(
        textwrap.dedent("""
        from whestbench import BaseEstimator, MLP
        from helper import zeros
        class Estimator(BaseEstimator):
            def predict(self, mlp, budget):
                return zeros(mlp)
    """)
    )
    est, meta = load_estimator_from_path(tmp_path / "estimator.py")
    assert meta.class_name == "Estimator"
