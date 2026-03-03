import pickle
import sys
from pathlib import Path
from textwrap import dedent

import pytest

from circuit_estimation.loader import load_estimator_from_path


def _write_estimator_module(tmp_path: Path, source: str) -> Path:
    module_path = tmp_path / "submission.py"
    module_path.write_text(dedent(source), encoding="utf-8")
    return module_path


def test_loader_prefers_default_estimator_class_name(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Alternative(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
                return np.zeros((1, 1), dtype=np.float32)

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
                return np.ones((1, 1), dtype=np.float32)
        """,
    )

    estimator, metadata = load_estimator_from_path(module_path)

    assert estimator.__class__.__name__ == "Estimator"
    assert metadata.class_name == "Estimator"


def test_loader_allows_explicit_class_override(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
                return np.ones((1, 1), dtype=np.float32)

        class CustomEstimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
                return np.full((1, 1), 2.0, dtype=np.float32)
        """,
    )

    estimator, metadata = load_estimator_from_path(module_path, class_name="CustomEstimator")

    assert estimator.__class__.__name__ == "CustomEstimator"
    assert metadata.class_name == "CustomEstimator"


def test_loader_errors_on_ambiguous_multiple_classes(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class AlphaEstimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
                return np.zeros((1, 1), dtype=np.float32)

        class BetaEstimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
                return np.zeros((1, 1), dtype=np.float32)
        """,
    )

    with pytest.raises(ValueError, match="Ambiguous estimator classes"):
        load_estimator_from_path(module_path)


def test_loader_deduplicates_alias_bindings_for_single_estimator_class(
    tmp_path: Path,
) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class CustomEstimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
                return np.ones((1, 1), dtype=np.float32)

        AliasEstimator = CustomEstimator
        """,
    )

    estimator, metadata = load_estimator_from_path(module_path)

    assert estimator.__class__.__name__ == "CustomEstimator"
    assert metadata.class_name == "CustomEstimator"


def test_loader_registers_module_in_sys_modules(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int) -> np.ndarray:
                return np.zeros((1, 1), dtype=np.float32)
        """,
    )

    estimator, metadata = load_estimator_from_path(module_path)

    assert metadata.module_name in sys.modules
    assert estimator.__class__.__module__ == metadata.module_name
    # Module registration should support pickling estimator instances.
    payload = pickle.dumps(estimator)
    restored = pickle.loads(payload)
    assert restored.__class__.__name__ == estimator.__class__.__name__


def test_loader_ignores_numpy_ndarray_generic_alias_in_module_globals(tmp_path: Path) -> None:
    module_path = _write_estimator_module(
        tmp_path,
        """
        import numpy as np
        from numpy.typing import NDArray
        from circuit_estimation import BaseEstimator, Circuit

        class Estimator(BaseEstimator):
            def predict(self, circuit: Circuit, budget: int) -> NDArray[np.float32]:
                return np.zeros((1, 1), dtype=np.float32)
        """,
    )

    estimator, metadata = load_estimator_from_path(module_path)

    assert estimator.__class__.__name__ == "Estimator"
    assert metadata.class_name == "Estimator"
