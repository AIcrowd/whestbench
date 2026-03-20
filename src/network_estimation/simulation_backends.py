"""Backend registry for simulation backends."""

from __future__ import annotations
import os
from typing import Dict, Optional, Type
from .simulation_backend import SimulationBackend
from .simulation_numpy import NumPyBackend


def _lazy_backends() -> Dict[str, Type[SimulationBackend]]:
    backends: Dict[str, Type[SimulationBackend]] = {"numpy": NumPyBackend}
    try:
        from .simulation_pytorch import PyTorchBackend
        backends["pytorch"] = PyTorchBackend
    except Exception:
        pass
    try:
        from .simulation_numba import NumbaBackend
        backends["numba"] = NumbaBackend
    except Exception:
        pass
    try:
        from .simulation_scipy import SciPyBackend
        backends["scipy"] = SciPyBackend
    except Exception:
        pass
    try:
        from .simulation_jax import JAXBackend
        backends["jax"] = JAXBackend
    except Exception:
        pass
    try:
        from .simulation_cython import CythonBackend
        backends["cython"] = CythonBackend
    except Exception:
        pass
    return backends


ALL_BACKEND_NAMES = ("numpy", "pytorch", "numba", "scipy", "jax", "cython")


def get_available_backends() -> Dict[str, Type[SimulationBackend]]:
    return {k: v for k, v in _lazy_backends().items() if v.is_available()}


def get_backend(name: Optional[str] = None) -> SimulationBackend:
    if name is None:
        name = os.environ.get("NESTIM_BACKEND", "numpy")
    if name not in ALL_BACKEND_NAMES:
        raise ValueError(f"Unknown backend: {name!r}. Valid backends: {list(ALL_BACKEND_NAMES)}")
    backends = _lazy_backends()
    cls = backends.get(name)
    if cls is None or not cls.is_available():
        hint = ""
        if cls is not None:
            hint = cls.install_hint()
        raise RuntimeError(f"Backend {name!r} is not available." + (f" Install: {hint}" if hint else ""))
    return cls()
