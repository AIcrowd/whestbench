"""Backend registry for simulation backends."""

from __future__ import annotations

import os
from typing import Dict, Optional, Type

from .simulation_backend import SimulationBackend
from .simulation_mechestim import MechestimBackend


def _lazy_backends() -> Dict[str, Type[SimulationBackend]]:
    return {"mechestim": MechestimBackend}


ALL_BACKEND_NAMES = ("mechestim",)

INSTALL_HINTS: Dict[str, str] = {
    "mechestim": "pip install git+https://github.com/AIcrowd/mechestim.git",
}


def get_available_backends() -> Dict[str, Type[SimulationBackend]]:
    return {k: v for k, v in _lazy_backends().items() if v.is_available()}


def get_backend(name: Optional[str] = None) -> SimulationBackend:
    if name is None:
        name = os.environ.get("WHEST_BACKEND", "mechestim")
    if name not in ALL_BACKEND_NAMES:
        raise ValueError(f"Unknown backend: {name!r}. Valid backends: {list(ALL_BACKEND_NAMES)}")
    backends = _lazy_backends()
    cls = backends.get(name)
    if cls is None or not cls.is_available():
        hint = INSTALL_HINTS.get(name, "")
        raise RuntimeError(
            f"Backend {name!r} is not available." + (f" Install: {hint}" if hint else "")
        )
    return cls()
