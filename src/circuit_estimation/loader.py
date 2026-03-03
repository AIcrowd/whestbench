"""Utilities for loading participant estimator implementations from Python files."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

from .sdk import BaseEstimator


@dataclass(frozen=True, slots=True)
class EstimatorClassMetadata:
    class_name: str
    module_name: str


def load_estimator_from_path(
    file_path: str | Path, class_name: str | None = None
) -> tuple[BaseEstimator, EstimatorClassMetadata]:
    module_path = Path(file_path).resolve()
    module = _import_module_from_path(module_path)
    estimator_class = _resolve_estimator_class(module, module_path, class_name=class_name)
    return estimator_class(), EstimatorClassMetadata(
        class_name=estimator_class.__name__,
        module_name=module.__name__,
    )


def _import_module_from_path(module_path: Path) -> ModuleType:
    if not module_path.is_file():
        raise FileNotFoundError(f"Estimator module file not found: {module_path}")

    module_hash = hashlib.sha1(str(module_path).encode("utf-8")).hexdigest()[:12]
    module_name = f"_circuit_estimation_submission_{module_hash}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import estimator module from path: {module_path}")

    module = importlib.util.module_from_spec(spec)
    loader = cast(importlib.abc.Loader, spec.loader)
    previous_module = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        loader.exec_module(module)
    except Exception:
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
        raise
    return module


def _resolve_estimator_class(
    module: ModuleType, module_path: Path, class_name: str | None
) -> type[BaseEstimator]:
    if class_name is not None:
        explicit_candidate = getattr(module, class_name, None)
        if (
            not _is_estimator_subclass(explicit_candidate)
            or explicit_candidate is BaseEstimator
        ):
            raise ValueError(
                f"Estimator class '{class_name}' was not found as a BaseEstimator subclass in {module_path}."
            )
        return cast(type[BaseEstimator], explicit_candidate)

    estimator_classes = _discover_estimator_classes(module)

    for estimator_class in estimator_classes:
        if estimator_class.__name__ == "Estimator":
            return estimator_class

    if len(estimator_classes) == 1:
        return estimator_classes[0]

    if not estimator_classes:
        raise ValueError(f"No BaseEstimator subclasses found in {module_path}.")

    class_names = ", ".join(estimator_class.__name__ for estimator_class in estimator_classes)
    raise ValueError(
        f"Ambiguous estimator classes in {module_path}: {class_names}. "
        "Pass class_name to select one explicitly."
    )


def _discover_estimator_classes(module: ModuleType) -> list[type[BaseEstimator]]:
    estimator_classes: list[type[BaseEstimator]] = []
    seen_class_ids: set[int] = set()
    for value in vars(module).values():
        if (
            _is_estimator_subclass(value)
            and value is not BaseEstimator
            and value.__module__ == module.__name__
        ):
            class_id = id(value)
            if class_id in seen_class_ids:
                continue
            seen_class_ids.add(class_id)
            estimator_classes.append(cast(type[BaseEstimator], value))
    return estimator_classes


def _is_estimator_subclass(candidate: object) -> bool:
    if not inspect.isclass(candidate) or not isinstance(candidate, type):
        return False
    try:
        return issubclass(candidate, BaseEstimator)
    except TypeError:
        return False
