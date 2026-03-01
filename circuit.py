"""Backward-compatible re-export surface for circuit runtime helpers."""

from tqdm import tqdm  # kept for legacy tests that monkeypatch this symbol

from circuit_estimation.domain import Circuit, Layer
from circuit_estimation.generation import random_circuit, random_gates
from circuit_estimation.simulation import empirical_mean, run_batched, run_on_random

__all__ = [
    "Circuit",
    "Layer",
    "random_gates",
    "random_circuit",
    "run_batched",
    "run_on_random",
    "empirical_mean",
    "tqdm",
]
