from __future__ import annotations

from itertools import product
from typing import Iterable, List

import numpy as np
from numpy.typing import NDArray

from circuit import Circuit, Layer, run_batched


def make_layer(
    first: Iterable[int],
    second: Iterable[int],
    first_coeff: Iterable[float],
    second_coeff: Iterable[float],
    const: Iterable[float],
    product_coeff: Iterable[float],
) -> Layer:
    return Layer(
        first=np.array(list(first), dtype=np.int32),
        second=np.array(list(second), dtype=np.int32),
        first_coeff=np.array(list(first_coeff), dtype=np.float32),
        second_coeff=np.array(list(second_coeff), dtype=np.float32),
        const=np.array(list(const), dtype=np.float32),
        product_coeff=np.array(list(product_coeff), dtype=np.float32),
    )


def make_circuit(n: int, layers: List[Layer]) -> Circuit:
    return Circuit(n=n, d=len(layers), gates=layers)


def exhaustive_means(circuit: Circuit) -> List[NDArray[np.float32]]:
    inputs = np.array(list(product([-1.0, 1.0], repeat=circuit.n)), dtype=np.float16)
    layer_outputs = list(run_batched(circuit, inputs))
    return [layer.astype(np.float32).mean(axis=0) for layer in layer_outputs]
