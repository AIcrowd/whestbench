from dataclasses import dataclass
import numpy as np
import random
from numpy.typing import NDArray
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeAlias
from tqdm import tqdm

@dataclass(slots=True)
class Layer:
    first: NDArray[np.int32]
    second: NDArray[np.int32]
    first_coeff: NDArray[np.float32]
    second_coeff: NDArray[np.float32]
    const: NDArray[np.float32]
    product_coeff: NDArray[np.float32]

@dataclass(slots=True)
class Circuit:
    n: int  # Number of inputs
    d: int  # Number of layers
    gates: List[Layer]

@dataclass(slots=True)
class Op:
    const: float = 0.0
    first_coeff: float = 0.0
    second_coeff: float = 0.0
    product_coeff: float = 0.0

def random_op() -> Op:
    """
    Generate an Op representing one of the 16 binary boolean operations at random.
    """
    is_simple:bool = np.random.choice([True, False])
    if is_simple: # Generate +- one of {x, y, 1, or xy} at random.
        sign:int = np.random.choice([-1, 1])
        basic_ops:List[Op] = [
            Op(first_coeff=sign), Op(second_coeff=sign), Op(const=sign), Op(product_coeff=sign)
        ]
        return random.choice(basic_ops)
    else: # Generate +- AND of (+-x, +-y)
        x_coeff:float = np.random.choice([-1, 1])
        y_coeff:float = np.random.choice([-1, 1])
        coeff:float = np.random.choice([-1, 1]) * 0.5
        return Op(
            const=-1 * coeff,
            first_coeff=x_coeff*coeff,
            second_coeff=y_coeff*coeff,
            product_coeff=x_coeff * y_coeff * coeff
        )

def random_gates(n: int, rng: Optional[np.random.Generator] = None) -> Layer:
    """
    Generate a random Layer for the circuit with n inputs.

    :param n: Number of inputs to the layer.
    :param rng: Optional random number generator for reproducibility.
    :return: A Layer object with random gates.
    """
    if rng is None:
        rng = np.random.default_rng()
    # Vectorized generation of operations
    is_simple = rng.choice([True, False], size=n)
    
    # Initialize arrays
    const = np.zeros(n, dtype=np.float32)
    first_coeff = np.zeros(n, dtype=np.float32)
    second_coeff = np.zeros(n, dtype=np.float32)
    product_coeff = np.zeros(n, dtype=np.float32)
    
    # Simple operations: Generate +- one of {x, y, 1, or xy}
    n_simple = np.sum(is_simple)
    if n_simple > 0:
        sign = rng.choice([-1, 1], size=n_simple)
        # Choose which basic operation: 0=first_coeff, 1=second_coeff, 2=const, 3=product_coeff
        op_type = rng.integers(0, 4, size=n_simple)
        
        simple_mask = is_simple
        const[simple_mask] = (op_type == 2) * sign
        first_coeff[simple_mask] = (op_type == 0) * sign
        second_coeff[simple_mask] = (op_type == 1) * sign
        product_coeff[simple_mask] = (op_type == 3) * sign
    
    # Complex operations: Generate +- AND of (+-x, +-y)
    n_complex = n - n_simple
    if n_complex > 0:
        complex_mask = ~is_simple
        x_coeff = rng.choice([-1, 1], size=n_complex)
        y_coeff = rng.choice([-1, 1], size=n_complex)
        coeff = rng.choice([-1, 1], size=n_complex) * 0.5
        
        const[complex_mask] = -coeff
        first_coeff[complex_mask] = x_coeff * coeff
        second_coeff[complex_mask] = y_coeff * coeff
        product_coeff[complex_mask] = x_coeff * y_coeff * coeff
    
    # Generate input indices
    first:NDArray[np.int32] = rng.integers(0, n, size=(n,), dtype=np.int32)
    second_raw:NDArray[np.int32] = rng.integers(0, n-1, size=(n,), dtype=np.int32)
    # Ensure no gate uses the same input twice
    second:NDArray[np.int32] = (second_raw + (second_raw >= first).astype(np.int32)).astype(np.int32)
    
    return Layer(
        first=first,
        second=second,
        const=const,
        first_coeff=first_coeff,
        second_coeff=second_coeff,
        product_coeff=product_coeff,
    )

def random_circuit(n: int, d: int, rng:Optional[np.random.Generator] = None) -> Circuit:
    if rng is None:
        rng = np.random.default_rng()
    return Circuit(
        n=n,
        d=d,
        gates=[random_gates(n, rng) for _ in range(d)]
    )

def run_batched(circuit, inputs: NDArray[np.float16]) -> List[NDArray[np.float16]]:
    """
    Execute the circuit on batched inputs.

    :param circuit: The Circuit object to execute.
    :param inputs: Batched input values as a numpy array of shape (batch, ...).
    :return: Batched output values after executing the circuit.
    """
    # inputs shape: (batch, n_values)
    x: NDArray[np.float16] = inputs  # (B, N)
    result: List[NDArray[np.float16]] = [x]
    for layer in tqdm(circuit.gates):
        # Each x[layer.first] and x[layer.second] now has shape (B,)
        x = (
            layer.const +
            layer.first_coeff * x[:, layer.first] +
            layer.second_coeff * x[:, layer.second] +
            layer.product_coeff * x[:, layer.first] * x[:, layer.second]
        )
        result.append(x)
    return result

def run_on_random(circuit: Circuit, trials: int) -> List[NDArray[np.float16]]:
    """
    Execute the circuit on random inputs.

    :param circuit: The Circuit object to execute.
    :param trials: Number of random input trials to perform.
    :return: Batched output values after executing the circuit.
    """
    n: int = circuit.n
    inputs: NDArray[np.float16] = np.random.choice([-1.0, 1.0], size=(trials, n))
    return run_batched(circuit, inputs)

# Returns empirical means at each layer.
def empirical_mean(circuit: Circuit, trials: int) -> List[NDArray[np.float32]]:
    """
    Compute the empirical mean output of the circuit over a number of random trials.

    :param circuit: The Circuit object to evaluate.
    :param trials: The number of random input trials to perform.
    :return: The empirical mean output as a numpy array.
    """
    outputs: List[NDArray[np.float16]] = run_on_random(circuit, trials)
    return [np.mean(output.astype(np.float32), axis=0) for output in outputs]

Info:TypeAlias = Dict[str, Any]

@dataclass
class Estimator:
    name: str
    estimate: Callable[[Circuit], Tuple[List[NDArray[np.float32]], Info]]