from circuit import *

# Runs mean propagation, returns the estimated means at each layer.
def propagate_means(circuit: Circuit, verbose:bool = False) -> Tuple[List[NDArray[np.float32]], Dict[str, Any]]:
    """
    Compute the mean output of the circuit using mean propagation.

    :param circuit: The Circuit object to evaluate.
    :return: The mean output as a numpy array.
    """
    n: int = circuit.n
    x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
    result: List[NDArray[np.float32]] = [x_mean]
    if verbose: print("Initial mean:", x_mean)
    for layer in circuit.gates:
        x_mean = (
            layer.first_coeff * x_mean[layer.first] +
            layer.second_coeff * x_mean[layer.second] +
            layer.const + 
            layer.product_coeff * x_mean[layer.first] * x_mean[layer.second]
        ) #pyright: ignore
        if verbose: print("Mean propagation after layer:", x_mean)
        result.append(x_mean)
    return result, {}

mean_propagation = Estimator(
    name='meanprop',
    estimate=propagate_means
)