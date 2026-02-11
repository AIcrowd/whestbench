from circuit import *
from itertools import chain
from meanprop import mean_propagation
from investigate import *
from covprop import clipped_covariance_propagation
import argparse
import os

# Returns the mean squared error from using n_trials samples to estimate each entry of a +-1 vector with the given means
def sampling_baseline_error(means: NDArray[np.float32], n_trials:int = 1) -> float:
    return np.mean(1 - means * means) / n_trials #pyright: ignore

Errors: TypeAlias = Dict[str, float]

def test_all_depths(
        circuit:Circuit, trials: int, estimators: List[Estimator],
        empirical: Optional[List[NDArray[np.float32]]] = None
) ->Tuple[List[Errors], Dict[str, List[NDArray[np.float32]]], Dict[str, Info]]:
    if empirical is None:
        empirical = empirical_mean(circuit, trials)
    estimator_outputs = [estimator.estimate(circuit) for estimator in estimators]
    estimates: Dict[str, List[NDArray[np.float32]]] = {estimator.name: output[0] for estimator, output in zip(estimators, estimator_outputs)}
    estimator_infos: Dict[str, Dict[str, Any]] = {estimator.name: output[1] for estimator, output in zip(estimators, estimator_outputs)}

    errors: List[Errors] = []
    for i in range(len(empirical)):
        error:Errors = dict(
            sampling=sampling_baseline_error(empirical[i]),
        )
        for name in estimates.keys():
            error[name] = float(np.mean((empirical[i] - estimates[name][i]) ** 2))
        errors.append(error)
    estimates['empirical'] = empirical
    return errors, estimates, estimator_infos

if __name__ == "__main__":
    master_seed = 42
    np.random.seed(master_seed)
    print(f"Using master seed: {master_seed}")
    n: int = 100
    d: int = 100
    trials: int = 10000
    num_circuits: int = 10
    # Generate random seeds for each circuit
    circuit_seeds: List[int] = [int(np.random.randint(0, 1000000)) for _ in range(num_circuits)]

    estimators: List[Estimator] = [clipped_covariance_propagation, mean_propagation]
    all_names = ['sampling'] + [e.name for e in estimators]

    # Collect results per circuit, using cache where available
    all_circuit_results: List[List[Errors]] = []

    for i, seed in enumerate(circuit_seeds):
        print(f"Processing circuit {i+1}/{num_circuits} (seed={seed})")

        rng = np.random.default_rng(seed)
        circuit = random_circuit(n, d, rng)

        errors, estimates, infos = test_all_depths(circuit, trials, estimators)
        all_circuit_results.append(errors)

    # Transpose the data and get the average error for each estimator at each depth
    errors_by_depth: Dict[str, List[float]] = {name: [] for name in all_names}
    for depth in range(d + 1):
        depth_errors = {name: [] for name in all_names}
        for circuit_results in all_circuit_results:
            for name in all_names:
                depth_errors[name].append(circuit_results[depth][name])
        mean_depth_errors = {name: float(np.mean(depth_errors[name])) for name in all_names}
        for name in all_names:
            errors_by_depth[name].append(mean_depth_errors[name])

    # Make a plot with one line for each estimator, showing error vs depth.
    import matplotlib.pyplot as plt
    for name in all_names:
        plt.plot(errors_by_depth[name], label=name)
    plt.yscale('log')
    plt.xlabel('Depth')
    plt.ylabel('Mean Squared Error')
    plt.title('Estimator Error vs Depth')
    plt.legend()
    plt.show()