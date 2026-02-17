from typing import Callable, Iterator, List, Tuple, TypedDict, Any, TypeVar
from circuit import Circuit, empirical_mean, random_circuit, run_batched, run_on_random
import numpy as np
from numpy.typing import NDArray

class ContestParams(TypedDict):
    width: int
    max_depth: int
    budgets: List[int]
    time_tolerance: float

T = TypeVar('T')
# returns the time used for each step
def profile_fn(fn: Callable[[], Iterator[T]]) -> Iterator[Tuple[float, T]]:
    import time
    start_time = time.time()
    for output in fn():
        yield time.time() - start_time, output
    
def sampling_baseline_time(n_samples, width, depth) -> List[float]:
    circuit = random_circuit(width, depth)
    inputs = np.random.choice([-1.0, 1.0], size=(n_samples, width)).astype(np.float16)
    return [time for time, output in profile_fn(lambda: run_batched(circuit, inputs))]

default_context_params: ContestParams = ContestParams(
    width=1000,
    max_depth=300,
    budgets = [10**i for i in range(2, 6)],
    time_tolerance=0.1,
)

def score_estimator(
    estimator: Callable[[Circuit, int], Iterator[NDArray[np.float32]]],
    n_circuits: int,
    n_samples: int,
    contest_params: ContestParams = default_context_params
) -> float:
    n = contest_params['width']
    d = contest_params['max_depth']
    tolerance = contest_params['time_tolerance']
    circuits = [random_circuit(n, d) for _ in range(n_circuits)]
    # circuits x depth x wires
    means:NDArray[np.float32] = np.array([list(empirical_mean(circuit, n_samples)) for circuit in circuits])
    # average variance for each depth
    variances = (1 - means * means).mean(axis=(0,2))
    performance_by_budget: List[float] = []
    for budget in contest_params['budgets']:
        # time for each depth
        baseline_times: NDArray[np.float32] = np.array(sampling_baseline_time(budget, n, d))
        # mse for sampling at each depth
        baseline_performance = variances / budget
        runtimes = np.zeros(d, dtype=np.float32)
        all_outputs:List[List[NDArray[np.float32]]] = []
        for circuit in circuits:
            outputs: List[NDArray[np.float32]] = []
            for i, (time, output) in enumerate(profile_fn(lambda: estimator(circuit, budget))):
                baseline_time = baseline_times[i]
                effective_time = max(time, (1 - tolerance) * baseline_time) # can't use less time than (1 - tolerance)
                effective_output = output if time <= baseline_time * (1 + tolerance) else np.zeros_like(output) # if we use more than (1 + tolerance) time, zero the output
                runtimes[i] += effective_time
                outputs.append(effective_output)
            all_outputs.append(outputs)
        estimates = np.array(all_outputs) # circuits x depth x wires
        average_times = runtimes / n_circuits
        time_ratios = average_times / baseline_times
        mse = ((estimates - means) ** 2).mean(axis=(0,2)) # depth
        adjusted_mse = mse * time_ratios
        performance_by_budget.append(np.mean(adjusted_mse))
    return sum(performance_by_budget) / len(performance_by_budget)