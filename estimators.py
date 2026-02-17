from circuit import *

from typing import Iterator


def mean_propagation(circuit: Circuit) -> Iterator[NDArray[np.float32]]:
    """
    Compute the mean output of the circuit using mean propagation.

    :param circuit: The Circuit object to evaluate.
    :return: The mean output as a numpy array.
    """
    x_mean: NDArray[np.float32] = np.zeros(circuit.n, dtype=np.float32)
    for layer in circuit.gates:
        x_mean = (
            layer.first_coeff * np.take(x_mean, layer.first)  +
            layer.second_coeff * np.take(x_mean, layer.second) +
            layer.const + 
            layer.product_coeff * np.take(x_mean, layer.first) * np.take(x_mean, layer.second)
        ) #pyright: ignore
        yield x_mean

# Result[i, j] = Cov(x[a[i]], x[b[j]] * x[c[j]])
def one_v_two_covariance(a: NDArray[np.int32], b: NDArray[np.int32], c: NDArray[np.int32], x_cov: NDArray[np.float32], x_mean: NDArray[np.float32]) -> NDArray[np.float32]:
    return (
        x_mean[b][None, :] * x_cov[np.ix_(a, c)] +
        x_mean[c][None, :] * x_cov[np.ix_(a, b)]
    ) #pyright: ignore

# Result[i, j] = Cov(x[a[i]] * x[b[i]], x[c[j]] * x[d[j]])
def two_v_two_covariance(a: NDArray[np.int32], b: NDArray[np.int32], c: NDArray[np.int32], d: NDArray[np.int32], cov: NDArray[np.float32], mean: NDArray[np.float32]) -> NDArray[np.float32]:
    C_ac = cov[np.ix_(a, c)]
    C_ad = cov[np.ix_(a, d)]
    C_bc = cov[np.ix_(b, c)]
    C_bd = cov[np.ix_(b, d)]

    # Means, shaped for broadcasting across columns
    mu_a = mean[a][:, None]  # (k,1)
    mu_b = mean[b][:, None]  # (k,1)
    mu_c = mean[c][None, :]  # (1,k)
    mu_d = mean[d][None, :]  # (1,k)

    return (
        (mu_a * mu_c) * C_bd + (mu_a * mu_d) * C_bc +
        (mu_b * mu_c) * C_ad + (mu_b * mu_d) * C_ac
    ) #pyright: ignore

def clip(mean: NDArray[np.float32], cov: NDArray[np.float32]) -> None:
    n = len(mean)
    np.clip(mean, -1.0, 1.0, out=mean)
    var = 1 - mean * mean
    cov[np.arange(n), np.arange(n)] = var

    # Compute max absolute covariance for each pair
    std = np.sqrt(var)
    max_cov = np.outer(std, std)

    # Clip off-diagonal elements
    np.clip(cov, -max_cov, max_cov, out=cov)

def covariance_propagation(circuit: Circuit) -> Iterator[NDArray[np.float32]]:
    """
    Compute the covariance matrix of the circuit outputs using covariance propagation.

    :param circuit: The Circuit object to evaluate.
    :return: The covariance matrix as a numpy array.
    """
    n: int = circuit.n
    x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
    x_cov: NDArray[np.float32] = np.eye(n, dtype=np.float32)
    for layer in circuit.gates:

        # --- Compute means

        new_mean: NDArray[np.float32] = (
            layer.first_coeff * x_mean[layer.first] +
            layer.second_coeff * x_mean[layer.second] +
            layer.const + 
            layer.product_coeff * x_mean[layer.first] * x_mean[layer.second] +
            layer.product_coeff * x_cov[layer.first, layer.second]
        ) #pyright: ignore

        # --- Compute covariances

        new_cov: NDArray[np.float32] = np.zeros((n, n), dtype=np.float32)
        # 1v1 terms: Cov(linear, linear)
        new_cov += np.outer(layer.first_coeff, layer.first_coeff) * x_cov[np.ix_(layer.first, layer.first)]
        new_cov += np.outer(layer.second_coeff, layer.second_coeff) * x_cov[np.ix_(layer.second, layer.second)]
        new_cov += np.outer(layer.first_coeff, layer.second_coeff) * x_cov[np.ix_(layer.first, layer.second)]
        new_cov += np.outer(layer.second_coeff, layer.first_coeff) * x_cov[np.ix_(layer.second, layer.first)]
        # 1v2 terms: Cov(linear, product) - symmetric contribution
        result_1v2_first: NDArray[np.float32] = np.outer(layer.first_coeff, layer.product_coeff) * one_v_two_covariance(
            layer.first, layer.first, layer.second, x_cov, x_mean
        )
        new_cov += result_1v2_first + result_1v2_first.T
        result_1v2_second: NDArray[np.float32] = np.outer(layer.second_coeff, layer.product_coeff) * one_v_two_covariance(
            layer.second, layer.first, layer.second, x_cov, x_mean
        )
        new_cov += result_1v2_second + result_1v2_second.T
        # 2v2 terms: Cov(product, product)
        new_cov += np.outer(layer.product_coeff, layer.product_coeff) * two_v_two_covariance(
            layer.first, layer.second, layer.first, layer.second, x_cov, x_mean
        )

        # --- Clip and update covariances

        clip(new_mean, new_cov)
        x_mean, x_cov = new_mean, new_cov

        yield x_mean

def combined_estimator(circuit: Circuit, budget: int) -> Iterator[NDArray[np.float32]]:
    if budget >= 30*circuit.n: #Rough guess for how much budget we need.
        yield from covariance_propagation(circuit)
    else:
        yield from mean_propagation(circuit)