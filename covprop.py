from circuit import *
from enum import Enum
from typing import Literal


def propagate_layer(
    x_cov: NDArray[np.float32],
    x_mean: NDArray[np.float32],
    layer: Layer,
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Propagate covariance and mean through a single layer.

    Args:
        x_cov: Covariance matrix of inputs (n x n)
        x_mean: Mean vector of inputs (n,)
        layer: The layer to propagate through

    Returns:
        Tuple of (new_cov, new_mean) after propagating through the layer
    """
    n = len(x_mean)

    new_mean: NDArray[np.float32] = (
        layer.const +
        layer.first_coeff * x_mean[layer.first] +
        layer.second_coeff * x_mean[layer.second] +
        layer.product_coeff * (x_cov[layer.first, layer.second] + x_mean[layer.first] * x_mean[layer.second])
    )  # pyright: ignore

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

    # Fix diagonal: Var(y_i) = 1 - E[y_i]^2 for +-1 variables
    new_cov[np.arange(n), np.arange(n)] = 1 - new_mean * new_mean

    return new_cov, new_mean


# Runs covariance propagation, returns the estimated means at each layer.
def propagate_covariances(circuit: Circuit, verbose: bool = False) -> Tuple[List[NDArray[np.float32]], Dict[str, Any]]:
    """
    Compute the covariance matrix of the circuit outputs using covariance propagation.

    :param circuit: The Circuit object to evaluate.
    :return: The covariance matrix as a numpy array.
    """
    n: int = circuit.n
    x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
    x_cov: NDArray[np.float32] = np.eye(n, dtype=np.float32)
    result: List[NDArray[np.float32]] = [x_mean]
    covs: List[NDArray[np.float32]] = [x_cov]

    def dump_state():
        print("Mean:")
        print(x_mean)
        print("Covariance:")
        print(x_cov)

    for i, layer in enumerate(circuit.gates):
        x_cov, x_mean = propagate_layer(x_cov, x_mean, layer)
        result.append(x_mean)
        covs.append(x_cov)

    return result, {'cov': covs}

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
        C_ac * C_bd + C_ad * C_bc +
        (mu_a * mu_c) * C_bd + (mu_a * mu_d) * C_bc +
        (mu_b * mu_c) * C_ad + (mu_b * mu_d) * C_ac
    ) #pyright: ignore

# Result[i, j] = Cov(x[a[i]], x[b[j]] * x[c[j]])
def one_v_two_covariance(a: NDArray[np.int32], b: NDArray[np.int32], c: NDArray[np.int32], x_cov: NDArray[np.float32], x_mean: NDArray[np.float32]) -> NDArray[np.float32]:
    return (
        x_mean[b][None, :] * x_cov[np.ix_(a, c)] +
        x_mean[c][None, :] * x_cov[np.ix_(a, b)]
    ) #pyright: ignore

covariance_propagation:Estimator = Estimator(
    name='covprop',
    estimate=lambda circuit : propagate_covariances(circuit)
)

def clip_covariance(cov: NDArray[np.float32], mean: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Clip covariance matrix to maintain valid structure for +-1 variables.

    1. Set diagonal to 1 - E[a]^2
    2. Clip off-diagonal to [-sqrt(Var(a)*Var(b)), sqrt(Var(a)*Var(b))]
    """
    n = len(mean)

    mean = np.clip(mean, -1.0, 1.0)

    # Set diagonal: Var(a) = 1 - E[a]^2
    var = 1 - mean * mean
    cov[np.arange(n), np.arange(n)] = var

    # Compute max absolute covariance for each pair
    std = np.sqrt(np.maximum(var, 0))  # Clip to avoid sqrt of negative
    max_cov = np.outer(std, std)

    # Clip off-diagonal elements
    cov = np.clip(cov, -max_cov, max_cov)

    return cov


def clip(
    x_cov: NDArray[np.float32],
    x_mean: NDArray[np.float32],
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Clip means and covariances."""
    new_mean = np.clip(x_mean, -1.0, 1.0)
    new_cov = clip_covariance(x_cov, new_mean)
    return new_cov, new_mean


def propagate_covariances_clipped(
    circuit: Circuit,
    verbose: bool = False
) -> Tuple[List[NDArray[np.float32]], Dict[str, Any]]:
    """
    Covariance propagation with clipping after each layer.
    """
    n: int = circuit.n
    x_mean: NDArray[np.float32] = np.zeros(n, dtype=np.float32)
    x_cov: NDArray[np.float32] = np.eye(n, dtype=np.float32)
    result: List[NDArray[np.float32]] = [x_mean]
    covs: List[NDArray[np.float32]] = [x_cov]

    for i, layer in enumerate(circuit.gates):

        x_cov, x_mean = propagate_layer(x_cov, x_mean, layer)
        x_cov, x_mean = clip(x_cov, x_mean)

        result.append(x_mean)
        covs.append(x_cov)

    return result, {'cov': covs}


clipped_covariance_propagation: Estimator = Estimator(
    name='covprop_clipped',
    estimate=lambda circuit: propagate_covariances_clipped(circuit)
)