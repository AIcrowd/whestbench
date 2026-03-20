"""JAX CPU backend for MLP forward pass simulation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray

from .domain import MLP
from .simulation_backend import SimulationBackend

try:
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_platform_name", "cpu")
    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _pick_chunk_size(width: int) -> int:
    """Choose chunk size targeting a 2-8 MB working set for L2/L3 cache."""
    return max(1024, min(16384, 2**20 // width))


if _HAS_JAX:

    @jax.jit
    def _jax_forward(inputs, weights):
        """JIT-compiled forward pass through all layers with ReLU."""
        x = inputs
        for w in weights:
            x = jnp.maximum(x @ w, 0.0)
        return x


class JAXBackend(SimulationBackend):
    @property
    def name(self) -> str:
        return "jax"

    @classmethod
    def is_available(cls) -> bool:
        return _HAS_JAX

    @classmethod
    def install_hint(cls) -> str:
        return "pip install 'jax[cpu]>=0.4'"

    def run_mlp(self, mlp: MLP, inputs: NDArray[np.float32]) -> NDArray[np.float32]:
        jax_weights = [jnp.array(w) for w in mlp.weights]
        result = _jax_forward(inputs, jax_weights)
        return np.asarray(result, dtype=np.float32)

    def run_mlp_all_layers(
        self, mlp: MLP, inputs: NDArray[np.float32]
    ) -> List[NDArray[np.float32]]:
        x = jnp.array(inputs)
        layers: List[NDArray[np.float32]] = []
        for w in mlp.weights:
            x = jnp.maximum(x @ jnp.array(w), 0.0)
            layers.append(np.asarray(x, dtype=np.float32))
        return layers

    def output_stats(
        self, mlp: MLP, n_samples: int
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32], float]:
        width = mlp.width
        depth = mlp.depth
        chunk_size = _pick_chunk_size(width)

        jax_weights = [jnp.array(w) for w in mlp.weights]

        layer_sums = jnp.zeros((depth, width), dtype=jnp.float32)
        final_sum_sq = jnp.zeros(width, dtype=jnp.float32)

        # Seed JAX PRNG from numpy RNG for reproducibility
        seed = int(np.random.default_rng().integers(0, 2**31))
        key = jax.random.PRNGKey(seed)

        n_processed = 0
        for start in range(0, n_samples, chunk_size):
            n = min(chunk_size, n_samples - start)
            key, subkey = jax.random.split(key)
            x = jax.random.normal(subkey, shape=(n, width), dtype=jnp.float32)

            for layer_idx, w in enumerate(jax_weights):
                x = jnp.maximum(x @ w, 0.0)
                layer_sums = layer_sums.at[layer_idx].add(x.sum(axis=0))

            final_sum_sq = final_sum_sq + (x * x).sum(axis=0)
            n_processed += n

        layer_means = np.asarray(layer_sums / n_processed, dtype=np.float32)
        final_mean = layer_means[-1].copy()
        avg_variance = float(
            jnp.mean(final_sum_sq / n_processed - jnp.array(final_mean) ** 2)
        )
        return layer_means, final_mean, avg_variance
