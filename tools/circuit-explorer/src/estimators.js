/**
 * estimators.js — JS port of mean propagation from estimators.py
 *
 * Mean propagation: propagates wire means through each layer,
 * approximating E[output] using E[x] * E[y] ≈ E[xy].
 * This is the simplest analytic estimator — fast but approximate.
 */

/**
 * Compute mean propagation estimates for each layer.
 * Returns: Float32Array[] — one per layer, each of length n
 *
 * Mirrors estimators.py::mean_propagation
 */
export function meanPropagation(circuit) {
  const n = circuit.n;
  let xMean = new Float32Array(n); // inputs are uniform ±1 → mean = 0

  const results = [];
  for (const layer of circuit.gates) {
    const newMean = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const muFirst = xMean[layer.first[i]];
      const muSecond = xMean[layer.second[i]];
      newMean[i] =
        layer.firstCoeff[i] * muFirst +
        layer.secondCoeff[i] * muSecond +
        layer.const[i] +
        layer.productCoeff[i] * muFirst * muSecond;
    }
    xMean = newMean;
    results.push(Float32Array.from(xMean));
  }
  return results;
}
