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

/**
 * Covariance propagation estimator — tracks mean + full covariance matrix.
 *
 * Mirrors estimators.py::CovariancePropagationEstimator.
 * More accurate than mean propagation because it accounts for wire
 * correlations via E[x_f * x_s] = m_f * m_s + C_fs.
 *
 * State: mean (n), covariance (n×n flat row-major Float32Array).
 * Runtime: O(depth × n²) per layer.
 *
 * Returns: Float32Array[] — one per layer, each of length n.
 */
export function covariancePropagation(circuit) {
  const n = circuit.n;
  let xMean = new Float32Array(n);           // E[x] = 0 for uniform ±1
  let xCov = new Float32Array(n * n);        // Cov = I  (variance 1 each)
  for (let i = 0; i < n; i++) xCov[i * n + i] = 1.0;

  const results = [];
  for (const layer of circuit.gates) {
    const out = propagateLayerCov(n, layer, xMean, xCov);
    xMean = out.mean;
    xCov = out.cov;
    results.push(Float32Array.from(xMean));
  }
  return results;
}

// ── internal helpers (flat row-major cov) ──

/** Propagate one layer: mean + covariance. */
function propagateLayerCov(n, layer, xMean, xCov) {
  const first = layer.first;
  const second = layer.second;
  const a = layer.firstCoeff;
  const b = layer.secondCoeff;
  const c = layer.const;
  const p = layer.productCoeff;

  // Pre-gather means and pair covariances
  const mF = new Float32Array(n);
  const mS = new Float32Array(n);
  const pairCov = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    mF[i] = xMean[first[i]];
    mS[i] = xMean[second[i]];
    pairCov[i] = xCov[first[i] * n + second[i]];
  }

  // Step 1: mean update (with covariance correction)
  const newMean = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    newMean[i] = a[i] * mF[i]
               + b[i] * mS[i]
               + c[i]
               + p[i] * (mF[i] * mS[i] + pairCov[i]);
  }

  // Step 2: linear-linear covariance
  const newCov = new Float32Array(n * n); // zeros
  linearLinearCov(n, layer, xCov, newCov);

  // Step 3: 1v2 cross terms (first-coeff × product-coeff path)
  addOneVTwoCross(n, a, p, first, first, second, xCov, xMean, newCov);
  // Step 3b: second-coeff × product-coeff path
  addOneVTwoCross(n, b, p, second, first, second, xCov, xMean, newCov);

  // Step 4: 2v2 bilinear-bilinear
  addTwoVTwo(n, p, first, second, xCov, xMean, newCov);

  // Step 5: clip moments
  clipMoments(n, newMean, newCov);

  return { mean: newMean, cov: newCov };
}

/** Linear-linear covariance: 4 outer-product-like terms. */
function linearLinearCov(n, layer, xCov, out) {
  const f = layer.first, s = layer.second;
  const a = layer.firstCoeff, b = layer.secondCoeff;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const idx = i * n + j;
      out[idx] +=
        a[i] * a[j] * xCov[f[i] * n + f[j]]
      + b[i] * b[j] * xCov[s[i] * n + s[j]]
      + a[i] * b[j] * xCov[f[i] * n + s[j]]
      + b[i] * a[j] * xCov[s[i] * n + f[j]];
    }
  }
}

/**
 * Add 1v2 cross-covariance term: outer(coeffRow, coeffCol) * Cov(x[aIdx], x[bIdx]*x[cIdx]).
 * Cov(x[a], x[b]*x[c]) ≈ mean[b] * C[a,c] + mean[c] * C[a,b]  (pairwise closure).
 * Adds both the term and its transpose (symmetry).
 */
function addOneVTwoCross(n, coeffRow, coeffCol, aIdx, bIdx, cIdx, xCov, xMean, out) {
  // Compute the n×n one-v-two block
  // result[i][j] = coeffRow[i] * coeffCol[j] * (mean[bIdx[j]] * C[aIdx[i], cIdx[j]]
  //                                            + mean[cIdx[j]] * C[aIdx[i], bIdx[j]])
  for (let i = 0; i < n; i++) {
    const cr = coeffRow[i];
    if (cr === 0) continue;
    const ai = aIdx[i];
    for (let j = 0; j < n; j++) {
      const cc = coeffCol[j];
      if (cc === 0) continue;
      const val = cr * cc * (
        xMean[bIdx[j]] * xCov[ai * n + cIdx[j]]
      + xMean[cIdx[j]] * xCov[ai * n + bIdx[j]]
      );
      out[i * n + j] += val;
      out[j * n + i] += val; // transpose
    }
  }
}

/**
 * Add 2v2 bilinear-bilinear term:
 * outer(p, p) * Cov(x[a]*x[b], x[c]*x[d])
 * ≈ μa·μc·C[b,d] + μa·μd·C[b,c] + μb·μc·C[a,d] + μb·μd·C[a,c]  (Isserlis/Wick)
 */
function addTwoVTwo(n, p, aIdx, bIdx, xCov, xMean, out) {
  for (let i = 0; i < n; i++) {
    const pi = p[i];
    if (pi === 0) continue;
    const ai = aIdx[i], bi = bIdx[i];
    const muA = xMean[ai], muB = xMean[bi];
    for (let j = 0; j < n; j++) {
      const pj = p[j];
      if (pj === 0) continue;
      const cj = aIdx[j], dj = bIdx[j];
      const muC = xMean[cj], muD = xMean[dj];
      out[i * n + j] += pi * pj * (
        muA * muC * xCov[bi * n + dj]
      + muA * muD * xCov[bi * n + cj]
      + muB * muC * xCov[ai * n + dj]
      + muB * muD * xCov[ai * n + cj]
      );
    }
  }
}

/** Clip moments to feasible bounds for {-1, +1} wire values. */
function clipMoments(n, mean, cov) {
  // Clip means to [-1, 1]
  for (let i = 0; i < n; i++) {
    mean[i] = Math.max(-1, Math.min(1, mean[i]));
  }
  // Set diagonal = 1 - μ²
  const vari = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    vari[i] = 1 - mean[i] * mean[i];
    cov[i * n + i] = vari[i];
  }
  // Clip off-diagonal by ±√(var_i * var_j)
  for (let i = 0; i < n; i++) {
    const si = Math.sqrt(Math.max(0, vari[i]));
    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const maxC = si * Math.sqrt(Math.max(0, vari[j]));
      const idx = i * n + j;
      cov[idx] = Math.max(-maxC, Math.min(maxC, cov[idx]));
    }
  }
}
