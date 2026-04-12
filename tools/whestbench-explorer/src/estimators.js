/**
 * estimators.js — ReLU moment-propagation estimators for MLP networks
 *
 * Ports estimators.py (MeanPropagationEstimator, CovariancePropagationEstimator)
 * to JavaScript for use in WhestBench Explorer.
 *
 * Weight matrix convention: W[i * width + j] where i = input neuron, j = output neuron.
 * Forward pass: x_next = ReLU(x @ W)  (row-vector convention, matching mlp.js).
 *
 * Exports:
 *   meanPropagation(mlp)       — diagonal-variance first-moment propagation
 *   covariancePropagation(mlp) — full-covariance first+second-moment propagation
 */

import { normalPdf, normalCdf } from './math-utils.js';

/**
 * Mean propagation estimator.
 *
 * Propagates both E[x] (mean) and Var[x] (diagonal variance) per neuron,
 * assuming independence across neurons within each layer.
 *
 * Per layer, for each output neuron j:
 *   Pre-activation:
 *     mu_j  = sum_i  W[i,j] * mean_i
 *     var_j = sum_i  W[i,j]^2 * var_i
 *   Numerical floor: var_j = max(var_j, 1e-12)
 *   sigma_j = sqrt(var_j),  alpha_j = mu_j / sigma_j
 *   Post-ReLU mean:
 *     mean_j = mu_j * Phi(alpha_j) + sigma_j * phi(alpha_j)
 *   Post-ReLU variance:
 *     E[z^2] = (mu_j^2 + var_j) * Phi(alpha_j) + mu_j * sigma_j * phi(alpha_j)
 *     var_j  = max(E[z^2] - mean_j^2, 0)
 *
 * Initial state: mean = 0, var = 1 for each neuron (Gaussian N(0,1) inputs).
 *
 * @param {{ width: number, depth: number, weights: Float32Array[] }} mlp
 * @returns {Float32Array} shape (depth × width), per-layer post-ReLU means
 */
export function meanPropagation(mlp) {
  const { width, weights } = mlp;

  // Initial state: N(0,1) inputs
  let mean = new Float64Array(width);   // all zeros
  let vari = new Float64Array(width);   // all ones
  for (let i = 0; i < width; i++) vari[i] = 1.0;

  // Output: one row per layer, each row has `width` predicted means
  const result = new Float32Array(weights.length * width);

  for (let layerIdx = 0; layerIdx < weights.length; layerIdx++) {
    const W = weights[layerIdx]; // Float32Array, row-major: W[i * width + j]

    const muPre  = new Float64Array(width);
    const varPre = new Float64Array(width);

    // Pre-activation: mu_pre[j] = sum_i W[i,j] * mean[i]
    //                 var_pre[j] = sum_i W[i,j]^2 * var[i]
    for (let i = 0; i < width; i++) {
      const m = mean[i];
      const v = vari[i];
      for (let j = 0; j < width; j++) {
        const w = W[i * width + j];
        muPre[j]  += w * m;
        varPre[j] += w * w * v;
      }
    }

    const newMean = new Float64Array(width);
    const newVar  = new Float64Array(width);

    for (let j = 0; j < width; j++) {
      const mu  = muPre[j];
      const vp  = Math.max(varPre[j], 1e-12);
      const sig = Math.sqrt(vp);
      const alpha = mu / sig;

      const phi = normalPdf(alpha);
      const Phi = normalCdf(alpha);

      const postMean = mu * Phi + sig * phi;
      const ez2 = (mu * mu + vp) * Phi + mu * sig * phi;
      const postVar = Math.max(ez2 - postMean * postMean, 0.0);

      newMean[j] = postMean;
      newVar[j]  = postVar;
    }

    mean = newMean;
    vari = newVar;

    // Store this layer's means into the flat output
    const base = layerIdx * width;
    for (let j = 0; j < width; j++) {
      result[base + j] = mean[j];
    }
  }

  return result;
}

/**
 * Covariance propagation estimator.
 *
 * Tracks full covariance matrix and mean vector per layer.
 * More accurate than meanPropagation for deep networks because it
 * accounts for inter-neuron correlations when computing pre-activation variance.
 *
 * Per layer:
 *   Pre-activation mean:       mu = W^T @ mean  (mu[j] = sum_i W[i,j] * mean[i])
 *   Pre-activation covariance: Cov_pre = W^T @ Cov @ W
 *   Diagonal variances:        var_j = max(Cov_pre[j,j], 1e-12)
 *   sigma_j = sqrt(var_j),     alpha_j = mu_j / sigma_j
 *   Post-ReLU means:           same formula as mean propagation
 *   Post-ReLU covariance:      diag = postVar,
 *                              off-diag[a,b] = gain_a * gain_b * Cov_pre[a,b]
 *                              where gain_j = Phi(alpha_j)
 *
 * Initial state: mean = 0, Cov = I (identity).
 * Uses Float64Array for internal computation.
 * Covariance is stored flat row-major: Cov[a * width + b].
 *
 * @param {{ width: number, depth: number, weights: Float32Array[] }} mlp
 * @returns {Float32Array} shape (depth × width), per-layer post-ReLU means
 */
export function covariancePropagation(mlp) {
  const { width, weights } = mlp;

  // Initial state: mean = 0, Cov = I
  let mean = new Float64Array(width);  // all zeros
  let cov  = new Float64Array(width * width);
  for (let i = 0; i < width; i++) cov[i * width + i] = 1.0;

  const result = new Float32Array(weights.length * width);

  for (let layerIdx = 0; layerIdx < weights.length; layerIdx++) {
    const W = weights[layerIdx]; // Float32Array, row-major: W[i * width + j]

    // Pre-activation mean: mu_pre[j] = sum_i W[i,j] * mean[i]
    const muPre = new Float64Array(width);
    for (let i = 0; i < width; i++) {
      const m = mean[i];
      for (let j = 0; j < width; j++) {
        muPre[j] += W[i * width + j] * m;
      }
    }

    // Pre-activation covariance: Cov_pre = W^T @ Cov @ W
    // Step 1: tmp = Cov @ W  (shape width × width)
    //   tmp[a,j] = sum_b Cov[a,b] * W[b,j]
    const tmp = new Float64Array(width * width);
    for (let a = 0; a < width; a++) {
      for (let b = 0; b < width; b++) {
        const c = cov[a * width + b];
        if (c === 0) continue;
        for (let j = 0; j < width; j++) {
          tmp[a * width + j] += c * W[b * width + j];
        }
      }
    }

    // Step 2: Cov_pre = W^T @ tmp  (shape width × width)
    //   covPre[i,j] = sum_a W[a,i] * tmp[a,j]  (W^T[i,a] = W[a,i])
    const covPre = new Float64Array(width * width);
    for (let a = 0; a < width; a++) {
      for (let i = 0; i < width; i++) {
        const w = W[a * width + i];
        if (w === 0) continue;
        for (let j = 0; j < width; j++) {
          covPre[i * width + j] += w * tmp[a * width + j];
        }
      }
    }

    // Per-neuron: extract diagonal variances, compute ReLU moments and gains
    const varPre  = new Float64Array(width);
    const alpha   = new Float64Array(width);
    const phi     = new Float64Array(width);
    const Phi     = new Float64Array(width);
    const newMean = new Float64Array(width);
    const postVar = new Float64Array(width);
    const gain    = new Float64Array(width);

    for (let j = 0; j < width; j++) {
      const vp  = Math.max(covPre[j * width + j], 1e-12);
      const sig = Math.sqrt(vp);
      const a   = muPre[j] / sig;

      varPre[j] = vp;
      alpha[j]  = a;
      phi[j]    = normalPdf(a);
      Phi[j]    = normalCdf(a);

      const mu = muPre[j];
      const pm = mu * Phi[j] + sig * phi[j];
      const ez2 = (mu * mu + vp) * Phi[j] + mu * sig * phi[j];
      newMean[j] = pm;
      postVar[j] = Math.max(ez2 - pm * pm, 0.0);

      // gain_j = Phi(alpha_j) (or 0 if sigma was negligible before flooring)
      gain[j] = Phi[j];
    }

    // Post-ReLU covariance:
    //   off-diagonal: newCov[a,b] = gain[a] * gain[b] * covPre[a,b]
    //   diagonal:     newCov[j,j] = postVar[j]
    const newCov = new Float64Array(width * width);
    for (let a = 0; a < width; a++) {
      for (let b = 0; b < width; b++) {
        newCov[a * width + b] = gain[a] * gain[b] * covPre[a * width + b];
      }
      // Override diagonal with proper post-ReLU variance
      newCov[a * width + a] = postVar[a];
    }

    mean = newMean;
    cov  = newCov;

    const base = layerIdx * width;
    for (let j = 0; j < width; j++) {
      result[base + j] = mean[j];
    }
  }

  return result;
}
