/**
 * mlp.worker.js — Web Worker for off-thread MLP computation.
 *
 * Receives messages with { id, type, params } and posts back { id, result }
 * where result includes a `time` field (milliseconds).
 *
 * Message types:
 *   sampleMLP       — { width, depth, seed }       → { mlp }
 *   outputStats     — { mlp, nSamples, seed }       → { means, variances }
 *   sampling        — { mlp, budget, seed }         → { estimates: means }
 *   meanPropagation — { mlp }                       → { estimates }
 *   covPropagation  — { mlp }                       → { estimates }
 */
import { sampleMLP, outputStats } from './mlp.js';
import { meanPropagation, covariancePropagation } from './estimators.js';

self.onmessage = function ({ data }) {
  const { id, type, params } = data;

  const t0 = performance.now();
  let payload;

  try {
    switch (type) {
      case 'sampleMLP': {
        const mlp = sampleMLP(params.width, params.depth, params.seed);
        payload = { mlp };
        break;
      }
      case 'outputStats': {
        const onProgress1 = (p) => self.postMessage({ id, progress: p });
        const { means, variances } = outputStats(params.mlp, params.nSamples, params.seed, onProgress1);
        payload = { means, variances };
        break;
      }
      case 'sampling': {
        // Monte Carlo sampling: run outputStats and return means as estimates
        const onProgress2 = (p) => self.postMessage({ id, progress: p });
        const { means } = outputStats(params.mlp, params.budget, params.seed, onProgress2);
        payload = { estimates: means };
        break;
      }
      case 'meanPropagation': {
        const estimates = meanPropagation(params.mlp);
        payload = { estimates };
        break;
      }
      case 'covPropagation': {
        const estimates = covariancePropagation(params.mlp);
        payload = { estimates };
        break;
      }
      default:
        self.postMessage({ id, error: `Unknown type: ${type}` });
        return;
    }

    const time = performance.now() - t0;
    self.postMessage({ id, result: { ...payload, time } });
  } catch (err) {
    self.postMessage({ id, error: err.message });
  }
};
