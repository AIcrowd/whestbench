/**
 * circuit.worker.js — Web Worker for off-thread circuit computation.
 *
 * Receives messages with { id, type, params } and posts back { id, result }.
 * Typed arrays (Float32Array, Int32Array) survive structured clone via postMessage.
 */
import { empiricalMean } from './circuit';
import { meanPropagation } from './estimators';

self.onmessage = function (e) {
  const { id, type, params } = e.data;
  const circuit = params.circuit;

  let result;
  const t0 = performance.now();

  switch (type) {
    case 'empiricalMean': {
      result = { estimates: empiricalMean(circuit, params.trials, params.seed) };
      break;
    }
    case 'meanPropagation': {
      result = { estimates: meanPropagation(circuit) };
      break;
    }
    default:
      self.postMessage({ id, error: `Unknown type: ${type}` });
      return;
  }

  result.time = performance.now() - t0;
  self.postMessage({ id, result });
};
