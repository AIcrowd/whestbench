/**
 * circuit.worker.js — Web Worker for off-thread circuit computation.
 *
 * Receives messages with { id, type, params } and posts back { id, result }.
 * Typed arrays (Float32Array, Int32Array) survive structured clone via postMessage.
 */
import { empiricalMean, empiricalStats, randomCircuit, runSingleTrial } from './circuit';
import { meanPropagation } from './estimators';

self.onmessage = function (e) {
  const { id, type, params } = e.data;

  let result;
  const t0 = performance.now();

  switch (type) {
    case 'randomCircuit': {
      result = { circuit: randomCircuit(params.width, params.depth, params.seed) };
      break;
    }
    case 'empiricalMean': {
      const circuit = params.circuit;
      result = { estimates: empiricalMean(circuit, params.trials, params.seed) };
      break;
    }
    case 'empiricalStats': {
      const circuit = params.circuit;
      const stats = empiricalStats(circuit, params.trials, params.seed);
      result = { estimates: stats.means, stds: stats.stds, mins: stats.mins, maxs: stats.maxs };
      break;
    }
    case 'runSingleTrial': {
      const circuit = params.circuit;
      result = { layerValues: runSingleTrial(circuit, params.seed) };
      break;
    }
    case 'meanPropagation': {
      const circuit = params.circuit;
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

