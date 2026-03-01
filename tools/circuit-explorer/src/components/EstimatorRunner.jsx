import { useCallback, useState } from "react";
import { empiricalMean } from "../circuit";
import { meanPropagation } from "../estimators";

/**
 * EstimatorRunner — lets users run estimators one at a time,
 * with timing stats. No artificial animation delays.
 *
 * Ground Truth = empiricalMean with 10k samples (high-fidelity reference).
 * Sampling = empiricalMean with user-chosen budget.
 * Mean Propagation = analytic layer-wise estimation (instant).
 */
export default function EstimatorRunner({ circuit, onResult }) {
  const [budget, setBudget] = useState(1000);
  const [running, setRunning] = useState(null);
  const [results, setResults] = useState({});

  const runEstimator = useCallback((key, computeFn) => {
    if (!circuit) return;
    setRunning(key);

    // Use requestAnimationFrame to let the "Running…" state render,
    // then run the computation synchronously.
    requestAnimationFrame(() => {
      const t0 = performance.now();
      const result = computeFn();
      result.time = performance.now() - t0;

      setResults((prev) => ({ ...prev, [key]: result }));
      onResult(key, result);
      setRunning(null);
    });
  }, [circuit, onResult]);

  const runGroundTruth = useCallback(() => {
    runEstimator("groundTruth", () => {
      const gtBudget = 10000;
      const estimates = empiricalMean(circuit, gtBudget, 7777);
      return { name: "Ground Truth (10k samples)", estimates, budget: gtBudget };
    });
  }, [circuit, runEstimator]);

  const runSampling = useCallback(() => {
    runEstimator("sampling", () => {
      const estimates = empiricalMean(circuit, budget, 1234);
      return { name: "Sampling", estimates, budget };
    });
  }, [circuit, budget, runEstimator]);

  const runMeanProp = useCallback(() => {
    runEstimator("meanprop", () => {
      const estimates = meanPropagation(circuit);
      return { name: "Mean Propagation", estimates };
    });
  }, [circuit, runEstimator]);

  const formatTime = (ms) => {
    if (ms < 0.01) return "<0.01ms";
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  return (
    <div className="estimator-runner">
      <h2>Run Estimators</h2>

      {/* Budget slider for sampling */}
      <div className="control-row">
        <label>
          <span className="control-label">Sampling budget</span>
          <span className="control-value">{budget.toLocaleString()}</span>
        </label>
        <input
          type="range"
          min={100}
          max={50000}
          step={100}
          value={budget}
          onChange={(e) => setBudget(Number(e.target.value))}
        />
      </div>

      {/* Estimator buttons */}
      <div className="estimator-buttons">
        <button
          className="run-btn run-btn-gt"
          onClick={runGroundTruth}
          disabled={!!running}
        >
          {running === "groundTruth" ? "Running…" : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Ground Truth (10k)</>}
        </button>

        <button
          className="run-btn run-btn-sampling"
          onClick={runSampling}
          disabled={!!running}
        >
          {running === "sampling" ? "Running…" : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Sampling ({budget.toLocaleString()})</>}
        </button>

        <button
          className="run-btn run-btn-meanprop"
          onClick={runMeanProp}
          disabled={!!running}
        >
          {running === "meanprop" ? "Running…" : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Mean Propagation</>}
        </button>
      </div>

      {/* Results */}
      {Object.keys(results).length > 0 && (
        <div className="estimator-results">
          <h3>Results</h3>
          {results.groundTruth && (
            <div className="result-row result-gt">
              <span className="result-name">Ground Truth</span>
              <span className="result-stat">{formatTime(results.groundTruth.time)}</span>
              <span className="result-detail">{results.groundTruth.budget.toLocaleString()} samples</span>
            </div>
          )}
          {results.sampling && (
            <div className="result-row result-sampling">
              <span className="result-name">Sampling</span>
              <span className="result-stat">{formatTime(results.sampling.time)}</span>
              <span className="result-detail">{results.sampling.budget.toLocaleString()} samples</span>
            </div>
          )}
          {results.meanprop && (
            <div className="result-row result-meanprop">
              <span className="result-name">Mean Prop</span>
              <span className="result-stat">{formatTime(results.meanprop.time)}</span>
              <span className="result-detail">analytic</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
