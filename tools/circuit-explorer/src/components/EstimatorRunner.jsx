import { useCallback, useRef, useState } from "react";
import { empiricalMean } from "../circuit";
import { meanPropagation } from "../estimators";

/**
 * EstimatorRunner — lets users run estimators one at a time,
 * with timing stats and a progress bar.
 *
 * Timing measures only the actual computation, not the animation.
 * Ground Truth = empiricalMean with 10k samples (high-fidelity reference).
 * Sampling = empiricalMean with user-chosen budget (what participants optimize).
 * Mean Propagation = analytic layer-wise estimation (no sampling).
 */
export default function EstimatorRunner({ circuit, onResult }) {
  const [budget, setBudget] = useState(1000);
  const [running, setRunning] = useState(null);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState({});
  const cancelRef = useRef(false);

  // Animate progress, then run computation at the end
  const animateAndRun = useCallback((label, totalSteps, stepDelay, computeFn, key) => {
    setRunning(key);
    setProgress(0);
    cancelRef.current = false;
    let step = 0;

    const tick = () => {
      if (cancelRef.current) { setRunning(null); return; }
      step++;
      setProgress(step / totalSteps);

      if (step >= totalSteps) {
        // Measure ONLY computation time
        const t0 = performance.now();
        const result = computeFn();
        result.time = performance.now() - t0;

        setResults((prev) => ({ ...prev, [key]: result }));
        onResult(key, result);
        setRunning(null);
        setProgress(1);
      } else {
        if (stepDelay > 0) setTimeout(tick, stepDelay);
        else requestAnimationFrame(tick);
      }
    };

    requestAnimationFrame(tick);
  }, [onResult]);

  const runGroundTruth = useCallback(() => {
    if (!circuit) return;
    const steps = 20;
    animateAndRun("Computing ground truth", steps, 0, () => {
      const gtBudget = 10000;
      const estimates = empiricalMean(circuit, gtBudget, 7777);
      return { name: "Ground Truth (10k samples)", estimates, budget: gtBudget };
    }, "groundTruth");
  }, [circuit, animateAndRun]);

  const runSampling = useCallback(() => {
    if (!circuit) return;
    const steps = 20;
    animateAndRun("Sampling", steps, 0, () => {
      const estimates = empiricalMean(circuit, budget, 1234);
      return { name: "Sampling", estimates, budget };
    }, "sampling");
  }, [circuit, budget, animateAndRun]);

  const runMeanProp = useCallback(() => {
    if (!circuit) return;
    const steps = circuit.d;
    animateAndRun("Propagating means", steps, 30, () => {
      const estimates = meanPropagation(circuit);
      return { name: "Mean Propagation", estimates };
    }, "meanprop");
  }, [circuit, animateAndRun]);

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
          {running === "groundTruth" ? "Running…" : "⏱ Ground Truth (10k)"}
        </button>

        <button
          className="run-btn run-btn-sampling"
          onClick={runSampling}
          disabled={!!running}
        >
          {running === "sampling" ? "Running…" : `⏱ Sampling (${budget.toLocaleString()})`}
        </button>

        <button
          className="run-btn run-btn-meanprop"
          onClick={runMeanProp}
          disabled={!!running}
        >
          {running === "meanprop" ? "Running…" : "⏱ Mean Propagation"}
        </button>
      </div>

      {/* Progress bar */}
      {running && (
        <div className="progress-container">
          <div className="progress-label">
            {typeof running === "string" && running !== "groundTruth" && running !== "sampling" && running !== "meanprop"
              ? running
              : running === "groundTruth" ? "Computing ground truth"
              : running === "sampling" ? "Sampling"
              : "Propagating means"}
            … {Math.round(progress * 100)}%
          </div>
          <div className="progress-track">
            <div
              className="progress-fill"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      )}

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
