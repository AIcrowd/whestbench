import { useCallback, useState } from "react";

/**
 * EstimatorRunner — lets users run estimators one at a time,
 * with timing stats. Computation runs in a Web Worker (off main thread).
 *
 * Ground Truth = empiricalMean with 10k samples (high-fidelity reference).
 * Sampling = empiricalMean with user-chosen budget.
 * Mean Propagation = analytic layer-wise estimation (instant).
 */
export default function EstimatorRunner({ circuit, onResult, worker }) {
  const [budget, setBudget] = useState(1000);
  const [running, setRunning] = useState(null);
  const [results, setResults] = useState({});

  const runEstimator = useCallback(async (key, type, workerParams, displayInfo) => {
    if (!circuit || !worker) return;
    setRunning(key);

    try {
      const result = await worker.run(type, workerParams);
      const enriched = { ...displayInfo, estimates: result.estimates, time: result.time };
      setResults((prev) => ({ ...prev, [key]: enriched }));
      onResult(key, enriched);
    } catch (err) {
      console.error(`Estimator ${key} failed:`, err);
    } finally {
      setRunning(null);
    }
  }, [circuit, worker, onResult]);

  const runGroundTruth = useCallback(() => {
    const gtBudget = 10000;
    runEstimator("groundTruth", "empiricalMean",
      { circuit, trials: gtBudget, seed: 7777 },
      { name: "Ground Truth (10k samples)", budget: gtBudget }
    );
  }, [circuit, runEstimator]);

  const runSampling = useCallback(() => {
    runEstimator("sampling", "empiricalMean",
      { circuit, trials: budget, seed: 1234 },
      { name: "Sampling", budget }
    );
  }, [circuit, budget, runEstimator]);

  const runMeanProp = useCallback(() => {
    runEstimator("meanprop", "meanPropagation",
      { circuit },
      { name: "Mean Propagation" }
    );
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

      {/* ① Ground Truth */}
      <div className="estimator-card estimator-card--gt">
        <div className="estimator-card-header">
          <span className="estimator-badge estimator-badge--gt">1</span>
          <span className="estimator-card-title">Ground Truth</span>
        </div>
        <p className="estimator-card-desc">
          Brute-force: samples <strong>10,000</strong> random inputs and averages
          each wire. Slow but accurate.
        </p>
        <button
          className="run-btn run-btn-gt"
          onClick={runGroundTruth}
          disabled={!!running}
        >
          {running === "groundTruth" ? "Running…" : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Ground Truth (10k)</>}
        </button>
        {running === "groundTruth" && (
          <div className="estimator-progress"><div className="estimator-progress-bar" /></div>
        )}
        {results.groundTruth && !running && (
          <div className="estimator-card-result">
            <span className="result-stat">{formatTime(results.groundTruth.time)}</span>
            <span className="result-detail">{results.groundTruth.budget.toLocaleString()} samples</span>
          </div>
        )}
      </div>

      {/* ② Sampling */}
      <div className="estimator-card estimator-card--sampling">
        <div className="estimator-card-header">
          <span className="estimator-badge estimator-badge--sampling">2</span>
          <span className="estimator-card-title">Sampling</span>
        </div>
        <p className="estimator-card-desc">
          Same idea, fewer samples. Faster but noisier — tune the budget below.
        </p>
        <div className="control-row">
          <label>
            <span className="control-label">Budget</span>
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
        <button
          className="run-btn run-btn-sampling"
          onClick={runSampling}
          disabled={!!running}
        >
          {running === "sampling" ? "Running…" : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Sampling ({budget.toLocaleString()})</>}
        </button>
        {running === "sampling" && (
          <div className="estimator-progress"><div className="estimator-progress-bar" /></div>
        )}
        {results.sampling && !running && (
          <div className="estimator-card-result">
            <span className="result-stat">{formatTime(results.sampling.time)}</span>
            <span className="result-detail">{results.sampling.budget.toLocaleString()} samples</span>
          </div>
        )}
      </div>

      {/* ③ Mean Propagation */}
      <div className="estimator-card estimator-card--meanprop">
        <div className="estimator-card-header">
          <span className="estimator-badge estimator-badge--meanprop">3</span>
          <span className="estimator-card-title">Mean Propagation</span>
        </div>
        <p className="estimator-card-desc">
          Analytic: propagates E[x] through each layer. Instant, no sampling needed.
        </p>
        <button
          className="run-btn run-btn-meanprop"
          onClick={runMeanProp}
          disabled={!!running}
        >
          {running === "meanprop" ? "Running…" : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Mean Propagation</>}
        </button>
        {running === "meanprop" && (
          <div className="estimator-progress"><div className="estimator-progress-bar" /></div>
        )}
        {results.meanprop && !running && (
          <div className="estimator-card-result">
            <span className="result-stat">{formatTime(results.meanprop.time)}</span>
            <span className="result-detail">analytic</span>
          </div>
        )}
      </div>
    </div>
  );
}
