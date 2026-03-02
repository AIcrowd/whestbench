import { useCallback, useEffect, useState } from "react";
import { empiricalMeanTF, empiricalStatsTF, initTF } from "../circuit-tf";
import { perfEnd, perfStart } from "../perf";

/**
 * EstimatorRunner — lets users run estimators one at a time,
 * with timing stats. GPU estimators run on main thread via TF.js
 * (WebGL/WebGPU need DOM context). CPU fallback via Web Worker.
 *
 * Ground Truth = empiricalMean with 10k samples (high-fidelity reference).
 * Sampling = empiricalMean with user-chosen budget.
 * Mean Propagation = analytic layer-wise estimation (instant, always via worker).
 */
export default function EstimatorRunner({ circuit, onResult, worker }) {
  const [budget, setBudget] = useState(1000);
  const [running, setRunning] = useState(null);
  const [progress, setProgress] = useState(0); // 0..1 real progress
  const [results, setResults] = useState({});
  const [tfBackend, setTfBackend] = useState(null);

  // Initialize TF.js on mount
  useEffect(() => {
    initTF()
      .then((backend) => setTfBackend(backend))
      .catch(() => setTfBackend('unavailable'));
  }, []);

  // Run empiricalMean (sampling) or empiricalStats (ground truth) via GPU/worker
  const runEmpirical = useCallback(async (key, trials, seed, displayInfo) => {
    if (!circuit) return;
    setRunning(key);
    setProgress(0);
    const useStats = key === 'groundTruth'; // Enriched stats only for GT

    try {
      let estimates, time, extraStats = null;

      if (tfBackend && tfBackend !== 'unavailable') {
        // GPU path: TF.js on main thread with progress reporting
        perfStart(`estimator-${key}`);
        const t0 = performance.now();
        if (useStats) {
          const stats = await empiricalStatsTF(circuit, trials, seed, (p) => setProgress(p));
          estimates = stats.means;
          extraStats = { stds: stats.stds, mins: stats.mins, maxs: stats.maxs };
        } else {
          estimates = await empiricalMeanTF(circuit, trials, seed, (p) => setProgress(p));
        }
        time = performance.now() - t0;
        perfEnd(`estimator-${key}`);
      } else {
        // CPU fallback: worker
        perfStart(`estimator-${key}`);
        if (useStats) {
          const result = await worker.run('empiricalStats', { circuit, trials, seed });
          estimates = result.estimates;
          extraStats = { stds: result.stds, mins: result.mins, maxs: result.maxs };
          time = result.time;
        } else {
          const result = await worker.run('empiricalMean', { circuit, trials, seed });
          estimates = result.estimates;
          time = result.time;
        }
        perfEnd(`estimator-${key}`);
      }

      setProgress(1);
      const enriched = { ...displayInfo, estimates, time, ...(extraStats || {}) };
      setResults((prev) => ({ ...prev, [key]: enriched }));
      onResult(key, enriched);
    } catch (err) {
      console.error(`Estimator ${key} failed:`, err);
      // Try CPU fallback if GPU failed
      if (tfBackend && tfBackend !== 'unavailable') {
        console.warn(`[EstimatorRunner] TF.js failed, falling back to worker`);
        try {
          const workerType = useStats ? 'empiricalStats' : 'empiricalMean';
          const result = await worker.run(workerType, { circuit, trials, seed });
          const fallbackExtra = useStats
            ? { stds: result.stds, mins: result.mins, maxs: result.maxs }
            : {};
          const enriched = { ...displayInfo, estimates: result.estimates, time: result.time, ...fallbackExtra };
          setResults((prev) => ({ ...prev, [key]: enriched }));
          onResult(key, enriched);
        } catch (err2) {
          console.error(`Worker fallback also failed:`, err2);
        }
      }
    } finally {
      setRunning(null);
      setProgress(0);
    }
  }, [circuit, worker, onResult, tfBackend]);

  // Mean propagation always uses the worker (analytic, instant, no GPU needed)
  const runMeanProp = useCallback(async () => {
    if (!circuit || !worker) return;
    setRunning("meanprop");
    setProgress(0);
    try {
      perfStart('estimator-meanprop');
      const result = await worker.run('meanPropagation', { circuit });
      perfEnd('estimator-meanprop');
      setProgress(1);
      const enriched = { name: "Mean Propagation", estimates: result.estimates, time: result.time };
      setResults((prev) => ({ ...prev, meanprop: enriched }));
      onResult("meanprop", enriched);
    } catch (err) {
      console.error("Mean propagation failed:", err);
    } finally {
      setRunning(null);
      setProgress(0);
    }
  }, [circuit, worker, onResult]);

  const runGroundTruth = useCallback(() => {
    const gtBudget = 10000;
    runEmpirical("groundTruth", gtBudget, 7777,
      { name: "Ground Truth (10k samples)", budget: gtBudget }
    );
  }, [runEmpirical]);

  const runSampling = useCallback(() => {
    runEmpirical("sampling", budget, 1234,
      { name: "Sampling", budget }
    );
  }, [runEmpirical, budget]);

  const formatTime = (ms) => {
    if (ms < 0.01) return "<0.01ms";
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const backendBadge = tfBackend === 'unavailable' ? 'CPU (worker)'
    : tfBackend ? `GPU (${tfBackend})`
    : 'loading…';

  const progressPct = Math.round(progress * 100);

  const renderProgressBar = (key) => {
    if (running !== key) return null;
    return (
      <div className="estimator-progress">
        <div
          className="estimator-progress-bar"
          style={{
            width: `${progressPct}%`,
            transition: 'width 0.15s ease-out',
          }}
        />
      </div>
    );
  };

  return (
    <div className="estimator-runner">
      <h2>Run Estimators</h2>
      <div className="estimator-backend-badge" style={{
        fontSize: 10, color: '#9CA3AF', fontFamily: "'IBM Plex Mono', monospace",
        marginBottom: 8, padding: '2px 0'
      }}>
        Backend: <span style={{
          color: tfBackend === 'webgpu' ? '#10B981' : tfBackend === 'webgl' ? '#3B82F6' : '#F59E0B'
        }}>{backendBadge}</span>
      </div>

      {/* ① Ground Truth */}
      <div className="estimator-card estimator-card--gt">
        <div className="estimator-card-header">
          <span className="estimator-badge estimator-badge--gt">1</span>
          <span className="estimator-card-title">Ground Truth</span>
        </div>
        <p className="estimator-card-desc">
          Brute-force: samples <strong>10,000</strong> random inputs and averages
          each wire. {tfBackend && tfBackend !== 'unavailable' ? 'GPU-accelerated.' : 'Slow but accurate.'}
        </p>
        <button
          className="run-btn run-btn-gt"
          onClick={runGroundTruth}
          disabled={!!running}
        >
          {running === "groundTruth" ? `Running… ${progressPct}%` : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Ground Truth (10k)</>}
        </button>
        {renderProgressBar("groundTruth")}
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
          {running === "sampling" ? `Running… ${progressPct}%` : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Sampling ({budget.toLocaleString()})</>}
        </button>
        {renderProgressBar("sampling")}
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
        {renderProgressBar("meanprop")}
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
