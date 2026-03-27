import { useCallback, useState } from "react";
import { perfEnd, perfStart } from "../perf";
import InfoTip from "./InfoTip";

/** Reshape flat Float32Array(depth × width) → Array of Float32Array, one per layer */
function reshapeFlat(flat, depth, width) {
  const result = [];
  for (let l = 0; l < depth; l++) {
    result.push(flat.slice(l * width, (l + 1) * width));
  }
  return result;
}

/**
 * EstimatorRunner — lets users run estimators one at a time,
 * with timing stats. All estimators route through the MLP Web Worker.
 *
 * Ground Truth = outputStats with nSamples=10000 (high-fidelity reference).
 * Sampling     = sampling with user-controlled budget.
 * Mean Prop    = meanPropagation (analytic, instant).
 * Cov Prop     = covPropagation (analytic, O(n²)).
 */
export default function EstimatorRunner({ mlp, onResult, worker }) {
  const [budget, setBudget] = useState(1000);
  const [running, setRunning] = useState(null);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState({});

  const formatTime = (ms) => {
    if (ms < 0.01) return "<0.01ms";
    if (ms < 1) return `${(ms * 1000).toFixed(0)}μs`;
    if (ms < 1000) return `${ms.toFixed(1)}ms`;
    return `${(ms / 1000).toFixed(2)}s`;
  };

  const progressPct = Math.round(progress * 100);

  const renderProgressBar = (key) => {
    if (running !== key) return null;
    return (
      <div className="estimator-progress">
        <div
          className="estimator-progress-bar"
          style={{ width: `${progressPct}%`, transition: "width 0.15s ease-out" }}
        />
      </div>
    );
  };

  /* ---- Ground Truth ---- */
  const runGroundTruth = useCallback(async () => {
    if (!mlp || !worker) return;
    setRunning("groundTruth");
    setProgress(0);
    try {
      perfStart("estimator-groundTruth");
      const result = await worker.run("outputStats", { mlp, nSamples: 10000, seed: 7777 }, setProgress);
      perfEnd("estimator-groundTruth");
      setProgress(1);
      const { depth, width } = mlp;
      const meansArr = reshapeFlat(result.means, depth, width);
      const stdsArr = result.variances
        ? reshapeFlat(result.variances, depth, width).map((layer) =>
            Float32Array.from(layer, (v) => Math.sqrt(v))
          )
        : null;
      const enriched = {
        name: "Ground Truth (10k samples)",
        budget: 10000,
        estimates: meansArr,
        stds: stdsArr,
        time: result.time,
      };
      setResults((prev) => ({ ...prev, groundTruth: enriched }));
      onResult("groundTruth", enriched);
    } catch (err) {
      console.error("Ground Truth failed:", err);
    } finally {
      setRunning(null);
      setProgress(0);
    }
  }, [mlp, worker, onResult]);

  /* ---- Sampling ---- */
  const runSampling = useCallback(async () => {
    if (!mlp || !worker) return;
    setRunning("sampling");
    setProgress(0);
    try {
      perfStart("estimator-sampling");
      const result = await worker.run("sampling", { mlp, budget, seed: 1234 }, setProgress);
      perfEnd("estimator-sampling");
      setProgress(1);
      const enriched = {
        name: "Sampling",
        budget,
        estimates: reshapeFlat(result.estimates, mlp.depth, mlp.width),
        time: result.time,
      };
      setResults((prev) => ({ ...prev, sampling: enriched }));
      onResult("sampling", enriched);
    } catch (err) {
      console.error("Sampling failed:", err);
    } finally {
      setRunning(null);
      setProgress(0);
    }
  }, [mlp, worker, onResult, budget]);

  /* ---- Mean Propagation ---- */
  const runMeanProp = useCallback(async () => {
    if (!mlp || !worker) return;
    setRunning("meanprop");
    setProgress(0);
    try {
      perfStart("estimator-meanprop");
      const result = await worker.run("meanPropagation", { mlp });
      perfEnd("estimator-meanprop");
      setProgress(1);
      const enriched = { name: "Mean Propagation", estimates: reshapeFlat(result.estimates, mlp.depth, mlp.width), time: result.time };
      setResults((prev) => ({ ...prev, meanprop: enriched }));
      onResult("meanprop", enriched);
    } catch (err) {
      console.error("Mean propagation failed:", err);
    } finally {
      setRunning(null);
      setProgress(0);
    }
  }, [mlp, worker, onResult]);

  /* ---- Covariance Propagation ---- */
  const runCovProp = useCallback(async () => {
    if (!mlp || !worker) return;
    setRunning("covprop");
    setProgress(0);
    try {
      perfStart("estimator-covprop");
      const result = await worker.run("covPropagation", { mlp });
      perfEnd("estimator-covprop");
      setProgress(1);
      const enriched = { name: "Cov Propagation", estimates: reshapeFlat(result.estimates, mlp.depth, mlp.width), time: result.time };
      setResults((prev) => ({ ...prev, covprop: enriched }));
      onResult("covprop", enriched);
    } catch (err) {
      console.error("Covariance propagation failed:", err);
    } finally {
      setRunning(null);
      setProgress(0);
    }
  }, [mlp, worker, onResult]);

  return (
    <div className="estimator-runner">
      <h2>Run Estimators</h2>

      {/* ① Ground Truth */}
      <div className="estimator-card estimator-card--gt">
        <InfoTip
          trigger={
            <div className="estimator-card-header" style={{ cursor: "pointer" }}>
              <span className="estimator-badge estimator-badge--gt">1</span>
              <span className="estimator-card-title">Ground Truth</span>
              <button className="info-tip-btn" aria-label="Show explanation" title="What does this mean?">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
                  <text x="8" y="12" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">i</text>
                </svg>
              </button>
            </div>
          }
        >
          <span className="tip-title">Ground Truth</span>
          <p className="tip-desc">
            High-fidelity reference: draws <span className="tip-mono">10,000</span> random Gaussian input vectors and averages each neuron&apos;s output.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Method</span><span className="tip-kv-val">Monte Carlo sampling (brute-force)</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Samples</span><span className="tip-kv-val">10,000 random Gaussian inputs</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Output</span><span className="tip-kv-val">E[neuron] per neuron per layer</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            Treated as the <span className="tip-highlight">gold standard</span> — other estimators are scored against this.
          </p>
        </InfoTip>
        <p className="estimator-card-desc">
          Brute-force: samples <strong>10,000</strong> random inputs and averages each neuron.
        </p>
        <button
          className="run-btn run-btn-gt"
          onClick={runGroundTruth}
          disabled={!!running}
        >
          {running === "groundTruth"
            ? `Running… ${progressPct}%`
            : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Ground Truth (10k)</>}
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
        <InfoTip
          trigger={
            <div className="estimator-card-header" style={{ cursor: "pointer" }}>
              <span className="estimator-badge estimator-badge--sampling">2</span>
              <span className="estimator-card-title">Sampling</span>
              <button className="info-tip-btn" aria-label="Show explanation" title="What does this mean?">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
                  <text x="8" y="12" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">i</text>
                </svg>
              </button>
            </div>
          }
        >
          <span className="tip-title">Sampling Estimator</span>
          <p className="tip-desc">
            Same Monte Carlo approach as Ground Truth, but with a <span className="tip-highlight">user-controlled budget</span> — fewer samples means faster but noisier estimates.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Method</span><span className="tip-kv-val">Monte Carlo sampling</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Budget</span><span className="tip-kv-val">100 – 50,000 samples (adjustable)</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Tradeoff</span><span className="tip-kv-val">More samples → less noise, more time</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            Estimation error scales as <span className="tip-mono">O(1/√budget)</span>.
          </p>
        </InfoTip>
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
          {running === "sampling"
            ? `Running… ${progressPct}%`
            : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Sampling ({budget.toLocaleString()})</>}
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
        <InfoTip
          trigger={
            <div className="estimator-card-header" style={{ cursor: "pointer" }}>
              <span className="estimator-badge estimator-badge--meanprop">3</span>
              <span className="estimator-card-title">Mean Propagation</span>
              <button className="info-tip-btn" aria-label="Show explanation" title="What does this mean?">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
                  <text x="8" y="12" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">i</text>
                </svg>
              </button>
            </div>
          }
        >
          <span className="tip-title">Mean Propagation</span>
          <p className="tip-desc">
            Propagates <span className="tip-mono">E[neuron]</span> through each layer analytically — no sampling required.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Formula</span><span className="tip-kv-val">E[ReLU(z)] = μΦ(μ/σ) + σφ(μ/σ)</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Key approx.</span><span className="tip-kv-val">Neurons are independent</span></div>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Runtime</span><span className="tip-kv-val">O(depth × width²) — instant</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            <span className="tip-highlight">Limitation</span>: ignores neuron correlations. Approximation <span className="tip-highlight">drifts at deeper layers</span>.
          </p>
        </InfoTip>
        <p className="estimator-card-desc">
          Analytic: propagates E[neuron] layer by layer. Instant, no sampling needed.
        </p>
        <button
          className="run-btn run-btn-meanprop"
          onClick={runMeanProp}
          disabled={!!running}
        >
          {running === "meanprop"
            ? "Running…"
            : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Mean Propagation</>}
        </button>
        {renderProgressBar("meanprop")}
        {results.meanprop && !running && (
          <div className="estimator-card-result">
            <span className="result-stat">{formatTime(results.meanprop.time)}</span>
            <span className="result-detail">analytic</span>
          </div>
        )}
      </div>

      {/* ④ Covariance Propagation */}
      <div className="estimator-card estimator-card--covprop">
        <InfoTip
          trigger={
            <div className="estimator-card-header" style={{ cursor: "pointer" }}>
              <span className="estimator-badge estimator-badge--covprop">4</span>
              <span className="estimator-card-title">Cov Propagation</span>
              <button className="info-tip-btn" aria-label="Show explanation" title="What does this mean?">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                  <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.5" />
                  <text x="8" y="12" textAnchor="middle" fontSize="10" fontWeight="700" fill="currentColor">i</text>
                </svg>
              </button>
            </div>
          }
        >
          <span className="tip-title">Covariance Propagation</span>
          <p className="tip-desc">
            Propagates <span className="tip-mono">E[neuron]</span> and <span className="tip-mono">Cov(neuron_i, neuron_j)</span> analytically — no sampling required.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Runtime</span><span className="tip-kv-val">O(depth × width²)</span></div>
          <div className="tip-kv"><span className="tip-kv-key">State</span><span className="tip-kv-val">O(width²) — covariance matrix per layer</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            More accurate than Mean Propagation because it tracks how neuron outputs co-vary.
          </p>
        </InfoTip>
        <p className="estimator-card-desc">
          Tracks mean + covariance: corrects for neuron correlations. O(n²) per layer.
        </p>
        <button
          className="run-btn run-btn-covprop"
          onClick={runCovProp}
          disabled={!!running}
        >
          {running === "covprop"
            ? "Running…"
            : <><svg className="btn-icon" width="13" height="13" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="5 3 19 12 5 21 5 3" /></svg> Cov Propagation</>}
        </button>
        {renderProgressBar("covprop")}
        {results.covprop && !running && (
          <div className="estimator-card-result">
            <span className="result-stat">{formatTime(results.covprop.time)}</span>
            <span className="result-detail">analytic</span>
          </div>
        )}
      </div>
    </div>
  );
}
