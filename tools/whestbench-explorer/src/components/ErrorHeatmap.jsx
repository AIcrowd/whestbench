/**
 * ErrorHeatmap — Estimation error per neuron per layer.
 * Shows |groundTruth - estimate| as a heatmap for either Mean Propagation
 * or Sampling estimator, selectable via dropdown.
 * Orientation: X-axis = Layer, Y-axis = Neuron (matches network layout).
 * Uses putImageData for gap-free pixel-perfect rendering.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import HeatmapTooltip from "./HeatmapTooltip";
import InfoTip from "./InfoTip";

function errorToRGB(err, maxErr) {
  const t = Math.min(1, err / Math.max(0.001, maxErr));
  const mapped = t * 2 - 1;
  if (mapped < 0) {
    const s = 1 + mapped;
    return [51 + (204 * s) | 0, 65 + (190 * s) | 0, 85 + (170 * s) | 0];
  } else {
    return [255 - (15 * mapped) | 0, 255 - (173 * mapped) | 0, 255 - (178 * mapped) | 0];
  }
}

const ESTIMATORS = {
  meanprop: { label: "Mean Propagation", key: "meanprop" },
  covprop:  { label: "Cov Propagation",  key: "covprop" },
  sampling: { label: "Sampling", key: "sampling" },
};

export default function ErrorHeatmap({
  groundTruth,
  meanPropEstimates,
  covPropEstimates,
  samplingEstimates,
  width: n,
  depth: d,
}) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [dims, setDims] = useState({ width: 0, height: 0 });
  const [selectedEstimator, setSelectedEstimator] = useState(null);

  // Determine which estimators are available
  const availableEstimators = useMemo(() => {
    const avail = [];
    if (groundTruth && meanPropEstimates) avail.push("meanprop");
    if (groundTruth && covPropEstimates) avail.push("covprop");
    if (groundTruth && samplingEstimates) avail.push("sampling");
    return avail;
  }, [groundTruth, meanPropEstimates, covPropEstimates, samplingEstimates]);

  // Auto-select when availability changes
  useEffect(() => {
    if (availableEstimators.length === 0) {
      setSelectedEstimator(null);
    } else if (!selectedEstimator || !availableEstimators.includes(selectedEstimator)) {
      setSelectedEstimator(availableEstimators[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableEstimators]);

  // Pick the active estimates based on selection
  const activeEstimates = useMemo(() => {
    if (selectedEstimator === "meanprop") return meanPropEstimates;
    if (selectedEstimator === "covprop") return covPropEstimates;
    if (selectedEstimator === "sampling") return samplingEstimates;
    return null;
  }, [selectedEstimator, meanPropEstimates, covPropEstimates, samplingEstimates]);

  const { errors, maxErr } = useMemo(() => {
    if (!groundTruth || !activeEstimates) return { errors: null, maxErr: 0 };
    const layers = Math.min(d, groundTruth.length, activeEstimates.length);
    const errs = [];
    let mx = 0;
    for (let l = 0; l < layers; l++) {
      const layerErr = new Float32Array(n);
      for (let w = 0; w < n; w++) {
        const e = Math.abs((groundTruth[l][w] || 0) - (activeEstimates[l][w] || 0));
        layerErr[w] = e;
        if (e > mx) mx = e;
      }
      errs.push(layerErr);
    }
    return { errors: errs, maxErr: mx };
  }, [groundTruth, activeEstimates, n, d]);

  useEffect(() => {
    if (!errors || !canvasRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const canvas = canvasRef.current;
    const rect = container.getBoundingClientRect();
    const width = Math.floor(rect.width);
    const MAX_HEIGHT = 300;
    const height = Math.min(MAX_HEIGHT, Math.max(100, Math.floor(n * 2)));

    const dpr = window.devicePixelRatio || 1;
    const rawCellW = width / d;
    const rawCellH = height / n;
    const renderScale = (rawCellW < 1 || rawCellH < 1)
      ? Math.max(1, Math.min(dpr, 2))
      : dpr;

    const canvasW = Math.round(width * renderScale);
    const canvasH = Math.round(height * renderScale);

    canvas.width = canvasW;
    canvas.height = canvasH;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");

    // putImageData path — pixel-perfect, no gaps
    const imgData = ctx.createImageData(canvasW, canvasH);
    const pixels = imgData.data;

    for (let py = 0; py < canvasH; py++) {
      const neuron = Math.floor((py / canvasH) * n);
      for (let px = 0; px < canvasW; px++) {
        const layer = Math.floor((px / canvasW) * d);
        const idx = (py * canvasW + px) * 4;
        const val = (layer < errors.length) ? (errors[layer][neuron] || 0) : 0;
        const [r, g, b] = errorToRGB(val, maxErr);
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    setDims({ width, height });
  }, [errors, maxErr, n, d]);

  const getData = useCallback((layer, neuron) => {
    if (!errors || layer >= errors.length) return 0;
    return errors[layer][neuron] || 0;
  }, [errors]);

  const getColor = useCallback((layer, neuron) => {
    if (!errors || layer >= errors.length) return [30, 41, 59];
    return errorToRGB(errors[layer][neuron] || 0, maxErr);
  }, [errors, maxErr]);

  const hasData = !!errors;

  return (
    <div
      className="panel"
      ref={containerRef}
      style={!hasData ? { display: 'flex', flexDirection: 'column', height: '100%' } : {}}
    >
      <div className="error-heatmap-header">
        <h2>
          Estimation Error
          <InfoTip>
            <span className="tip-title">Estimation Error</span>
            <p className="tip-desc">
              Heatmap of <span className="tip-mono">|ground truth − estimate|</span> for each neuron at each layer.
            </p>
            <div className="tip-sep" />
            <div className="tip-kv"><span className="tip-kv-key">Bright coral</span><span className="tip-kv-val">Large estimation error</span></div>
            <div className="tip-kv"><span className="tip-kv-key">Dark</span><span className="tip-kv-val">Small or zero error</span></div>
            <div className="tip-sep" />
            <p className="tip-desc">
              Use the dropdown to switch between <span className="tip-highlight">Mean Propagation</span> and <span className="tip-highlight">Sampling</span> estimators.
            </p>
          </InfoTip>
        </h2>
        {availableEstimators.length > 1 && (
          <select
            className="estimator-select"
            value={selectedEstimator || ""}
            onChange={(e) => setSelectedEstimator(e.target.value)}
          >
            {availableEstimators.map((key) => (
              <option key={key} value={key}>
                {ESTIMATORS[key].label}
              </option>
            ))}
          </select>
        )}
        {availableEstimators.length === 1 && (
          <span className="estimator-badge-label">
            {ESTIMATORS[availableEstimators[0]].label}
          </span>
        )}
      </div>

      {!hasData ? (
        <div className="error-heatmap-empty" style={{ flex: 1 }}>
          <div className="error-heatmap-empty-grid">
            {Array.from({ length: 6 }).map((_, r) => (
              <div key={r} className="error-heatmap-empty-row">
                {Array.from({ length: 10 }).map((_, c) => (
                  <div key={c} className="error-heatmap-empty-cell" />
                ))}
              </div>
            ))}
          </div>
          <p className="error-heatmap-empty-msg">
            Run Ground Truth sampling and an estimator to populate this plot
          </p>
        </div>
      ) : (
        <>
          <div className="heatmap-canvas-container">
            <canvas ref={canvasRef} style={{ display: "block" }} />
            {dims.width > 0 && (
              <HeatmapTooltip
                width={dims.width}
                height={dims.height}
                n={n}
                d={d}
                getData={getData}
                getColor={getColor}
                valueLabel="Abs. Error"
                showZoom={n * d > 4096}
              />
            )}
          </div>
          <div className="heatmap-axes">
            <span className="axis-label-x">Layer →</span>
            <span className="axis-label-y">Neuron ↓</span>
          </div>
          <div className="heatmap-legend">
            <span className="legend-label">0</span>
            <div className="legend-gradient" />
            <span className="legend-label">max = {maxErr.toFixed(4)}</span>
          </div>
        </>
      )}
    </div>
  );
}
