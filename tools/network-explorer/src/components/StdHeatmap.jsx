/**
 * StdHeatmap — Neuron × Layer standard deviation heatmap.
 * Shows where activations are most variable (input-dependent).
 * Low σ = dark (predictable), High σ = bright coral (variable).
 * Orientation: X-axis = Layer, Y-axis = Neuron (matches network layout).
 * Uses putImageData for gap-free pixel-perfect rendering.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import HeatmapTooltip from "./HeatmapTooltip";
import InfoTip from "./InfoTip";

function stdToRGB(std, maxStd) {
  const t = Math.min(1, std / Math.max(0.01, maxStd));
  const mapped = t * 2 - 1;
  if (mapped < 0) {
    const s = 1 + mapped;
    return [51 + (204 * s) | 0, 65 + (190 * s) | 0, 85 + (170 * s) | 0];
  } else {
    return [255 - (15 * mapped) | 0, 255 - (173 * mapped) | 0, 255 - (178 * mapped) | 0];
  }
}

export default function StdHeatmap({ stds, width: n, depth: d }) {
  const containerRef = useRef(null);
  const canvasRef = useRef(null);
  const [dims, setDims] = useState({ width: 0, height: 0 });
  const maxStdRef = useRef(0);

  useEffect(() => {
    if (!stds || stds.length === 0 || !canvasRef.current || !containerRef.current) return;

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

    // Find global max std
    let maxStd = 0;
    for (let l = 0; l < d && l < stds.length; l++) {
      for (let w = 0; w < n; w++) {
        if (stds[l][w] > maxStd) maxStd = stds[l][w];
      }
    }
    maxStdRef.current = maxStd;

    // putImageData path — pixel-perfect, no gaps
    const imgData = ctx.createImageData(canvasW, canvasH);
    const pixels = imgData.data;

    for (let py = 0; py < canvasH; py++) {
      const neuron = Math.floor((py / canvasH) * n);
      for (let px = 0; px < canvasW; px++) {
        const layer = Math.floor((px / canvasW) * d);
        const idx = (py * canvasW + px) * 4;
        const val = (layer < stds.length) ? (stds[layer][neuron] || 0) : 0;
        const [r, g, b] = stdToRGB(val, maxStd);
        pixels[idx] = r;
        pixels[idx + 1] = g;
        pixels[idx + 2] = b;
        pixels[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);

    setDims({ width, height });
  }, [stds, n, d]);

  const getData = useCallback((layer, neuron) => {
    if (!stds || layer >= stds.length) return 0;
    return stds[layer][neuron] || 0;
  }, [stds]);

  const getColor = useCallback((layer, neuron) => {
    if (!stds || layer >= stds.length) return [30, 41, 59];
    return stdToRGB(stds[layer][neuron] || 0, maxStdRef.current);
  }, [stds]);

  if (!stds || stds.length === 0) return null;

  return (
    <div className="panel" ref={containerRef}>
      <h2>
        Signal Variability <small>(σ)</small>
        <InfoTip>
          <span className="tip-title">Signal Variability</span>
          <p className="tip-desc">
            Per-neuron standard deviation <span className="tip-highlight">σ</span> across random inputs. Each cell is one neuron at one layer.
          </p>
          <div className="tip-sep" />
          <div className="tip-kv"><span className="tip-kv-key">Bright coral</span><span className="tip-kv-val">High σ — activation varies a lot with inputs</span></div>
          <div className="tip-kv"><span className="tip-kv-key">Dark</span><span className="tip-kv-val">Low σ — nearly constant regardless of input</span></div>
          <div className="tip-sep" />
          <p className="tip-desc">
            High-variability neurons are harder to estimate from few samples.
          </p>
        </InfoTip>
      </h2>
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
            valueLabel="σ"
            showZoom={n * d > 4096}
          />
        )}
      </div>
      <div className="heatmap-axes">
        <span className="axis-label-x">Layer →</span>
        <span className="axis-label-y">Neuron ↓</span>
      </div>
      <div className="heatmap-legend">
        <span className="legend-label">Low σ</span>
        <div className="legend-gradient" />
        <span className="legend-label">High σ</span>
      </div>
    </div>
  );
}
