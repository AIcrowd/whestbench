/**
 * SignalHeatmap — Wire means heatmap using TF.js Vis.
 * Replaces the SVG rect-based heatmap with tfvis.render.heatmap.
 */
import * as tfvis from "@tensorflow/tfjs-vis";
import { useEffect, useRef } from "react";

export default function SignalHeatmap({ means, width: n, depth: d, source }) {
  const heatmapRef = useRef(null);

  useEffect(() => {
    if (!means || means.length === 0 || !heatmapRef.current) return;
    heatmapRef.current.innerHTML = '';

    // Build 2D values array (layers × wires) for the heatmap
    const values = [];
    for (let l = 0; l < d && l < means.length; l++) {
      const row = [];
      for (let w = 0; w < n; w++) {
        row.push(means[l][w] || 0);
      }
      values.push(row);
    }

    // Generate tick labels
    const yTickLabels = [];
    const labelStep = d > 16 ? Math.ceil(d / 8) : 1;
    for (let l = 0; l < d; l++) {
      yTickLabels.push(l % labelStep === 0 ? `${l}` : '');
    }

    const xTickLabels = [];
    const wireLabelStep = n > 16 ? Math.ceil(n / 8) : 1;
    for (let w = 0; w < n; w++) {
      xTickLabels.push(w % wireLabelStep === 0 ? `${w}` : '');
    }

    tfvis.render.heatmap(heatmapRef.current, {
      values,
      xTickLabels,
      yTickLabels,
    }, {
      width: heatmapRef.current.offsetWidth || 500,
      height: Math.min(400, Math.max(180, d * 14 + 40)),
      xLabel: 'Wire',
      yLabel: 'Layer',
      domain: [-1, 1],
      colorMap: 'blues',
      rowMajor: true,
    });
  }, [means, n, d]);

  if (!means || means.length === 0) return null;

  return (
    <div className="panel">
      <h2>
        Wire Means Heatmap
        {source && <span className="source-badge">{source}</span>}
      </h2>
      <div ref={heatmapRef} style={{ width: '100%', minHeight: 180 }} />
      <div className="heatmap-legend">
        <span className="legend-item" style={{ color: "#334155" }}>◆ −1</span>
        <span className="legend-item" style={{ color: "#9CA3AF" }}>◆ 0</span>
        <span className="legend-item" style={{ color: "#F0524D" }}>◆ +1</span>
      </div>
    </div>
  );
}
