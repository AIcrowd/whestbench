/**
 * WireStats — per-layer wire mean distribution using TF.js Vis.
 * Line chart for μ ± σ band, scatter for individual wire means.
 */
import * as tfvis from "@tensorflow/tfjs-vis";
import { useEffect, useRef } from "react";

export default function WireStats({ means, width: n, depth: d, source, activeLayer }) {
  const lineRef = useRef(null);
  const scatterRef = useRef(null);

  useEffect(() => {
    if (!means || means.length === 0) return;
    if (!lineRef.current || !scatterRef.current) return;

    // Compute per-layer stats
    const meanSeries = [];
    const upperSeries = [];
    const lowerSeries = [];
    const minSeries = [];
    const maxSeries = [];

    for (let l = 0; l < d && l < means.length; l++) {
      let sum = 0, sumSq = 0;
      let mn = Infinity, mx = -Infinity;
      for (let w = 0; w < n; w++) {
        const val = means[l][w] || 0;
        sum += val;
        sumSq += val * val;
        if (val < mn) mn = val;
        if (val > mx) mx = val;
      }
      const avg = sum / n;
      const variance = sumSq / n - avg * avg;
      const std = Math.sqrt(Math.max(0, variance));
      meanSeries.push({ x: l, y: avg });
      upperSeries.push({ x: l, y: Math.min(1, avg + std) });
      lowerSeries.push({ x: l, y: Math.max(-1, avg - std) });
      minSeries.push({ x: l, y: mn });
      maxSeries.push({ x: l, y: mx });
    }

    // Line chart: mean, upper, lower, min, max
    lineRef.current.innerHTML = '';
    tfvis.render.linechart(lineRef.current, {
      values: [meanSeries, upperSeries, lowerSeries, minSeries, maxSeries],
      series: ['μ (mean)', 'μ + σ', 'μ − σ', 'min', 'max'],
    }, {
      width: lineRef.current.offsetWidth || 500,
      height: 220,
      xLabel: 'Layer',
      yLabel: 'E[wire]',
      yAxisDomain: [-1, 1],
      seriesColors: ['#F0524D', '#F7A09D', '#F7A09D', '#94A3B8', '#334155'],
    });

    // Scatter plot: individual wire means
    const scatterData = [];
    for (let l = 0; l < d && l < means.length; l++) {
      for (let w = 0; w < n; w++) {
        scatterData.push({ x: l, y: means[l][w] || 0 });
      }
    }

    scatterRef.current.innerHTML = '';
    tfvis.render.scatterplot(scatterRef.current, {
      values: [scatterData],
      series: ['Wire Means'],
    }, {
      width: scatterRef.current.offsetWidth || 500,
      height: 140,
      xLabel: 'Layer',
      yLabel: 'E[wire]',
      yAxisDomain: [-1, 1],
      seriesColors: ['#F0524D'],
    });
  }, [means, n, d, activeLayer]);

  if (!means || means.length === 0) return null;

  return (
    <div className="panel">
      <h2>
        Wire Mean Distribution
        {source && <span className="source-badge">{source}</span>}
      </h2>
      <div ref={lineRef} style={{ width: '100%', minHeight: 220 }} />

      <h3 className="subheading" style={{ marginTop: 12 }}>Individual Wire Means</h3>
      <div ref={scatterRef} style={{ width: '100%', minHeight: 140 }} />
    </div>
  );
}
