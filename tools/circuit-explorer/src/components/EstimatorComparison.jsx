/**
 * EstimatorComparison — per-layer MSE comparison using TF.js Vis.
 * Shows bar chart of estimation error per layer for each estimator.
 */
import * as tfvis from "@tensorflow/tfjs-vis";
import { useEffect, useRef } from "react";

export default function EstimatorComparison({
  groundTruth,
  samplingEstimates,
  meanPropEstimates,
  depth,
  activeLayer,
}) {
  const chartRef = useRef(null);

  useEffect(() => {
    if (!groundTruth || !chartRef.current) return;
    chartRef.current.innerHTML = '';

    const hasSampling = !!samplingEstimates;
    const hasMeanProp = !!meanPropEstimates;

    const layers = Math.min(
      depth,
      groundTruth.length,
      hasSampling ? samplingEstimates.length : Infinity,
      hasMeanProp ? meanPropEstimates.length : Infinity
    );

    // Build data for each estimator as separate series
    const series = [];
    const seriesNames = [];
    const colors = [];

    if (hasSampling) {
      const samplingData = [];
      for (let l = 0; l < layers; l++) {
        const n = groundTruth[l].length;
        let mse = 0;
        for (let i = 0; i < n; i++) {
          mse += (samplingEstimates[l][i] - groundTruth[l][i]) ** 2;
        }
        samplingData.push({ x: l, y: mse / n });
      }
      series.push(samplingData);
      seriesNames.push('Sampling Error');
      colors.push('#F0524D');
    }

    if (hasMeanProp) {
      const meanPropData = [];
      for (let l = 0; l < layers; l++) {
        const n = groundTruth[l].length;
        let mse = 0;
        for (let i = 0; i < n; i++) {
          mse += (meanPropEstimates[l][i] - groundTruth[l][i]) ** 2;
        }
        meanPropData.push({ x: l, y: mse / n });
      }
      series.push(meanPropData);
      seriesNames.push('Mean Prop Error');
      colors.push('#94A3B8');
    }

    if (series.length === 0) return;

    // Use linechart with markers to show MSE per layer
    tfvis.render.linechart(chartRef.current, {
      values: series,
      series: seriesNames,
    }, {
      width: chartRef.current.offsetWidth || 600,
      height: 240,
      xLabel: 'Layer',
      yLabel: 'MSE',
      seriesColors: colors,
      zoomToFit: true,
    });
  }, [groundTruth, samplingEstimates, meanPropEstimates, depth, activeLayer]);

  if (!groundTruth) return null;

  return (
    <div className="panel">
      <h2>Estimation Error (MSE per Layer)</h2>
      <div ref={chartRef} style={{ width: '100%', minHeight: 240 }} />
    </div>
  );
}
