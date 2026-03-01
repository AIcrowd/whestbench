/**
 * GateDetailOverlay — Shows gate neighborhood on heatmap hover.
 * Left band (layer l-1) → Gate element → Right band (layer l+1)
 */
import { useMemo } from "react";
import { classifyGate, meanToColor } from "./gateShapes";

const BAND_WIDTH = 24;
const BAND_HEIGHT = 280;

export default function GateDetailOverlay({
  circuit,
  means,
  hoveredWire,
  hoveredLayer,
  position,
}) {
  const n = circuit.n;
  const d = circuit.d;
  const l = hoveredLayer;
  const w = hoveredWire;

  // Gate info — always called (no conditional hooks)
  const gateInfo = useMemo(() => {
    if (l < 0 || l >= d || w < 0 || w >= n) return null;
    return classifyGate(circuit.gates[l], w);
  }, [circuit, l, w, n, d]);

  // Compute consumers: wires in layer l+1 that use wire w as input
  const consumers = useMemo(() => {
    if (l < 0 || l >= d - 1 || w < 0 || w >= n) return [];
    const nextGate = circuit.gates[l + 1];
    const result = [];
    for (let wi = 0; wi < n; wi++) {
      if (nextGate.first[wi] === w || nextGate.second[wi] === w) {
        result.push(wi);
      }
    }
    return result;
  }, [circuit, l, w, n, d]);

  if (!gateInfo) return null;

  const gate = circuit.gates[l];
  const inputFirst = gate.first[w];
  const inputSecond = gate.second[w];

  // Wire height in bands
  const wireH = Math.max(1, Math.min(4, BAND_HEIGHT / n));
  const bandTotalH = wireH * n;

  // Compute overlay position — try to keep it visible
  const overlayStyle = {
    position: "absolute",
    left: Math.max(10, position.x - 100),
    top: Math.max(10, position.y - bandTotalH - 80),
    zIndex: 100,
  };

  const currentMean = means && means[l] ? means[l][w] : null;

  return (
    <div className="gate-detail-overlay" style={overlayStyle}>
      <div className="detail-content">
        {/* Left band — layer l-1 */}
        {l > 0 && (
          <div className="detail-band">
            <div className="band-label">L{l - 1}</div>
            <svg width={BAND_WIDTH} height={bandTotalH}>
              {Array.from({ length: n }, (_, wi) => {
                const mean = means && means[l - 1] ? means[l - 1][wi] : null;
                const isInput = wi === inputFirst || wi === inputSecond;
                return (
                  <g key={wi}>
                    <rect
                      x={isInput ? 0 : 4}
                      y={wi * wireH}
                      width={isInput ? BAND_WIDTH : BAND_WIDTH - 8}
                      height={wireH}
                      fill={meanToColor(mean)}
                      stroke={isInput ? "#F0524D" : "none"}
                      strokeWidth={isInput ? 1.5 : 0}
                      rx={isInput ? 2 : 0}
                    />
                    {isInput && (
                      <text
                        x={BAND_WIDTH + 2}
                        y={wi * wireH + wireH / 2 + 3}
                        fontSize={7}
                        fill="#F0524D"
                        fontWeight="bold"
                      >
                        ▶
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        )}

        {/* Center — gate element */}
        <div className="detail-gate">
          <div className="gate-header">
            Layer {l}, Wire {w}
          </div>
          <div
            className="gate-symbol"
            style={{ borderColor: gateInfo.color }}
          >
            <div className="gate-op">{gateInfo.label}</div>
          </div>
          <div className="gate-coefficients">
            <div className="coeff-row">
              <span className="coeff-label">c</span>
              <span className="coeff-val">{gate.const[w].toFixed(3)}</span>
            </div>
            <div className="coeff-row">
              <span className="coeff-label">a·x[{inputFirst}]</span>
              <span className="coeff-val">{gate.firstCoeff[w].toFixed(3)}</span>
            </div>
            <div className="coeff-row">
              <span className="coeff-label">b·y[{inputSecond}]</span>
              <span className="coeff-val">{gate.secondCoeff[w].toFixed(3)}</span>
            </div>
            <div className="coeff-row">
              <span className="coeff-label">p·xy</span>
              <span className="coeff-val">{gate.productCoeff[w].toFixed(3)}</span>
            </div>
          </div>
          {currentMean !== null && (
            <div className="gate-mean">
              E[wire] = <strong>{currentMean.toFixed(4)}</strong>
            </div>
          )}
        </div>

        {/* Right band — layer l+1 */}
        {l < d - 1 && (
          <div className="detail-band">
            <div className="band-label">L{l + 1}</div>
            <svg width={BAND_WIDTH} height={bandTotalH}>
              {Array.from({ length: n }, (_, wi) => {
                const mean = means && means[l + 1] ? means[l + 1][wi] : null;
                const isConsumer = consumers.includes(wi);
                return (
                  <g key={wi}>
                    <rect
                      x={isConsumer ? 0 : 4}
                      y={wi * wireH}
                      width={isConsumer ? BAND_WIDTH : BAND_WIDTH - 8}
                      height={wireH}
                      fill={meanToColor(mean)}
                      stroke={isConsumer ? "#3B82F6" : "none"}
                      strokeWidth={isConsumer ? 1.5 : 0}
                      rx={isConsumer ? 2 : 0}
                    />
                    {isConsumer && (
                      <text
                        x={-6}
                        y={wi * wireH + wireH / 2 + 3}
                        fontSize={7}
                        fill="#3B82F6"
                        fontWeight="bold"
                      >
                        ◀
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>
        )}
      </div>
    </div>
  );
}
