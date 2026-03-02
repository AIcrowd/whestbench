/**
 * GateDetailOverlay — Shows gate neighborhood on heatmap hover.
 * Left band (layer l-1) → Gate element → Right band (layer l+1)
 *
 * Each band is a continuous vertical strip of ALL wires in the adjacent layer,
 * color-coded by E[wire]. Highlighted wires (inputs / consumers) are marked
 * with bold coral triangles pointing inward.
 */
import { useEffect, useMemo, useRef } from "react";
import { classifyGate, meanToColor } from "./gateShapes";

const BAND_WIDTH = 28;
const BAND_HEIGHT = 280;
const ARROW_SIZE = 12;
const ARROW_PAD = 14;  // left padding for arrow triangles
const ARROW_COLOR = "#F0524D";

/**
 * Draw a continuous wire band to a canvas element.
 * All N wires are rendered as a single vertical strip.
 * @param {'left'|'right'} side — which side the arrows appear on
 */
function drawBand(canvas, { n, means, layerIdx, highlightWires, side = 'left' }) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  const wireH = BAND_HEIGHT / n;
  const totalW = BAND_WIDTH + ARROW_PAD;
  const bandX = side === 'left' ? ARROW_PAD : 0;

  canvas.width = totalW * dpr;
  canvas.height = BAND_HEIGHT * dpr;
  canvas.style.width = `${totalW}px`;
  canvas.style.height = `${BAND_HEIGHT}px`;

  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, totalW, BAND_HEIGHT);

  // Draw all wires as a continuous strip
  for (let wi = 0; wi < n; wi++) {
    const y = wi * wireH;
    const mean = means && means[layerIdx] ? means[layerIdx][wi] : null;

    ctx.fillStyle = mean !== null ? meanToColor(mean) : "#E5E7EB";
    ctx.fillRect(bandX, y, BAND_WIDTH, Math.max(wireH, 0.5));
  }

  // Draw coral arrow indicators for highlighted wires
  for (const wi of highlightWires) {
    const y = wi * wireH;
    const cy = y + wireH / 2;

    // Highlight stripe with coral border
    ctx.strokeStyle = ARROW_COLOR;
    ctx.lineWidth = 2;
    ctx.strokeRect(bandX + 1, y, BAND_WIDTH - 2, Math.max(wireH, 3));

    // Coral triangle pointing inward
    ctx.fillStyle = ARROW_COLOR;
    ctx.beginPath();
    if (side === 'left') {
      // Arrow on left, pointing right into the band
      ctx.moveTo(0, cy - ARROW_SIZE / 2);
      ctx.lineTo(ARROW_PAD - 2, cy);
      ctx.lineTo(0, cy + ARROW_SIZE / 2);
    } else {
      // Arrow on right, pointing left into the band
      ctx.moveTo(totalW, cy - ARROW_SIZE / 2);
      ctx.lineTo(BAND_WIDTH + 2, cy);
      ctx.lineTo(totalW, cy + ARROW_SIZE / 2);
    }
    ctx.closePath();
    ctx.fill();
  }
}

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

  const leftCanvasRef = useRef(null);
  const rightCanvasRef = useRef(null);

  // Gate info
  const gateInfo = useMemo(() => {
    if (l < 0 || l >= d || w < 0 || w >= n) return null;
    return classifyGate(circuit.gates[l], w);
  }, [circuit, l, w, n, d]);

  // Input wires for this gate
  const gate = (l >= 0 && l < d) ? circuit.gates[l] : null;
  const inputFirst = gate ? gate.first[w] : 0;
  const inputSecond = gate ? gate.second[w] : 0;

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

  // Draw left band (layer l-1)
  useEffect(() => {
    if (!leftCanvasRef.current || l <= 0) return;
    drawBand(leftCanvasRef.current, {
      n,
      means,
      layerIdx: l - 1,
      highlightWires: [inputFirst, inputSecond],
      side: 'left',
    });
  }, [n, means, l, inputFirst, inputSecond]);

  // Draw right band (layer l+1)
  useEffect(() => {
    if (!rightCanvasRef.current || l >= d - 1) return;
    drawBand(rightCanvasRef.current, {
      n,
      means,
      layerIdx: l + 1,
      highlightWires: consumers,
      side: 'right',
    });
  }, [n, means, l, d, consumers]);

  if (!gateInfo) return null;

  // Compute overlay position — try to keep it visible
  const overlayStyle = {
    position: "absolute",
    left: Math.max(10, Math.min(position.x - 100, (typeof window !== 'undefined' ? window.innerWidth - 320 : 800))),
    top: Math.max(10, position.y - 200),
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
            <canvas ref={leftCanvasRef} />
          </div>
        )}

        {/* Center — gate element */}
        <div className="detail-gate">
          <div className="canvas-tip-header" style={{ border: "none", padding: "6px 0" }}>
            Layer <span className="layer-num">{l}</span>
            {" · "}
            Wire <span className="layer-num">{w}</span>
          </div>
          <div
            className="gate-symbol"
            style={{
              borderColor: currentMean !== null ? meanToColor(currentMean) : gateInfo.color,
              background: currentMean !== null ? meanToColor(currentMean) + '18' : undefined,
            }}
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
            <canvas ref={rightCanvasRef} />
          </div>
        )}
      </div>
    </div>
  );
}
