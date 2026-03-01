/**
 * GateDetailOverlay — Shows gate neighborhood on heatmap hover.
 * Left band (layer l-1) → Gate element → Right band (layer l+1)
 *
 * Performance notes:
 * - Bands use canvas instead of SVG (instant draw for 1024+ wires)
 * - Focus window: only shows ~60 wires around inputs for large circuits
 * - Split window: if inputs are far apart, shows two mini-windows with gap indicator
 */
import { useEffect, useMemo, useRef } from "react";
import { classifyGate, meanToColor } from "./gateShapes";

const BAND_WIDTH = 28;
const BAND_HEIGHT = 280;
const MAX_VISIBLE = 60;

/**
 * Compute which wires to display in the focus window.
 */
function computeFocusWindow(n, inputFirst, inputSecond) {
  if (n <= MAX_VISIBLE) {
    return { type: "full", start: 0, end: n };
  }
  const lo = Math.min(inputFirst, inputSecond);
  const hi = Math.max(inputFirst, inputSecond);
  const gap = hi - lo;

  if (gap <= MAX_VISIBLE - 10) {
    // Inputs close enough — single window centered on both
    const center = Math.floor((lo + hi) / 2);
    const half = Math.floor(MAX_VISIBLE / 2);
    const start = Math.max(0, Math.min(n - MAX_VISIBLE, center - half));
    return { type: "single", start, end: Math.min(n, start + MAX_VISIBLE) };
  } else {
    // Inputs far apart — two mini-windows
    const halfWin = Math.floor(MAX_VISIBLE / 2) - 2;
    return {
      type: "split",
      windowA: {
        start: Math.max(0, lo - Math.floor(halfWin / 2)),
        end: Math.min(n, lo + Math.ceil(halfWin / 2)),
      },
      windowB: {
        start: Math.max(0, hi - Math.floor(halfWin / 2)),
        end: Math.min(n, hi + Math.ceil(halfWin / 2)),
      },
    };
  }
}

/**
 * Draw a wire band to a canvas element.
 * @param {HTMLCanvasElement} canvas
 * @param {object} params
 */
function drawBand(canvas, {
  n, means, layerIdx, highlightWires, highlightColor, focusWindow
}) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;

  // Determine visible wires and layout
  let wireRanges = [];
  let totalVisible = 0;
  const hasContext = focusWindow.type !== "full";

  if (focusWindow.type === "full") {
    wireRanges = [{ start: focusWindow.start, end: focusWindow.end }];
    totalVisible = focusWindow.end - focusWindow.start;
  } else if (focusWindow.type === "single") {
    wireRanges = [{ start: focusWindow.start, end: focusWindow.end }];
    totalVisible = focusWindow.end - focusWindow.start;
  } else {
    wireRanges = [focusWindow.windowA, focusWindow.windowB];
    totalVisible = (focusWindow.windowA.end - focusWindow.windowA.start) +
                   (focusWindow.windowB.end - focusWindow.windowB.start) + 2; // +2 for gap
  }

  const wireH = Math.max(1, Math.min(5, BAND_HEIGHT / totalVisible));
  const bandH = wireH * totalVisible;
  const contextBarH = hasContext ? 3 : 0;
  const totalH = bandH + (hasContext ? contextBarH * 2 + 4 : 0);

  canvas.width = BAND_WIDTH * dpr;
  canvas.height = totalH * dpr;
  canvas.style.width = `${BAND_WIDTH}px`;
  canvas.style.height = `${totalH}px`;

  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, BAND_WIDTH, totalH);

  let y = 0;

  // Top context bar — compressed gradient of wires above
  if (hasContext && wireRanges[0].start > 0) {
    drawContextBar(ctx, means, layerIdx, 0, wireRanges[0].start, BAND_WIDTH, contextBarH, y);
    y += contextBarH + 2;
  }

  // Draw wire ranges
  for (let ri = 0; ri < wireRanges.length; ri++) {
    const range = wireRanges[ri];
    for (let wi = range.start; wi < range.end; wi++) {
      const mean = means && means[layerIdx] ? means[layerIdx][wi] : null;
      const isHighlight = highlightWires.includes(wi);
      const x = isHighlight ? 0 : 3;
      const w = isHighlight ? BAND_WIDTH : BAND_WIDTH - 6;

      ctx.fillStyle = mean !== null ? meanToColor(mean) : "#E5E7EB";
      ctx.fillRect(x, y, w, wireH);

      if (isHighlight) {
        ctx.strokeStyle = highlightColor;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(x + 0.75, y + 0.75, w - 1.5, wireH - 1.5);
        // Wire label
        ctx.fillStyle = highlightColor;
        ctx.font = "bold 7px 'IBM Plex Mono', monospace";
        ctx.textAlign = "left";
        ctx.fillText(`w${wi}`, BAND_WIDTH + 2, y + wireH / 2 + 3);
      }
      y += wireH;
    }

    // Gap indicator between split windows
    if (ri < wireRanges.length - 1) {
      ctx.fillStyle = "#E5E7EB";
      ctx.fillRect(0, y, BAND_WIDTH, wireH * 0.5);
      ctx.fillStyle = "#9CA3AF";
      ctx.font = "7px 'IBM Plex Mono', monospace";
      ctx.textAlign = "center";
      ctx.fillText("⋮", BAND_WIDTH / 2, y + wireH * 0.5 - 1);
      y += wireH;
      // Second gap row
      ctx.fillStyle = "#E5E7EB";
      ctx.fillRect(0, y, BAND_WIDTH, wireH * 0.5);
      y += wireH;
    }
  }

  // Bottom context bar
  if (hasContext) {
    const lastRange = wireRanges[wireRanges.length - 1];
    if (lastRange.end < n) {
      y += 2;
      drawContextBar(ctx, means, layerIdx, lastRange.end, n, BAND_WIDTH, contextBarH, y);
    }
  }
}

/**
 * Draw a compressed 3px context bar showing the overall gradient.
 */
function drawContextBar(ctx, means, layerIdx, startWire, endWire, width, height, y) {
  const count = endWire - startWire;
  const step = Math.max(1, Math.floor(count / width));
  for (let px = 0; px < width; px++) {
    const wi = startWire + Math.floor(px * count / width);
    const mean = means && means[layerIdx] ? means[layerIdx][wi] : null;
    ctx.fillStyle = mean !== null ? meanToColor(mean) : "#E5E7EB";
    ctx.fillRect(px, y, 1, height);
  }
  // Subtle border
  ctx.strokeStyle = "rgba(0,0,0,0.1)";
  ctx.lineWidth = 0.5;
  ctx.strokeRect(0, y, width, height);
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

  // Focus windows for left and right bands
  const leftFocus = useMemo(() =>
    computeFocusWindow(n, inputFirst, inputSecond),
    [n, inputFirst, inputSecond]
  );

  const rightFocus = useMemo(() => {
    if (consumers.length === 0) return computeFocusWindow(n, w, w);
    if (consumers.length === 1) return computeFocusWindow(n, w, consumers[0]);
    // Use first and last consumer for window range
    const sorted = [...consumers].sort((a, b) => a - b);
    return computeFocusWindow(n, sorted[0], sorted[sorted.length - 1]);
  }, [n, w, consumers]);

  // Draw left band (layer l-1)
  useEffect(() => {
    if (!leftCanvasRef.current || l <= 0) return;
    drawBand(leftCanvasRef.current, {
      n,
      means,
      layerIdx: l - 1,
      highlightWires: [inputFirst, inputSecond],
      highlightColor: "#F0524D",
      focusWindow: leftFocus,
    });
  }, [n, means, l, inputFirst, inputSecond, leftFocus]);

  // Draw right band (layer l+1)
  useEffect(() => {
    if (!rightCanvasRef.current || l >= d - 1) return;
    drawBand(rightCanvasRef.current, {
      n,
      means,
      layerIdx: l + 1,
      highlightWires: consumers,
      highlightColor: "#94A3B8",
      focusWindow: rightFocus,
    });
  }, [n, means, l, d, consumers, rightFocus]);

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
            <canvas ref={rightCanvasRef} />
          </div>
        )}
      </div>
    </div>
  );
}
