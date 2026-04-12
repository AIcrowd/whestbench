/**
 * NetworkGraph — JointJS-based visualization for small MLPs (width ≤ 8).
 *
 * Layout: one column of nodes per layer (input + hidden layers + output).
 * - Input layer: open/outlined circles with dashed border (visually distinct)
 * - Hidden layers: solid circles colored by activation magnitude (gray → coral)
 * - Output layer: double-bordered circles (visually distinct from hidden)
 * - Edges: weight value → color (negative=dark slate, zero=gray, positive=coral),
 *          thickness proportional to |weight|
 * - Interactivity: click neuron to highlight incoming/outgoing connections
 */
import { dia, shapes } from "@joint/core";
import { useCallback, useEffect, useRef, useState } from "react";

/* ------------------------------------------------------------------ */
/*  Layout constants                                                   */
/* ------------------------------------------------------------------ */
const NODE_R   = 20;       // neuron circle radius
const COL_GAP  = 120;      // horizontal gap between columns
const ROW_GAP  = 10;       // vertical gap between nodes
const PAD_X    = 44;
const PAD_Y    = 34;

/* ------------------------------------------------------------------ */
/*  Color helpers                                                      */
/* ------------------------------------------------------------------ */
// Activation magnitude → color (gray-200 → coral)
// Matches app palette: --gray-200 (#D9DCDC) at 0, --coral (#F0524D) at high
function activationColor(v) {
  if (v === null || v === undefined) return "#D9DCDC";
  const t = Math.max(0, Math.min(1, v));
  // Interpolate: #D9DCDC (217,220,220) → #F0524D (240,82,77)
  const r = Math.round(217 + (240 - 217) * t);
  const g = Math.round(220 + (82 - 220) * t);
  const b = Math.round(220 + (77 - 220) * t);
  return `rgb(${r},${g},${b})`;
}

// Weight value → color using app palette
// Negative: dark slate (#334155), zero: gray (#94A3B8), positive: coral (#F0524D)
function weightColor(w) {
  if (Math.abs(w) < 0.001) return "#AAACAD"; // --gray-400
  if (w < 0) {
    // Interpolate gray-400 → dark slate as magnitude increases
    const t = Math.min(1, Math.abs(w) * 2);
    // #AAACAD (170,172,173) → #334155 (51,65,85)
    const r = Math.round(170 + (51 - 170) * t);
    const g = Math.round(172 + (65 - 172) * t);
    const b = Math.round(173 + (85 - 173) * t);
    return `rgb(${r},${g},${b})`;
  } else {
    // Interpolate gray-400 → coral as magnitude increases
    const t = Math.min(1, w * 2);
    // #AAACAD (170,172,173) → #F0524D (240,82,77)
    const r = Math.round(170 + (240 - 170) * t);
    const g = Math.round(172 + (82 - 172) * t);
    const b = Math.round(173 + (77 - 173) * t);
    return `rgb(${r},${g},${b})`;
  }
}

function weightWidth(w) {
  return 0.5 + Math.min(3, Math.abs(w) * 3);
}

const EDGE_OPACITY = 0.25;  // base edge opacity — keeps strong weights visible, reduces clutter

// Pick a contrasting label color (white or dark) for a given fill
function contrastLabel(fill) {
  if (!fill) return "#1E293B";
  const m = fill.match(/(\d+)/g);
  if (!m || m.length < 3) return "#1E293B";
  const [r, g, b] = m.map(Number);
  const lum = 0.299 * r + 0.587 * g + 0.114 * b;
  return lum > 160 ? "#1E293B" : "#FFFFFF";
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
export default function NetworkGraph({ mlp, means, activeLayer, onNodeSelect }) {
  const containerRef = useRef(null);
  const paperRef = useRef(null);
  const graphRef = useRef(null);
  const [highlighted, setHighlighted] = useState(null); // { col, row }
  const [zoomPct, setZoomPct] = useState(100);

  const { width, depth, weights } = mlp;
  // Total columns: 1 input + depth hidden
  const numCols = depth + 1;

  /* ---- Build/update graph ---- */
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Compute canvas dimensions
    const nodeH = NODE_R * 2;
    const totalH = PAD_Y * 2 + width * nodeH + (width - 1) * ROW_GAP;
    const totalW = PAD_X * 2 + numCols * NODE_R * 2 + (numCols - 1) * COL_GAP;

    /* ---- Init or reset JointJS graph ---- */
    if (!graphRef.current) {
      graphRef.current = new dia.Graph();
    } else {
      graphRef.current.clear();
    }

    if (!paperRef.current) {
      paperRef.current = new dia.Paper({
        el: container,
        model: graphRef.current,
        width: totalW,
        height: totalH,
        gridSize: 1,
        interactive: false,
        background: { color: "transparent" },
        defaultConnector: { name: "smooth" },
        defaultRouter: { name: "normal" },
        defaultConnectionPoint: { name: "anchor" },
        sorting: "sorting-exact",
      });
    } else {
      paperRef.current.setDimensions(totalW, totalH);
    }

    const graph = graphRef.current;

    // Add drop-shadow filter to SVG for node depth
    const svgEl = paperRef.current.svg;
    if (svgEl && !svgEl.querySelector("#node-shadow")) {
      const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
      defs.innerHTML = `
        <filter id="node-shadow" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="1" stdDeviation="2" flood-color="#000" flood-opacity="0.12" />
        </filter>
      `;
      svgEl.insertBefore(defs, svgEl.firstChild);
    }

    // Node IDs: nodeId(col, row) for quick lookup
    const nodeMap = {}; // key: `${col},${row}` → cell id
    const allCells = []; // batch all cells for a single resetCells() call

    /* ---- Create neuron nodes ---- */
    for (let col = 0; col < numCols; col++) {
      const cx = PAD_X + NODE_R + col * (NODE_R * 2 + COL_GAP);

      for (let row = 0; row < width; row++) {
        const cy = PAD_Y + NODE_R + row * (nodeH + ROW_GAP);

        // Determine fill, stroke, and style per layer type
        const isInput = col === 0;
        const isOutput = col === numCols - 1;
        const isActive = activeLayer !== undefined && activeLayer !== null && col === activeLayer + 1;

        let fillColor, strokeColor, strokeW, strokeDash, labelColor;

        if (isInput) {
          // Input layer — open circles with dashed border
          fillColor = "#FFFFFF";
          strokeColor = isActive ? "#F0524D" : "#5D5F60"; // --gray-600
          strokeW = isActive ? 2.5 : 2;
          strokeDash = "4,3";
          labelColor = "#5D5F60";
        } else if (isOutput) {
          // Output layer — double-border effect (thicker stroke + filled)
          const layerIdx = col - 1;
          const activation = means ? means[layerIdx]?.[row] ?? null : null;
          fillColor = activationColor(activation);
          strokeColor = isActive ? "#F0524D" : "#292C2D"; // --gray-900
          strokeW = isActive ? 3 : 3;
          strokeDash = "";
          labelColor = contrastLabel(fillColor);
        } else {
          // Hidden layers — solid circles colored by activation
          const layerIdx = col - 1;
          const activation = means ? means[layerIdx]?.[row] ?? null : null;
          fillColor = activationColor(activation);
          strokeColor = isActive ? "#F0524D" : "#292C2D"; // --gray-900
          strokeW = isActive ? 2.5 : 1.5;
          strokeDash = "";
          labelColor = contrastLabel(fillColor);
        }

        const ellipse = new shapes.standard.Ellipse({
          position: { x: cx - NODE_R, y: cy - NODE_R },
          size: { width: NODE_R * 2, height: NODE_R * 2 },
          attrs: {
            body: {
              fill: fillColor,
              stroke: strokeColor,
              strokeWidth: strokeW,
              strokeDasharray: strokeDash,
              cursor: "pointer",
              filter: "url(#node-shadow)",
            },
            label: {
              text: isInput ? `x${row}` : isOutput ? `y${row}` : `${row}`,
              fontSize: 10,
              fill: labelColor,
              fontFamily: "'IBM Plex Mono', monospace",
            },
          },
        });
        ellipse.set("nodeKey", { col, row });
        ellipse.set("z", 10);
        allCells.push(ellipse);
        nodeMap[`${col},${row}`] = ellipse.id;
      }
    }

    /* ---- Create weight edges ---- */
    // weights: Array of Float32Array, one per layer, each width×width row-major
    // weights[l][i * width + j] = weight from input neuron i to output neuron j
    // (row-vector convention: x @ W, so W[i,j] connects input i to output j)
    for (let l = 0; l < depth; l++) {
      const srcCol = l;       // input column for this layer
      const dstCol = l + 1;   // output column for this layer
      const W = weights[l];

      for (let j = 0; j < width; j++) {
        // destination neuron j
        for (let i = 0; i < width; i++) {
          // source neuron i
          const wVal = W[i * width + j];
          if (Math.abs(wVal) < 0.05) continue; // skip near-zero weights for clarity

          const srcId = nodeMap[`${srcCol},${i}`];
          const dstId = nodeMap[`${dstCol},${j}`];
          if (!srcId || !dstId) continue;

          const link = new shapes.standard.Link({
            source: { id: srcId },
            target: { id: dstId },
            attrs: {
              line: {
                stroke: weightColor(wVal),
                strokeWidth: weightWidth(wVal),
                strokeLinecap: "round",
                targetMarker: { type: "none" },
                opacity: EDGE_OPACITY,
              },
            },
            z: -1, // behind nodes
          });
          link.set("edgeKey", { l, i, j, w: wVal });
          allCells.push(link);
        }
      }
    }

    /* ---- Batch-add all cells (single render pass) ---- */
    graph.resetCells(allCells);

    /* ---- Fit network to viewport ---- */
    const paper = paperRef.current;
    paper.scaleContentToFit({ padding: 20, maxScale: 1 });
    const scale = paper.scale().sx;
    setZoomPct(Math.round(scale * 100));

    /* ---- Click handler ---- */
    paperRef.current.off("element:pointerclick");
    paperRef.current.on("element:pointerclick", (cellView) => {
      const key = cellView.model.get("nodeKey");
      if (!key) return;
      setHighlighted((prev) => {
        if (prev && prev.col === key.col && prev.row === key.row) return null;
        return key;
      });
    });

    /* ---- Blank click to deselect ---- */
    paperRef.current.off("blank:pointerclick");
    paperRef.current.on("blank:pointerclick", () => setHighlighted(null));

    // Update container style
    container.style.width = `${totalW}px`;
    container.style.height = `${totalH}px`;
  }, [mlp, means, activeLayer, width, depth, numCols, weights]);

  /* ---- Apply highlight / dim logic ---- */
  useEffect(() => {
    if (!graphRef.current || !paperRef.current) return;
    const graph = graphRef.current;

    graph.getCells().forEach((cell) => {
      if (!highlighted) {
        // Reset all
        if (cell.isLink()) {
          cell.attr("line/opacity", EDGE_OPACITY);
          cell.attr("line/strokeWidth", weightWidth(cell.get("edgeKey")?.w ?? 0));
        } else {
          cell.attr("body/opacity", 1);
        }
        return;
      }

      const { col, row } = highlighted;

      if (cell.isLink()) {
        const ek = cell.get("edgeKey");
        if (!ek) return;
        // Check if this edge is connected to the highlighted neuron
        const srcCol = ek.l;
        const dstCol = ek.l + 1;
        const isConnected = (srcCol === col && ek.i === row) || (dstCol === col && ek.j === row);
        cell.attr("line/opacity", isConnected ? 1 : 0.08);
        cell.attr("line/strokeWidth", isConnected ? weightWidth(ek.w) * 2 : 0.5);
      } else {
        const nk = cell.get("nodeKey");
        if (!nk) return;
        // Keep selected node and its direct neighbors fully opaque
        // so edges don't show through semi-transparent node fills
        const isSelected = nk.col === col && nk.row === row;
        const isNeighbor = nk.col === col - 1 || nk.col === col + 1;
        cell.attr("body/opacity", isSelected || isNeighbor ? 1 : 0.3);
      }
    });

    // Re-assert node-on-top ordering after attr changes
    // (JointJS attr mutations can disturb SVG DOM order)
    graph.getElements().forEach((el) => el.toFront());
  }, [highlighted]);

  /* ---- Notify parent of node selection ---- */
  useEffect(() => {
    if (onNodeSelect) onNodeSelect(highlighted);
  }, [highlighted, onNodeSelect]);

  /* ---- Zoom (mousewheel) ---- */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const onWheel = (e) => {
      e.preventDefault();
      e.stopPropagation();
      const paper = paperRef.current;
      if (!paper) return;
      const f = e.deltaY > 0 ? 0.92 : 1.08;
      const sx = paper.scale().sx;
      const ns = Math.max(0.15, Math.min(4, sx * f));
      paper.scale(ns, ns);
      setZoomPct(Math.round(ns * 100));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  /* ---- Pan (drag on blank area) ---- */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let panning = false;
    let startX = 0, startY = 0, origTx = 0, origTy = 0;

    const onDown = (e) => {
      if (e.target.closest(".joint-element") || e.target.closest(".joint-link")) return;
      panning = true;
      startX = e.clientX;
      startY = e.clientY;
      const paper = paperRef.current;
      if (paper) {
        const t = paper.translate();
        origTx = t.tx;
        origTy = t.ty;
      }
      el.style.cursor = "grabbing";
    };
    const onMove = (e) => {
      if (!panning) return;
      const paper = paperRef.current;
      if (!paper) return;
      paper.translate(origTx + (e.clientX - startX), origTy + (e.clientY - startY));
    };
    const onUp = () => {
      panning = false;
      el.style.cursor = "";
    };
    el.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      el.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, []);

  return (
    <div className="panel" style={{ overflowX: "auto" }}>
      <h2>
        Network Graph
        <span className="mode-badge">
          width={width} · depth={depth}
        </span>
        {zoomPct !== 100 && (
          <span className="mode-badge" style={{ marginLeft: 6 }}>
            {zoomPct}%
          </span>
        )}
      </h2>
      <div style={{ position: "relative", overflowX: "auto" }}>
        <div ref={containerRef} style={{ display: "inline-block" }} />
      </div>
      <div className="formula-legend" style={{ marginTop: 6 }}>
        <span style={{ color: "#334155" }}>━ negative weight</span>
        <span style={{ color: "#AAACAD" }}>━ ~zero</span>
        <span style={{ color: "#F0524D" }}>━ positive weight</span>
        <span style={{ marginLeft: 12 }}>
          <span style={{ border: "2px dashed #5D5F60", borderRadius: "50%", display: "inline-block", width: 10, height: 10, verticalAlign: "middle" }} /> input
        </span>
        <span style={{ marginLeft: 6 }}>
          <span style={{ border: "3px solid #292C2D", borderRadius: "50%", display: "inline-block", width: 10, height: 10, verticalAlign: "middle", background: "#F7A09D" }} /> output
        </span>
        <span style={{ color: "#F0524D", marginLeft: 6 }}>● high activation</span>
        <span style={{ color: "#9CA3AF", marginLeft: 16, fontSize: 11 }}>Scroll to zoom · Drag to pan</span>
      </div>
      {highlighted && (
        <p style={{ fontSize: 11, color: "#9CA3AF", margin: "4px 0 0" }}>
          Showing connections for neuron {highlighted.row} in layer {highlighted.col === 0 ? "input" : highlighted.col - 1}.
          Click again or click blank to deselect.
        </p>
      )}
    </div>
  );
}
