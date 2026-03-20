/**
 * NetworkGraph — JointJS-based visualization for small MLPs (width ≤ 8).
 *
 * Layout: one column of nodes per layer (input + hidden layers).
 * - Input layer: neutral gray nodes
 * - Hidden layers: nodes colored by activation magnitude (dark → coral)
 * - Edges: weight value → color (negative=blue, zero=gray, positive=red),
 *          thickness proportional to |weight|
 * - Interactivity: click neuron to highlight incoming/outgoing connections
 */
import { dia, shapes } from "@joint/core";
import { useCallback, useEffect, useRef, useState } from "react";

/* ------------------------------------------------------------------ */
/*  Layout constants                                                   */
/* ------------------------------------------------------------------ */
const NODE_R   = 16;       // neuron circle radius
const COL_GAP  = 110;      // horizontal gap between columns
const ROW_GAP  = 14;       // vertical gap between nodes
const PAD_X    = 40;
const PAD_Y    = 30;

/* ------------------------------------------------------------------ */
/*  Color helpers                                                      */
/* ------------------------------------------------------------------ */
function activationColor(v) {
  if (v === null || v === undefined) return "#CBD5E1";
  const t = Math.max(0, Math.min(1, v));
  const r = Math.round(51 + 204 * t);
  const g = Math.round(65 - 65 * t);
  const b = Math.round(85 - 85 * t);
  return `rgb(${r},${g},${b})`;
}

function weightColor(w) {
  if (Math.abs(w) < 0.001) return "#94A3B8";
  if (w < 0) {
    const t = Math.min(1, Math.abs(w) * 2);
    const r = Math.round(59 + (1 - t) * (200 - 59));
    const g = Math.round(130 + (1 - t) * (200 - 130));
    const b = Math.round(246);
    return `rgb(${r},${g},${b})`;
  } else {
    const t = Math.min(1, w * 2);
    return `rgb(${Math.round(240 - (1 - t) * 80)},${Math.round(82 - (1 - t) * 40)},${Math.round(77 - (1 - t) * 30)})`;
  }
}

function weightWidth(w) {
  return 0.5 + Math.min(4, Math.abs(w) * 4);
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
export default function NetworkGraph({ mlp, means, activeLayer }) {
  const containerRef = useRef(null);
  const paperRef = useRef(null);
  const graphRef = useRef(null);
  const [highlighted, setHighlighted] = useState(null); // { col, row }

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
        defaultConnector: { name: "straight" },
        defaultRouter: { name: "normal" },
      });
    } else {
      paperRef.current.setDimensions(totalW, totalH);
    }

    const graph = graphRef.current;

    // Node IDs: nodeId(col, row) for quick lookup
    const nodeMap = {}; // key: `${col},${row}` → cell id

    /* ---- Create neuron nodes ---- */
    for (let col = 0; col < numCols; col++) {
      const cx = PAD_X + NODE_R + col * (NODE_R * 2 + COL_GAP);

      for (let row = 0; row < width; row++) {
        const cy = PAD_Y + NODE_R + row * (nodeH + ROW_GAP);

        // Determine fill color
        let fillColor;
        if (col === 0) {
          // Input layer — neutral
          fillColor = "#CBD5E1";
        } else {
          // Hidden layer — colored by activation mean
          const layerIdx = col - 1;
          const activation = means ? means[layerIdx * width + row] : null;
          fillColor = activationColor(activation);
        }

        const isActive = activeLayer !== undefined && activeLayer !== null && col === activeLayer + 1;
        const strokeColor = isActive ? "#F0524D" : "#1E293B";
        const strokeW = isActive ? 2.5 : 1.5;

        const ellipse = new shapes.standard.Ellipse({
          position: { x: cx - NODE_R, y: cy - NODE_R },
          size: { width: NODE_R * 2, height: NODE_R * 2 },
          attrs: {
            body: {
              fill: fillColor,
              stroke: strokeColor,
              strokeWidth: strokeW,
              cursor: "pointer",
            },
            label: {
              text: `${row}`,
              fontSize: 9,
              fill: col === 0 ? "#475569" : "#fff",
              fontFamily: "'IBM Plex Mono', monospace",
            },
          },
        });
        ellipse.set("nodeKey", { col, row });
        graph.addCell(ellipse);
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
          if (Math.abs(wVal) < 0.01) continue; // skip near-zero weights for clarity

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
                targetMarker: { type: "none" },
                opacity: 0.65,
              },
            },
            z: -1, // behind nodes
          });
          link.set("edgeKey", { l, i, j, w: wVal });
          graph.addCell(link);
        }
      }
    }

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
          cell.attr("line/opacity", 0.65);
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
        cell.attr("body/opacity", nk.col === col && nk.row === row ? 1 : 0.3);
      }
    });
  }, [highlighted]);

  return (
    <div className="panel" style={{ overflowX: "auto" }}>
      <h2>
        Network Graph
        <span className="mode-badge">
          width={width} · depth={depth}
        </span>
      </h2>
      <div style={{ position: "relative", overflowX: "auto" }}>
        <div ref={containerRef} style={{ display: "inline-block" }} />
      </div>
      <div className="formula-legend" style={{ marginTop: 6 }}>
        <span style={{ color: "#3B82F6" }}>━ negative weight</span>
        <span style={{ color: "#94A3B8" }}>━ ~zero</span>
        <span style={{ color: "#F0524D" }}>━ positive weight</span>
        <span style={{ color: "#CBD5E1", marginLeft: 12 }}>● input</span>
        <span style={{ color: "#F0524D" }}>● high activation</span>
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
