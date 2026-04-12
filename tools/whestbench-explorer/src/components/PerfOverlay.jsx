import { useEffect, useState } from 'react';
import { onPerfUpdate, resetPerf } from '../perf';

function fmt(ms) {
  if (ms < 0.01) return '<0.01ms';
  if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function badge(ms) {
  if (ms < 10) return '✅';
  if (ms < 100) return '🟡';
  return '🔴';
}

/**
 * Build a clean, standalone SVG blob from the JointJS paper element.
 * Returns null if no network SVG is found.
 */
function buildNetworkSVGBlob() {
  const svgEl = document.querySelector('.joint-paper svg');
  if (!svgEl) return null;

  const clone = svgEl.cloneNode(true);

  // Remove root style attr — it contains CSS positioning (e.g. "position: absolute")
  // which is invalid in standalone SVG and causes rendering errors.
  clone.removeAttribute('style');

  // Remove non-standard JointJS attributes
  clone.removeAttribute('joint-selector');
  clone.querySelectorAll('[joint-selector]').forEach(el =>
    el.removeAttribute('joint-selector')
  );

  // Add proper SVG namespaces
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.setAttribute('xmlns:xlink', 'http://www.w3.org/1999/xlink');

  // Tight viewBox from bounding box + padding
  const bbox = svgEl.getBBox();
  const pad = 25;
  clone.setAttribute('viewBox',
    `${bbox.x - pad} ${bbox.y - pad} ${bbox.width + 2 * pad} ${bbox.height + 2 * pad}`
  );
  clone.setAttribute('width', Math.round(bbox.width + 2 * pad));
  clone.setAttribute('height', Math.round(bbox.height + 2 * pad));

  // Insert white background rect
  const bg = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  bg.setAttribute('x', bbox.x - pad);
  bg.setAttribute('y', bbox.y - pad);
  bg.setAttribute('width', bbox.width + 2 * pad);
  bg.setAttribute('height', bbox.height + 2 * pad);
  bg.setAttribute('fill', 'white');
  clone.insertBefore(bg, clone.firstChild);

  const svgString = new XMLSerializer().serializeToString(clone);
  return new Blob([svgString], { type: 'image/svg+xml' });
}

const SVG_FILENAME = 'whestbench-explorer-visualization.svg';

/**
 * Download the network SVG. Uses the File System Access API (showSaveFilePicker)
 * for a native save dialog with proper filename; falls back to blob URL.
 */
async function downloadSVG() {
  const blob = buildNetworkSVGBlob();
  if (!blob) {
    alert('No network SVG found on the page.');
    return;
  }

  // Prefer native save dialog (Chrome/Edge 86+)
  if (typeof window.showSaveFilePicker === 'function') {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName: SVG_FILENAME,
        types: [{ description: 'SVG Image', accept: { 'image/svg+xml': ['.svg'] } }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return;
    } catch (err) {
      if (err.name === 'AbortError') return; // user cancelled
      // fall through to legacy approach
    }
  }

  // Fallback: blob URL with delayed cleanup
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = SVG_FILENAME;
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, 500);
}

export default function PerfOverlay() {
  const [timings, setTimings] = useState(new Map());
  const [open, setOpen] = useState(false);

  useEffect(() => onPerfUpdate(setTimings), []);

  if (!import.meta.env.DEV) return null;

  return (
    <div className="perf-overlay" data-open={open}>
      <button className="perf-toggle" onClick={downloadSVG} title="Download network as SVG">
        ↓ SVG
      </button>
      <button className="perf-toggle" onClick={() => setOpen(!open)}>
        ⚡ Perf {!open && timings.size > 0 && `(${timings.size})`}
      </button>
      {open && (
        <div className="perf-panel">
          {timings.size === 0 ? (
            <p style={{ color: 'var(--text-muted)', margin: '8px 0' }}>
              No measurements yet. Interact with the app to see timings.
            </p>
          ) : (
            <table>
              <thead>
                <tr><th>Marker</th><th>Last</th><th>Avg</th><th>N</th><th></th></tr>
              </thead>
              <tbody>
                {[...timings.entries()].map(([name, t]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{fmt(t.last)}</td>
                    <td>{fmt(t.avg)}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{t.count}</td>
                    <td>{badge(t.last)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <button className="perf-reset" onClick={resetPerf}>Reset</button>
        </div>
      )}
    </div>
  );
}
