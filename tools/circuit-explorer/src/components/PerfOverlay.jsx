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

export default function PerfOverlay() {
  const [timings, setTimings] = useState(new Map());
  const [open, setOpen] = useState(false);

  useEffect(() => onPerfUpdate(setTimings), []);

  if (!import.meta.env.DEV) return null;
  if (timings.size === 0 && !open) return null;

  return (
    <div className="perf-overlay" data-open={open}>
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
