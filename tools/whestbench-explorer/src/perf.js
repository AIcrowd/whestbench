/**
 * perf.js — Lightweight performance instrumentation.
 * Uses performance.mark/measure under the hood.
 * Only active in dev mode (import.meta.env.DEV).
 * Zero overhead in production — all calls are no-ops.
 */

const enabled = typeof window !== 'undefined' && import.meta.env.DEV;
const timings = new Map();   // name → { last, avg, count }
const listeners = new Set();

export function perfStart(name) {
  if (!enabled) return;
  performance.mark(`${name}-start`);
}

export function perfEnd(name) {
  if (!enabled) return;
  const startMark = `${name}-start`;
  const endMark = `${name}-end`;
  performance.mark(endMark);

  try {
    const measure = performance.measure(name, startMark, endMark);
    const ms = measure.duration;

    const prev = timings.get(name) || { last: 0, avg: 0, count: 0 };
    prev.count++;
    prev.last = ms;
    prev.avg = prev.avg + (ms - prev.avg) / prev.count;
    timings.set(name, prev);

    // Defer listener notifications to avoid React render-phase state update warnings
    queueMicrotask(() => {
      listeners.forEach(fn => fn(new Map(timings)));
    });
  } catch {
    // marks may have been cleared
  } finally {
    performance.clearMarks(startMark);
    performance.clearMarks(endMark);
    performance.clearMeasures(name);
  }
}

/** Subscribe to timing updates. Returns unsubscribe function. */
export function onPerfUpdate(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

/** Get all current timings. */
export function getPerfTimings() {
  return new Map(timings);
}

/** Reset all timings. */
export function resetPerf() {
  timings.clear();
  listeners.forEach(fn => fn(new Map()));
}
