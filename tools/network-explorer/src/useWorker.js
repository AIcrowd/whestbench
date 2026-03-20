/**
 * useCircuitWorker — React hook wrapping the circuit Web Worker.
 *
 * Returns { run, isRunning } where run(type, params) returns a Promise
 * that resolves with the worker's result.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export function useCircuitWorker() {
  const workerRef = useRef(null);
  const callbackRef = useRef(null);
  const idRef = useRef(0);
  const [isRunning, setIsRunning] = useState(false);

  useEffect(() => {
    const worker = new Worker(
      new URL('./circuit.worker.js', import.meta.url),
      { type: 'module' }
    );
    worker.onmessage = (e) => {
      const { id, result, error } = e.data;
      if (callbackRef.current?.id === id) {
        if (error) callbackRef.current.reject(new Error(error));
        else callbackRef.current.resolve(result);
        callbackRef.current = null;
        setIsRunning(false);
      }
    };
    workerRef.current = worker;
    return () => worker.terminate();
  }, []);

  const run = useCallback((type, params) => {
    return new Promise((resolve, reject) => {
      const id = ++idRef.current;
      callbackRef.current = { id, resolve, reject };
      setIsRunning(true);
      workerRef.current.postMessage({ id, type, params });
    });
  }, []);

  // Memoize to keep stable reference — prevents consumer useEffects
  // from re-firing when isRunning toggles
  return useMemo(() => ({ run, isRunning }), [run, isRunning]);
}
