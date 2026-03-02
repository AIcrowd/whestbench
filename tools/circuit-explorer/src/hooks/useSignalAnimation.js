/**
 * useSignalAnimation — Hook for animated signal flow.
 * Separated into its own file to support React Fast Refresh
 * (files exporting hooks alongside components break HMR).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { runSingleTrial } from "../circuit";

export function useSignalAnimation(circuit, depth) {
  const [isAnimating, setIsAnimating] = useState(false);
  const [animLayer, setAnimLayer] = useState(-1);
  const [trialValues, setTrialValues] = useState(null);
  const timerRef = useRef(null);
  const seedRef = useRef(1);

  const startAnimation = useCallback(() => {
    if (!circuit || isAnimating) return;

    const seed = seedRef.current++;
    const values = runSingleTrial(circuit, seed);
    setTrialValues(values);
    setAnimLayer(-1);
    setIsAnimating(true);

    let layer = 0;
    const tick = () => {
      setAnimLayer(layer);
      layer++;
      if (layer < depth) {
        timerRef.current = setTimeout(tick, Math.max(80, 400 / depth));
      } else {
        timerRef.current = setTimeout(() => {
          setIsAnimating(false);
        }, 600);
      }
    };
    timerRef.current = setTimeout(tick, 200);
  }, [circuit, depth, isAnimating]);

  // Cleanup on unmount
  useEffect(() => {
    return () => { if (timerRef.current) clearTimeout(timerRef.current); };
  }, []);

  return { isAnimating, animLayer, trialValues, startAnimation };
}
