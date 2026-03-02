import { useCallback, useRef, useState } from "react";

export default function Controls({ params, onParamsChange }) {
  const [localSeed, setLocalSeed] = useState(String(params.seed));
  // Local slider state for instant visual feedback while debouncing actual generation
  const [localParams, setLocalParams] = useState(params);
  const debounceRef = useRef(null);

  // When parent params change (e.g., tour mode), sync local state
  if (params.width !== localParams.width || params.depth !== localParams.depth) {
    if (!debounceRef.current) {
      setLocalParams(params);
    }
  }

  const handleSliderChange = useCallback((key, value) => {
    const next = { ...localParams, [key]: value };
    setLocalParams(next);

    // Debounce: wait 300ms after last change before triggering circuit generation
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      debounceRef.current = null;
      onParamsChange(next);
    }, 300);
  }, [localParams, onParamsChange]);

  const slider = (label, key, min, max, step = 1, tooltip = "") => (
    <div className="control-row">
      <label title={tooltip}>
        <span className="control-label">{label}</span>
        <span className="control-value">{localParams[key]}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={localParams[key]}
        onChange={(e) => handleSliderChange(key, Number(e.target.value))}
      />
    </div>
  );

  return (
    <div className="controls-panel">
      <h2>Circuit</h2>
      {slider(<>Depth <code>d</code> (Layers)</>, "depth", 1, 256, 1, "Number of gate layers in the circuit")}
      {slider(<>Width <code>n</code> (Wires)</>, "width", 2, 1024, 1, "Number of wires (parallel values) per layer")}

      <div className="control-row">
        <label>
          <span className="control-label">Seed</span>
        </label>
        <input
          type="number"
          className="seed-input"
          value={localSeed}
          onChange={(e) => setLocalSeed(e.target.value)}
          onBlur={() =>
            onParamsChange({ ...localParams, seed: Number(localSeed) || 42 })
          }
          onKeyDown={(e) => {
            if (e.key === "Enter")
              onParamsChange({ ...localParams, seed: Number(localSeed) || 42 });
          }}
        />
      </div>

      <button
        className="regenerate-btn"
        onClick={() => {
          const newSeed = Math.floor(Math.random() * 100000);
          setLocalSeed(String(newSeed));
          const next = { ...localParams, seed: newSeed };
          setLocalParams(next);
          onParamsChange(next);
        }}
      >
        <svg className="btn-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" /><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" /></svg> Regenerate
      </button>

      <div className="controls-help">
        <p className="controls-hint">
          Circuit regenerates automatically when you change parameters.
        </p>
      </div>
    </div>
  );
}
