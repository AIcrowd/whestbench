import { useState } from "react";

export default function Controls({ params, onParamsChange }) {
  const [localSeed, setLocalSeed] = useState(String(params.seed));

  const slider = (label, key, min, max, step = 1) => (
    <div className="control-row">
      <label>
        <span className="control-label">{label}</span>
        <span className="control-value">{params[key]}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={params[key]}
        onChange={(e) =>
          onParamsChange({ ...params, [key]: Number(e.target.value) })
        }
      />
    </div>
  );

  return (
    <div className="controls-panel">
      <h2>Circuit</h2>
      {slider("Width (n)", "width", 2, 1024)}
      {slider("Depth (d)", "depth", 1, 256)}

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
            onParamsChange({ ...params, seed: Number(localSeed) || 42 })
          }
          onKeyDown={(e) => {
            if (e.key === "Enter")
              onParamsChange({ ...params, seed: Number(localSeed) || 42 });
          }}
        />
      </div>

      <button
        className="regenerate-btn"
        onClick={() => {
          const newSeed = Math.floor(Math.random() * 100000);
          setLocalSeed(String(newSeed));
          onParamsChange({ ...params, seed: newSeed });
        }}
      >
        <svg className="btn-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 4 23 10 17 10" /><polyline points="1 20 1 14 7 14" /><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15" /></svg> Regenerate
      </button>

      <div className="controls-help">
        <p>
          <strong>Width</strong> — wires per layer
        </p>
        <p>
          <strong>Depth</strong> — gate layers
        </p>
        <p className="controls-hint">
          Circuit regenerates automatically when you change parameters.
        </p>
      </div>
    </div>
  );
}
