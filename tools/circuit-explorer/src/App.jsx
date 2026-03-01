import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";
import { randomCircuit } from "./circuit";
import CircuitGraphJoint from "./components/CircuitGraphJoint";
import CircuitHeatmap from "./components/CircuitHeatmap";
import Controls from "./components/Controls";
import EstimatorComparison from "./components/EstimatorComparison";
import EstimatorRunner from "./components/EstimatorRunner";
import GateStats from "./components/GateStats";
import SignalHeatmap from "./components/SignalHeatmap";
import WireStats from "./components/WireStats";

const DEFAULT_PARAMS = { width: 8, depth: 6, seed: 42 };
const GRAPH_MODE_THRESHOLD = 4096; // n×d threshold for JointJS vs heatmap

export default function App() {
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [activeLayer, setActiveLayer] = useState(undefined);
  const [estimatorResults, setEstimatorResults] = useState({});

  // Auto-regenerate circuit when params change
  const circuit = useMemo(
    () => randomCircuit(params.width, params.depth, params.seed),
    [params.width, params.depth, params.seed]
  );

  // Determine rendering mode
  const totalGates = params.width * params.depth;
  const useGraphMode = totalGates <= GRAPH_MODE_THRESHOLD;

  // Clear estimator results when circuit changes
  useEffect(() => {
    setEstimatorResults({});
    setActiveLayer(undefined);
  }, [circuit]);

  const handleEstimatorResult = useCallback((key, result) => {
    setEstimatorResults((prev) => ({ ...prev, [key]: result }));
  }, []);

  // Derive visualization data from results
  const groundTruth = estimatorResults.groundTruth?.estimates || null;
  const samplingEst = estimatorResults.sampling?.estimates || null;
  const meanPropEst = estimatorResults.meanprop?.estimates || null;
  const hasAnyEstimate = groundTruth || samplingEst || meanPropEst;
  const displayMeans = groundTruth || samplingEst || meanPropEst;

  return (
    <div className="app">
      <header className="app-header">
        <h1>
          <span className="header-icon">⚡</span> Circuit Explorer
        </h1>
        <p className="subtitle">
          Interactive visualization of Boolean circuits for the{" "}
          <em>Mechanistic Estimation</em> challenge
        </p>
      </header>

      <div className="app-layout">
        <aside className="sidebar">
          <Controls params={params} onParamsChange={setParams} />

          <EstimatorRunner
            circuit={circuit}
            onResult={handleEstimatorResult}
          />

          {/* Layer stepper — only useful in graph mode */}
          {useGraphMode && (
            <div className="layer-stepper">
              <h3>Step Through</h3>
              <div className="stepper-buttons">
                <button
                  onClick={() =>
                    setActiveLayer((l) =>
                      l === undefined ? 0 : Math.max(0, l - 1)
                    )
                  }
                >
                  ◀ Prev
                </button>
                <span className="layer-indicator">
                  {activeLayer === undefined
                    ? "All layers"
                    : `Layer ${activeLayer}`}
                </span>
                <button
                  onClick={() =>
                    setActiveLayer((l) =>
                      l === undefined
                        ? 0
                        : Math.min(circuit.d - 1, l + 1)
                    )
                  }
                >
                  Next ▶
                </button>
              </div>
              <button
                className="reset-btn"
                onClick={() => setActiveLayer(undefined)}
              >
                Show All
              </button>
            </div>
          )}
        </aside>

        <main className="main-content">
          {/* Circuit visualization — adaptive mode */}
          {useGraphMode ? (
            <CircuitGraphJoint
              circuit={circuit}
              means={displayMeans}
              activeLayer={activeLayer}
            />
          ) : (
            <CircuitHeatmap
              circuit={circuit}
              means={displayMeans}
            />
          )}

          {/* Gate structure analysis — always visible */}
          <GateStats circuit={circuit} />

          {/* After estimates are available */}
          {hasAnyEstimate && (
            <>
              <div className="panels-row">
                <SignalHeatmap
                  means={displayMeans}
                  width={params.width}
                  depth={params.depth}
                />
                <WireStats
                  means={displayMeans}
                  width={params.width}
                  depth={params.depth}
                />
              </div>

              {groundTruth && (samplingEst || meanPropEst) && (
                <EstimatorComparison
                  groundTruth={groundTruth}
                  samplingEstimates={samplingEst}
                  meanPropEstimates={meanPropEst}
                  depth={params.depth}
                />
              )}
            </>
          )}

          {!hasAnyEstimate && (
            <div className="empty-state">
              <div className="empty-state-inner">
                <span className="empty-icon">📊</span>
                <h3>Ready to Explore</h3>
                <p>
                  This circuit has <strong>{params.width}</strong> wires and{" "}
                  <strong>{params.depth}</strong> layers of random gates.
                </p>
                <p className="empty-hint">
                  Your goal: estimate the mean output E[wire] of each wire
                  over uniform ±1 inputs.
                </p>
                <p className="empty-hint">
                  Start with <strong>Ground Truth</strong> (10k samples) to
                  see exact means, then compare{" "}
                  <strong>Sampling</strong> and{" "}
                  <strong>Mean Propagation</strong>.
                </p>
              </div>
            </div>
          )}
        </main>
      </div>

      <footer className="app-footer">
        <p>
          Part of the{" "}
          <a
            href="https://www.alignment.org/"
            target="_blank"
            rel="noopener noreferrer"
          >
            ARC
          </a>{" "}
          × AIcrowd Mechanistic Estimation Challenge •{" "}
          <code>tools/circuit-explorer</code>
        </p>
      </footer>
    </div>
  );
}
