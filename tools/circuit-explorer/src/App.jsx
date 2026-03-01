import { useCallback, useEffect, useMemo, useState } from "react";
import "./App.css";
import { randomCircuit } from "./circuit";
import CircuitGraph from "./components/CircuitGraph";
import Controls from "./components/Controls";
import EstimatorComparison from "./components/EstimatorComparison";
import EstimatorRunner from "./components/EstimatorRunner";
import GateStats from "./components/GateStats";
import SignalHeatmap from "./components/SignalHeatmap";
import WireStats from "./components/WireStats";

const DEFAULT_PARAMS = { width: 8, depth: 6, seed: 42 };

export default function App() {
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const [activeLayer, setActiveLayer] = useState(undefined);
  const [estimatorResults, setEstimatorResults] = useState({});

  // Auto-regenerate circuit when params change
  const circuit = useMemo(
    () => randomCircuit(params.width, params.depth, params.seed),
    [params.width, params.depth, params.seed]
  );

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
        </aside>

        <main className="main-content">
          {/* Circuit structure — always visible */}
          <CircuitGraph
            circuit={circuit}
            means={displayMeans}
            activeLayer={activeLayer}
          />

          {/* Gate structure analysis — always visible (structural info) */}
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
                <p>Run an estimator from the sidebar to see signal analysis.</p>
                <p className="empty-hint">
                  Start with <strong>Ground Truth</strong> to establish a baseline,
                  then compare <strong>Sampling</strong> and <strong>Mean Propagation</strong>.
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
