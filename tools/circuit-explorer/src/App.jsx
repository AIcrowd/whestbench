import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { randomCircuit } from "./circuit";
import CircuitGraphJoint from "./components/CircuitGraphJoint";
import CircuitHeatmap from "./components/CircuitHeatmap";
import Controls from "./components/Controls";
import EstimatorComparison from "./components/EstimatorComparison";
import EstimatorRunner from "./components/EstimatorRunner";
import GateStats from "./components/GateStats";
import NarrativeCard, { Ewire } from "./components/NarrativeCard";
import SignalHeatmap from "./components/SignalHeatmap";
import StepIndicator from "./components/StepIndicator";
import WireStats from "./components/WireStats";
import { useCircuitWorker } from "./useWorker";

const DEFAULT_PARAMS = { width: 8, depth: 6, seed: 42 };
const TOUR_PARAMS = { width: 8, depth: 6, seed: 42 };
const GRAPH_MODE_THRESHOLD = 4096;

function formatTime(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export default function App() {
  // ── Web Worker ──
  const worker = useCircuitWorker();

  // ── Step state (tour) ──
  const [step, setStep] = useState(() => {
    const saved = localStorage.getItem("circuit-explorer-tour-step");
    return saved === "done" ? 6 : 1;
  });

  const isTour = step < 6;

  // ── Circuit params ──
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const effectiveParams = isTour ? TOUR_PARAMS : params;

  const [activeLayer, setActiveLayer] = useState(undefined);
  const [estimatorResults, setEstimatorResults] = useState({});

  // ── Tour auto-run state ──
  const [tourGroundTruth, setTourGroundTruth] = useState(null);
  const [tourGroundTruthTime, setTourGroundTruthTime] = useState(null);
  const [tourSampling, setTourSampling] = useState(null);
  const [tourMeanProp, setTourMeanProp] = useState(null);
  const [tourBudget, setTourBudget] = useState(1000);
  const autoRunDone = useRef({ gt: false, sampling: false, meanprop: false });

  // ── Circuit ──
  // Tour circuit is tiny (8×6=48 gates) — keep synchronous
  const tourCircuit = useMemo(
    () => randomCircuit(TOUR_PARAMS.width, TOUR_PARAMS.depth, TOUR_PARAMS.seed),
    []
  );

  // Explore circuit: async via worker for large sizes, sync for small
  const [circuit, setCircuit] = useState(null);
  const [circuitLoading, setCircuitLoading] = useState(false);
  const circuitGenIdRef = useRef(0);

  useEffect(() => {
    if (isTour) return;
    const { width, depth, seed } = effectiveParams;
    const size = width * depth;

    if (size <= GRAPH_MODE_THRESHOLD) {
      // Small circuit — generate synchronously (instant)
      setCircuit(randomCircuit(width, depth, seed));
      setCircuitLoading(false);
      setEstimatorResults({});
      setActiveLayer(undefined);
    } else {
      // Large circuit — generate in worker
      const genId = ++circuitGenIdRef.current;
      setCircuitLoading(true);
      setEstimatorResults({});
      setActiveLayer(undefined);
      worker.run('randomCircuit', { width, depth, seed })
        .then(({ circuit: c }) => {
          // Only apply if this is still the latest request
          if (genId === circuitGenIdRef.current) {
            setCircuit(c);
            setCircuitLoading(false);
          }
        });
    }
  }, [effectiveParams.width, effectiveParams.depth, effectiveParams.seed, isTour, worker]);

  // When entering tour mode, clear explore circuit
  useEffect(() => {
    if (isTour) {
      setCircuit(null);
      setCircuitLoading(false);
    }
  }, [isTour]);

  const totalGates = effectiveParams.width * effectiveParams.depth;
  const useGraphMode = totalGates <= GRAPH_MODE_THRESHOLD;

  // Effective circuit for display
  const displayCircuit = isTour ? tourCircuit : circuit;

  // Escape key clears activeLayer
  useEffect(() => {
    const handler = (e) => {
      if (e.key === 'Escape') setActiveLayer(undefined);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // ── Tour auto-run effects (via worker) ──
  // Step 3: auto-run ground truth
  useEffect(() => {
    if (step === 3 && !autoRunDone.current.gt) {
      autoRunDone.current.gt = true;
      worker.run('empiricalMean', { circuit: tourCircuit, trials: 10000, seed: 99 })
        .then(({ estimates, time }) => {
          setTourGroundTruth(estimates);
          setTourGroundTruthTime(time);
        });
    }
  }, [step, tourCircuit, worker]);

  // Step 4: auto-run sampling
  useEffect(() => {
    if (step >= 4 && !autoRunDone.current.sampling) {
      autoRunDone.current.sampling = true;
      worker.run('empiricalMean', { circuit: tourCircuit, trials: tourBudget, seed: 77 })
        .then(({ estimates }) => {
          setTourSampling(estimates);
        });
    }
  }, [step, tourCircuit, tourBudget, worker]);

  const handleTourBudgetChange = useCallback(
    (newBudget) => {
      setTourBudget(newBudget);
      worker.run('empiricalMean', { circuit: tourCircuit, trials: newBudget, seed: 77 })
        .then(({ estimates }) => {
          setTourSampling(estimates);
        });
    },
    [tourCircuit, worker]
  );

  // Step 5: auto-run mean propagation
  useEffect(() => {
    if (step >= 5 && !autoRunDone.current.meanprop) {
      autoRunDone.current.meanprop = true;
      worker.run('meanPropagation', { circuit: tourCircuit })
        .then(({ estimates }) => {
          setTourMeanProp(estimates);
        });
    }
  }, [step, tourCircuit, worker]);

  // ── Step navigation ──
  const nextStep = useCallback(() => {
    setStep((s) => {
      const next = Math.min(6, s + 1);
      if (next === 6) {
        localStorage.setItem("circuit-explorer-tour-step", "done");
      }
      return next;
    });
  }, []);

  const prevStep = useCallback(() => {
    setStep((s) => Math.max(1, s - 1));
  }, []);

  const handleSkipTour = useCallback(
    (action) => {
      if (action === "restart") {
        localStorage.removeItem("circuit-explorer-tour-step");
        autoRunDone.current = { gt: false, sampling: false, meanprop: false };
        setTourGroundTruth(null);
        setTourGroundTruthTime(null);
        setTourSampling(null);
        setTourMeanProp(null);
        setStep(1);
      } else {
        localStorage.setItem("circuit-explorer-tour-step", "done");
        setStep(6);
      }
    },
    []
  );

  // ── Explore mode estimator handler ──
  const handleEstimatorResult = useCallback((key, result) => {
    setEstimatorResults((prev) => ({ ...prev, [key]: result }));
  }, []);

  // ── Derived data ──
  const groundTruth = estimatorResults.groundTruth?.estimates || null;
  const samplingEst = estimatorResults.sampling?.estimates || null;
  const meanPropEst = estimatorResults.meanprop?.estimates || null;

  // Tour display data
  const tourDisplayMeans =
    step >= 3 ? tourGroundTruth || tourSampling || tourMeanProp : null;

  // Explore mode display data
  const exploreDisplayMeans = groundTruth || samplingEst || meanPropEst;
  const hasAnyExploreEstimate = !!(groundTruth || samplingEst || meanPropEst);

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

      {/* Step indicator — always visible */}
      <StepIndicator currentStep={step} onSkipTour={handleSkipTour} />

      <div className="app-layout">
        <aside className="sidebar">
          {/* Controls — locked during tour */}
          <div
            className={`controls-panel ${isTour ? "controls-panel--locked" : ""}`}
          >
            <Controls params={isTour ? TOUR_PARAMS : params} onParamsChange={setParams} />
            {isTour && (
              <p className="locked-hint">🔒 Unlocks in Explore mode</p>
            )}
          </div>

          {/* Tour budget slider (step 4) */}
          {isTour && step >= 4 && (
            <div className="controls-panel panel-reveal">
              <h2>Sampling Budget</h2>
              <div className="control-row">
                <label>
                  <span className="control-label">Samples</span>
                  <span className="control-value">
                    {tourBudget.toLocaleString()}
                  </span>
                </label>
                <input
                  type="range"
                  min={100}
                  max={10000}
                  step={100}
                  value={tourBudget}
                  onChange={(e) =>
                    handleTourBudgetChange(Number(e.target.value))
                  }
                />
              </div>
            </div>
          )}

          {/* Estimator runner — locked during tour */}
          <div
            className={isTour ? "estimator-runner--locked" : ""}
          >
            <EstimatorRunner
              circuit={displayCircuit}
              onResult={handleEstimatorResult}
              worker={worker}
            />
          </div>
        </aside>

        <main className="main-content">
          {/* ── Narrative Card ── */}
          {step === 3 ? (
            <NarrativeCard step={3} onNext={nextStep} onBack={prevStep}>
              {tourGroundTruthTime ? (
                <>
                  ✅ We sampled <strong>10,000 random inputs</strong> and
                  averaged each wire. The circuit is now colored by{" "}
                  <Ewire />.
                  Accurate, but took{" "}
                  <strong>{formatTime(tourGroundTruthTime)}</strong>. Now
                  imagine <strong>1,000 wires × 256 layers</strong>…
                </>
              ) : (
                <>Computing ground truth…</>
              )}
            </NarrativeCard>
          ) : (
            <NarrativeCard
              step={step}
              onNext={nextStep}
              onBack={prevStep}
            />
          )}

          {/* ── Circuit visualization ── */}
          {circuitLoading && (
            <div className="panel" style={{ textAlign: 'center', padding: '60px 20px', color: '#9CA3AF' }}>
              <div style={{ fontSize: 24, marginBottom: 8 }}>⏳</div>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                Generating {effectiveParams.width}×{effectiveParams.depth} circuit…
              </div>
            </div>
          )}
          {isTour ? (
            <CircuitGraphJoint
              circuit={tourCircuit}
              means={tourDisplayMeans}
              activeLayer={activeLayer}
              pulseOutputs={step === 2}
            />
          ) : !circuitLoading && displayCircuit && useGraphMode ? (
            <CircuitGraphJoint
              circuit={displayCircuit}
              means={exploreDisplayMeans}
              activeLayer={activeLayer}
            />
          ) : !circuitLoading && displayCircuit ? (
            <CircuitHeatmap
              circuit={displayCircuit}
              means={exploreDisplayMeans}
              activeLayer={activeLayer}
              onLayerClick={setActiveLayer}
            />
          ) : null}

          {/* ── Tour: MSE Comparison (steps 4-5) ── */}
          {isTour && step >= 4 && tourGroundTruth && (
            <div className="panel-reveal">
              <EstimatorComparison
                groundTruth={tourGroundTruth}
                samplingEstimates={tourSampling}
                meanPropEstimates={step >= 5 ? tourMeanProp : null}
                depth={TOUR_PARAMS.depth}
              />
            </div>
          )}

          {/* ── Explore mode: all panels ── */}
          {!isTour && (
            <>
              {displayCircuit && <GateStats circuit={displayCircuit} activeLayer={activeLayer} />}

              {hasAnyExploreEstimate && (
                <>
                  <div className="panels-row panel-reveal">
                    {useGraphMode && (
                      <SignalHeatmap
                        means={exploreDisplayMeans}
                        width={params.width}
                        depth={params.depth}
                      />
                    )}
                    <WireStats
                      means={exploreDisplayMeans}
                      width={params.width}
                      depth={params.depth}
                      activeLayer={activeLayer}
                    />
                  </div>

                  {groundTruth && (samplingEst || meanPropEst) && (
                    <div className="panel-reveal">
                      <EstimatorComparison
                        groundTruth={groundTruth}
                        samplingEstimates={samplingEst}
                        meanPropEstimates={meanPropEst}
                        depth={params.depth}
                        activeLayer={activeLayer}
                      />
                    </div>
                  )}
                </>
              )}

              {!hasAnyExploreEstimate && (
                <div className="empty-state">
                  <div className="empty-state-inner">
                    <span className="empty-icon">📊</span>
                    <h3>Ready to Explore</h3>
                    <p>
                      This circuit has{" "}
                      <strong>{params.width}</strong> wires and{" "}
                      <strong>{params.depth}</strong> layers of random gates.
                    </p>
                    <p className="empty-hint">
                      Run <strong>Ground Truth</strong> to see means, then
                      compare <strong>Sampling</strong> and{" "}
                      <strong>Mean Propagation</strong>.
                    </p>
                  </div>
                </div>
              )}
            </>
          )}
        </main>
      </div>

      <footer className="app-footer">
        {step === 6 && (
          <p style={{ marginBottom: 4 }}>
            Ready to compete? →{" "}
            <a
              href="https://www.aicrowd.com/"
              target="_blank"
              rel="noopener noreferrer"
            >
              Join the Challenge on AIcrowd
            </a>
          </p>
        )}
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
