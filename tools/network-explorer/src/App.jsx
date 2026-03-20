import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import ActivationRibbon from "./components/ActivationRibbon";
import CoeffHistograms from "./components/CoeffHistograms";
import Controls from "./components/Controls";
import ErrorHeatmap from "./components/ErrorHeatmap";
import EstimatorComparison from "./components/EstimatorComparison";
import EstimatorRunner from "./components/EstimatorRunner";
import NarrativeCard, { Eneuron } from "./components/NarrativeCard";
import NetworkGraph from "./components/NetworkGraph";
import NetworkHeatmap from "./components/NetworkHeatmap";
import PerfOverlay from "./components/PerfOverlay";
import SignalHeatmap from "./components/SignalHeatmap";
import StdHeatmap from "./components/StdHeatmap";
import StepIndicator from "./components/StepIndicator";

import { perfEnd, perfStart } from "./perf";
import { useMLPWorker } from "./useWorker";

const DEFAULT_PARAMS = { width: 8, depth: 6, seed: 42 };
const TOUR_PARAMS = { width: 8, depth: 6, seed: 42 };
const GRAPH_MODE_MAX_WIDTH = 8;

function formatTime(ms) {
  if (ms < 1) return `${(ms * 1000).toFixed(0)}µs`;
  if (ms < 1000) return `${ms.toFixed(1)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

export default function App() {
  // ── Web Worker ──
  const worker = useMLPWorker();

  // ── Step state (tour) ──
  const [step, setStep] = useState(() => {
    const saved = localStorage.getItem("network-explorer-tour-step");
    return saved === "done" ? 6 : 1;
  });

  const isTour = step < 6;

  // ── MLP params ──
  const [params, setParams] = useState(DEFAULT_PARAMS);
  const effectiveParams = isTour ? TOUR_PARAMS : params;

  const [activeLayer, setActiveLayer] = useState(undefined);
  const [estimatorResults, setEstimatorResults] = useState({});

  // ── Tour auto-run state ──
  const [tourMeans, setTourMeans] = useState(null);
  const [tourMeansTime, setTourMeansTime] = useState(null);
  const [tourSampling, setTourSampling] = useState(null);
  const [tourMeanProp, setTourMeanProp] = useState(null);
  const [tourBudget, setTourBudget] = useState(1000);
  const autoRunDone = useRef({ gt: false, sampling: false, meanprop: false });

  // ── MLP: always generated via worker ──
  const [mlp, setMlp] = useState(null);
  const [mlpLoading, setMlpLoading] = useState(false);
  const mlpGenIdRef = useRef(0);
  const { run: workerRun } = worker;

  useEffect(() => {
    const { width, depth, seed } = effectiveParams;
    const genId = ++mlpGenIdRef.current;
    setMlpLoading(true);
    perfStart('mlp-gen');
    workerRun('sampleMLP', { width, depth, seed })
      .then(({ mlp: m }) => {
        perfEnd('mlp-gen');
        if (genId === mlpGenIdRef.current) {
          setMlp(m);
          setMlpLoading(false);
        }
      })
      .catch((err) => {
        console.error("MLP generation failed:", err);
        setMlpLoading(false);
      });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectiveParams.width, effectiveParams.depth, effectiveParams.seed, workerRun]);

  // Clear estimator results when MLP changes
  const prevMlpRef = useRef(mlp);
  useEffect(() => {
    if (prevMlpRef.current !== mlp && !isTour) {
      setEstimatorResults({});
      setActiveLayer(undefined);
    }
    prevMlpRef.current = mlp;
  }, [mlp, isTour]);

  // Tour MLP — synchronously reuse main mlp when in tour
  const tourMlp = useMemo(() => mlp, [mlp]);

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
    if (step === 3 && tourMlp && !autoRunDone.current.gt) {
      autoRunDone.current.gt = true;
      worker.run('outputStats', { mlp: tourMlp, nSamples: 10000, seed: 99 })
        .then(({ means, time }) => {
          setTourMeans(means);
          setTourMeansTime(time);
        });
    }
  }, [step, tourMlp, worker]);

  // Step 4: auto-run sampling
  useEffect(() => {
    if (step >= 4 && tourMlp && !autoRunDone.current.sampling) {
      autoRunDone.current.sampling = true;
      worker.run('sampling', { mlp: tourMlp, budget: tourBudget, seed: 77 })
        .then(({ estimates }) => {
          setTourSampling(estimates);
        });
    }
  }, [step, tourMlp, tourBudget, worker]);

  const handleTourBudgetChange = useCallback(
    (newBudget) => {
      if (!tourMlp) return;
      setTourBudget(newBudget);
      worker.run('sampling', { mlp: tourMlp, budget: newBudget, seed: 77 })
        .then(({ estimates }) => {
          setTourSampling(estimates);
        });
    },
    [tourMlp, worker]
  );

  // Step 5: auto-run mean propagation
  useEffect(() => {
    if (step >= 5 && tourMlp && !autoRunDone.current.meanprop) {
      autoRunDone.current.meanprop = true;
      worker.run('meanPropagation', { mlp: tourMlp })
        .then(({ estimates }) => {
          setTourMeanProp(estimates);
        });
    }
  }, [step, tourMlp, worker]);

  // ── Step navigation ──
  const nextStep = useCallback(() => {
    setStep((s) => {
      const next = Math.min(6, s + 1);
      if (next === 6) {
        localStorage.setItem("network-explorer-tour-step", "done");
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
        localStorage.removeItem("network-explorer-tour-step");
        autoRunDone.current = { gt: false, sampling: false, meanprop: false };
        setTourMeans(null);
        setTourMeansTime(null);
        setTourSampling(null);
        setTourMeanProp(null);
        setStep(1);
      } else {
        localStorage.setItem("network-explorer-tour-step", "done");
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
  const groundTruthStats = estimatorResults.groundTruth || null;
  const samplingEst = estimatorResults.sampling?.estimates || null;
  const meanPropEst = estimatorResults.meanprop?.estimates || null;
  const covPropEst = estimatorResults.covprop?.estimates || null;

  // Tour display data
  const tourDisplayMeans =
    step >= 3 ? tourMeans || tourSampling || tourMeanProp : null;

  // Explore mode display data
  const exploreDisplayMeans = groundTruth || samplingEst || meanPropEst || covPropEst;
  const hasAnyExploreEstimate = !!(groundTruth || samplingEst || meanPropEst || covPropEst);

  // Graph mode: use NetworkGraph for small networks, NetworkHeatmap for larger ones
  const useGraphMode = effectiveParams.width <= GRAPH_MODE_MAX_WIDTH;

  return (
    <div className="app">
      <header className="app-header">
        <h1>
          <img src="/logo.png" alt="Network Explorer" className="header-logo" /> Network Explorer
        </h1>
        <p className="subtitle">
          Interactive visualization of MLPs for the{" "}
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
              <p className="locked-hint">Unlocks in Explore mode</p>
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
            {mlp && (
              <EstimatorRunner
                mlp={mlp}
                onResult={handleEstimatorResult}
                worker={worker}
              />
            )}
          </div>
        </aside>

        <main className="main-content">
          {/* ── Narrative Card ── */}
          {step === 3 ? (
            <NarrativeCard step={3} onNext={nextStep} onBack={prevStep}>
              {tourMeansTime ? (
                <>
                  We sampled <strong>10,000 random inputs</strong> and averaged
                  each neuron. The network is now colored by <Eneuron />.
                  Accurate, but took{" "}
                  <strong>{formatTime(tourMeansTime)}</strong>. Now imagine{" "}
                  <strong>1,000 neurons × 256 layers</strong>…
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

          {/* ── MLP visualization ── */}
          {mlpLoading && (
            <div className="panel" style={{ textAlign: 'center', padding: '60px 20px', color: '#9CA3AF' }}>
              <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                Generating {effectiveParams.width}×{effectiveParams.depth} network…
              </div>
            </div>
          )}
          {!mlpLoading && mlp && (
            isTour ? (
              useGraphMode ? (
                <NetworkGraph
                  mlp={tourMlp}
                  means={tourDisplayMeans}
                  activeLayer={activeLayer}
                />
              ) : (
                <NetworkHeatmap
                  mlp={tourMlp}
                  means={tourDisplayMeans}
                  activeLayer={activeLayer}
                  onLayerClick={setActiveLayer}
                />
              )
            ) : useGraphMode ? (
              <NetworkGraph
                mlp={mlp}
                means={exploreDisplayMeans}
                activeLayer={activeLayer}
              />
            ) : (
              <NetworkHeatmap
                mlp={mlp}
                means={exploreDisplayMeans}
                activeLayer={activeLayer}
                onLayerClick={setActiveLayer}
              />
            )
          )}

          {/* ── Tour: MSE Comparison (steps 4-5) ── */}
          {isTour && step >= 4 && tourMeans && (
            <div className="panel-reveal">
              <EstimatorComparison
                groundTruth={tourMeans}
                samplingEstimates={tourSampling}
                meanPropEstimates={step >= 5 ? tourMeanProp : null}
                depth={TOUR_PARAMS.depth}
              />
            </div>
          )}

          {/* ── Explore mode: all panels ── */}
          {!isTour && (
            <>
              {hasAnyExploreEstimate && (
                <>
                  {/* Signal heatmap */}
                  {exploreDisplayMeans && (
                    <div className="panel-reveal">
                      <SignalHeatmap
                        means={exploreDisplayMeans}
                        width={params.width}
                        depth={params.depth}
                      />
                    </div>
                  )}

                  {/* Output variance ribbon */}
                  {groundTruthStats?.stds && (
                    <div className="panel-reveal">
                      <ActivationRibbon
                        means={groundTruth}
                        stds={groundTruthStats.stds}
                        depth={params.depth}
                        width={params.width}
                      />
                    </div>
                  )}

                  {/* Estimation Error (full width) */}
                  {groundTruth && (samplingEst || meanPropEst) && (
                    <div className="panel-reveal">
                      <EstimatorComparison
                        groundTruth={groundTruth}
                        samplingEstimates={samplingEst}
                        meanPropEstimates={meanPropEst}
                        covPropEstimates={covPropEst}
                        depth={params.depth}
                        activeLayer={activeLayer}
                      />
                    </div>
                  )}

                  {/* Signal Variability (σ) | Error heatmap */}
                  <div className="panels-row panel-reveal">
                    {groundTruthStats?.stds && (
                      <StdHeatmap
                        stds={groundTruthStats.stds}
                        width={params.width}
                        depth={params.depth}
                      />
                    )}
                    <ErrorHeatmap
                      groundTruth={groundTruth}
                      meanPropEstimates={meanPropEst}
                      covPropEstimates={covPropEst}
                      samplingEstimates={samplingEst}
                      width={params.width}
                      depth={params.depth}
                    />
                  </div>

                  {/* Weight Distributions */}
                  {mlp && (
                    <div className="panel-reveal">
                      <CoeffHistograms mlp={mlp} />
                    </div>
                  )}
                </>
              )}

              {!hasAnyExploreEstimate && (
                <div className="empty-state">
                  <div className="empty-state-inner">
                    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg" className="empty-state-motif">
                      <rect x="8" y="8" width="48" height="48" rx="12" className="motif-bg" strokeWidth="1.5" />
                      <path d="M 8 32 h 48 M 32 8 v 48" className="motif-wire-dash" strokeWidth="1.5" />
                      <path d="M 20 20 v 24 M 44 20 v 24 M 20 20 h 24 M 20 44 h 24" className="motif-wire" strokeWidth="1.5" />
                      <path d="M 20 32 L 32 20 L 44 32 L 32 44 Z" fill="var(--white)" />
                      <path d="M 20 32 L 32 20 L 44 32 L 32 44 Z" className="motif-link" strokeWidth="1.5" strokeLinejoin="round" />
                      <path d="M 32 20 L 32 44 M 20 32 L 44 32" className="motif-wire" strokeWidth="1.5" />
                      <circle cx="32" cy="20" r="4.5" className="motif-node motif-node-mean" strokeWidth="2" />
                      <circle cx="20" cy="32" r="4.5" className="motif-node motif-node-cov" strokeWidth="2" />
                      <circle cx="44" cy="32" r="4.5" className="motif-node motif-node-samp" strokeWidth="2" />
                      <circle cx="32" cy="44" r="4.5" className="motif-node motif-node-gt" strokeWidth="2" />
                      <circle cx="32" cy="32" r="3" className="motif-node-center" />
                    </svg>
                    <h3>Ready to Explore</h3>
                    <p>
                      This network has{" "}
                      <strong>{params.width}</strong> neurons wide and{" "}
                      <strong>{params.depth}</strong> layers.
                    </p>
                    <p className="empty-hint">
                      Use the <strong>Run Estimators</strong> panel on the left. Start with <strong>Ground Truth</strong> to establish a baseline, then try the other estimators to compare accuracy and speed.
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
          x AIcrowd Mechanistic Estimation Challenge
        </p>
      </footer>

      <PerfOverlay />
    </div>
  );
}
