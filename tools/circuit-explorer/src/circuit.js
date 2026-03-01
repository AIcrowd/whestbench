/**
 * circuit.js — JS port of circuit.py
 *
 * Mirrors the Python circuit generation and forward pass logic.
 * Operates on typed arrays for small circuits (n ≤ 32).
 */

// Seedable PRNG (xoshiro128** — good enough for visualization)
function makeRng(seed = 42) {
  let s = [seed, seed ^ 0xdeadbeef, seed ^ 0xcafebabe, seed ^ 0x12345678];
  function next() {
    const result = (s[1] * 5) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = (s[3] << 11) | (s[3] >>> 21);
    return (result >>> 0) / 0x100000000;
  }
  // warm up
  for (let i = 0; i < 20; i++) next();
  return {
    /** Returns float in [0, 1) */
    random: next,
    /** Returns int in [0, max) */
    randInt(max) {
      return Math.floor(next() * max);
    },
    /** Returns -1 or +1 */
    randSign() {
      return next() < 0.5 ? -1 : 1;
    },
    /** Returns true/false */
    randBool() {
      return next() < 0.5;
    },
  };
}

/**
 * Generate a random layer for a circuit with n wires.
 * Mirrors circuit.py::random_gates
 */
function randomGates(n, rng) {
  const first = new Int32Array(n);
  const second = new Int32Array(n);
  const constArr = new Float32Array(n);
  const firstCoeff = new Float32Array(n);
  const secondCoeff = new Float32Array(n);
  const productCoeff = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    // Pick two distinct random inputs
    first[i] = rng.randInt(n);
    const secondRaw = rng.randInt(n - 1);
    second[i] = secondRaw >= first[i] ? secondRaw + 1 : secondRaw;

    const isSimple = rng.randBool();

    if (isSimple) {
      // Simple: ±{x, y, 1, xy}
      const sign = rng.randSign();
      const opType = rng.randInt(4);
      if (opType === 0) firstCoeff[i] = sign;
      else if (opType === 1) secondCoeff[i] = sign;
      else if (opType === 2) constArr[i] = sign;
      else productCoeff[i] = sign;
    } else {
      // Complex: ± AND(±x, ±y) = coeff*(1 + xc*x + yc*y + xc*yc*xy)
      const xc = rng.randSign();
      const yc = rng.randSign();
      const coeff = rng.randSign() * 0.5;
      constArr[i] = -coeff;
      firstCoeff[i] = xc * coeff;
      secondCoeff[i] = yc * coeff;
      productCoeff[i] = xc * yc * coeff;
    }
  }

  return { first, second, const: constArr, firstCoeff, secondCoeff, productCoeff };
}

/**
 * Generate a random circuit with n wires and d layers.
 * Mirrors circuit.py::random_circuit
 */
export function randomCircuit(n, d, seed = 42) {
  const rng = makeRng(seed);
  const gates = [];
  for (let i = 0; i < d; i++) {
    gates.push(randomGates(n, rng));
  }
  return { n, d, gates };
}

/**
 * Run the circuit on batched inputs. Returns array of layer outputs.
 * inputs: Array of Float32Array (each of length n), or 2D array [trials][n]
 * Returns: Array<Float32Array[]> — one entry per layer, each a batch of wire values
 *
 * Mirrors circuit.py::run_batched (but returns all at once, not a generator)
 */
export function runBatched(circuit, inputs) {
  const batchSize = inputs.length;
  let x = inputs.map((row) => Float32Array.from(row));
  const results = [];

  for (const layer of circuit.gates) {
    const newX = [];
    for (let b = 0; b < batchSize; b++) {
      const row = new Float32Array(circuit.n);
      for (let i = 0; i < circuit.n; i++) {
        const xi = x[b][layer.first[i]];
        const yi = x[b][layer.second[i]];
        row[i] =
          layer.const[i] +
          layer.firstCoeff[i] * xi +
          layer.secondCoeff[i] * yi +
          layer.productCoeff[i] * xi * yi;
      }
      newX.push(row);
    }
    x = newX;
    results.push(x);
  }
  return results;
}

/**
 * Run circuit on random ±1 inputs and return the mean per wire per layer.
 * Returns: Float32Array[] — one per layer, each of length n
 *
 * Mirrors circuit.py::empirical_mean
 */
export function empiricalMean(circuit, trials, seed = 99) {
  const rng = makeRng(seed);
  const inputs = [];
  for (let t = 0; t < trials; t++) {
    const row = new Float32Array(circuit.n);
    for (let i = 0; i < circuit.n; i++) {
      row[i] = rng.randBool() ? 1.0 : -1.0;
    }
    inputs.push(row);
  }

  const layerOutputs = runBatched(circuit, inputs);
  return layerOutputs.map((batch) => {
    const mean = new Float32Array(circuit.n);
    for (let b = 0; b < trials; b++) {
      for (let i = 0; i < circuit.n; i++) {
        mean[i] += batch[b][i];
      }
    }
    for (let i = 0; i < circuit.n; i++) {
      mean[i] /= trials;
    }
    return mean;
  });
}

/**
 * Describe the gate operation for display purposes.
 * Returns a human-readable string like "AND(x, y)", "x", "-y", etc.
 */
export function describeGate(layer, i) {
  const c = layer.const[i];
  const fc = layer.firstCoeff[i];
  const sc = layer.secondCoeff[i];
  const pc = layer.productCoeff[i];

  // Simple gates: only one nonzero coeff
  if (fc !== 0 && sc === 0 && c === 0 && pc === 0)
    return fc > 0 ? "x" : "-x";
  if (sc !== 0 && fc === 0 && c === 0 && pc === 0)
    return sc > 0 ? "y" : "-y";
  if (c !== 0 && fc === 0 && sc === 0 && pc === 0)
    return c > 0 ? "+1" : "-1";
  if (pc !== 0 && fc === 0 && sc === 0 && c === 0)
    return pc > 0 ? "xy" : "-xy";

  // Complex gates: AND variants
  // coeff*(−1 + xc*x + yc*y + xc*yc*xy) = coeff * AND(xc*x, yc*y) roughly
  if (pc !== 0) {
    const xSign = fc / pc > 0 ? "" : "-";
    const ySign = sc > 0 === pc > 0 ? "" : "-";
    const gateSign = c < 0 === pc > 0 ? "" : "-";
    return `${gateSign}AND(${xSign}x,${ySign}y)`;
  }

  return "?";
}
