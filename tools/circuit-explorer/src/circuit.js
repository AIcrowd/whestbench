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
  const gateType = new Array(n);   // e.g. "AND", "XOR", "BUF"
  const gateLabel = new Array(n);  // e.g. "AND(-x, y)", "XOR(x, y)", "x"

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
      if (opType === 0) {
        firstCoeff[i] = sign;
        gateType[i] = sign > 0 ? "BUF" : "NOT";
        gateLabel[i] = sign > 0 ? "x" : "−x";
      } else if (opType === 1) {
        secondCoeff[i] = sign;
        gateType[i] = sign > 0 ? "BUF" : "NOT";
        gateLabel[i] = sign > 0 ? "y" : "−y";
      } else if (opType === 2) {
        constArr[i] = sign;
        gateType[i] = "CONST";
        gateLabel[i] = sign > 0 ? "+1" : "−1";
      } else {
        productCoeff[i] = sign;
        gateType[i] = sign > 0 ? "XNOR" : "XOR";
        gateLabel[i] = sign > 0 ? "XNOR(x, y)" : "XOR(x, y)";
      }
    } else {
      // Complex: ± AND(±x, ±y) = coeff*(−1 + xc*x + yc*y + xc*yc*xy)
      const xc = rng.randSign();
      const yc = rng.randSign();
      const coeff = rng.randSign() * 0.5;
      constArr[i] = -coeff;
      firstCoeff[i] = xc * coeff;
      secondCoeff[i] = yc * coeff;
      productCoeff[i] = xc * yc * coeff;

      // Determine gate type from signs
      const outerPositive = coeff < 0;  // -coeff > 0 means outer is positive
      const xNeg = xc < 0;
      const yNeg = yc < 0;
      const xStr = xNeg ? "−x" : "x";
      const yStr = yNeg ? "−y" : "y";

      if (outerPositive) {
        if (!xNeg && !yNeg) { gateType[i] = "AND";  gateLabel[i] = "AND(x, y)"; }
        else if (xNeg && yNeg) { gateType[i] = "NOR";  gateLabel[i] = "NOR(x, y)"; }
        else { gateType[i] = "AND"; gateLabel[i] = `AND(${xStr}, ${yStr})`; }
      } else {
        if (!xNeg && !yNeg) { gateType[i] = "NAND"; gateLabel[i] = "NAND(x, y)"; }
        else if (xNeg && yNeg) { gateType[i] = "OR";   gateLabel[i] = "OR(x, y)"; }
        else { gateType[i] = "NAND"; gateLabel[i] = `NAND(${xStr}, ${yStr})`; }
      }
    }
  }

  return { first, second, const: constArr, firstCoeff, secondCoeff, productCoeff, gateType, gateLabel };
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
 * Run circuit on random ±1 inputs and return rich stats per wire per layer.
 * Returns: { means, stds, mins, maxs } — each Float32Array[] (one per layer, length n)
 */
export function empiricalStats(circuit, trials, seed = 99) {
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
  const means = [];
  const stds = [];
  const mins = [];
  const maxs = [];

  for (const batch of layerOutputs) {
    const n = circuit.n;
    const mean = new Float32Array(n);
    const sumSq = new Float32Array(n);
    const min = new Float32Array(n);
    const max = new Float32Array(n);
    for (let i = 0; i < n; i++) { min[i] = Infinity; max[i] = -Infinity; }

    for (let b = 0; b < trials; b++) {
      for (let i = 0; i < n; i++) {
        const v = batch[b][i];
        mean[i] += v;
        sumSq[i] += v * v;
        if (v < min[i]) min[i] = v;
        if (v > max[i]) max[i] = v;
      }
    }

    const std = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      mean[i] /= trials;
      const variance = sumSq[i] / trials - mean[i] * mean[i];
      std[i] = Math.sqrt(Math.max(0, variance));
    }

    means.push(mean);
    stds.push(std);
    mins.push(min);
    maxs.push(max);
  }

  return { means, stds, mins, maxs };
}

/**
 * Run circuit with a single random ±1 input.
 * Returns: Float32Array[] — one per layer, each of length n (wire values at that layer).
 */
export function runSingleTrial(circuit, seed = 42) {
  const rng = makeRng(seed);
  const input = new Float32Array(circuit.n);
  for (let i = 0; i < circuit.n; i++) {
    input[i] = rng.randBool() ? 1.0 : -1.0;
  }
  const results = runBatched(circuit, [input]);
  return results.map(batch => batch[0]);
}


