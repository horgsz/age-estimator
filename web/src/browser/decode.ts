/**
 * Turning 101 logits into an age, a range and a confidence.
 *
 * A PORT OF ``server/predictor.py``. The decode is not an implementation
 * detail: choosing the distribution's MEDIAN over its expectation was worth
 * ~0.7 years of MAE on these weights, and it was only findable because both
 * were measurable on the same model. A browser port that quietly used the
 * expectation -- the more obvious of the two, and the one most "age
 * regression" code uses -- would give slightly different numbers to the server
 * for every face, with nothing to indicate it.
 *
 * The interval is the 0.16/0.84 CDF quantiles, which is the +/- 1 sigma mass of
 * a normal but needs no symmetry assumption; these distributions are visibly
 * skewed at the tails. Confidence is a function of that interval's WIDTH.
 *
 * Interval coverage is measured at ~61% against a nominal 68%. That figure is
 * a property of this construction and corpus rather than of either checkpoint,
 * which is why it is served alongside the model list rather than inside it.
 */

export const MAX_AGE = 100;
export const LOW_Q = 0.16;
export const HIGH_Q = 0.84;

/** Interval half-width (years) at which confidence is 0.5. */
export const CONFIDENCE_STD_SCALE = 6.0;

export interface DecodedFace {
  age: number;
  low: number;
  high: number;
  confidence: number;
}

/** `confidence_from_width`: monotonically decreasing map to [0, 1]. */
export function confidenceFromWidth(width: number): number {
  const w = Math.max(width, 0);
  return round(1 / (1 + w / (2 * CONFIDENCE_STD_SCALE)), 4);
}

/**
 * Python's `round(x, n)`, which is round-half-to-even on the decimal value.
 *
 * `toFixed` rounds halves away from zero, so a face whose confidence lands on
 * exactly 0.56785 would be reported differently by the two implementations.
 * That is a cosmetic difference, but the parity harness compares the numbers a
 * user actually sees, and a cosmetic mismatch there is indistinguishable from a
 * real one until someone investigates it.
 */
function round(value: number, digits: number): number {
  const factor = 10 ** digits;
  const scaled = value * factor;
  const floor = Math.floor(scaled);
  // Only treat it as a tie when the scaled value is exactly representable as
  // one; float noise otherwise makes this fire on values that are not ties.
  let rounded: number;
  if (Math.abs(scaled - floor - 0.5) < Number.EPSILON * Math.abs(scaled)) {
    rounded = floor % 2 === 0 ? floor : floor + 1;
  } else {
    rounded = Math.round(scaled);
  }
  return rounded / factor;
}

/**
 * `softmax` over one row of logits, in float64.
 *
 * torch computes this in float32. The difference is ~1e-7 per bin and cannot
 * move a CDF quantile, which is an integer bin index -- it would take a bin
 * sitting within 1e-7 of the 0.5 crossing. The parity harness reports the
 * measured logit and age deltas so this reasoning is checked rather than
 * assumed.
 */
export function softmax(logits: Float32Array | number[]): Float64Array {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i]! > max) max = logits[i]!;
  const out = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i]! - max);
    out[i] = e;
    sum += e;
  }
  for (let i = 0; i < out.length; i++) out[i] = out[i]! / sum;
  return out;
}

/**
 * The smallest index whose CDF has reached `level`.
 *
 * The Python computes `(cdf < level).sum()`, which counts the leading bins that
 * have NOT yet reached `level` -- and that count *is* the index of the first
 * bin that has. Reproduced as the same count rather than as a "first index
 * where cdf >= level" search, so the two agree on the boundary case where a
 * bin's CDF is exactly `level`.
 */
function quantileIndex(cdf: Float64Array, level: number): number {
  let count = 0;
  for (let i = 0; i < cdf.length; i++) if (cdf[i]! < level) count++;
  return Math.min(Math.max(count, 0), cdf.length - 1);
}

export interface DecodeStats {
  median: number;
  low: number;
  high: number;
  expectation: number;
  std: number;
}

/** Everything `Decoded` carries, for one face. */
export function decodeLogits(logits: Float32Array | number[]): DecodeStats {
  const probs = softmax(logits);

  let expectation = 0;
  for (let i = 0; i < probs.length; i++) expectation += probs[i]! * i;
  let variance = 0;
  for (let i = 0; i < probs.length; i++) variance += probs[i]! * (i - expectation) ** 2;
  const std = Math.sqrt(Math.max(variance, 0));

  const cdf = new Float64Array(probs.length);
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i]!;
    cdf[i] = acc;
  }

  return {
    median: quantileIndex(cdf, 0.5),
    low: quantileIndex(cdf, LOW_Q),
    high: quantileIndex(cdf, HIGH_Q),
    expectation,
    std,
  };
}

/** `build_result`: clamp, keep the range coherent, derive confidence. */
export function buildResult(age: number, low: number, high: number): DecodedFace {
  const a = Math.min(Math.max(age, 0), MAX_AGE);
  let lo = Math.min(Math.max(low, 0), MAX_AGE);
  let hi = Math.min(Math.max(high, 0), MAX_AGE);
  // The median can sit outside a degenerate interval; keep the range coherent.
  lo = Math.min(lo, a);
  hi = Math.max(hi, a);
  return {
    age: round(a, 1),
    low: round(lo, 1),
    high: round(hi, 1),
    confidence: confidenceFromWidth(hi - lo),
  };
}

/** The shipped decode: median point estimate, quantile interval. */
export function decode(logits: Float32Array | number[]): DecodedFace {
  const stats = decodeLogits(logits);
  return buildResult(stats.median, stats.low, stats.high);
}
