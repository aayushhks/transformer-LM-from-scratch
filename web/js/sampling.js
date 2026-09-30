// Decoding rules matching scratchlm/sampling.py: greedy, temperature, top-k and nucleus sampling.

export function softmax(logits) {
  let max = -Infinity;
  for (const x of logits) if (x > max) max = x;
  const out = new Float64Array(logits.length);
  let total = 0;
  for (let i = 0; i < logits.length; i++) {
    out[i] = Math.exp(logits[i] - max);
    total += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= total;
  return out;
}

// First index of the largest value, like torch.argmax.
export function argmax(logits) {
  let best = 0;
  for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
  return best;
}

function applyTopK(logits, topK) {
  if (!topK || topK <= 0) return logits;
  const k = Math.min(topK, logits.length);
  const threshold = Float64Array.from(logits).sort()[logits.length - k];
  return logits.map((x) => (x < threshold ? -Infinity : x));
}

function applyTopP(logits, topP) {
  if (topP === null || topP === undefined || topP >= 1) return logits;
  const order = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a] || a - b);
  const probs = softmax(order.map((i) => logits[i]));
  const out = Float64Array.from(logits);
  let cumulative = 0;
  // A token is dropped once the tokens ranked above it already exceed top_p, so the crossing token stays.
  for (let r = 0; r < order.length; r++) {
    if (r > 0 && cumulative > topP) out[order[r]] = -Infinity;
    cumulative += probs[r];
  }
  return out;
}

export function nextTokenProbs(logits, { temperature = 1, topK = null, topP = null } = {}) {
  const scaled = Float64Array.from(logits, (x) => x / Math.max(temperature, 1e-8));
  return softmax(applyTopP(applyTopK(scaled, topK), topP));
}

export function sampleNextToken(logits, options = {}, random = Math.random) {
  if (options.greedy) return argmax(logits);
  const probs = nextTokenProbs(logits, options);
  const u = random();
  let cumulative = 0;
  let last = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] <= 0) continue;
    last = i;
    cumulative += probs[i];
    if (u < cumulative) return i;
  }
  return last;
}
