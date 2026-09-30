// Forward passes matching scratchlm/transformer.py and scratchlm/rnn.py, run on exported float32 weights.

export function createModel(manifest, buffer) {
  const weights = new Float32Array(buffer);
  const tensor = (name) => {
    const entry = manifest.tensors[name];
    if (!entry) throw new Error(`${manifest.name} has no tensor named ${name}`);
    const size = entry.shape.reduce((a, b) => a * b, 1);
    return weights.subarray(entry.offset, entry.offset + size);
  };
  if (manifest.model_type === "transformer") return new Transformer(manifest, tensor);
  if (manifest.model_type === "rnn") return new Recurrent(manifest, tensor, false);
  if (manifest.model_type === "lstm") return new Recurrent(manifest, tensor, true);
  throw new Error(`unsupported model type: ${manifest.model_type}`);
}

// y = W x + b for an nn.Linear weight stored as (out, in).
function linear(x, weight, bias, inDim, outDim) {
  const y = new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let sum = bias ? bias[o] : 0;
    const row = o * inDim;
    for (let i = 0; i < inDim; i++) sum += weight[row + i] * x[i];
    y[o] = sum;
  }
  return y;
}

// y = x M for a raw (in, out) parameter, as used by the hand-written recurrent cells.
function vecMat(x, matrix, inDim, outDim) {
  const sums = new Float64Array(outDim);
  for (let i = 0; i < inDim; i++) {
    const xi = x[i];
    const row = i * outDim;
    for (let j = 0; j < outDim; j++) sums[j] += xi * matrix[row + j];
  }
  return Float32Array.from(sums);
}

function layerNorm(x, gamma, beta, eps = 1e-5) {
  const n = x.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += x[i];
  mean /= n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (x[i] - mean) ** 2;
  const denom = Math.sqrt(variance / n + eps);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = (gamma[i] * (x[i] - mean)) / denom + beta[i];
  return out;
}

function add(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + b[i];
  return out;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

class Transformer {
  constructor(manifest, tensor) {
    this.type = "transformer";
    this.params = manifest.params;
    this.vocabSize = manifest.vocab_size;
    this.hidden = manifest.hidden_dim;
    this.contextLen = manifest.context_len;
    this.heads = manifest.num_heads;
    this.headDim = this.hidden / this.heads;
    this.embedding = tensor("embedding.weight");
    this.positions = tensor("pos_embedding.weight");
    this.layers = [];
    for (let l = 0; l < manifest.num_layers; l++) {
      this.layers.push({
        wq: tensor(`layers.${l}.w_q.weight`),
        wk: tensor(`layers.${l}.w_k.weight`),
        wv: tensor(`layers.${l}.w_v.weight`),
        wo: tensor(`layers.${l}.w_o.weight`),
        wUp: tensor(`layers.${l}.w_up.weight`),
        bUp: tensor(`layers.${l}.w_up.bias`),
        wDown: tensor(`layers.${l}.w_down.weight`),
        bDown: tensor(`layers.${l}.w_down.bias`),
        gammaAttn: tensor(`gamma_attn.${l}`),
        betaAttn: tensor(`beta_attn.${l}`),
        gammaMlp: tensor(`gamma_mlp.${l}`),
        betaMlp: tensor(`beta_mlp.${l}`),
      });
    }
  }

  // Next-token logits for ids, which must fit in the context window; positions start at zero.
  logits(ids) {
    const H = this.hidden;
    let h = ids.map((id, t) => {
      const row = new Float32Array(H);
      for (let d = 0; d < H; d++) row[d] = this.embedding[id * H + d] + this.positions[t * H + d];
      return row;
    });

    for (const layer of this.layers) {
      const attn = this.attention(layer, h);
      h = h.map((row, t) => layerNorm(add(row, attn[t]), layer.gammaAttn, layer.betaAttn));
      h = h.map((row) => {
        const up = linear(row, layer.wUp, layer.bUp, H, 4 * H).map((x) => Math.max(x, 0));
        const mlp = linear(up, layer.wDown, layer.bDown, 4 * H, H);
        return layerNorm(add(row, mlp), layer.gammaMlp, layer.betaMlp);
      });
    }

    // Weight tying: project the last hidden state back through the embedding matrix.
    const last = h[h.length - 1];
    const logits = new Float32Array(this.vocabSize);
    for (let v = 0; v < this.vocabSize; v++) {
      let sum = 0;
      for (let d = 0; d < H; d++) sum += last[d] * this.embedding[v * H + d];
      logits[v] = sum;
    }
    return logits;
  }

  attention(layer, h) {
    const H = this.hidden;
    const D = this.headDim;
    const scale = Math.sqrt(D);
    const q = h.map((row) => linear(row, layer.wq, null, H, H));
    const k = h.map((row) => linear(row, layer.wk, null, H, H));
    const v = h.map((row) => linear(row, layer.wv, null, H, H));
    const context = h.map(() => new Float32Array(H));
    const scores = new Float64Array(h.length);

    for (let head = 0; head < this.heads; head++) {
      const base = head * D;
      for (let i = 0; i < h.length; i++) {
        // Causal mask: position i attends only to positions 0..i.
        let max = -Infinity;
        for (let j = 0; j <= i; j++) {
          let s = 0;
          for (let d = 0; d < D; d++) s += q[i][base + d] * k[j][base + d];
          scores[j] = s / scale;
          if (scores[j] > max) max = scores[j];
        }
        let total = 0;
        for (let j = 0; j <= i; j++) {
          scores[j] = Math.exp(scores[j] - max);
          total += scores[j];
        }
        for (let j = 0; j <= i; j++) {
          const w = scores[j] / total;
          for (let d = 0; d < D; d++) context[i][base + d] += w * v[j][base + d];
        }
      }
    }
    return context.map((row) => linear(row, layer.wo, null, H, H));
  }

  start(ids) {
    const tokens = [...ids];
    return {
      logits: () => this.logits(tokens.slice(-this.contextLen)),
      push: (id) => tokens.push(id),
    };
  }
}

class Recurrent {
  constructor(manifest, tensor, lstm) {
    this.type = lstm ? "lstm" : "rnn";
    this.lstm = lstm;
    this.params = manifest.params;
    this.vocabSize = manifest.vocab_size;
    this.hidden = manifest.hidden_dim;
    // Recurrent models carry the whole history in their state, so there is no context window.
    this.contextLen = null;
    this.embedding = tensor("embedding.weight");
    this.embedDim = this.embedding.length / this.vocabSize;
    const suffix = lstm ? "_lstm" : "";
    this.wIh = tensor(`W_ih${suffix}`);
    this.wHh = tensor(`W_hh${suffix}`);
    this.bIh = tensor(`b_ih${suffix}`);
    this.bHh = tensor(`b_hh${suffix}`);
    this.fcWeight = tensor("fc.weight");
    this.fcBias = tensor("fc.bias");
  }

  step(state, id) {
    const H = this.hidden;
    const width = this.lstm ? 4 * H : H;
    const x = this.embedding.subarray(id * this.embedDim, (id + 1) * this.embedDim);
    const fromInput = vecMat(x, this.wIh, this.embedDim, width);
    const fromHidden = vecMat(state.h, this.wHh, H, width);
    const pre = new Float32Array(width);
    for (let j = 0; j < width; j++) pre[j] = fromInput[j] + this.bIh[j] + fromHidden[j] + this.bHh[j];

    if (!this.lstm) return { h: pre.map(Math.tanh), c: null };

    // Gates are stacked as input, forget, cell, output.
    const h = new Float32Array(H);
    const c = new Float32Array(H);
    for (let j = 0; j < H; j++) {
      const i = sigmoid(pre[j]);
      const f = sigmoid(pre[H + j]);
      const g = Math.tanh(pre[2 * H + j]);
      const o = sigmoid(pre[3 * H + j]);
      c[j] = f * state.c[j] + i * g;
      h[j] = o * Math.tanh(c[j]);
    }
    return { h, c };
  }

  start(ids) {
    let state = { h: new Float32Array(this.hidden), c: this.lstm ? new Float32Array(this.hidden) : null };
    for (const id of ids) state = this.step(state, id);
    return {
      logits: () => linear(state.h, this.fcWeight, this.fcBias, this.hidden, this.vocabSize),
      push: (id) => {
        state = this.step(state, id);
      },
    };
  }

  logits(ids) {
    return this.start(ids).logits();
  }
}
