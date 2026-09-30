// Loads the exported demo assets and the PyTorch reference outputs from disk.

import { readFileSync } from "node:fs";
import { createModel } from "../../web/js/models.js";
import { BPETokenizer } from "../../web/js/tokenizer.js";

const MODELS = new URL("../../web/models/", import.meta.url);
const readJson = (url) => JSON.parse(readFileSync(url, "utf8"));

export const reference = readJson(new URL("reference.json", import.meta.url));
export const tokenizer = new BPETokenizer(readJson(new URL("tokenizer.json", MODELS)));

export function loadModel(name) {
  const manifest = readJson(new URL(`${name}.json`, MODELS));
  const bytes = readFileSync(new URL(manifest.weights, MODELS));
  return createModel(manifest, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

export function maxAbsDiff(a, b) {
  let worst = 0;
  for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]));
  return worst;
}

// Small deterministic generator so sampling tests are repeatable.
export function seededRandom(seed) {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
