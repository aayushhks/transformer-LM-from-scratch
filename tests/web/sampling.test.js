import assert from "node:assert/strict";
import { test } from "node:test";
import { argmax, nextTokenProbs, sampleNextToken } from "../../web/js/sampling.js";
import { maxAbsDiff, reference, seededRandom } from "./helpers.js";

test("temperature, top-k and top-p probabilities match pytorch", () => {
  for (const c of reference.filters) {
    const label = `temperature ${c.temperature} top_k ${c.top_k} top_p ${c.top_p}`;
    const probs = nextTokenProbs(c.logits, { temperature: c.temperature, topK: c.top_k, topP: c.top_p });
    const kept = (p) => p.flatMap((x, i) => (x > 0 ? [i] : []));
    assert.deepEqual(kept(Array.from(probs)), kept(c.probs), `${label}: kept tokens differ`);
    const worst = maxAbsDiff(probs, c.probs);
    assert.ok(worst < 1e-6, `${label}: max abs diff ${worst}`);
  }
});

test("greedy picks the first maximum like torch.argmax", () => {
  assert.equal(argmax([0.1, 5.0, -1.0, 5.0]), 1);
  assert.equal(sampleNextToken([0.1, 5.0, -1.0, 2.0], { greedy: true }), 1);
});

test("sampling only returns tokens that survive top-k", () => {
  const logits = [0.5, 3.0, -2.0, 2.5, 1.0, 2.9];
  const random = seededRandom(1);
  for (let i = 0; i < 500; i++) {
    assert.ok([1, 3, 5].includes(sampleNextToken(logits, { topK: 3 }, random)));
  }
});

test("sampling follows the softmax distribution", () => {
  const logits = [Math.log(0.5), Math.log(0.3), Math.log(0.2)];
  const counts = [0, 0, 0];
  const random = seededRandom(42);
  const n = 20000;
  for (let i = 0; i < n; i++) counts[sampleNextToken(logits, { temperature: 1 }, random)]++;
  [0.5, 0.3, 0.2].forEach((p, i) => assert.ok(Math.abs(counts[i] / n - p) < 0.015, `token ${i}: ${counts[i] / n}`));
});
