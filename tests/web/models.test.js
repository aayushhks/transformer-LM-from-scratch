import assert from "node:assert/strict";
import { test } from "node:test";
import { generate, generateTokens } from "../../web/js/generate.js";
import { loadModel, maxAbsDiff, reference, seededRandom, tokenizer } from "./helpers.js";

test("next-token logits match pytorch for every model", () => {
  for (const [name, cases] of Object.entries(reference.logits)) {
    const model = loadModel(name);
    for (const { ids, logits } of cases) {
      const got = model.logits(ids);
      assert.equal(got.length, logits.length);
      const worst = maxAbsDiff(got, logits);
      assert.ok(worst < 1e-4, `${name} on ${ids.length} tokens: max abs diff ${worst}`);
    }
  }
});

test("greedy generation matches pytorch token for token", () => {
  for (const [name, cases] of Object.entries(reference.greedy)) {
    const model = loadModel(name);
    for (const { prompt, max_new_tokens: maxNewTokens, text } of cases) {
      const got = generate(model, tokenizer, prompt, { maxNewTokens, greedy: true });
      assert.equal(got, text, `${name}: ${prompt}`);
    }
  }
});

test("streaming yields one growing text per token and ends on the final text", () => {
  const model = loadModel("transformer_tiny");
  const options = { maxNewTokens: 25, temperature: 0.8 };
  const steps = [...generateTokens(model, tokenizer, "The world", options, seededRandom(7))];
  assert.ok(steps.length >= 1 && steps.length <= 25);
  const final = generate(model, tokenizer, "The world", options, seededRandom(7));
  assert.equal(steps.at(-1), final);
});

test("an empty prompt generates nothing", () => {
  const model = loadModel("rnn_tiny");
  assert.deepEqual([...generateTokens(model, tokenizer, "   ", { maxNewTokens: 5 })], []);
});
