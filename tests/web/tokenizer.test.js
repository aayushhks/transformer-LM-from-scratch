import assert from "node:assert/strict";
import { test } from "node:test";
import { reference, tokenizer } from "./helpers.js";

test("encode and decode match the python tokenizer", () => {
  assert.ok(reference.tokenizer.length > 0);
  for (const { text, ids, decoded } of reference.tokenizer) {
    assert.deepEqual(tokenizer.encode(text), ids, JSON.stringify(text));
    assert.equal(tokenizer.decode(ids), decoded, JSON.stringify(text));
  }
});

test("decode drops special and unknown ids like python", () => {
  for (const { ids, text } of reference.decode) assert.equal(tokenizer.decode(ids), text);
});

test("vocabulary size matches the checkpoints", () => {
  assert.equal(tokenizer.vocabSize, 300);
  assert.equal(tokenizer.eosId, 3);
});
