// Autoregressive generation matching generate_stream and generate in scratchlm/sampling.py.

import { sampleNextToken } from "./sampling.js";

// Yields the decoded text after each new token so callers can stream it.
export function* generateTokens(model, tokenizer, prompt, options = {}, random = Math.random) {
  const { maxNewTokens = 50 } = options;
  const tokens = tokenizer.encode(prompt);
  if (tokens.length === 0) return;
  const session = model.start(tokens);
  for (let i = 0; i < maxNewTokens; i++) {
    const next = sampleNextToken(session.logits(), options, random);
    tokens.push(next);
    session.push(next);
    yield tokenizer.decode(tokens);
    if (next === tokenizer.eosId) return;
  }
}

export function generate(model, tokenizer, prompt, options = {}, random = Math.random) {
  let text = tokenizer.decode(tokenizer.encode(prompt));
  for (const next of generateTokens(model, tokenizer, prompt, options, random)) text = next;
  return text;
}
