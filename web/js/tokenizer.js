// Byte-pair-encoding tokenizer matching scratchlm/tokenizer.py.

// Python's \s and str.strip() whitespace set, which differs from JavaScript's \s.
const PY_SPACE = "\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000";
const WORD_RE = new RegExp(`[\\p{L}\\p{N}_]+|[^\\p{L}\\p{N}_${PY_SPACE}]`, "gu");
const STRIP_RE = new RegExp(`^[${PY_SPACE}]+|[${PY_SPACE}]+$`, "gu");

export function basicTokenize(text) {
  return text.match(WORD_RE) ?? [];
}

export class BPETokenizer {
  constructor({ special_tokens, end_of_word, vocab, merges }) {
    this.specialTokens = new Set(special_tokens);
    this.endOfWord = end_of_word;
    this.vocab = new Map(Object.entries(vocab));
    this.inverse = new Map([...this.vocab].map(([token, id]) => [id, token]));
    this.ranks = new Map();
    merges.forEach(([a, b], rank) => {
      if (!this.ranks.has(a)) this.ranks.set(a, new Map());
      this.ranks.get(a).set(b, rank);
    });
    this.unkId = this.vocab.get("<unk>");
    this.eosId = this.vocab.get("</s>");
  }

  get vocabSize() {
    return this.vocab.size;
  }

  applyBpe(symbols) {
    // Merge the lowest-ranked adjacent pair everywhere it occurs until no pair is known.
    for (;;) {
      let best = -1;
      let bestRank = Infinity;
      for (let i = 0; i < symbols.length - 1; i++) {
        const rank = this.ranks.get(symbols[i])?.get(symbols[i + 1]);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          best = i;
        }
      }
      if (best < 0) return symbols;

      const [a, b] = [symbols[best], symbols[best + 1]];
      const merged = [];
      for (let i = 0; i < symbols.length; ) {
        if (i < symbols.length - 1 && symbols[i] === a && symbols[i + 1] === b) {
          merged.push(a + b);
          i += 2;
        } else {
          merged.push(symbols[i]);
          i += 1;
        }
      }
      symbols = merged;
    }
  }

  encode(text) {
    const ids = [];
    for (const word of basicTokenize(text)) {
      for (const symbol of this.applyBpe([...Array.from(word), this.endOfWord])) {
        ids.push(this.vocab.get(symbol) ?? this.unkId);
      }
    }
    return ids;
  }

  decode(ids) {
    let text = "";
    for (const id of ids) {
      const token = this.inverse.get(id);
      if (token !== undefined && !this.specialTokens.has(token)) text += token;
    }
    return text.replaceAll(this.endOfWord, " ").replace(STRIP_RE, "");
  }
}
