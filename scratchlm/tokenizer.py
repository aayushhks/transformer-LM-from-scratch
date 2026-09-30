"""Tokenizers: a whitespace baseline and a byte-pair-encoding tokenizer.

Both expose a small encode/decode interface and the usual special-token ids so
they can be swapped for a Hugging Face tokenizer in the generation code.
"""

import collections
import re

# Matches word characters or single punctuation symbols; deterministic and
# dependency-free so training and tests do not need any downloaded data.
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def basic_tokenize(text, lower=False):
    """Split text into word and punctuation tokens."""
    if lower:
        text = text.lower()
    return _TOKEN_RE.findall(text)


class BaseTokenizer:
    """Shared special-token handling for the concrete tokenizers."""

    special_tokens = ("<pad>", "<unk>", "<s>", "</s>")
    end_of_word = "▁"  # marks the end of a word so suffixes can merge

    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_token_id(self):
        return self.vocab["<pad>"]

    @property
    def unk_token_id(self):
        return self.vocab["<unk>"]

    @property
    def bos_token_id(self):
        return self.vocab["<s>"]

    @property
    def eos_token_id(self):
        return self.vocab["</s>"]

    def _init_special_tokens(self):
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = idx
            self.inverse_vocab[idx] = token


class WordTokenizer(BaseTokenizer):
    """Whitespace/punctuation tokenizer keeping the most frequent words."""

    def __init__(self, target_vocab_size=None):
        super().__init__()
        self.target_vocab_size = target_vocab_size

    def train(self, text):
        counts = collections.Counter(basic_tokenize(text))
        self._init_special_tokens()
        if self.target_vocab_size is None:
            keep = len(counts)
        else:
            keep = self.target_vocab_size - len(self.special_tokens)

        next_id = len(self.vocab)
        for word, _ in counts.most_common(keep):
            if word not in self.vocab:
                self.vocab[word] = next_id
                self.inverse_vocab[next_id] = word
                next_id += 1

    def encode(self, text):
        unk = self.unk_token_id
        return [self.vocab.get(word, unk) for word in basic_tokenize(text)]

    def decode(self, ids):
        words = [self.inverse_vocab.get(i, "<unk>") for i in ids]
        return " ".join(w for w in words if w not in self.special_tokens)


class BPETokenizer(BaseTokenizer):
    """Byte-pair-encoding tokenizer trained by iteratively merging frequent pairs."""

    def __init__(self, vocab_size=1000):
        super().__init__()
        self.target_vocab_size = vocab_size
        self.merges = []  # ordered list of ((a, b), "ab")
        self.bpe_ranks = {}  # (a, b) -> rank, lower rank merges first

    def get_stats(self, vocab_counts):
        """Count the frequency of every adjacent symbol pair in the corpus."""
        pairs = collections.defaultdict(int)
        for word, freq in vocab_counts.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab_counts):
        """Replace every occurrence of ``pair`` with the merged symbol."""
        merged = {}
        bigram = re.escape(" ".join(pair))
        pattern = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word, freq in vocab_counts.items():
            merged[pattern.sub("".join(pair), word)] = freq
        return merged

    def train(self, text):
        # Represent each word as space-separated characters plus an end marker.
        word_freqs = collections.Counter(basic_tokenize(text))
        vocab_counts = {
            " ".join(list(word) + [self.end_of_word]): freq
            for word, freq in word_freqs.items()
        }

        # The base alphabet is every character symbol that appears in the corpus.
        alphabet = set()
        for word in vocab_counts:
            alphabet.update(word.split())

        # Merge the most frequent pair until we hit the target vocabulary size.
        num_merges = self.target_vocab_size - len(self.special_tokens) - len(alphabet)
        self.merges = []
        for _ in range(max(0, num_merges)):
            stats = self.get_stats(vocab_counts)
            if not stats:
                break
            best = max(stats, key=stats.get)
            vocab_counts = self.merge_vocab(best, vocab_counts)
            self.merges.append((best, "".join(best)))

        # Build the vocabulary: special tokens, then the alphabet, then merges.
        self._init_special_tokens()
        next_id = len(self.vocab)
        for token in sorted(alphabet):
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.inverse_vocab[next_id] = token
                next_id += 1
        for _, token in self.merges:
            if token not in self.vocab:
                self.vocab[token] = next_id
                self.inverse_vocab[next_id] = token
                next_id += 1

        self.bpe_ranks = {pair: rank for rank, (pair, _) in enumerate(self.merges)}

    def _apply_bpe(self, symbols):
        """Greedily apply the learned merges to a list of symbols, in rank order."""
        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [(self.bpe_ranks[p], p) for p in pairs if p in self.bpe_ranks]
            if not ranked:
                break
            _, best = min(ranked)

            merged = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        return symbols

    def encode(self, text):
        ids = []
        for word in basic_tokenize(text):
            symbols = self._apply_bpe(list(word) + [self.end_of_word])
            ids.extend(self.vocab.get(s, self.unk_token_id) for s in symbols)
        return ids

    def decode(self, ids):
        tokens = [self.inverse_vocab.get(i, "") for i in ids]
        tokens = [t for t in tokens if t not in self.special_tokens]
        return "".join(tokens).replace(self.end_of_word, " ").strip()

    def to_dict(self):
        """Serialize the trained tokenizer to plain Python types."""
        return {
            "target_vocab_size": self.target_vocab_size,
            "vocab": self.vocab,
            "merges": [[list(pair), token] for pair, token in self.merges],
        }

    @classmethod
    def from_dict(cls, state):
        """Rebuild a tokenizer saved with :meth:`to_dict`."""
        tok = cls(vocab_size=state["target_vocab_size"])
        tok.vocab = state["vocab"]
        tok.inverse_vocab = {idx: token for token, idx in tok.vocab.items()}
        tok.merges = [((a, b), token) for (a, b), token in state["merges"]]
        tok.bpe_ranks = {pair: rank for rank, (pair, _) in enumerate(tok.merges)}
        return tok
