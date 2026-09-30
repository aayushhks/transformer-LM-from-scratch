"""Export the trained checkpoints to static files for the browser demo and its parity tests."""

import argparse
import json
import os

import torch

from scratchlm.models import build_lm
from scratchlm.sampling import _apply_top_k, _apply_top_p, generate

NAMES = ["transformer_tiny", "lstm_tiny", "rnn_tiny"]
PROMPTS = ["The world", "In the beginning", "Scientists have discovered", "Hello, world! 42"]
TEXTS = [
    "The world",
    "Hello, world! It's 2026.",
    "  leading and trailing spaces  ",
    "tabs\tand\nnewlines\r\nmixed",
    "snake_case_name and CamelCase",
    "numbers 3.14159 and 1,000,000",
    "café naïve résumé Zürich",
    "東京 and Москва",
    "\U0001d400\U0001d401 bold and \U0001d11e clef",
    "odd\x1cspace\x85nel\xa0nbsp﻿bom　wide",
    "the marker ▁ itself",
    "é combining",
    "!!!???...,,,",
    "",
]
DECODE_IDS = [[2, 265, 204, 3, 999, 1], [0, 1, 2, 3], []]
FILTERS = [
    (0.8, None, None),
    (1.0, 5, None),
    (1.0, None, 0.9),
    (0.7, 20, 0.8),
    (1.5, 1, None),
    (1.0, None, 0.05),
    (2.0, 50, 0.95),
]


def load(path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    cfg, tokenizer = bundle["config"], bundle["tokenizer"]
    model = build_lm(
        bundle["model_type"], tokenizer.vocab_size,
        cfg["hidden_dim"], cfg["context_len"], cfg["num_heads"], cfg["num_layers"],
    )
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return bundle, model


def write_json(path, data, indent=2):
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)
        f.write("\n")


def export_weights(name, bundle, out_dir):
    tensors, blobs, offset = {}, [], 0
    for key, value in bundle["model_state"].items():
        array = value.detach().to(torch.float32).contiguous().numpy().astype("<f4")
        tensors[key] = {"offset": offset, "shape": list(array.shape)}
        offset += array.size
        blobs.append(array.tobytes())

    with open(os.path.join(out_dir, f"{name}.bin"), "wb") as f:
        f.write(b"".join(blobs))

    cfg = bundle["config"]
    write_json(os.path.join(out_dir, f"{name}.json"), {
        "name": name,
        "model_type": bundle["model_type"],
        "vocab_size": bundle["tokenizer"].vocab_size,
        "hidden_dim": cfg["hidden_dim"],
        "context_len": cfg["context_len"],
        "num_heads": cfg["num_heads"],
        "num_layers": cfg["num_layers"],
        "params": offset,
        "weights": f"{name}.bin",
        "tensors": tensors,
    })


def export_tokenizer(tokenizer, out_dir):
    write_json(os.path.join(out_dir, "tokenizer.json"), {
        "special_tokens": tokenizer.special_tokens,
        "end_of_word": tokenizer.end_of_word,
        "vocab": tokenizer.vocab,
        "merges": [list(pair) for pair, _ in tokenizer.merges],
    })


def rounded(values):
    return [float(f"{v:.8g}") for v in values]


@torch.no_grad()
def reference_outputs(models, tokenizer):
    ref = {"tokenizer": [], "decode": [], "logits": {}, "greedy": {}, "filters": []}
    for text in TEXTS:
        ids = tokenizer.encode(text)
        ref["tokenizer"].append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})
    for ids in DECODE_IDS:
        ref["decode"].append({"ids": ids, "text": tokenizer.decode(ids)})

    generator = torch.Generator().manual_seed(0)
    for name, model in models.items():
        # Recurrent models have no context window, so test them on a longer sequence.
        longest = getattr(model, "context_len", 50)
        cases = []
        for length in (1, 7, longest):
            ids = torch.randint(tokenizer.vocab_size, (length,), generator=generator).tolist()
            logits = model.logits(torch.tensor([ids]))[0, -1]
            cases.append({"ids": ids, "logits": rounded(logits.tolist())})
        ref["logits"][name] = cases
        ref["greedy"][name] = [
            {"prompt": p, "max_new_tokens": 60, "text": generate(model, tokenizer, p, 60, greedy=True)}
            for p in PROMPTS
        ]

    for temperature, top_k, top_p in FILTERS:
        logits = torch.tensor(rounded((torch.randn(tokenizer.vocab_size, generator=generator) * 3).tolist()))
        scaled = logits / max(temperature, 1e-8)
        probs = torch.softmax(_apply_top_p(_apply_top_k(scaled, top_k), top_p), dim=-1)
        ref["filters"].append({
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "logits": logits.tolist(), "probs": rounded(probs.tolist()),
        })
    return ref


def main():
    parser = argparse.ArgumentParser(description="export checkpoints for the browser demo")
    parser.add_argument("--checkpoint_dir", default="checkpoints")
    parser.add_argument("--out_dir", default="web/models")
    parser.add_argument("--reference", default="tests/web/reference.json")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    bundles, models = {}, {}
    for name in NAMES:
        bundles[name], models[name] = load(os.path.join(args.checkpoint_dir, f"{name}.pt"))
        export_weights(name, bundles[name], args.out_dir)

    tokenizer = bundles[NAMES[0]]["tokenizer"]
    for name in NAMES[1:]:
        other = bundles[name]["tokenizer"]
        if other.vocab != tokenizer.vocab or other.merges != tokenizer.merges:
            raise ValueError(f"{name} uses a different tokenizer; the demo expects one shared tokenizer")
    export_tokenizer(tokenizer, args.out_dir)
    write_json(os.path.join(args.out_dir, "index.json"), {"models": NAMES, "default": NAMES[0]})

    if args.reference:
        os.makedirs(os.path.dirname(args.reference), exist_ok=True)
        write_json(args.reference, reference_outputs(models, tokenizer), indent=None)


if __name__ == "__main__":
    main()
