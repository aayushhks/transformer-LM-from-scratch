"""Decoder-only transformer language models built from scratch.

``Transformer`` implements single-head causal self-attention; ``MultiHeadTransformer``
reuses the same block and only overrides the attention computation. Layer norm,
the causal mask and weight tying are all written explicitly.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class Transformer(nn.Module):
    """Multi-layer decoder with single-head causal self-attention."""

    def __init__(self, vocab_size, hidden_dim, context_len, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(context_len, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "w_q": nn.Linear(hidden_dim, hidden_dim, bias=False),
                        "w_k": nn.Linear(hidden_dim, hidden_dim, bias=False),
                        "w_v": nn.Linear(hidden_dim, hidden_dim, bias=False),
                        "w_o": nn.Linear(hidden_dim, hidden_dim, bias=False),
                        "w_up": nn.Linear(hidden_dim, 4 * hidden_dim),
                        "w_down": nn.Linear(4 * hidden_dim, hidden_dim),
                    }
                )
            )

        # Per-layer learnable layer-norm parameters for the attention and MLP blocks.
        self.gamma_attn = nn.ParameterList(
            [nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_layers)]
        )
        self.beta_attn = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)]
        )
        self.gamma_mlp = nn.ParameterList(
            [nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_layers)]
        )
        self.beta_mlp = nn.ParameterList(
            [nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)]
        )

    def layer_norm(self, x, gamma, beta, eps=1e-5):
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return gamma * (x - mu) / torch.sqrt(var + eps) + beta

    def _attention(self, layer, h, seq_len, device):
        # Single-head scaled dot-product attention with a causal mask.
        q = layer["w_q"](h)
        k = layer["w_k"](h)
        v = layer["w_v"](h)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.hidden_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        return layer["w_o"](weights @ v)

    def forward(self, x):
        _, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        for i, layer in enumerate(self.layers):
            attn = self._attention(layer, h, seq_len, x.device)
            h = self.layer_norm(h + attn, self.gamma_attn[i], self.beta_attn[i])

            mlp = layer["w_down"](F.relu(layer["w_up"](h)))
            h = self.layer_norm(h + mlp, self.gamma_mlp[i], self.beta_mlp[i])

        return h

    def lm_head(self, h):
        # Weight tying: project hidden states back through the embedding matrix.
        return h @ self.embedding.weight.t()

    def logits(self, x):
        """Raw next-token logits, shared interface with the other LMs."""
        return self.lm_head(self.forward(x))


class MultiHeadTransformer(Transformer):
    """Same decoder block but with multi-head causal self-attention."""

    def __init__(self, vocab_size, hidden_dim, context_len, num_heads=4, num_layers=2):
        super().__init__(vocab_size, hidden_dim, context_len, num_layers)
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def _attention(self, layer, h, seq_len, device):
        batch_size = h.size(0)

        # Project then split the hidden dimension across heads.
        q = layer["w_q"](h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = layer["w_k"](h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = layer["w_v"](h).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        context = weights @ v

        # Concatenate the heads back into the hidden dimension.
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return layer["w_o"](context)
