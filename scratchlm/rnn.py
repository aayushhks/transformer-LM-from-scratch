"""Recurrent language models with the RNN and LSTM recurrences written by hand."""

import math

import torch
from torch import nn


class RNNLM(nn.Module):
    """Elman RNN language model.

    The recurrence ``h_t = tanh(x_t W_ih + b_ih + h_{t-1} W_hh + b_hh)`` is built
    from raw parameters rather than ``nn.RNN`` so the update is fully visible.
    """

    def __init__(self, vocab_size, embed_size=128, hidden_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.W_ih = nn.Parameter(torch.empty(embed_size, hidden_size))
        self.W_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_ih = nn.Parameter(torch.empty(hidden_size))
        self.b_hh = nn.Parameter(torch.empty(hidden_size))
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -std, std)

    def rnn_cell(self, x, h_prev):
        return torch.tanh(x @ self.W_ih + self.b_ih + h_prev @ self.W_hh + self.b_hh)

    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)

        embedded = self.embedding(x)
        outputs = []
        for t in range(seq_len):
            hidden = self.rnn_cell(embedded[:, t, :], hidden)
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden)
        logits = self.fc(outputs)  # (batch, seq_len, vocab)
        return logits, hidden

    def logits(self, x):
        """Raw next-token logits, shared interface with the other LMs."""
        return self.forward(x)[0]


class LSTMLM(RNNLM):
    """LSTM language model with the four gates computed from raw parameters."""

    def __init__(self, vocab_size, embed_size=128, hidden_size=128):
        super().__init__(vocab_size, embed_size, hidden_size)
        # One stacked matrix per projection holds the input, forget, cell and
        # output gates side by side for a single efficient matmul.
        self.W_ih_lstm = nn.Parameter(torch.empty(embed_size, 4 * hidden_size))
        self.W_hh_lstm = nn.Parameter(torch.empty(hidden_size, 4 * hidden_size))
        self.b_ih_lstm = nn.Parameter(torch.empty(4 * hidden_size))
        self.b_hh_lstm = nn.Parameter(torch.empty(4 * hidden_size))
        self.reset_parameters()

    def lstm_cell(self, x, h_prev, c_prev):
        gates = x @ self.W_ih_lstm + self.b_ih_lstm + h_prev @ self.W_hh_lstm + self.b_hh_lstm
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def forward(self, x, states=None):
        batch_size, seq_len = x.size()
        if states is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
        else:
            h, c = states

        embedded = self.embedding(x)
        outputs = []
        for t in range(seq_len):
            h, c = self.lstm_cell(embedded[:, t, :], h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        logits = self.fc(outputs)
        return logits, (h, c)
