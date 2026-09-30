"""Training and evaluation loop shared by the RNN, LSTM and transformer models."""

import math

import torch
from torch import nn, optim
from torch.utils.data import DataLoader


def evaluate_perplexity(model, loader, device="cpu"):
    """Token-level perplexity of a model over a data loader."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model.logits(x)
            total_loss += criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
            total_tokens += y.numel()

    model.train()
    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def train_lm(model, train_dataset, dev_dataset, config, save_path=None):
    """Train a neural language model and return the per-epoch history.

    Works for any model exposing ``logits(x) -> (batch, seq_len, vocab)``, which
    covers the RNN, LSTM and transformer implementations in this package.
    """
    device = config.device
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)

    history = {"train_loss": [], "dev_perplexity": []}
    best_ppl = float("inf")

    for epoch in range(config.epochs):
        model.train()
        epoch_loss, num_batches = 0.0, 0

        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model.logits(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            if config.log_every and step % config.log_every == 0:
                print(f"epoch {epoch + 1} step {step} loss {loss.item():.4f}")

        avg_loss = epoch_loss / max(1, num_batches)
        dev_ppl = evaluate_perplexity(model, dev_loader, device)
        history["train_loss"].append(avg_loss)
        history["dev_perplexity"].append(dev_ppl)
        print(f"epoch {epoch + 1} done | train loss {avg_loss:.4f} | dev perplexity {dev_ppl:.4f}")

        if save_path and dev_ppl < best_ppl:
            best_ppl = dev_ppl
            torch.save(model.state_dict(), save_path)

    return history
