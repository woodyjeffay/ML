"""Train the pitch CNN on the train split, checking progress on the val split.

Run prepare_data.py first. Then:
    .venv/bin/python train.py

The model from the epoch with the best validation accuracy is saved to
checkpoints/best_model.pt. The test split is not touched here; use
evaluate_test.py once, at the end.
"""

import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from pitchnet import (
    CHECKPOINT_PATH,
    PitchCNN,
    SpectrogramDataset,
    error_breakdown,
    format_breakdown,
    loudest_bin_baseline,
    pick_device,
    predict_classes,
)

EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
SEED = 42


def main():
    torch.manual_seed(SEED)
    device = pick_device()
    print(f"Training on: {device}")

    train_loader = DataLoader(SpectrogramDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SpectrogramDataset("val"), batch_size=BATCH_SIZE)
    print(f"Clips: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val")
    print(f"Baseline (loudest slot, val): {format_breakdown(loudest_bin_baseline('val'))}")

    model = PitchCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)
    best_val_accuracy = -1.0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        train_loss, train_correct, seen = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y)

            optimizer.zero_grad()
            loss.backward()   # Work out how each weight affected the loss...
            optimizer.step()  # ...and nudge the weights to reduce it.

            train_loss += loss.item() * len(y)
            train_correct += (scores.argmax(dim=1) == y).sum().item()
            seen += len(y)

        predicted, actual, val_loss = predict_classes(model, val_loader, device)
        val_stats = error_breakdown(predicted, actual)
        saved = ""
        if val_stats["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_stats["accuracy"]
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            saved = "  [saved best]"

        print(
            f"Epoch {epoch:2d}/{EPOCHS} ({time.time() - start:.0f}s) | "
            f"train loss {train_loss / seen:.3f}, acc {100 * train_correct / seen:.1f}% | "
            f"val loss {val_loss:.3f}, {format_breakdown(val_stats)}{saved}",
            flush=True,
        )

    print(f"Best val accuracy: {best_val_accuracy:.1f}% (saved to {CHECKPOINT_PATH})")


if __name__ == "__main__":
    main()
