"""Shared pieces for training and using the pitch model: data, network, metrics."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from prepare_data import MAX_PITCH, MIN_PITCH, N_BINS

PROCESSED_DIR = Path("data/processed")
CHECKPOINT_PATH = Path("checkpoints/best_model.pt")
N_CLASSES = MAX_PITCH - MIN_PITCH + 1  # 88 piano keys.


def pick_device():
    """Use the Mac's GPU (MPS) when available, otherwise the CPU."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def normalize(spectrogram):
    """Map decibels (-80 to 0) to roughly -1 to 1, a range networks train well on."""
    return (spectrogram + 40.0) / 40.0


class SpectrogramDataset(Dataset):
    """Serves (spectrogram, class) pairs for one split from data/processed/.

    The spectrogram file is memory-mapped: clips are read from disk as needed,
    so the whole split never has to fit in RAM.
    """

    def __init__(self, split):
        self.x = np.load(PROCESSED_DIR / f"{split}_x.npy", mmap_mode="r")
        # Classes are 0-87: MIDI 21 (A0) is class 0.
        self.y = np.load(PROCESSED_DIR / f"{split}_y.npy") - MIN_PITCH

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        spectrogram = torch.from_numpy(normalize(np.array(self.x[i])))
        return spectrogram.unsqueeze(0), int(self.y[i])  # Add a channel dimension.


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
    )


class PitchCNN(nn.Module):
    """A small CNN that reads a [1, 264, time] CQT and scores the 88 notes.

    Pooling shrinks frequency and time between blocks. At the end, time is
    averaged away (a held note barely changes), but frequency positions are
    kept, because *where* the energy sits is what tells the pitch.
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(1, 32), nn.MaxPool2d(2),    # 264 -> 132 freq bins
            conv_block(32, 64), nn.MaxPool2d(2),   # 132 -> 66
            conv_block(64, 128), nn.MaxPool2d(2),  # 66 -> 33
            conv_block(128, 128),
            nn.AdaptiveAvgPool2d((N_BINS // 8, 1)),  # Average over time only.
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128 * (N_BINS // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, N_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def error_breakdown(predicted, actual):
    """Return accuracy and error types, each as a % of all clips.

    predicted and actual are arrays of class numbers (0-87).
    """
    predicted = np.asarray(predicted)
    actual = np.asarray(actual)
    wrong = predicted != actual
    diff = np.abs(predicted - actual)
    octave = wrong & (diff % 12 == 0)
    near = wrong & (diff == 1)
    total = len(actual)
    return {
        "accuracy": 100 * (~wrong).sum() / total,
        "octave_errors": 100 * octave.sum() / total,
        "near_misses": 100 * near.sum() / total,
        "other_errors": 100 * (wrong & ~octave & ~near).sum() / total,
    }


def loudest_bin_baseline(split):
    """The no-ML rule: guess the key whose CQT slot is loudest early in the note."""
    x = np.load(PROCESSED_DIR / f"{split}_x.npy", mmap_mode="r")
    y = np.load(PROCESSED_DIR / f"{split}_y.npy") - MIN_PITCH
    guesses = [int(np.argmax(x[i, :, :40].mean(axis=1))) // 3 for i in range(len(y))]
    return error_breakdown(guesses, y)


@torch.no_grad()
def predict_classes(model, loader, device):
    """Run the model over a DataLoader; return (predicted, actual, mean loss)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    predicted, actual, total_loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        scores = model(x)
        total_loss += loss_fn(scores, y).item()
        predicted.extend(scores.argmax(dim=1).cpu().tolist())
        actual.extend(y.cpu().tolist())
    return predicted, actual, total_loss / len(actual)


def format_breakdown(stats):
    return (
        f"accuracy {stats['accuracy']:.1f}% | octave errors {stats['octave_errors']:.1f}% "
        f"| near misses {stats['near_misses']:.1f}% | other {stats['other_errors']:.1f}%"
    )
