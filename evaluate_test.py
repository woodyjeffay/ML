"""Final, one-time check of the best saved model on the untouched test split.

Run only when you're done changing the model:
    .venv/bin/python evaluate_test.py
"""

import torch
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


def main():
    device = pick_device()
    model = PitchCNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    loader = DataLoader(SpectrogramDataset("test"), batch_size=64)
    predicted, actual, _ = predict_classes(model, loader, device)
    print(f"Test clips: {len(actual)}")
    print(f"Baseline (loudest slot): {format_breakdown(loudest_bin_baseline('test'))}")
    print(f"CNN:                     {format_breakdown(error_breakdown(predicted, actual))}")


if __name__ == "__main__":
    main()
