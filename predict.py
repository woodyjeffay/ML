"""Name the note in an audio file, e.g. "A4".

Usage:
    .venv/bin/python predict.py path/to/note.wav

Version 1 expects NSynth-style audio: one clean note. Only the first 4 seconds
are used; shorter files are padded with silence.
"""

import sys

import librosa
import numpy as np
import torch

from pitchnet import CHECKPOINT_PATH, PitchCNN, normalize, pick_device
from prepare_data import (
    BINS_PER_OCTAVE,
    FMIN,
    HOP_LENGTH,
    MIN_PITCH,
    N_BINS,
    SAMPLE_RATE,
)

CLIP_SECONDS = 4


def load_spectrogram(path):
    """Load any audio file as a 4-second, 16 kHz mono CQT in decibels."""
    samples, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)
    samples = librosa.util.fix_length(samples, size=SAMPLE_RATE * CLIP_SECONDS)
    cqt = librosa.cqt(
        samples,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    return librosa.amplitude_to_db(np.abs(cqt), ref=np.max).astype(np.float32)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)

    device = pick_device()
    model = PitchCNN().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    x = torch.from_numpy(normalize(load_spectrogram(sys.argv[1])))[None, None].to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)[0].cpu()

    top = torch.topk(probabilities, 3)
    for rank, (p, cls) in enumerate(zip(top.values, top.indices)):
        name = librosa.midi_to_note(int(cls) + MIN_PITCH, unicode=False)
        label = "Note:" if rank == 0 else "  or:"
        print(f"{label} {name}  ({100 * p:.0f}% confident)")


if __name__ == "__main__":
    main()
