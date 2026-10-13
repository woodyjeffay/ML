"""Turn the downloaded NSynth clips into constant-Q spectrograms, split by instrument.

Run download_data.py first. Then:
    .venv/bin/python prepare_data.py
"""

import json
from pathlib import Path

import librosa
import numpy as np

RAW_DIRS = [Path("data/raw/nsynth-valid"), Path("data/raw/nsynth-test")]
OUT_DIR = Path("data/processed")

# Only keep notes on an 88-key piano: A0 (MIDI 21) to C8 (MIDI 108).
MIN_PITCH = 21
MAX_PITCH = 108

# Fractions of *instruments* (not clips) that go into each split.
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42  # Fixed so the split comes out the same every run.

# Constant-Q spectrogram settings.
SAMPLE_RATE = 16000                      # NSynth's sample rate.
FMIN = librosa.note_to_hz("A0")          # 27.5 Hz, the lowest piano key.
BINS_PER_OCTAVE = 36                     # 3 bins per key (12 keys per octave).
N_BINS = 88 * 3                          # 264 bins: all 88 keys.
HOP_LENGTH = 512                         # Samples between spectrogram columns.


def load_examples():
    """Return a list of clips that are in range, each a dict of NSynth labels.

    Each dict also gets a "wav_path" key pointing at its audio file.
    """
    examples = []
    for raw_dir in RAW_DIRS:
        with open(raw_dir / "examples.json") as f:
            labels = json.load(f)
        for name, info in labels.items():
            if MIN_PITCH <= info["pitch"] <= MAX_PITCH:
                info["wav_path"] = raw_dir / "audio" / f"{name}.wav"
                examples.append(info)
    return examples


def split_by_instrument(examples, fractions, seed):
    """Divide clips into train / val / test so no instrument is in two splits.

    Args:
        examples: list of clip dicts from load_examples(). Each has an
            "instrument_str" key, e.g. "keyboard_acoustic_004".
        fractions: dict like {"train": 0.70, "val": 0.15, "test": 0.15}.
        seed: number to make the shuffle repeatable.

    Returns:
        dict mapping each split name to its list of clip dicts, e.g.
        {"train": [...], "val": [...], "test": [...]}.

    """
    instruments = sorted({ex["instrument_str"] for ex in examples})
    rng = np.random.default_rng(seed)
    rng.shuffle(instruments)

    # Give each split its share of instruments; the last split takes the rest
    # so rounding never drops or double-counts one.
    split_of = {}
    start = 0
    names = list(fractions)
    for i, name in enumerate(names):
        if i == len(names) - 1:
            count = len(instruments) - start
        else:
            count = round(fractions[name] * len(instruments))
        for instrument in instruments[start:start + count]:
            split_of[instrument] = name
        start += count

    splits = {name: [] for name in names}
    for ex in examples:
        splits[split_of[ex["instrument_str"]]].append(ex)
    return splits


def compute_cqt(wav_path):
    """Load one WAV file and return its constant-Q spectrogram.

    Returns:
        A NumPy array of shape (N_BINS, number_of_time_steps), about
        (264, 126) for a 4-second NSynth clip, in decibels.
    """
    samples, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    cqt = librosa.cqt(
        samples,
        sr=SAMPLE_RATE,
        hop_length=HOP_LENGTH,
        fmin=FMIN,
        n_bins=N_BINS,
        bins_per_octave=BINS_PER_OCTAVE,
    )
    # Strengths span a huge range, so convert to decibels (0 = loudest in the clip).
    return librosa.amplitude_to_db(np.abs(cqt), ref=np.max).astype(np.float32)


def save_split(name, examples):
    """Compute every clip's spectrogram and save them, plus labels, to OUT_DIR.

    Writes <name>_x.npy (spectrograms, shape [clips, N_BINS, time_steps]),
    <name>_y.npy (MIDI pitch per clip) and <name>_instruments.json.
    """
    first = compute_cqt(examples[0]["wav_path"])
    # open_memmap writes straight to disk, so the whole split never has to
    # fit in memory at once.
    x = np.lib.format.open_memmap(
        OUT_DIR / f"{name}_x.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(examples), *first.shape),
    )
    for i, ex in enumerate(examples):
        x[i] = first if i == 0 else compute_cqt(ex["wav_path"])
        if (i + 1) % 500 == 0 or i + 1 == len(examples):
            print(f"\r  {name}: {i + 1}/{len(examples)}", end="", flush=True)
    print()
    x.flush()

    y = np.array([ex["pitch"] for ex in examples], dtype=np.int64)
    np.save(OUT_DIR / f"{name}_y.npy", y)

    instruments = sorted({ex["instrument_str"] for ex in examples})
    with open(OUT_DIR / f"{name}_instruments.json", "w") as f:
        json.dump(instruments, f, indent=1)


def main():
    examples = load_examples()
    splits = split_by_instrument(examples, SPLIT_FRACTIONS, SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_examples in splits.items():
        n_instruments = len({ex["instrument_str"] for ex in split_examples})
        print(f"{name}: {len(split_examples)} clips from {n_instruments} instruments")
        save_split(name, split_examples)
    print("Done.")


if __name__ == "__main__":
    main()
