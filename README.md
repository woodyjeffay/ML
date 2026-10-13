## Perfect Pitch

Machine learnt perfect pitch.

Give it an audio file of a single musical note and it tells you which note it is, e.g. `A4`.

A small convolutional neural network (PyTorch) reads a constant-Q spectrogram of the
audio and picks one of the 88 piano keys (A0 to C8). It was trained on part of the
[NSynth](https://magenta.withgoogle.com/datasets/nsynth) dataset.

**Current accuracy:** 92.6% on instruments it never saw in training (validation split),
vs. 62.5% for a simple "loudest frequency" rule. Most remaining mistakes are the right
note in the wrong octave. Version 1 is trained on clean studio notes only, so expect
worse results on real-world recordings. Training on the full set (289,205 examples) would provide greater accuracy.

## Setup

Requires Python 3.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Name a note

A trained model is included (`checkpoints/best_model.pt`), so you can try it straight away:

```
.venv/bin/python predict.py path/to/note.wav
```

```
Note: A4  (97% confident)
  or: A3  (2% confident)
  or: A#4  (0% confident)
```

- Reads WAV, MP3, FLAC, OGG and AIFF. On a Mac, convert `.m4a` first:
  `afconvert -f WAVE -d LEI16 note.m4a note.wav`
- Only the first 4 seconds are used, so start the note at the beginning of the file.

## Train it yourself

Needs about 5 GB of free disk space.

```
.venv/bin/python download_data.py   # NSynth validation + test splits (~1.4 GB download)
.venv/bin/python prepare_data.py    # spectrograms, split by instrument 70/15/15
.venv/bin/python train.py           # ~20 min on an Apple M2; saves the best model
.venv/bin/python evaluate_test.py   # final score on held-out instruments; run once
```

## Files

| File | What it does |
|---|---|
| `download_data.py` | Downloads and unpacks NSynth into `data/raw/` |
| `prepare_data.py` | Drops notes outside the piano range, splits by instrument, computes spectrograms into `data/processed/` |
| `pitchnet.py` | The CNN, data loading and accuracy measures |
| `train.py` | Trains the model, reporting accuracy and error types each epoch |
| `evaluate_test.py` | One-time final check on the test split |
| `predict.py` | Names the note in an audio file |

## Credits

Trained on the NSynth dataset by Google Magenta, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
