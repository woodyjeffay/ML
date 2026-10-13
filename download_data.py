"""Download and unpack the NSynth validation and test splits into data/raw/.

NSynth dataset by Google Magenta, licensed CC BY 4.0:
https://magenta.withgoogle.com/datasets/nsynth

Usage:
    .venv/bin/python download_data.py
"""

import tarfile
import urllib.request
from pathlib import Path

BASE_URL = "http://download.magenta.tensorflow.org/datasets/nsynth"
SPLITS = ["valid", "test"]
RAW_DIR = Path("data/raw")


def report_progress(blocks_done, block_size, total_size):
    percent = min(100, blocks_done * block_size * 100 // total_size)
    print(f"\r  {percent}%", end="", flush=True)


def download_split(split):
    folder = RAW_DIR / f"nsynth-{split}"
    if folder.exists():
        print(f"{folder} already exists, skipping.")
        return

    archive = RAW_DIR / f"nsynth-{split}.jsonwav.tar.gz"
    print(f"Downloading {split} split...")
    urllib.request.urlretrieve(
        f"{BASE_URL}/{archive.name}", archive, reporthook=report_progress
    )
    print()

    print(f"Unpacking {archive.name}...")
    with tarfile.open(archive) as tar:
        tar.extractall(RAW_DIR, filter="data")
    archive.unlink()  # The unpacked files are all we need; free the space.


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        download_split(split)
    print("Done.")


if __name__ == "__main__":
    main()
