"""Download MIND-small (train + dev) [wu2020mind]; Phase 5 plan Q1, Q4.

Source: the open Hugging Face copy used by Microsoft's `recommenders` library
(`recommenders/datasets/mind.py`). The official page (msnews.github.io) links to a gated copy.
Use is governed by the Microsoft Research License Terms (research only, no redistribution):
the data stays in data/ (gitignored). Prints SHA-256 of every zip and unzipped file.
"""

import argparse
import hashlib
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/Recommenders/MIND/resolve/main/MIND{size}_{split}.zip"
root = Path(__file__).resolve().parent.parent / "data/raw/mind"
parser = argparse.ArgumentParser()
parser.add_argument("--size", choices=["small", "large"], default="small")
size = parser.parse_args().size


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:  # chunked: large files are GBs
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


for split in ["train", "dev"]:  # no test split: its click labels are not public
    out = root / size / split
    zip_path = root / f"MIND{size}_{split}.zip"
    out.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():  # also lets a hand-downloaded zip be used
        print("downloading", zip_path.name)
        urllib.request.urlretrieve(URL.format(size=size, split=split), zip_path)
    zipfile.ZipFile(zip_path).extractall(out)
    print(f"{zip_path.name}  {zip_path.stat().st_size:>12,} B  sha256 {sha256(zip_path)}")
    for f in sorted(out.rglob("*")):
        if f.is_file():
            print(f"  {f.relative_to(root)}  {f.stat().st_size:>12,} B  sha256 {sha256(f)}")
