"""
precompute_preprocessing.py -- hair removal and CLAHE are DETERMINISTIC
(unlike random flip/rotation), so there is no reason to recompute them
from scratch on every single epoch. Measured cost: ~0.76s/image on this
machine -- at 200 epochs x 2000 images, doing it on-the-fly would add
roughly 40+ hours to a run that otherwise takes under an hour.

This script runs the same hair_removal/CLAHE transforms ONCE per image
and writes the results to a new folder. Point your config's
train_images/test_images at the *_preprocessed folder this creates, and
turn OFF hair_removal/clahe in configs/base_preprocessing.yaml's
data.preprocessing block (they're already baked into the cached images;
leaving them on would apply them twice).

Usage:
    python precompute_preprocessing.py \
        --src data_root/ISIC2017/train/images \
        --dst data_root/ISIC2017_preprocessed/train/images \
        --hair_removal --clahe
"""
import argparse
import os
import time
from pathlib import Path

from PIL import Image
from data.transforms import remove_hair, apply_clahe

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="source image folder")
    parser.add_argument("--dst", required=True, help="destination folder for preprocessed images")
    parser.add_argument("--hair_removal", action="store_true")
    parser.add_argument("--clahe", action="store_true")
    args = parser.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(src.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    print(f"Preprocessing {len(files)} images from {src} -> {dst}")
    print(f"hair_removal={args.hair_removal}  clahe={args.clahe}")

    t0 = time.time()
    for i, path in enumerate(files):
        out_path = dst / path.name
        if out_path.exists():
            continue  # resume-friendly: skip already-processed images
        img = Image.open(path).convert("RGB")
        if args.hair_removal:
            img = remove_hair(img)
        if args.clahe:
            img = apply_clahe(img)
        img.save(out_path)
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(files) - i - 1) / rate if rate > 0 else 0
            print(f"  {i+1}/{len(files)}  ({rate:.1f} img/s, ETA {eta/60:.1f} min)")

    print(f"Done in {(time.time()-t0)/60:.1f} min. Preprocessed images: {dst}")


if __name__ == "__main__":
    main()
