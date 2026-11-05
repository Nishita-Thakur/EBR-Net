"""
failure_mode_subset.py -- Section 4.6 of the paper: does the adaptive model
actually help on the baseline's own documented failure modes (hair-obscured
lesions, low-contrast boundaries), even if the full-dataset average doesn't
show it?

This does NOT re-run the model. It only needs the test images/masks (to
flag which ones are hair-heavy / low-contrast) and the per_sample_json
files evaluate.py already produced. Because evaluate.py's DataLoader uses
shuffle=False and ISICDataset lists files with sorted(folder.iterdir()),
per_sample["dice"][i] always corresponds to the i-th file in that sorted
listing -- so the subset masks computed here line up directly with the
already-saved JSONs, for ANY checkpoint evaluated on the same dataset split.
"""
import argparse
import json

import cv2
import numpy as np
import yaml
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(folder):
    folder = Path(folder)
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]


def find_matching_mask(image_path: Path, mask_folder: Path):
    stem = image_path.stem
    for suffix in ["_segmentation.png", ".png", "_Segmentation.png"]:
        cand = mask_folder / f"{stem}{suffix}"
        if cand.exists():
            return cand
    matches = list(mask_folder.glob(f"{stem}*"))
    return matches[0] if matches else None


def hair_score(image_bgr):
    gray = image_bgr[:, :, 1]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    return float((mask > 0).mean())


def contrast_score(image_bgr, gt_mask):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    fg = gray[gt_mask > 127]
    bg = gray[gt_mask <= 127]
    if fg.size == 0 or bg.size == 0:
        return np.nan
    return abs(fg.mean() - bg.mean())


def subset_stats(per_sample, indices, metric):
    vals = np.array([per_sample[metric][i] for i in indices], dtype=np.float64)
    vals = vals[~np.isnan(vals)]
    scale = 100.0 if metric in ("dice", "iou", "acc", "se", "sp") else 1.0
    return float(vals.mean()) * scale, float(vals.std()) * scale, len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--dataset", default="isic2017")
    parser.add_argument("--results", action="append", required=True,
                         help="path/to/per_sample.json:Label, repeatable")
    parser.add_argument("--hair_quantile", type=float, default=0.75,
                         help="images at/above this hair-score quantile are 'hair-heavy'")
    parser.add_argument("--contrast_quantile", type=float, default=0.25,
                         help="images at/below this contrast-score quantile are 'low-contrast'")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    ds_cfg = base_cfg["data"]["datasets"][args.dataset]

    images = list_images(ds_cfg["test_images"])
    masks_dir = Path(ds_cfg["test_masks"])

    hair_scores, contrast_scores = [], []
    for img_path in images:
        mask_path = find_matching_mask(img_path, masks_dir)
        image_bgr = cv2.imread(str(img_path))
        gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path else None
        if image_bgr is None or gt_mask is None:
            hair_scores.append(np.nan)
            contrast_scores.append(np.nan)
            continue
        hair_scores.append(hair_score(image_bgr))
        contrast_scores.append(contrast_score(image_bgr, gt_mask))

    hair_scores = np.array(hair_scores)
    contrast_scores = np.array(contrast_scores)

    hair_thresh = np.nanquantile(hair_scores, args.hair_quantile)
    contrast_thresh = np.nanquantile(contrast_scores, args.contrast_quantile)

    hair_idx = [i for i, s in enumerate(hair_scores) if not np.isnan(s) and s >= hair_thresh]
    lowcon_idx = [i for i, s in enumerate(contrast_scores) if not np.isnan(s) and s <= contrast_thresh]
    all_idx = list(range(len(images)))

    print(f"Dataset: {args.dataset}  ({len(images)} test images)")
    print(f"Hair-heavy subset:     {len(hair_idx)} images (hair-score >= {hair_thresh:.4f}, top {int((1-args.hair_quantile)*100)}%)")
    print(f"Low-contrast subset:   {len(lowcon_idx)} images (contrast <= {contrast_thresh:.2f}, bottom {int(args.contrast_quantile*100)}%)")
    print()

    for spec in args.results:
        path, label = spec.split(":", 1)
        with open(path) as f:
            per_sample = json.load(f)

        print(f"=== {label} ===")
        for name, idx in [("Full dataset", all_idx), ("Hair-heavy subset", hair_idx), ("Low-contrast subset", lowcon_idx)]:
            d_mean, d_std, n = subset_stats(per_sample, idx, "dice")
            i_mean, i_std, _ = subset_stats(per_sample, idx, "iou")
            h_mean, h_std, _ = subset_stats(per_sample, idx, "hd95")
            print(f"  {name:20s} (n={n:3d})  Dice={d_mean:6.2f}±{d_std:5.2f}  IoU={i_mean:6.2f}±{i_std:5.2f}  HD95={h_mean:6.2f}±{h_std:5.2f}")
        print()


if __name__ == "__main__":
    main()
