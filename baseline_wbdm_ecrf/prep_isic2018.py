"""Build a local ISIC 2018 Task 1 benchmark from the PUBLIC labelled images.
The official test masks are withheld by the challenge, so the 2594 training images and the 100
validation images (both with masks) are pooled and split 80/20 with a fixed seed.

usage: python prep_isic2018.py RAW_DIR OUT_DIR      (e.g. data_root/ISIC2018_raw data_root/ISIC2018)
"""
import glob, os, random, sys
import cv2, numpy as np

raw, out = sys.argv[1], sys.argv[2]
MAX_SIDE, TEST_FRAC, SEED = 1024, 0.2, 1234
imgs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(f"{raw}/**/*.jpg", recursive=True)}
msks = {os.path.basename(p)[:-len("_segmentation.png")]: p for p in glob.glob(f"{raw}/**/*_segmentation.png", recursive=True)}
stems = sorted(set(imgs) & set(msks))
print(f"{len(imgs)} images, {len(msks)} masks, {len(stems)} labelled pairs")
assert stems, "no labelled pairs found - check RAW_DIR"
random.Random(SEED).shuffle(stems)
n_test = int(round(len(stems) * TEST_FRAC))
split = {"test": stems[:n_test], "train": stems[n_test:]}
for name, lst in split.items():
    os.makedirs(f"{out}/{name}/images", exist_ok=True); os.makedirs(f"{out}/{name}/masks", exist_ok=True)
    for s in lst:
        im = cv2.imread(imgs[s], cv2.IMREAD_COLOR); mk = cv2.imread(msks[s], cv2.IMREAD_GRAYSCALE)
        h, w = im.shape[:2]; k = MAX_SIDE / max(h, w)
        if k < 1:
            im = cv2.resize(im, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
            mk = cv2.resize(mk, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(f"{out}/{name}/images/{s}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(f"{out}/{name}/masks/{s}_segmentation.png", ((mk > 127) * 255).astype(np.uint8))
    open(f"{out}/{name}_ids.txt", "w").write("\n".join(lst))
    print(name, len(lst))
