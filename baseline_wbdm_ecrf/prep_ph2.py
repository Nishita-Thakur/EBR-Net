"""Lay out the PH2 dataset (200 images) for zero-shot evaluation.
usage: python prep_ph2.py RAW_DIR OUT_DIR      (e.g. data_root/PH2_raw data_root/PH2)
RAW_DIR is the unzipped PH2 download, which holds one folder per case (IMD002, ...)."""
import glob, os, sys
import cv2, numpy as np

raw, out = sys.argv[1], sys.argv[2]
os.makedirs(f"{out}/images", exist_ok=True); os.makedirs(f"{out}/masks", exist_ok=True)
n = 0
for d in sorted(glob.glob(f"{raw}/**/*_Dermoscopic_Image", recursive=True)):
    case = os.path.basename(d)[:-len("_Dermoscopic_Image")]
    ims = glob.glob(f"{d}/*.bmp") + glob.glob(f"{d}/*.BMP")
    mks = glob.glob(f"{os.path.dirname(d)}/{case}_lesion/*.bmp") + glob.glob(f"{os.path.dirname(d)}/{case}_lesion/*.BMP")
    if not ims or not mks:
        print("skipping", case); continue
    cv2.imwrite(f"{out}/images/{case}.bmp", cv2.imread(ims[0], cv2.IMREAD_COLOR))
    cv2.imwrite(f"{out}/masks/{case}.png", ((cv2.imread(mks[0], cv2.IMREAD_GRAYSCALE) > 0) * 255).astype(np.uint8))
    n += 1
print("PH2 cases written:", n, "(expected 200)")
