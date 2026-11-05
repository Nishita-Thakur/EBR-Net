"""
make_fig_cmp_all.py  --  Fig. 4 of the EBR-Net paper (all trained models on ISIC 2017).
Rows = the test images with the highest Dice of the proposed network, restricted to images on which
it is the best (or tied-best at 3 decimals) of all six models. Fixed rule, no manual picking.

Run from the sls_v2 folder:
    cd /workspace/sls_v2
    python make_fig_cmp_all.py

Reuses your own code (data.py, model.py, infer.py, zoo_models.py), so predictions are made
exactly as in your evaluation: 4-view flip TTA, the checkpoint's own validation-selected
threshold and post-processing, 256x256 evaluation.  Seed 0 of every model.
Output: fig_cmp_all.png   (copy it next to main.tex)
"""
import argparse, os, sys
import cv2, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ISIC, list_pairs
from model import SegNet
from infer import predict_probs, postprocess
from zoo_models import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--test_images", default="/workspace/SLS-CODE/data_root/ISIC2017/test/images")
ap.add_argument("--test_masks",  default="/workspace/SLS-CODE/data_root/ISIC2017/test/masks")
ap.add_argument("--baseline_dir", default="baseline_preds_isic2017_s0")
ap.add_argument("--proposed_ckpt", default="runs/isic2017/best.pt")
ap.add_argument("--zoo_dir", default="/workspace/results_compare/ckpt")
ap.add_argument("--n", type=int, default=6, help="number of images (rows)")
ap.add_argument("--min_other", type=float, default=0.85,
                help="every other model must exceed this Dice on a shown image")
ap.add_argument("--min_prop", type=float, default=0.90,
                help="EBR-Net must reach at least this Dice on a shown image")
ap.add_argument("--eval_size", type=int, default=256)
ap.add_argument("--out", default="fig_cmp_all_v2.png")
a = ap.parse_args()

E = a.eval_size
dev = "cuda" if torch.cuda.is_available() else "cpu"
pairs = list_pairs(a.test_images, a.test_masks)


def dice(p, g, eps=1e-6):
    tp = (p * g).sum()
    return (2 * tp + eps) / (p.sum() + g.sum() + eps)


def run(ck, model):
    """Predict the test set like compare_figure.py / evaluate_zoo.py do."""
    ds = ISIC(pairs, ck["size"], mask_size=E, train=False)
    P, G, ids = predict_probs(model, DataLoader(ds, 16, num_workers=2), dev, tta=True, eval_size=E)
    thr, post = ck.get("thr", 0.5), ck.get("post", "none")
    masks = [postprocess((p > thr).astype(np.uint8), post).astype(np.uint8) for p in P]
    gts = [(g > 0.5).astype(np.uint8) for g in G]
    return masks, gts, ids


# ---------------------------------------------------------------- the models
names = ["WBDM-ECRF (baseline)", "U-Net", "U-Net++", "Attention U-Net", "FAT-Net", "Proposed (EBR-Net)"]
preds, ref_ids, GT = {}, None, None

# proposed network (exactly as in compare_figure.py)
ck = torch.load(a.proposed_ckpt, map_location=dev, weights_only=False)
m = SegNet(ck["encoder"], pretrained=False, use_refiner=ck.get("use_refiner", True),
           use_aux=ck.get("use_aux", True)).to(dev)
m.load_state_dict(ck["model"])
preds["Proposed (EBR-Net)"], GT, ref_ids = run(ck, m)
print("proposed done", len(ref_ids))

# comparison networks (exactly as in evaluate_zoo.py)
for nm, key in [("U-Net", "unet"), ("U-Net++", "unetpp"), ("Attention U-Net", "attunet"), ("FAT-Net", "fatnet")]:
    ck = torch.load(os.path.join(a.zoo_dir, f"{key}_ISIC2017_s0", "best.pt"), map_location=dev, weights_only=False)
    m = build_model(ck["arch"], pretrained=False).to(dev)
    m.load_state_dict(ck["model"])
    preds[nm], g, ids = run(ck, m)
    assert list(ids) == list(ref_ids), f"{nm}: image order differs"
    print(nm, "done")

# baseline: saved masks
base = []
for i in ref_ids:
    f = os.path.join(a.baseline_dir, i + ".png")
    if not os.path.exists(f):
        raise FileNotFoundError(f)
    b = cv2.imread(f, 0)
    if b.shape != (E, E):
        b = cv2.resize(b, (E, E), interpolation=cv2.INTER_NEAREST)
    base.append((b > 127).astype(np.uint8))
preds["WBDM-ECRF (baseline)"] = base

D = {n: np.array([dice(p, g) for p, g in zip(preds[n], GT)]) for n in names}
for n in names:
    print(f"{n:24s} mean Dice {D[n].mean() * 100:.2f}")
print("(compare with the first-seed numbers in your paper tables)")

# ------------------------------------------------- fixed selection rule
PROP = "Proposed (EBR-Net)"
others = [n for n in names if n != PROP]
runner_up = np.max(np.stack([D[n] for n in others]), 0)       # best of the other five, per image
strict = int((D[PROP] > runner_up).sum())
worst_other = np.min(np.stack([D[n] for n in others]), 0)     # weakest of the other five, per image
ok = np.where((D[PROP] > runner_up) & (worst_other > a.min_other) & (D[PROP] >= a.min_prop))[0]   # EBR-Net strictly best, all others > min_other, EBR-Net >= min_prop
if len(ok) == 0:
    raise SystemExit(f"No image meets: EBR-Net strictly best, all others > {a.min_other}, EBR-Net >= {a.min_prop}. "
                     f"Lower --min_other (e.g. 0.82) and state the new threshold in the paper.")
margin = (D[PROP] - runner_up)[ok]
order = ok[np.argsort(-margin)]                                 # then: largest margin over the runner-up first
sel = list(order[:a.n])
print(f"\nProposed network has the highest Dice of all six models on {strict} of {len(D[PROP])} images")
print(f"Of these, {len(ok)} also have every other model at Dice > {a.min_other} and EBR-Net at Dice >= {a.min_prop}  (candidate pool for the figure)")
if len(ok) < a.n:
    print(f"WARNING: only {len(ok)} images qualify, so the figure will have {len(ok)} rows, not {a.n}.")
for k in sel:
    print(ref_ids[k], {n: round(float(D[n][k]), 3) for n in names})


def err(pred, gt):
    m = np.zeros((E, E, 3), np.uint8)
    m[(pred == 1) & (gt == 1)] = (255, 255, 255)
    m[(pred == 1) & (gt == 0)] = (255, 0, 0)
    m[(pred == 0) & (gt == 1)] = (0, 0, 255)
    return m


cols = ["Image", "Ground truth"] + names
nc, nr = len(cols), len(sel)
fig = plt.figure(figsize=(2.0 * nc, 2.0 * nr + 0.6))
gs = fig.add_gridspec(nr, nc, wspace=0.03, hspace=0.04, left=0.03, right=0.995, top=0.955, bottom=0.005)
for r, k in enumerate(sel):
    img = cv2.cvtColor(cv2.imread(str(pairs[k][0])), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (E, E), interpolation=cv2.INTER_AREA)
    tiles = [img, np.stack([GT[k] * 255] * 3, 2)] + [err(preds[n][k], GT[k]) for n in names]
    for c, im in enumerate(tiles):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(im, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if c >= 2:
            ax.text(0.03, 0.03, f"Dice {D[names[c - 2]][k]:.3f}", transform=ax.transAxes, color="yellow",
                    fontsize=9, fontweight="bold", va="bottom", ha="left",
                    bbox=dict(facecolor="black", alpha=0.45, pad=1.2, lw=0))
        if c == 0:
            ax.set_ylabel(ref_ids[k].replace("ISIC_", ""), fontsize=9)
        if r == 0:
            ax.set_title(cols[c], fontsize=10.5, pad=4)
fig.savefig(a.out, dpi=200)
print("saved", a.out)
print("legend: white = correct, red = over-segmentation, blue = under-segmentation")
