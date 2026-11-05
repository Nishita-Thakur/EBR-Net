"""infer.py - TTA prediction, post-processing, per-image metrics (shared by train + evaluate)."""
import cv2, numpy as np, torch, torch.nn.functional as F
from scipy.ndimage import binary_fill_holes, label


@torch.no_grad()
def predict_probs(model, loader, device, tta=True, eval_size=None):
    model.eval(); P, G, ids = [], [], []
    for b in loader:
        x = b["image"].to(device)
        views = [(lambda t: t, lambda t: t)]
        if tta:
            views += [(lambda t: t.flip(-1), lambda t: t.flip(-1)), (lambda t: t.flip(-2), lambda t: t.flip(-2)),
                      (lambda t: t.flip(-1).flip(-2), lambda t: t.flip(-1).flip(-2))]
        p = 0
        for f, inv in views:
            with torch.autocast(device_type=device.split(":")[0], enabled=device.startswith("cuda")):
                lg = model(f(x))["logits"]
            p = p + inv(torch.sigmoid(lg.float()))
        p = p / len(views)
        g = b["mask"]
        if eval_size and p.shape[-1] != eval_size:
            p = F.interpolate(p, size=(eval_size, eval_size), mode="bilinear", align_corners=False)
        if g.shape[-1] != p.shape[-1]:
            g = F.interpolate(g, size=p.shape[-2:], mode="nearest")
        P.append(p.cpu()); G.append(g); ids += list(b["id"])
    return torch.cat(P)[:, 0].numpy(), torch.cat(G)[:, 0].numpy(), ids


def postprocess(binm, mode):
    if mode == "none" or binm.sum() == 0:
        return binm
    lab, n = label(binm)
    if n > 1:  # keep components >= 20% of the largest one (handles rare multi-lesion images)
        sizes = np.bincount(lab.ravel())[1:]
        keep = [i + 1 for i, s in enumerate(sizes) if s >= 0.2 * sizes.max()]
        binm = np.isin(lab, keep)
    return binary_fill_holes(binm).astype(np.uint8)


def per_image_metrics(pred, gt, eps=1e-6):
    tp = (pred * gt).sum(); fp = (pred * (1 - gt)).sum(); fn = ((1 - pred) * gt).sum(); tn = ((1 - pred) * (1 - gt)).sum()
    return dict(dice=(2 * tp + eps) / (2 * tp + fp + fn + eps), iou=(tp + eps) / (tp + fp + fn + eps),
                se=(tp + eps) / (tp + fn + eps), sp=(tn + eps) / (tn + fp + eps),
                acc=(tp + tn + eps) / (tp + tn + fp + fn + eps))


def score(P, G, thr, post):
    rows = []
    for p, g in zip(P, G):
        rows.append(per_image_metrics(postprocess((p > thr).astype(np.uint8), post), (g > 0.5).astype(np.uint8)))
    return rows


def tune_on_val(P, G):
    """Pick threshold + post-processing on the VALIDATION split only (never on test)."""
    best = (-1, 0.5, "none")
    for post in ("none", "cc_fill"):
        for thr in (0.35, 0.4, 0.45, 0.5, 0.55, 0.6):
            d = np.mean([r["dice"] for r in score(P, G, thr, post)])
            if d > best[0]: best = (d, thr, post)
    return {"val_dice": float(best[0]), "thr": best[1], "post": best[2]}
