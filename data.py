"""data.py - ISIC loader with RAM cache + fully self-contained augmentation (cv2/numpy only).
Same folder layout as your old project: images/*.jpg, masks/<stem>[_segmentation].png"""
import random
from pathlib import Path
import cv2, numpy as np, torch
from torch.utils.data import Dataset

EXT = (".jpg", ".jpeg", ".png", ".bmp")
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def list_pairs(img_dir, mask_dir):
    img_dir, mask_dir = Path(img_dir), Path(mask_dir)
    pairs = []
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in EXT:
            continue
        m = None
        for c in (f"{p.stem}_segmentation.png", f"{p.stem}.png", f"{p.stem}_Segmentation.png"):
            if (mask_dir / c).exists():
                m = mask_dir / c
                break
        if m is None:
            hits = list(mask_dir.glob(f"{p.stem}*"))
            if not hits:
                raise FileNotFoundError(f"no mask for {p.name}")
            m = hits[0]
        pairs.append((p, m))
    return pairs


def split_pairs(pairs, val_frac, seed):
    idx = list(range(len(pairs)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(len(idx) * val_frac))
    val = [pairs[i] for i in idx[:n_val]]
    tr = [pairs[i] for i in idx[n_val:]]
    return tr, val


def _load(path_img, path_mask, size, mask_size):
    img = cv2.cvtColor(cv2.imread(str(path_img), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    m = cv2.imread(str(path_mask), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
    return img, (m > 127).astype(np.uint8)


# ----------------------------- augmentation -----------------------------
def add_hair(img, rng):
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(2, 10))):
        p = rng.uniform([0, 0], [w, h])
        ang = rng.uniform(0, 2 * np.pi)
        pts = [p.copy()]
        for _ in range(4):
            ang += rng.normal(0, 0.35)
            p = p + np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(0.08, 0.2) * w
            pts.append(p.copy())
        pts = np.array(pts, np.int32).reshape(-1, 1, 2)
        col = tuple(int(c) for c in rng.integers(10, 70, 3) * rng.uniform(0.8, 1.6))
        cv2.polylines(out, [pts], False, col, int(rng.integers(1, 3)), cv2.LINE_AA)
    return cv2.addWeighted(out, 0.85, cv2.GaussianBlur(out, (3, 3), 0), 0.15, 0)


def augment(img, m, rng):
    h, w = img.shape[:2]
    if rng.random() < 0.5: img, m = img[:, ::-1], m[:, ::-1]
    if rng.random() < 0.5: img, m = img[::-1], m[::-1]
    k = int(rng.integers(0, 4))
    img, m = np.rot90(img, k), np.rot90(m, k)
    img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
    if rng.random() < 0.8:  # affine: rotation + scale + translation
        ang = rng.uniform(-30, 30); sc = rng.uniform(0.8, 1.25)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[:, 2] += rng.uniform(-0.08, 0.08, 2) * [w, h]
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    img = img.astype(np.float32)
    if rng.random() < 0.8:  # photometric
        img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)           # contrast / brightness
        img = img * rng.uniform(0.9, 1.1, 3)                                    # colour cast (devices differ!)
        g = img.mean(2, keepdims=True); img = g + (img - g) * rng.uniform(0.7, 1.3)  # saturation
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3: img = add_hair(img, rng)
    if rng.random() < 0.2: img = cv2.GaussianBlur(img, (5, 5), rng.uniform(0.3, 1.5))
    if rng.random() < 0.2:
        img = np.clip(img + rng.normal(0, rng.uniform(2, 8), img.shape), 0, 255).astype(np.uint8)
    return img, m


def augment_basic(img, m, rng):
    """Baseline-like augmentation only: horizontal/vertical flips and a random rotation of up to 15 degrees."""
    h, w = img.shape[:2]
    if rng.random() < 0.5: img, m = img[:, ::-1], m[:, ::-1]
    if rng.random() < 0.5: img, m = img[::-1], m[::-1]
    img, m = np.ascontiguousarray(img), np.ascontiguousarray(m)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-15, 15), 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    m = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
    return img, m


def to_tensor(img):
    x = (img.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


class ISIC(Dataset):
    def __init__(self, pairs, size=384, mask_size=None, train=False, aug="full"):
        self.size, self.train, self.aug = size, train, aug
        self.mask_size = size if train else (mask_size or size)
        self.items = [_load(i, m, size, self.mask_size) for i, m in pairs]
        self.ids = [p[0].stem for p in pairs]
        self._rng = None

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        img, m = self.items[i]
        if self.train:
            if self._rng is None:  # per-worker, per-epoch seed (workers are re-spawned each epoch)
                self._rng = np.random.default_rng(torch.initial_seed() % (2 ** 32))
            img, m = (augment_basic if self.aug == "basic" else augment)(img, m, self._rng)
        return {"image": to_tensor(img), "mask": torch.from_numpy(m.astype(np.float32))[None], "id": self.ids[i]}
