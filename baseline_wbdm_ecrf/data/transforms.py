"""
transforms.py — paired image/mask augmentation.

Paper (Sec 4.2) uses: resize to 256x256, horizontal/vertical flip,
random rotation. We keep image and mask in sync by applying the
same random state to both.

Also provides optional deterministic preprocessing targeting the two
failure modes the baseline paper documents (Sec 4.7, Fig. 8): hair
artifacts and low-contrast lesion boundaries. Both are applied to the
image only (never the mask), and MUST be applied identically at train
and eval time (get_train_transforms / get_eval_transforms both take the
same hair_removal / clahe flags) -- inconsistent preprocessing between
train and test is a distribution-shift bug, not a free accuracy gain.
"""
import random
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF


def _cap_resolution(pil_image: Image.Image, max_dim: int = 768) -> Image.Image:
    """Downscale so the longer side is at most max_dim, preserving aspect
    ratio, before any hair-removal/CLAHE processing. Real ISIC images can
    be several thousand pixels per side, and cv2.inpaint's runtime scales
    badly with both image size and the size of the masked region -- an
    unbounded resolution risks a single dense-hair or large image taking
    minutes to hours (this is what caused a real hang during preprocessing
    of the full ISIC2017 training set). Since training resizes to a much
    smaller image_size anyway (e.g. 256), this loses no information that
    would have survived to the network regardless.
    """
    w, h = pil_image.size
    longest = max(w, h)
    if longest <= max_dim:
        return pil_image
    scale = max_dim / longest
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return pil_image.resize(new_size, Image.BILINEAR)


def remove_hair(pil_image: Image.Image, kernel_size: int = 9, inpaint_radius: int = 3, threshold: int = 10, max_dim: int = 768) -> Image.Image:
    """DullRazor-style hair removal: morphological black-hat on the grayscale
    image highlights thin dark structures (hair) against lighter skin, then
    those pixels are removed via inpainting rather than left for the network
    to (unreliably) learn to ignore. Resolution is capped first (see
    _cap_resolution) to bound worst-case runtime.
    """
    pil_image = _cap_resolution(pil_image, max_dim)
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    hair_mask = cv2.dilate(hair_mask, np.ones((3, 3), np.uint8), iterations=1)
    inpainted = cv2.inpaint(img, hair_mask, inpaint_radius, cv2.INPAINT_TELEA)
    return Image.fromarray(inpainted)


def apply_clahe(pil_image: Image.Image, clip_limit: float = 2.5, tile_grid_size: int = 8, max_dim: int = 768) -> Image.Image:
    """Contrast Limited Adaptive Histogram Equalization on the L channel of
    LAB color space (so color/hue is untouched, only local lightness
    contrast is boosted) -- standard preprocessing for exactly the
    low-contrast, faint-boundary case the baseline paper documents as a
    failure mode (Sec 4.7, bottom two rows of Fig. 8). Resolution capped
    first for consistency/speed (see _cap_resolution).
    """
    pil_image = _cap_resolution(pil_image, max_dim)
    img = np.array(pil_image.convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return Image.fromarray(rgb_eq)


class PairedHairRemoval:
    def __init__(self, kernel_size=9, inpaint_radius=3, threshold=10):
        self.kernel_size = kernel_size
        self.inpaint_radius = inpaint_radius
        self.threshold = threshold

    def __call__(self, image, mask):
        image = remove_hair(image, self.kernel_size, self.inpaint_radius, self.threshold)
        return image, mask


class PairedCLAHE:
    def __init__(self, clip_limit=2.5, tile_grid_size=8):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image, mask):
        image = apply_clahe(image, self.clip_limit, self.tile_grid_size)
        return image, mask


class PairedCompose:
    """Applies a list of paired (image, mask) transforms in sequence."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image: Image.Image, mask: Image.Image):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class PairedResize:
    def __init__(self, size):
        self.size = (size, size) if isinstance(size, int) else size

    def __call__(self, image, mask):
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class PairedRandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        return image, mask


class PairedRandomVFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):
        if random.random() < self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        return image, mask


class PairedRandomRotation:
    def __init__(self, degrees=15):
        self.degrees = degrees

    def __call__(self, image, mask):
        angle = random.uniform(-self.degrees, self.degrees)
        image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
        return image, mask


class PairedToTensor:
    """Image -> float tensor in [0,1], CxHxW. Mask -> float tensor in {0,1}, 1xHxW."""

    def __call__(self, image, mask):
        image = TF.to_tensor(image)  # C,H,W in [0,1]
        mask_np = np.array(mask, dtype=np.float32)
        if mask_np.max() > 1.0:
            mask_np = mask_np / 255.0
        mask_np = (mask_np > 0.5).astype(np.float32)
        mask = TF.to_tensor(mask_np)
        return image, mask


def get_train_transforms(image_size=256, hflip=True, vflip=True, rotation_degrees=15, hair_removal=False, clahe=False):
    tfms = []
    if hair_removal:
        tfms.append(PairedHairRemoval())
    if clahe:
        tfms.append(PairedCLAHE())
    tfms.append(PairedResize(image_size))
    if hflip:
        tfms.append(PairedRandomHFlip(0.5))
    if vflip:
        tfms.append(PairedRandomVFlip(0.5))
    if rotation_degrees and rotation_degrees > 0:
        tfms.append(PairedRandomRotation(rotation_degrees))
    tfms.append(PairedToTensor())
    return PairedCompose(tfms)


def get_eval_transforms(image_size=256, hair_removal=False, clahe=False):
    tfms = []
    if hair_removal:
        tfms.append(PairedHairRemoval())
    if clahe:
        tfms.append(PairedCLAHE())
    tfms.append(PairedResize(image_size))
    tfms.append(PairedToTensor())
    return PairedCompose(tfms)
