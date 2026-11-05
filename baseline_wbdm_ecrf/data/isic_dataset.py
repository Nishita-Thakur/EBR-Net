"""
isic_dataset.py — loader for ISIC 2016/2017/2018.

Expected directory layout (paths come from configs/base.yaml):
    train_images/*.jpg (or .png)
    train_masks/*.png   (same stem as the matching image)

Dataset sizes per the paper (Sec 4.1):
    ISIC2016: 900 train / 379 test
    ISIC2017: 2000 train / 150 test
    ISIC2018: 2500 train / 1000 test
"""
import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


def _list_images(folder):
    folder = Path(folder)
    files = [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return files


def _find_matching_mask(image_path: Path, mask_folder: Path):
    """ISIC masks are usually named '<image_stem>_segmentation.png' or '<image_stem>.png'."""
    stem = image_path.stem
    candidates = [
        mask_folder / f"{stem}_segmentation.png",
        mask_folder / f"{stem}.png",
        mask_folder / f"{stem}_Segmentation.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    # fall back: any file starting with stem
    matches = list(mask_folder.glob(f"{stem}*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"No mask found for image {image_path.name} in {mask_folder}")


class ISICDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, boundary_fn=None):
        """
        Args:
            images_dir: directory of dermoscopic images
            masks_dir: directory of binary ground-truth masks
            transform: a PairedCompose (see data/transforms.py)
            boundary_fn: optional callable(mask_tensor) -> boundary_tensor
                         (see data/boundary_labels.py::mask_to_boundary).
                         Only needed when training the adaptive model's
                         boundary head.
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_paths = _list_images(self.images_dir)
        self.transform = transform
        self.boundary_fn = boundary_fn

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = _find_matching_mask(image_path, self.masks_dir)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        sample = {"image": image, "mask": mask, "id": image_path.stem}

        if self.boundary_fn is not None:
            sample["boundary"] = self.boundary_fn(mask)

        return sample