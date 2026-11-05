"""
ph2_dataset.py — PH2 dataset loader, used ONLY for zero-shot cross-dataset
evaluation (train on ISIC2018, evaluate here with no fine-tuning),
reproducing the paper's Table 3.
"""
from pathlib import Path
from .isic_dataset import ISICDataset, _list_images, _find_matching_mask
from PIL import Image


class PH2Dataset(ISICDataset):
    """Identical loading logic to ISICDataset; kept as a separate class
    for clarity in configs/scripts, and to make it obvious this dataset
    must never be used for training/fine-tuning."""

    def __init__(self, images_dir, masks_dir, transform=None, boundary_fn=None):
        super().__init__(images_dir, masks_dir, transform=transform, boundary_fn=boundary_fn)