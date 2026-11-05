"""
train.py — CLI entry point.

Usage:
    python training/train.py \
        --base_config configs/base.yaml \
        --model_config configs/model_adaptive.yaml \
        --dataset isic2018

    python training/train.py --model_config configs/model_baseline.yaml --dataset isic2016 --resume checkpoints/wbdm_ecrf_baseline/last.pt

Merges base.yaml (data/training/diffusion/device, shared across all
experiments) with a model_*.yaml (architecture-specific: dwt/bdm/
uncertainty/ecrf/loss_weights), builds the dataset/model/trainer, and
runs the training loop.
"""
import argparse
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import ISICDataset, PH2Dataset, get_train_transforms, get_eval_transforms, mask_to_boundary
from models import build_model_from_config
from training.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_merged_config(base_path: str, model_path: str) -> dict:
    with open(base_path) as f:
        base_cfg = yaml.safe_load(f)
    with open(model_path) as f:
        model_cfg = yaml.safe_load(f)
    merged = dict(base_cfg)
    merged["model"] = model_cfg
    return merged


def build_datasets(cfg: dict, dataset_name: str, boundary_needed: bool):
    ds_cfg = cfg["data"]["datasets"][dataset_name]
    image_size = cfg["data"]["image_size"]
    aug_cfg = cfg["data"]["augmentation"]

    prep_cfg = cfg["data"].get("preprocessing", {})
    train_tf = get_train_transforms(
        image_size=image_size,
        hflip=aug_cfg.get("horizontal_flip", True),
        vflip=aug_cfg.get("vertical_flip", True),
        rotation_degrees=aug_cfg.get("random_rotation_degrees", 15),
        hair_removal=prep_cfg.get("hair_removal", False),
        clahe=prep_cfg.get("clahe", False),
    )
    eval_tf = get_eval_transforms(image_size=image_size, hair_removal=prep_cfg.get("hair_removal", False), clahe=prep_cfg.get("clahe", False))

    boundary_fn = mask_to_boundary if boundary_needed else None

    train_set = ISICDataset(ds_cfg["train_images"], ds_cfg["train_masks"], transform=train_tf, boundary_fn=boundary_fn)
    val_set = ISICDataset(ds_cfg["test_images"], ds_cfg["test_masks"], transform=eval_tf, boundary_fn=boundary_fn)
    return train_set, val_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--dataset", default="isic2018", choices=["isic2016", "isic2017", "isic2018"])
    parser.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    args = parser.parse_args()

    cfg = load_merged_config(args.base_config, args.model_config)
    set_seed(cfg.get("seed", 42))

    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    boundary_needed = cfg["model"].get("bdm", {}).get("boundary_head", {}).get("enabled", False)
    train_set, val_set = build_datasets(cfg, args.dataset, boundary_needed)

    train_cfg = cfg["training"]
    train_loader = DataLoader(
        train_set, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg.get("num_workers", 4), drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
    )

    model = build_model_from_config(cfg)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_weights_cfg=cfg["model"].get("loss_weights", {}),
        learning_rate=train_cfg.get("learning_rate", 1e-4),
        epochs=train_cfg.get("epochs", 200),
        checkpoint_dir=train_cfg.get("checkpoint_dir", "checkpoints/"),
        log_dir=train_cfg.get("log_dir", "logs/"),
        device=device,
        experiment_name=cfg["model"].get("experiment_name", "wbdm_ecrf"),
    )

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume) + 1

    trainer.fit(start_epoch=start_epoch)


if __name__ == "__main__":
    main()
