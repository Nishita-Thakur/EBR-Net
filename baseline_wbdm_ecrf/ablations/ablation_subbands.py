"""
ablation_subbands.py — reproduces Table 6: LL-only (our strategy, 3
channels) vs. All-Subbands (naive concatenation of LL+LH+HL+HH across
RGB, 12 channels, no gating). The paper found the naive variant
catastrophically destabilizes training (Dice 94.45% -> 40.23%); this
script trains both variants and reports the same collapse should
reproduce given the AdaptiveDWT gate is disabled for the comparison arm.

Usage:
    python ablations/ablation_subbands.py --dataset isic2016
"""
import argparse
import copy
import json

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data import ISICDataset, get_train_transforms, get_eval_transforms
from models import build_model_from_config
from models.dwt.static_dwt import StaticDWT
from training.trainer import Trainer
from evaluation.evaluate import evaluate


class RawConcatDWT(nn.Module):
    """Comparison arm for Table 6: concatenates all 4 subbands x 3 color
    channels with NO gating/weighting -- the naive baseline the paper
    warns against."""

    def __init__(self, channels: int = 3):
        super().__init__()
        self.dwt = StaticDWT(channels=channels)

    def forward(self, x):
        subbands = self.dwt(x)
        fused = torch.cat([subbands["LL"], subbands["LH"], subbands["HL"], subbands["HH"]], dim=1)
        return fused, {}

    def extract_ll(self, x):
        return self.dwt.extract_ll(x)


def build_llonly_model(cfg):
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["dwt"] = {"type": "static", "wavelet": "haar", "keep_subbands": ["LL"]}
    local_cfg = dict(cfg)
    local_cfg["model"] = model_cfg
    return build_model_from_config(local_cfg)


def build_allsubbands_model(cfg):
    """Builds the adaptive model, then swaps in a 12-channel raw-concat
    DWT and widens the U-Net's first conv to accept 12 input channels."""
    model_cfg = copy.deepcopy(cfg["model"])
    model_cfg["bdm"]["unet"]["in_channels"] = 12
    local_cfg = dict(cfg)
    local_cfg["model"] = model_cfg
    model = build_model_from_config(local_cfg)
    model.dwt = RawConcatDWT(channels=cfg["data"]["channels"])
    model.dwt_type = "static"  # no gate-regularization path
    return model


def train_and_eval(model, cfg, dataset, device, exp_name):
    ds_cfg = cfg["data"]["datasets"][dataset]
    image_size = cfg["data"]["image_size"]
    train_tf = get_train_transforms(image_size=image_size)
    eval_tf = get_eval_transforms(image_size=image_size)

    train_set = ISICDataset(ds_cfg["train_images"], ds_cfg["train_masks"], transform=train_tf)
    val_set = ISICDataset(ds_cfg["test_images"], ds_cfg["test_masks"], transform=eval_tf)
    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["training"].get("num_workers", 4))
    val_loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"].get("num_workers", 4))

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_weights_cfg=cfg["model"].get("loss_weights", {}),
        learning_rate=cfg["training"].get("learning_rate", 1e-4),
        epochs=cfg["training"].get("epochs", 200),
        checkpoint_dir=cfg["training"].get("checkpoint_dir", "checkpoints/"),
        log_dir=cfg["training"].get("log_dir", "logs/"),
        device=device, experiment_name=exp_name,
    )
    trainer.fit()
    summary, _ = evaluate(model, val_loader, device)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--dataset", default="isic2016")
    parser.add_argument("--out_json", default="results/ablation_subbands.json")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    cfg = dict(base_cfg)
    cfg["model"] = model_cfg
    device = base_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    results = {}

    print("=== LL-only (our strategy, 3 channels) ===")
    ll_model = build_llonly_model(cfg)
    results["LL_only"] = train_and_eval(ll_model, cfg, args.dataset, device, f"ablation_llonly_{args.dataset}")

    print("=== All-Subbands (naive concat, 12 channels) ===")
    all_model = build_allsubbands_model(cfg)
    results["All_Subbands"] = train_and_eval(all_model, cfg, args.dataset, device, f"ablation_allsubbands_{args.dataset}")

    print(json.dumps(results, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
