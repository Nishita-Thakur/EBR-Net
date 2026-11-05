"""
ablation_dwt_ecrf.py — reproduces Table 5: contribution of the DWT and
ECRF blocks. Defines four variants:
    A = BDM only
    B = BDM + ECRF
    C = DWT + BDM
    D = DWT + BDM + ECRF   (the full adaptive model)

Each variant is built by toggling the relevant sections of a base model
config, trained independently (reusing training/trainer.py), and
evaluated with evaluation/evaluate.py.

Usage:
    python ablations/ablation_dwt_ecrf.py --dataset isic2018
"""
import argparse
import copy
import json

import torch
import yaml
from torch.utils.data import DataLoader

from data import ISICDataset, get_train_transforms, get_eval_transforms, mask_to_boundary
from models import build_model_from_config
from training.trainer import Trainer
from evaluation.evaluate import evaluate


VARIANTS = {
    "A_bdm_only": {"use_dwt": False, "use_ecrf": False},
    "B_bdm_ecrf": {"use_dwt": False, "use_ecrf": True},
    "C_dwt_bdm": {"use_dwt": True, "use_ecrf": False},
    "D_dwt_bdm_ecrf": {"use_dwt": True, "use_ecrf": True},
}


def make_variant_config(model_cfg: dict, use_dwt: bool, use_ecrf: bool) -> dict:
    cfg = copy.deepcopy(model_cfg)
    if not use_dwt:
        # bypass DWT: keep all subbands' worth of raw channels off, fall back
        # to identity by using a static (LL-only) extraction as the minimal
        # "no adaptive DWT" input rather than removing the block entirely.
        cfg["dwt"] = {"type": "static", "wavelet": cfg["dwt"].get("wavelet", "haar"), "keep_subbands": ["LL"]}
    if not use_ecrf:
        cfg["ecrf"] = {
            "type": "static",
            "window_size": cfg["ecrf"].get("window_choices", [11])[0] if "window_choices" in cfg["ecrf"] else cfg["ecrf"].get("window_size", 11),
            "sigma_spatial": cfg["ecrf"].get("sigma_spatial", 1.0),
            "sigma_color": cfg["ecrf"].get("sigma_color", 2.0),
            "sigma_ssim": cfg["ecrf"].get("sigma_ssim", 2.0),
            "ssim_window": cfg["ecrf"].get("ssim_window", 11),
            "prior_prob": cfg["ecrf"].get("prior_prob", 0.9),
            "postprocess": {"dilation": False, "gaussian_smoothing": False},
        }
    return cfg


def run_variant(name, variant_cfg, base_cfg, model_cfg, dataset, device):
    full_model_cfg = make_variant_config(model_cfg, **variant_cfg)
    cfg = dict(base_cfg)
    cfg["model"] = full_model_cfg
    cfg["model"]["experiment_name"] = f"ablation_{name}_{dataset}"

    boundary_needed = full_model_cfg.get("bdm", {}).get("boundary_head", {}).get("enabled", False)
    ds_cfg = cfg["data"]["datasets"][dataset]
    image_size = cfg["data"]["image_size"]
    train_tf = get_train_transforms(image_size=image_size)
    eval_tf = get_eval_transforms(image_size=image_size)
    boundary_fn = mask_to_boundary if boundary_needed else None

    train_set = ISICDataset(ds_cfg["train_images"], ds_cfg["train_masks"], transform=train_tf, boundary_fn=boundary_fn)
    val_set = ISICDataset(ds_cfg["test_images"], ds_cfg["test_masks"], transform=eval_tf, boundary_fn=boundary_fn)

    train_loader = DataLoader(train_set, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=cfg["training"].get("num_workers", 4), drop_last=True)
    val_loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=cfg["training"].get("num_workers", 4))

    model = build_model_from_config(cfg)
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_weights_cfg=full_model_cfg.get("loss_weights", {}),
        learning_rate=cfg["training"].get("learning_rate", 1e-4),
        epochs=cfg["training"].get("epochs", 200),
        checkpoint_dir=cfg["training"].get("checkpoint_dir", "checkpoints/"),
        log_dir=cfg["training"].get("log_dir", "logs/"),
        device=device, experiment_name=cfg["model"]["experiment_name"],
    )
    trainer.fit()

    summary, _ = evaluate(model, val_loader, device)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--dataset", default="isic2018")
    parser.add_argument("--out_json", default="results/ablation_dwt_ecrf.json")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    device = base_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    results = {}
    for name, variant in VARIANTS.items():
        print(f"=== Running variant {name} ({variant}) ===")
        results[name] = run_variant(name, variant, base_cfg, model_cfg, args.dataset, device)

    print(json.dumps(results, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
