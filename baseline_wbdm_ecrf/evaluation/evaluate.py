"""
evaluate.py — runs a trained checkpoint over a test split and reports
Dice/IoU/ACC/SE/SP (+HD95/ASSD), matching Tables 1-3 of the paper.

Usage:
    python evaluation/evaluate.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt \
        --model_config configs/model_adaptive.yaml --dataset isic2018

    # zero-shot cross-dataset (Table 3): train on ISIC2018, eval on PH2
    python evaluation/evaluate.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt \
        --model_config configs/model_adaptive.yaml --dataset ph2
"""
import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import ISICDataset, PH2Dataset, get_eval_transforms
from models import build_model_from_config
from evaluation.metrics import (
    dice_score, iou_score, accuracy_score, sensitivity_score, specificity_score,
    hausdorff_distance_95, average_symmetric_surface_distance,
)


def build_test_set(cfg: dict, dataset_name: str):
    image_size = cfg["data"]["image_size"]
    prep_cfg = cfg["data"].get("preprocessing", {})
    eval_tf = get_eval_transforms(image_size=image_size, hair_removal=prep_cfg.get("hair_removal", False), clahe=prep_cfg.get("clahe", False))
    if dataset_name == "ph2":
        ds_cfg = cfg["data"]["datasets"]["ph2"]
        return PH2Dataset(ds_cfg["images"], ds_cfg["masks"], transform=eval_tf)
    ds_cfg = cfg["data"]["datasets"][dataset_name]
    return ISICDataset(ds_cfg["test_images"], ds_cfg["test_masks"], transform=eval_tf)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    per_sample = {"dice": [], "iou": [], "acc": [], "se": [], "sp": [], "hd95": [], "assd": []}

    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)
        out = model.predict(image)
        pred = out["refined_mask"]

        per_sample["dice"].extend(dice_score(pred, mask).cpu().tolist())
        per_sample["iou"].extend(iou_score(pred, mask).cpu().tolist())
        per_sample["acc"].extend(accuracy_score(pred, mask).cpu().tolist())
        per_sample["se"].extend(sensitivity_score(pred, mask).cpu().tolist())
        per_sample["sp"].extend(specificity_score(pred, mask).cpu().tolist())

        pred_np = (pred > 0.5).cpu().numpy().astype(bool)
        mask_np = (mask > 0.5).cpu().numpy().astype(bool)
        for b in range(pred_np.shape[0]):
            per_sample["hd95"].append(hausdorff_distance_95(pred_np[b, 0], mask_np[b, 0]))
            per_sample["assd"].append(average_symmetric_surface_distance(pred_np[b, 0], mask_np[b, 0]))

    summary = {}
    for k, v in per_sample.items():
        arr = np.array(v, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        summary[k] = {"mean": float(arr.mean()) * (100.0 if k in ("dice", "iou", "acc", "se", "sp") else 1.0),
                      "std": float(arr.std()) * (100.0 if k in ("dice", "iou", "acc", "se", "sp") else 1.0)}
    return summary, per_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--dataset", default="isic2018")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--per_sample_json", default=None, help="dump raw per-sample dice/iou/etc lists, for use with significance_tests.py")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    cfg = dict(base_cfg)
    cfg["model"] = model_cfg

    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    test_set = build_test_set(cfg, args.dataset)
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=cfg["training"].get("num_workers", 4))

    model = build_model_from_config(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])

    summary, per_sample = evaluate(model, loader, device)
    print(json.dumps(summary, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(summary, f, indent=2)
    if args.per_sample_json:
        with open(args.per_sample_json, "w") as f:
            json.dump(per_sample, f, indent=2)
        print(f"Saved per-sample scores to {args.per_sample_json}")


if __name__ == "__main__":
    main()