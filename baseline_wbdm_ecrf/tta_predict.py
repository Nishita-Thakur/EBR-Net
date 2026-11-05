"""
tta_predict.py -- Test-Time Augmentation for the already-trained baseline
checkpoint. No retraining. Unlike every adaptive component tried so far,
this doesn't add any new learned parameters that need to converge on a
small dataset -- it only recombines outputs from a model that already
works, using well-established geometric + preprocessing views.

For each test image, run the model on several transformed views (original,
horizontal flip, vertical flip, 180-degree rotation, and -- since we
already have working hair-removal/CLAHE code -- hair-removed and
CLAHE-enhanced versions of the raw image), undo each geometric transform
on the resulting predicted probability map, then average all views before
thresholding. Geometric flips/rotations directly match this model's own
training augmentations (configs/base.yaml -> data.augmentation), so the
model has already seen this family of transformation during training --
TTA with matched augmentations is a standard, low-risk technique (used
by default in nnU-Net and most serious medical segmentation pipelines).

Usage:
    python tta_predict.py \
        --checkpoint checkpoints/wbdm_ecrf_baseline/best.pt \
        --model_config configs/model_baseline.yaml \
        --base_config configs/base.yaml --dataset isic2017 \
        --views flip_h flip_v rot180 hair_removal clahe \
        --per_sample_json results/baseline_tta_2017.json
"""
import argparse
import json

import numpy as np
import torch
import yaml

from data import ISICDataset, get_eval_transforms
from data.transforms import remove_hair, apply_clahe
from models import build_model_from_config
from evaluation.metrics import (
    dice_score, iou_score, accuracy_score, sensitivity_score, specificity_score,
    hausdorff_distance_95, average_symmetric_surface_distance,
)


def apply_view(image_tensor, view: str):
    """image_tensor: (C,H,W) float tensor in [0,1]. Returns the transformed
    tensor and a function that undoes the SAME transform on a (1,H,W)
    predicted probability map."""
    if view == "identity":
        return image_tensor, lambda x: x
    if view == "flip_h":
        return torch.flip(image_tensor, dims=[-1]), lambda x: torch.flip(x, dims=[-1])
    if view == "flip_v":
        return torch.flip(image_tensor, dims=[-2]), lambda x: torch.flip(x, dims=[-2])
    if view == "rot180":
        return torch.rot90(image_tensor, k=2, dims=[-2, -1]), lambda x: torch.rot90(x, k=2, dims=[-2, -1])
    if view in ("hair_removal", "clahe"):
        # operate in PIL space, then convert back
        from PIL import Image
        img_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        if view == "hair_removal":
            pil_img = remove_hair(pil_img)
        else:
            pil_img = apply_clahe(pil_img)
        pil_img = pil_img.resize((image_tensor.shape[-1], image_tensor.shape[-2]))
        out = torch.from_numpy(np.array(pil_img).astype(np.float32) / 255.0).permute(2, 0, 1)
        return out, lambda x: x
    raise ValueError(f"unknown view: {view}")


@torch.no_grad()
def tta_predict(model, image_tensor, views, device):
    """image_tensor: (C,H,W) on CPU. Returns averaged probability map (1,H,W) on CPU."""
    probs = []
    for view in views:
        view_img, undo = apply_view(image_tensor, view)
        view_img = view_img.unsqueeze(0).to(device)
        out = model.predict(view_img)
        prob = out["refined_mask"][0].cpu()  # (1,H,W)
        prob = undo(prob)
        probs.append(prob)
    return torch.stack(probs, dim=0).mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_baseline.yaml")
    parser.add_argument("--dataset", default="isic2017")
    parser.add_argument("--views", nargs="+", default=["identity", "flip_h", "flip_v", "rot180"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--per_sample_json", default=None)
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    cfg = dict(base_cfg)
    cfg["model"] = model_cfg
    device = base_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

    ds_cfg = cfg["data"]["datasets"][args.dataset]
    eval_tf = get_eval_transforms(image_size=cfg["data"]["image_size"])
    dataset = ISICDataset(ds_cfg["test_images"], ds_cfg["test_masks"], transform=eval_tf)

    model = build_model_from_config(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    if "identity" not in args.views:
        args.views = ["identity"] + args.views
    print(f"TTA views: {args.views}  ({len(args.views)} forward passes per image)")

    per_sample = {k: [] for k in ["dice", "iou", "acc", "se", "sp", "hd95", "assd"]}
    for i in range(len(dataset)):
        sample = dataset[i]
        image = sample["image"]  # (C,H,W) CPU
        gt = sample["mask"].unsqueeze(0).to(device)  # (1,1,H,W)

        avg_prob = tta_predict(model, image, args.views, device).unsqueeze(0).to(device)
        pred = (avg_prob > args.threshold).float()

        per_sample["dice"].append(dice_score(pred, gt).item())
        per_sample["iou"].append(iou_score(pred, gt).item())
        per_sample["acc"].append(accuracy_score(pred, gt).item())
        per_sample["se"].append(sensitivity_score(pred, gt).item())
        per_sample["sp"].append(specificity_score(pred, gt).item())

        pred_np = (pred > 0.5).cpu().numpy().astype(bool)[0, 0]
        gt_np = (gt > 0.5).cpu().numpy().astype(bool)[0, 0]
        per_sample["hd95"].append(hausdorff_distance_95(pred_np, gt_np))
        per_sample["assd"].append(average_symmetric_surface_distance(pred_np, gt_np))

        if (i + 1) % 50 == 0 or (i + 1) == len(dataset):
            print(f"  {i+1}/{len(dataset)}")

    summary = {}
    for k, vals in per_sample.items():
        arr = np.array(vals, dtype=np.float64)
        arr = arr[~np.isnan(arr)]
        scale = 100.0 if k in ("dice", "iou", "acc", "se", "sp") else 1.0
        summary[k] = {"mean": float(arr.mean()) * scale, "std": float(arr.std()) * scale}
    print(json.dumps(summary, indent=2))

    if args.per_sample_json:
        with open(args.per_sample_json, "w") as f:
            json.dump(per_sample, f, indent=2)
        print(f"Saved per-sample scores to {args.per_sample_json}")


if __name__ == "__main__":
    main()
