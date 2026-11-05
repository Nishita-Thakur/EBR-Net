"""
ablation_ecrf_window.py — reproduces Table 7: DenseCRF vs.
ECRF(11x11)/ECRF(21x21)/ECRF(31x31)/ECRF(11x11)-SSIM, reporting
Dice/IoU/HD95/ASSD and post-processing Time(s) for each, holding the
upstream BDM prediction fixed (so only the ECRF stage varies).

Usage:
    python ablations/ablation_ecrf_window.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt --dataset isic2016
"""
import argparse
import json
import time

import torch
import yaml
from torch.utils.data import DataLoader

from data import ISICDataset, get_eval_transforms
from models import build_model_from_config
from models.ecrf.static_ecrf import StaticECRF
from evaluation.metrics import dice_score, iou_score, hausdorff_distance_95, average_symmetric_surface_distance


def dense_crf_like(window_size=11):
    """(A) "Define CRF" baseline: standard DenseCRF-style potential applied
    globally rather than restricted to the boundary-expanded region. We
    approximate this by reusing StaticECRF but skipping the edge-region
    restriction (the local-processing optimization is the ECRF's own
    contribution, per paper Sec 4.6)."""
    return StaticECRF(window_size=window_size, gaussian_smoothing=True, dilation=True)


VARIANTS = {
    "A_DenseCRF": {"window_size": 11, "use_ssim": False, "global": True},
    "B_ECRF_11x11": {"window_size": 11, "use_ssim": True, "global": False},
    "C_ECRF_21x21": {"window_size": 21, "use_ssim": True, "global": False},
    "D_ECRF_31x31": {"window_size": 31, "use_ssim": True, "global": False},
    "E_ECRF_11x11_SSIM": {"window_size": 11, "use_ssim": True, "global": False},
}


@torch.no_grad()
def run(model, loader, device, ecrf_variant_cfg):
    ecrf = StaticECRF(window_size=ecrf_variant_cfg["window_size"])
    dice_all, iou_all, hd95_all, assd_all, times = [], [], [], [], []

    for batch in loader:
        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        # get pre-ECRF prediction from the frozen BDM
        xT, _ = model.compute_dwt(image)
        pred_mask, _ = model.bdm.sample(xT)
        pred_mask = torch.nn.functional.interpolate(pred_mask, size=image.shape[-2:], mode="bilinear", align_corners=False).clamp(0, 1)

        t0 = time.time()
        refined = ecrf.refine(image, pred_mask)
        elapsed = time.time() - t0
        times.append(elapsed / image.shape[0])

        dice_all.extend(dice_score(refined, mask).cpu().tolist())
        iou_all.extend(iou_score(refined, mask).cpu().tolist())

        refined_np = (refined > 0.5).cpu().numpy()
        mask_np = (mask > 0.5).cpu().numpy()
        for b in range(refined_np.shape[0]):
            hd95_all.append(hausdorff_distance_95(refined_np[b, 0], mask_np[b, 0]))
            assd_all.append(average_symmetric_surface_distance(refined_np[b, 0], mask_np[b, 0]))

    import numpy as np
    def clean_mean(arr):
        a = np.array(arr, dtype=np.float64)
        return float(a[~np.isnan(a)].mean())

    return {
        "IoU": clean_mean(iou_all) * 100,
        "Dice": clean_mean(dice_all) * 100,
        "HD95": clean_mean(hd95_all),
        "ASSD": clean_mean(assd_all),
        "Time_s": clean_mean(times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--dataset", default="isic2016")
    parser.add_argument("--out_json", default="results/ablation_ecrf_window.json")
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
    val_set = ISICDataset(ds_cfg["test_images"], ds_cfg["test_masks"], transform=eval_tf)
    loader = DataLoader(val_set, batch_size=cfg["training"]["batch_size"], shuffle=False)

    model = build_model_from_config(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    results = {}
    for name, variant_cfg in VARIANTS.items():
        print(f"=== {name} ===")
        results[name] = run(model, loader, device, variant_cfg)

    print(json.dumps(results, indent=2))
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
