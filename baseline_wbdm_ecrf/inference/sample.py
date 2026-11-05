"""
sample.py — run a trained checkpoint on a folder of images and save the
predicted (and ECRF-refined) masks as PNGs. This is the inference-time
counterpart to the "Bridge Diffusion Model Block-Sample" + "ECRF" stages
in paper Fig. 2.

Usage:
    python inference/sample.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt \
        --model_config configs/model_adaptive.yaml \
        --images_dir data_root/ISIC2018/test/images --out_dir results/isic2018_preds
"""
import argparse
import os

import torch
import yaml
from PIL import Image

from data.isic_dataset import _list_images
from data.transforms import get_eval_transforms
from models import build_model_from_config
from inference.postprocess import full_postprocess_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--images_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--save_raw", action="store_true", help="also save the pre-ECRF mask")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    cfg = dict(base_cfg)
    cfg["model"] = model_cfg

    device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    image_size = cfg["data"]["image_size"]

    model = build_model_from_config(cfg).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.save_raw:
        os.makedirs(os.path.join(args.out_dir, "raw"), exist_ok=True)

    eval_tf = get_eval_transforms(image_size=image_size)
    image_paths = _list_images(args.images_dir)

    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            dummy_mask = Image.new("L", img.size)  # transform expects a paired mask; unused here
            img_t, _ = eval_tf(img, dummy_mask)
            img_t = img_t.unsqueeze(0).to(device)

            out = model.predict(img_t)
            refined_png = full_postprocess_pipeline(out["refined_mask"][0])
            Image.fromarray(refined_png).save(os.path.join(args.out_dir, f"{path.stem}_mask.png"))

            if args.save_raw:
                raw_png = full_postprocess_pipeline(out["pred_mask"][0], keep_largest_only=False, fill=False)
                Image.fromarray(raw_png).save(os.path.join(args.out_dir, "raw", f"{path.stem}_raw.png"))

    print(f"Saved {len(image_paths)} predicted masks to {args.out_dir}")


if __name__ == "__main__":
    main()
