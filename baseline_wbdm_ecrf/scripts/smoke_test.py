"""
smoke_test.py — proves the pipeline runs end-to-end WITHOUT any real
dataset. It fabricates a tiny synthetic "lesion" dataset (random blobs
on noisy backgrounds + their binary masks), runs a couple of training
steps through the real WBDMECRFAdaptive model, saves a checkpoint, then
runs inference and writes a predicted mask PNG you can open and look at.

Runs on CPU in a minute or two. No internet access needed after
`pip install -r requirements.txt`.

Usage (from the SLS-CODE/ directory):
    python scripts/smoke_test.py
    python scripts/smoke_test.py --model_config configs/model_baseline.yaml
"""
import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import ISICDataset, get_train_transforms, get_eval_transforms, mask_to_boundary
from models import build_model_from_config
from training.trainer import Trainer


def make_synthetic_dataset(root: str, split: str, n: int, size: int = 64, seed: int = 0):
    rng = np.random.RandomState(seed)
    img_dir = os.path.join(root, split, "images")
    mask_dir = os.path.join(root, split, "masks")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(n):
        # noisy "skin" background
        bg = rng.randint(140, 200, size=(size, size, 3), dtype=np.uint8)
        noise = rng.randint(-20, 20, size=(size, size, 3))
        img_arr = np.clip(bg.astype(int) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, mode="RGB")

        mask = Image.new("L", (size, size), 0)
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)

        cx, cy = rng.randint(size // 4, 3 * size // 4, size=2)
        r = rng.randint(size // 8, size // 4)
        darker = tuple(int(c * 0.4) for c in rng.randint(80, 140, size=3))
        bbox = [cx - r, cy - r, cx + r, cy + r]
        draw_img.ellipse(bbox, fill=darker)
        draw_mask.ellipse(bbox, fill=255)

        img.save(os.path.join(img_dir, f"synthetic_{i:03d}.png"))
        mask.save(os.path.join(mask_dir, f"synthetic_{i:03d}_segmentation.png"))

    return img_dir, mask_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--n_train", type=int, default=8)
    parser.add_argument("--n_test", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--keep_tmp", action="store_true", help="don't delete the synthetic data / checkpoints afterward")
    args = parser.parse_args()

    with open(args.base_config) as f:
        base_cfg = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    tmp_root = tempfile.mkdtemp(prefix="wbdm_smoke_")
    print(f"Synthetic data + outputs going to: {tmp_root}")

    train_img_dir, train_mask_dir = make_synthetic_dataset(tmp_root, "train", args.n_train, size=args.image_size, seed=0)
    test_img_dir, test_mask_dir = make_synthetic_dataset(tmp_root, "test", args.n_test, size=args.image_size, seed=1)

    cfg = dict(base_cfg)
    cfg["model"] = model_cfg
    cfg["data"] = dict(base_cfg["data"])
    cfg["data"]["image_size"] = args.image_size
    cfg["training"] = dict(base_cfg["training"])
    cfg["training"]["batch_size"] = 2
    cfg["training"]["epochs"] = args.epochs
    cfg["training"]["num_workers"] = 0
    cfg["training"]["checkpoint_dir"] = os.path.join(tmp_root, "checkpoints")
    cfg["training"]["log_dir"] = os.path.join(tmp_root, "logs")
    cfg["diffusion"] = dict(base_cfg["diffusion"])
    cfg["diffusion"]["T_train"] = 50   # small T for a fast smoke test; real runs use 1000
    cfg["model"]["experiment_name"] = "smoke_test"

    boundary_needed = model_cfg.get("bdm", {}).get("boundary_head", {}).get("enabled", False)
    train_tf = get_train_transforms(image_size=args.image_size)
    eval_tf = get_eval_transforms(image_size=args.image_size)
    boundary_fn = mask_to_boundary if boundary_needed else None

    train_set = ISICDataset(train_img_dir, train_mask_dir, transform=train_tf, boundary_fn=boundary_fn)
    val_set = ISICDataset(test_img_dir, test_mask_dir, transform=eval_tf, boundary_fn=boundary_fn)

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

    print("Building model...")
    model = build_model_from_config(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model built OK. {n_params/1e6:.2f}M parameters.")

    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        loss_weights_cfg=model_cfg.get("loss_weights", {}),
        learning_rate=1e-4, epochs=args.epochs,
        checkpoint_dir=cfg["training"]["checkpoint_dir"], log_dir=cfg["training"]["log_dir"],
        device="cpu", experiment_name="smoke_test",
    )

    print(f"\nTraining for {args.epochs} epoch(s) on {args.n_train} synthetic images (CPU)...")
    trainer.fit(val_every=1, ckpt_every=args.epochs)
    print("Training loop completed without errors.\n")

    print("Running inference on a held-out synthetic image...")
    model.eval()
    sample = val_set[0]
    image = sample["image"].unsqueeze(0)
    with torch.no_grad():
        out = model.predict(image)

    out_dir = os.path.join(tmp_root, "smoke_outputs")
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image as PILImage
    pred_png = (out["refined_mask"][0, 0].numpy() > 0.5).astype(np.uint8) * 255
    PILImage.fromarray(pred_png).save(os.path.join(out_dir, "predicted_mask.png"))
    PILImage.fromarray((sample["mask"][0].numpy() * 255).astype(np.uint8)).save(os.path.join(out_dir, "ground_truth_mask.png"))

    print(f"\nSMOKE TEST PASSED.")
    print(f"  - Trained {args.epochs} epoch(s) end-to-end with no errors")
    print(f"  - Ran inference through DWT -> BDM -> uncertainty -> ECRF with no errors")
    print(f"  - Predicted mask saved to: {os.path.join(out_dir, 'predicted_mask.png')}")
    print(f"  - Ground-truth mask saved to: {os.path.join(out_dir, 'ground_truth_mask.png')}")

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"\n(Cleaned up {tmp_root}. Re-run with --keep_tmp to inspect the images/masks/checkpoints.)")
    else:
        print(f"\nAll files kept at: {tmp_root}")


if __name__ == "__main__":
    main()
