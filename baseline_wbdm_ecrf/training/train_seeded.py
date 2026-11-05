"""
train_seeded.py -- leak-free, multi-seed training for the WBDM-ECRF baseline.

Differences from training/train.py (which this file does NOT modify):
  * checkpoint selection uses a VALIDATION split carved out of the TRAINING images
    (train.py used the TEST set as its "validation" set, so best.pt was chosen on test data);
  * every (dataset, seed) gets its own checkpoint folder (train.py wrote all runs into one
    folder named after the config, so different datasets overwrote each other's best.pt);
  * only best.pt and last.pt are kept (the epoch_*.pt files are deleted at the end).

Usage (run from the SLS-CODE folder):
  PYTHONPATH=. python training/train_seeded.py --dataset isic2017 --seed 0
"""
import argparse, glob, os, random
import numpy as np, torch
from torch.utils.data import DataLoader, Subset

from data import ISICDataset, get_train_transforms, get_eval_transforms
from models import build_model_from_config
from training.trainer import Trainer
from training.train import set_seed, load_merged_config

ap = argparse.ArgumentParser()
ap.add_argument("--base_config", default="configs/base.yaml")
ap.add_argument("--model_config", default="configs/model_baseline.yaml")
ap.add_argument("--dataset", required=True, choices=["isic2016", "isic2017", "isic2018"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--val_frac", type=float, default=0.15)
ap.add_argument("--exp_name", default=None)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--image_size", type=int, default=None)
ap.add_argument("--batch_size", type=int, default=None)
ap.add_argument("--train_images", default=None); ap.add_argument("--train_masks", default=None)
a = ap.parse_args()

cfg = load_merged_config(a.base_config, a.model_config)
if a.image_size: cfg["data"]["image_size"] = a.image_size
if a.epochs: cfg["training"]["epochs"] = a.epochs
if a.batch_size: cfg["training"]["batch_size"] = a.batch_size
base_name = cfg["model"].get("experiment_name", "wbdm_ecrf")
exp = a.exp_name or f"{base_name}_{a.dataset}_s{a.seed}" + (f"_{a.image_size}" if a.image_size else "")
cfg["model"]["experiment_name"] = exp
set_seed(a.seed)
device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"

ds = cfg["data"]["datasets"][a.dataset]
timg, tmsk = a.train_images or ds["train_images"], a.train_masks or ds["train_masks"]
size, aug = cfg["data"]["image_size"], cfg["data"]["augmentation"]
prep = cfg["data"].get("preprocessing", {})
train_tf = get_train_transforms(image_size=size, hflip=aug.get("horizontal_flip", True), vflip=aug.get("vertical_flip", True),
                                rotation_degrees=aug.get("random_rotation_degrees", 15),
                                hair_removal=prep.get("hair_removal", False), clahe=prep.get("clahe", False))
eval_tf = get_eval_transforms(image_size=size, hair_removal=prep.get("hair_removal", False), clahe=prep.get("clahe", False))
full_train = ISICDataset(timg, tmsk, transform=train_tf)
full_val = ISICDataset(timg, tmsk, transform=eval_tf)          # same files, no augmentation
idx = list(range(len(full_train))); random.Random(a.seed).shuffle(idx)
n_val = int(round(len(idx) * a.val_frac)); val_idx, tr_idx = idx[:n_val], idx[n_val:]
print(f"[{exp}] train {len(tr_idx)} / val {len(val_idx)} (validation taken from the TRAINING images; test set untouched)")

tc = cfg["training"]
train_loader = DataLoader(Subset(full_train, tr_idx), batch_size=tc["batch_size"], shuffle=True,
                          num_workers=tc.get("num_workers", 4), drop_last=True)
val_loader = DataLoader(Subset(full_val, val_idx), batch_size=tc["batch_size"], shuffle=False,
                        num_workers=tc.get("num_workers", 4))
model = build_model_from_config(cfg)
trainer = Trainer(model=model, train_loader=train_loader, val_loader=val_loader,
                  loss_weights_cfg=cfg["model"].get("loss_weights", {}),
                  learning_rate=tc.get("learning_rate", 1e-4), epochs=tc.get("epochs", 200),
                  checkpoint_dir=tc.get("checkpoint_dir", "checkpoints/"), log_dir=tc.get("log_dir", "logs/"),
                  device=device, experiment_name=exp)
trainer.fit(start_epoch=0, val_every=5, ckpt_every=10 ** 9)
for f in glob.glob(os.path.join(trainer.checkpoint_dir, "epoch_*.pt")):
    os.remove(f)
print(f"[{exp}] done; kept best.pt / last.pt in {trainer.checkpoint_dir}")
