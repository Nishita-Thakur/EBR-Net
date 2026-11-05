"""Quick diagnostic: does the trained gate actually deviate from the LL-only prior?
Run on the DGX with: python check_gate.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt
"""
import argparse
import torch
import yaml
from data import ISICDataset, get_eval_transforms
from models import build_model_from_config

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--base_config", default="configs/base.yaml")
parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
parser.add_argument("--dataset", default="isic2016")
args = parser.parse_args()

with open(args.base_config) as f:
    base_cfg = yaml.safe_load(f)
with open(args.model_config) as f:
    model_cfg = yaml.safe_load(f)
cfg = dict(base_cfg)
cfg["model"] = model_cfg
device = "cuda" if torch.cuda.is_available() else "cpu"

model = build_model_from_config(cfg).to(device)
state = torch.load(args.checkpoint, map_location=device)
model.load_state_dict(state["model_state_dict"])
model.eval()

ds_cfg = cfg["data"]["datasets"][args.dataset]
eval_tf = get_eval_transforms(image_size=cfg["data"]["image_size"])
dataset = ISICDataset(ds_cfg["test_images"], ds_cfg["test_masks"], transform=eval_tf)

gates = []
with torch.no_grad():
    for i in range(min(50, len(dataset))):
        image = dataset[i]["image"].unsqueeze(0).to(device)
        _, aux = model.compute_dwt(image)
        gates.append(aux["gate"].mean(dim=(0, 2)).cpu())  # average over batch, channels -> (4,)

import torch as T
avg_gate = T.stack(gates).mean(dim=0)
print("Learned gate distribution, averaged over 50 test images (LL, LH, HL, HH):")
print(avg_gate.tolist())
print()
print("Prior it was initialized/regularized toward: [0.9987, 0.0004, 0.0004, 0.0005]")
