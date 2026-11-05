"""
ablation_gate_analysis.py — inspects the learned SEFrequencyGate weights
(models/dwt/frequency_gate.py) across a dataset, analogous to the paper's
own energy-distribution analysis (Fig. 4). This directly checks the
architecturenew.docx caveat: "the fusion gate needs real
regularization/pretraining, not just random init" — i.e. verifies the
gate hasn't simply collapsed back to ~(1,0,0,0) (== the paper's static
LL-only choice) nor drifted toward amplifying high-frequency noise.

Usage:
    python ablations/ablation_gate_analysis.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt --dataset isic2018
"""
import argparse
import json

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import ISICDataset, get_eval_transforms
from models import build_model_from_config


@torch.no_grad()
def collect_gate_stats(model, loader, device) -> dict:
    assert model.dwt_type == "adaptive", "gate analysis requires an AdaptiveDWT model"
    subband_names = model.dwt.gate.SUBBAND_ORDER
    all_gates = []  # list of (4,) mean-over-channel gate values per image

    for batch in loader:
        image = batch["image"].to(device)
        _, aux = model.compute_dwt(image)
        gate = aux["gate"]  # (B,4,C)
        per_image_mean = gate.mean(dim=2)  # (B,4)
        all_gates.append(per_image_mean.cpu().numpy())

    all_gates = np.concatenate(all_gates, axis=0)  # (N,4)
    normalized = all_gates / all_gates.sum(axis=1, keepdims=True)

    stats = {}
    for i, name in enumerate(subband_names):
        stats[name] = {
            "raw_gate_mean": float(all_gates[:, i].mean()),
            "raw_gate_std": float(all_gates[:, i].std()),
            "energy_share_mean_pct": float(normalized[:, i].mean() * 100),
            "energy_share_std_pct": float(normalized[:, i].std() * 100),
        }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--dataset", default="isic2018")
    parser.add_argument("--out_json", default="results/gate_analysis.json")
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

    stats = collect_gate_stats(model, loader, device)
    print(json.dumps(stats, indent=2))

    ll_share = stats["LL"]["energy_share_mean_pct"]
    if ll_share > 99.5:
        print(f"\nWARNING: LL energy share is {ll_share:.2f}% -- the gate may have "
              f"collapsed back toward the paper's static LL-only behavior.")
    elif ll_share < 50.0:
        print(f"\nWARNING: LL energy share dropped to {ll_share:.2f}% -- the gate may "
              f"be admitting destabilizing high-frequency noise (cf. Table 6's collapse).")

    with open(args.out_json, "w") as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":
    main()
