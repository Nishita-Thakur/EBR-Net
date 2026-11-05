"""
plot_uncertainty_maps.py — visualizes the deterministic/stochastic
uncertainty map (models/bdm/{deterministic,stochastic}_uncertainty.py)
alongside the image, GT, predicted mask, and boundary prediction —
useful for sanity-checking that the adaptive ECRF (architecturenew.docx
item 5) is concentrating effort on the paper's documented failure
regions (hair artifacts, low-contrast fading boundaries, Sec 4.7).

Usage:
    python visualization/plot_uncertainty_maps.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt \
        --dataset isic2018 --num_samples 4 --out results/uncertainty_maps.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data import ISICDataset, get_eval_transforms
from models import build_model_from_config


@torch.no_grad()
def make_figure(model, dataset, device, num_samples: int, out_path: str):
    ncols = 5 if model.uncertainty_enabled else 4
    fig, axes = plt.subplots(num_samples, ncols, figsize=(3 * ncols, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    col_titles = ["Image", "GT", "Pred Mask", "Boundary"] + (["Uncertainty"] if model.uncertainty_enabled else [])

    indices = np.linspace(0, len(dataset) - 1, num_samples).astype(int)
    for row, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)
        gt = sample["mask"].numpy()[0]

        out = model.predict(image)
        pred = out["pred_mask"][0, 0].cpu().numpy()
        boundary = out["boundary_prob"][0, 0].cpu().numpy() if out["boundary_prob"] is not None else np.zeros_like(pred)

        cols = [sample["image"].permute(1, 2, 0).numpy(), gt, pred, boundary]
        if model.uncertainty_enabled:
            unc = out["uncertainty"][0, 0].cpu().numpy() if out["uncertainty"] is not None else np.zeros_like(pred)
            cols.append(unc)

        for c, data in enumerate(cols):
            cmap = None if c == 0 else "magma" if col_titles[c] == "Uncertainty" else "gray"
            axes[row, c].imshow(data, cmap=cmap)
            if row == 0:
                axes[row, c].set_title(col_titles[c])
            axes[row, c].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved uncertainty-map figure to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--base_config", default="configs/base.yaml")
    parser.add_argument("--model_config", default="configs/model_adaptive.yaml")
    parser.add_argument("--dataset", default="isic2018")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--out", default="results/uncertainty_maps.png")
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

    make_figure(model, dataset, device, args.num_samples, args.out)


if __name__ == "__main__":
    main()
