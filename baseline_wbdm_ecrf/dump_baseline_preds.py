import argparse, os
from pathlib import Path
import cv2, numpy as np, torch, yaml
from torch.utils.data import DataLoader
from models import build_model_from_config
from evaluation.evaluate import build_test_set

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="checkpoints/wbdm_ecrf_baseline/best.pt")
ap.add_argument("--base_config", default="configs/base.yaml")
ap.add_argument("--model_config", default="configs/model_baseline.yaml")
ap.add_argument("--dataset", default="isic2017")
ap.add_argument("--out_dir", required=True)
a = ap.parse_args()

cfg = dict(yaml.safe_load(open(a.base_config))); cfg["model"] = yaml.safe_load(open(a.model_config))
device = cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
test_set = build_test_set(cfg, a.dataset)
folder = Path(cfg["data"]["datasets"][a.dataset]["test_images"])
ids = [p.stem for p in sorted(folder.iterdir()) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
assert len(ids) == len(test_set), f"{len(ids)} files vs {len(test_set)} dataset items"
model = build_model_from_config(cfg).to(device)
model.load_state_dict(torch.load(a.checkpoint, map_location=device)["model_state_dict"]); model.eval()
os.makedirs(a.out_dir, exist_ok=True)
k = 0
with torch.no_grad():
    for batch in DataLoader(test_set, batch_size=8, shuffle=False, num_workers=2):
        pred = (model.predict(batch["image"].to(device))["refined_mask"] > 0.5).cpu().numpy().astype(np.uint8)
        for b in range(pred.shape[0]):
            cv2.imwrite(os.path.join(a.out_dir, ids[k] + ".png"), pred[b, 0] * 255); k += 1
print("saved", k, "masks to", a.out_dir)
