"""
trainer.py — the epoch/step loop shared by all experiments. Reads its
knobs from base.yaml -> training (batch_size, epochs, optimizer,
learning_rate, checkpoint_dir, log_dir) and from the model config's
loss_weights section (via losses.LossWeighter).

Kept independent of argparse/config-loading (see training/train.py) so
it can also be driven from tests/ or ablations/ scripts directly.
"""
import os
import time
import json
import torch
from torch.utils.data import DataLoader

from losses.loss_weighting import LossWeighter
from losses.boundary_loss import boundary_bce_dice_loss


class Trainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        loss_weights_cfg: dict = None,
        learning_rate: float = 1e-4,
        epochs: int = 200,
        checkpoint_dir: str = "checkpoints/",
        log_dir: str = "logs/",
        device: str = "cuda",
        experiment_name: str = "wbdm_ecrf",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.experiment_name = experiment_name

        self.checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.loss_weighter = LossWeighter.from_config(loss_weights_cfg or {})

        self.log_path = os.path.join(self.log_dir, "train_log.jsonl")
        self.global_step = 0

    def _log(self, record: dict):
        record["timestamp"] = time.time()
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def train_one_epoch(self, epoch: int) -> dict:
        self.model.train()
        epoch_losses = {}
        num_batches = 0

        for batch in self.train_loader:
            image = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)
            boundary_gt = batch.get("boundary")
            if boundary_gt is not None:
                boundary_gt = boundary_gt.to(self.device)

            raw_losses = self.model.training_step(image, mask, boundary_gt, epoch=epoch)

            if "boundary_logits" in raw_losses and boundary_gt is not None:
                target_hw = raw_losses["boundary_logits"].shape[-2:]
                bgt = torch.nn.functional.interpolate(boundary_gt, size=target_hw, mode="nearest")
                raw_losses["boundary_loss"] = boundary_bce_dice_loss(raw_losses.pop("boundary_logits"), bgt)
            else:
                raw_losses.pop("boundary_logits", None)

            total_loss, components = self.loss_weighter.combine(raw_losses)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            for k, v in components.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()
            epoch_losses["total_loss"] = epoch_losses.get("total_loss", 0.0) + total_loss.item()
            num_batches += 1
            self.global_step += 1

        for k in epoch_losses:
            epoch_losses[k] /= max(1, num_batches)
        return epoch_losses

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        if self.val_loader is None:
            return {}
        self.model.eval()
        from evaluation.metrics import dice_score, iou_score

        dice_total, iou_total, n = 0.0, 0.0, 0
        for batch in self.val_loader:
            image = batch["image"].to(self.device)
            mask = batch["mask"].to(self.device)
            out = self.model.predict(image)
            pred = out["refined_mask"]
            dice_total += dice_score(pred, mask).sum().item()
            iou_total += iou_score(pred, mask).sum().item()
            n += image.shape[0]

        return {"val_dice": dice_total / max(1, n), "val_iou": iou_total / max(1, n)}

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        path = os.path.join(self.checkpoint_dir, f"epoch_{epoch:04d}.pt")
        torch.save(state, path)
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, "best.pt"))
        torch.save(state, os.path.join(self.checkpoint_dir, "last.pt"))

    def load_checkpoint(self, path: str) -> int:
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        return state.get("epoch", 0)

    def fit(self, start_epoch: int = 0, val_every: int = 5, ckpt_every: int = 10):
        best_dice = -1.0
        for epoch in range(start_epoch, self.epochs):
            t0 = time.time()
            train_losses = self.train_one_epoch(epoch)
            record = {"epoch": epoch, "phase": "train", "elapsed_s": time.time() - t0, **train_losses}

            if self.val_loader is not None and (epoch + 1) % val_every == 0:
                val_metrics = self.validate(epoch)
                record.update(val_metrics)
                is_best = val_metrics.get("val_dice", -1.0) > best_dice
                best_dice = max(best_dice, val_metrics.get("val_dice", best_dice))
            else:
                is_best = False

            self._log(record)
            print(f"[{self.experiment_name}] epoch {epoch}: {record}")

            if (epoch + 1) % ckpt_every == 0 or is_best or epoch == self.epochs - 1:
                self.save_checkpoint(epoch, is_best=is_best)
