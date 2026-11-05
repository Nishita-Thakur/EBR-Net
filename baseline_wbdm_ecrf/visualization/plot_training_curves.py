"""
plot_training_curves.py — plots IoU vs. training epoch across datasets,
matching paper Fig. 7 ("Impact of Varying the Number of Training Rounds
on the IoU of ISIC 2016/2017/2018"), and generic loss-curve plotting
from the JSONL logs written by training/trainer.py.

Usage:
    python visualization/plot_training_curves.py \
        --logs logs/wbdm_ecrf_adaptive_isic2016/train_log.jsonl:ISIC2016 \
               logs/wbdm_ecrf_adaptive_isic2017/train_log.jsonl:ISIC2017 \
               logs/wbdm_ecrf_adaptive_isic2018/train_log.jsonl:ISIC2018 \
        --metric val_iou --out results/training_curves_iou.png
"""
import argparse
import json

import matplotlib.pyplot as plt


def load_jsonl(path: str):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", nargs="+", required=True, help="path:label pairs, e.g. logs/x/train_log.jsonl:ISIC2016")
    parser.add_argument("--metric", default="val_iou")
    parser.add_argument("--out", default="results/training_curves.png")
    args = parser.parse_args()

    plt.figure(figsize=(6, 4))
    for entry in args.logs:
        path, _, label = entry.partition(":")
        label = label or path
        records = load_jsonl(path)
        epochs = [r["epoch"] for r in records if args.metric in r]
        values = [r[args.metric] * 100 if args.metric.startswith("val_") and r[args.metric] <= 1.0 else r[args.metric] for r in records if args.metric in r]
        plt.plot(epochs, values, marker="o", markersize=3, label=label)

    plt.xlabel("Number of Training Epochs")
    plt.ylabel(f"{args.metric} (%)" if args.metric.startswith("val_") else args.metric)
    plt.title(f"{args.metric} over training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved training-curve figure to {args.out}")


if __name__ == "__main__":
    main()
