"""
significance_tests.py — paired t-tests between WBDM-ECRF and a baseline
method's per-sample Dice/IoU scores, matching Table 4 of the paper.

Usage:
    python evaluation/significance_tests.py \
        --ours_json results/wbdm_ecrf_isic2016_per_sample.json \
        --baseline_json results/unet_isic2016_per_sample.json \
        --metric dice
"""
import argparse
import json

import numpy as np
from scipy import stats


def paired_ttest(ours: list, baseline: list) -> dict:
    ours = np.array(ours, dtype=np.float64)
    baseline = np.array(baseline, dtype=np.float64)
    n = min(len(ours), len(baseline))
    if len(ours) != len(baseline):
        # per-sample arrays must be aligned to the same test-set ordering;
        # if lengths differ we can only compare the overlapping prefix.
        ours, baseline = ours[:n], baseline[:n]

    t_stat, p_value = stats.ttest_rel(ours, baseline)
    return {
        "n": int(n),
        "mean_ours": float(ours.mean()),
        "mean_baseline": float(baseline.mean()),
        "mean_diff": float(ours.mean() - baseline.mean()),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant_at_0.05": bool(p_value < 0.05),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours_json", required=True, help="JSON with a list of per-sample scores, or {'dice':[...],'iou':[...]}")
    parser.add_argument("--baseline_json", required=True)
    parser.add_argument("--metric", default="dice", choices=["dice", "iou"])
    args = parser.parse_args()

    with open(args.ours_json) as f:
        ours_data = json.load(f)
    with open(args.baseline_json) as f:
        baseline_data = json.load(f)

    ours_scores = ours_data[args.metric] if isinstance(ours_data, dict) else ours_data
    baseline_scores = baseline_data[args.metric] if isinstance(baseline_data, dict) else baseline_data

    result = paired_ttest(ours_scores, baseline_scores)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
