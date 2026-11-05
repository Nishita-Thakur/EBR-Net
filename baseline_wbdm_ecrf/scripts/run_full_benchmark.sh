#!/usr/bin/env bash
# run_full_benchmark.sh — end-to-end reproduction of the paper's Tables
# 1-9: trains the baseline and adaptive models on ISIC 2016/2017/2018,
# evaluates them (+ zero-shot on PH2), runs the ablation studies, and
# benchmarks efficiency.
#
# Usage:
#   ./scripts/run_full_benchmark.sh
set -euo pipefail

DATASETS=("isic2016" "isic2017" "isic2018")
CONFIGS=("configs/model_baseline.yaml" "configs/model_adaptive.yaml")

mkdir -p results

for model_cfg in "${CONFIGS[@]}"; do
  exp_name="$(basename "${model_cfg}" .yaml)"
  for ds in "${DATASETS[@]}"; do
    echo "=============================================="
    echo " Training ${exp_name} on ${ds}"
    echo "=============================================="
    python training/train.py --base_config configs/base.yaml --model_config "${model_cfg}" --dataset "${ds}"

    ckpt="checkpoints/$(python -c "import yaml; print(yaml.safe_load(open('${model_cfg}'))['experiment_name'])")/best.pt"

    echo "-- Evaluating on ${ds} --"
    python evaluation/evaluate.py --checkpoint "${ckpt}" --model_config "${model_cfg}" --dataset "${ds}" \
      --out_json "results/${exp_name}_${ds}_metrics.json"
  done
done

echo "=============================================="
echo " Zero-shot PH2 evaluation (adaptive model, trained on ISIC2018)"
echo "=============================================="
python evaluation/evaluate.py \
  --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt \
  --model_config configs/model_adaptive.yaml --dataset ph2 \
  --out_json results/wbdm_ecrf_adaptive_ph2_metrics.json

echo "=============================================="
echo " Ablation studies (Tables 5-7) + gate analysis"
echo "=============================================="
python ablations/ablation_dwt_ecrf.py --dataset isic2018
python ablations/ablation_subbands.py --dataset isic2016
python ablations/ablation_ecrf_window.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt --dataset isic2016
python ablations/ablation_gate_analysis.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt --dataset isic2018

echo "=============================================="
echo " Efficiency benchmark (Tables 8-9)"
echo "=============================================="
python evaluation/efficiency_bench.py --model_config configs/model_adaptive.yaml \
  --sweep_steps 3 10 50 100 200 500 1000

echo "All done. See results/ for metrics and figures."
