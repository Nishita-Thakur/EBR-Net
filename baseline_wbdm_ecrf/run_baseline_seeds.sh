#!/bin/bash
# Retrain the WBDM-ECRF baseline with 3 seeds per dataset, selecting checkpoints on a validation split.
# usage (from /workspace/SLS-CODE):  nohup bash run_baseline_seeds.sh isic2016 isic2017 > baseline_seeds.log 2>&1 &
set -u
OUT=${OUT:-/workspace/sls_v2/baseline_seeded}
mkdir -p "$OUT" logs_seeded
DATASETS=${@:-isic2016 isic2017}
for ds in $DATASETS; do
  for s in 0 1 2; do
    PYTHONPATH=. python training/train_seeded.py --dataset $ds --seed $s > logs_seeded/${ds}_s${s}.log 2>&1 &
  done
  wait
  for s in 0 1 2; do
    ck=checkpoints/wbdm_ecrf_baseline_${ds}_s${s}/best.pt
    PYTHONPATH=. python evaluation/evaluate.py --checkpoint $ck --model_config configs/model_baseline.yaml --dataset $ds \
      --per_sample_json $OUT/${ds}_s${s}.json --out_json $OUT/${ds}_s${s}_summary.json > $OUT/${ds}_s${s}.txt 2>&1
    # zero-shot PH2 with the model trained on this dataset (only if PH2 has been prepared)
    if [ -d data_root/PH2/images ]; then
      PYTHONPATH=. python evaluation/evaluate.py --checkpoint $ck --model_config configs/model_baseline.yaml --dataset ph2 \
        --per_sample_json $OUT/ph2_from_${ds}_s${s}.json > $OUT/ph2_from_${ds}_s${s}.txt 2>&1
    fi
  done
  echo "finished $ds"
done
