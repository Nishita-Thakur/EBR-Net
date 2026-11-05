# WBDM-ECRF (Adaptive) — Skin Lesion Segmentation

This repository implements two variants of a bridge-diffusion + conditional-random-field
skin lesion segmentation pipeline:

- **Baseline** (`configs/model_baseline.yaml`): a faithful reproduction of
  *"WBDM-ECRF: A bridge diffusion model with efficient conditional random field for skin
  lesion segmentation"* (Ji et al., *Expert Systems With Applications*, 2026) —
  fixed LL-only DWT, a Brownian-Bridge Diffusion U-Net with FlashAttention, and a fixed
  11×11 boundary-expanded ECRF with an SSIM pairwise term.
- **Adaptive** (`configs/model_adaptive.yaml`): our modified architecture (see
  `architecturenew.docx`) that keeps all four Haar subbands with a learned per-subband
  gate, adds an auxiliary boundary head to the BDM, estimates per-pixel uncertainty from
  mask/boundary disagreement, and modulates the ECRF's boundary-expansion window size by
  that uncertainty.

## Layout

```
configs/        base.yaml (shared) + model_baseline.yaml / model_adaptive.yaml
data/           ISIC/PH2 datasets, paired augmentations, boundary pseudo-GT
models/
  dwt/          static (LL-only) and adaptive (gated multi-subband) DWT
  bdm/          FlashAttention, U-Net backbone, boundary head, bridge diffusion,
                deterministic/stochastic uncertainty
  ecrf/         energy terms (Eqs. 15-18), static (fixed window) and adaptive
                (uncertainty-modulated window) ECRF
losses/         diffusion MSE, boundary BCE+Dice, loss-weight combiner
training/       Trainer, LR/warmup schedules, train.py CLI
evaluation/     Dice/IoU/ACC/SE/SP/HD95/ASSD metrics, evaluate.py, paired t-tests,
                efficiency benchmarking (params/FLOPs/time)
inference/      post-processing (largest-CC, hole-filling), sample.py batch inference
ablations/      DWT+ECRF ablation (Table 5), subband ablation (Table 6),
                ECRF-window ablation (Table 7), frequency-gate analysis
visualization/  qualitative overlays, training curves, uncertainty maps
tests/          unit + shape tests for every module above
scripts/        dataset download helpers, full-benchmark reproduction script
```

## Setup

```bash
pip install -r requirements.txt
./scripts/download_isic.sh data_root all
./scripts/download_ph2.sh data_root         # manual step; see script output
```

Edit `configs/base.yaml -> data.datasets.*` if your data lives elsewhere.

## Training

```bash
# reproduce the paper's baseline on ISIC2018
python training/train.py --base_config configs/base.yaml \
    --model_config configs/model_baseline.yaml --dataset isic2018

# train our adaptive model
python training/train.py --base_config configs/base.yaml \
    --model_config configs/model_adaptive.yaml --dataset isic2018
```

Checkpoints land in `checkpoints/<experiment_name>/`, JSONL logs in
`logs/<experiment_name>/train_log.jsonl`.

## Evaluation

```bash
python evaluation/evaluate.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt \
    --model_config configs/model_adaptive.yaml --dataset isic2018

# zero-shot cross-dataset (paper Table 3): train on isic2018, eval on ph2
python evaluation/evaluate.py --checkpoint checkpoints/wbdm_ecrf_adaptive/best.pt \
    --model_config configs/model_adaptive.yaml --dataset ph2
```

## Ablations & full benchmark

```bash
python ablations/ablation_dwt_ecrf.py --dataset isic2018      # Table 5
python ablations/ablation_subbands.py --dataset isic2016      # Table 6
python ablations/ablation_ecrf_window.py --checkpoint <ckpt> --dataset isic2016  # Table 7
python ablations/ablation_gate_analysis.py --checkpoint <ckpt> --dataset isic2018

./scripts/run_full_benchmark.sh   # everything above, end to end
```

## Tests

```bash
pytest tests/ -v
```

## Notes / honest caveats (see `architecturenew.docx`)

- The paper's own ablation (Table 6) shows naive high-frequency-subband concatenation is
  catastrophic (Dice 94.45% → 40.23%); the adaptive DWT gate is initialized near the
  paper's measured LL-dominant energy prior and KL-regularized toward it during a warmup
  period specifically to avoid this failure mode. `ablations/ablation_gate_analysis.py`
  checks the gate hasn't collapsed back to (or past) that prior.
- The boundary head and uncertainty module add parameters and inference time; re-run
  `evaluation/efficiency_bench.py` against the baseline before claiming the paper's 0.12s
  inference advantage is preserved.
- ISIC datasets ship no boundary annotations; boundary pseudo-GT is derived from the mask
  via `data/boundary_labels.py` (Sobel or morphological erosion-difference).
