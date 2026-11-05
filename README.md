# EBR-Net

Code for **EBR-Net**, a deterministic skin-lesion segmentation network, and for the comparison models used in the accompanying paper.

> Paper: `<EBR-Net: A pretrained encoder–decoder network with residual boundary
refinement for skin lesion segmentation, Nishita Thakur>`

## Overview

EBR-Net consists of:

- an ImageNet-pretrained **EfficientNetV2-S** encoder,
- a **self-attention** block,
- an **scSE** decoder,
- a **boundary head**, and
- a **full-resolution residual refinement head**.

It is deterministic and needs **one forward pass per view**. It does not use stochastic sampling or a CRF. It has 23.4M parameters, against 125.3M reported for WBDM-ECRF.

## Results (Dice, %)

All models were trained under the same recipe on ISIC 2016, ISIC 2017 and a local ISIC 2018 split.

| Model | ISIC 2016 | ISIC 2017 | ISIC 2018 (local split) |
|---|---|---|---|
| WBDM-ECRF (our retraining) | 88.46 | 81.56 | 84.65 |
| **EBR-Net** | **92.73** | **85.59** | **90.86** |

- EBR-Net is 1.1 to 1.6 Dice above U-Net, U-Net++ and Attention U-Net.
- Against FAT-Net, the differences are not statistically distinguishable.
- Ablations: removing pretraining, the refinement head or the auxiliary supervision changes test Dice by no more than seed noise (about 0.5).

See the paper for the full tables, the protocol and the statistical tests.

## Repository layout

```
.
├── data.py, model.py, infer.py, zoo_models.py   # EBR-Net and the comparison models (U-Net, U-Net++, Attention U-Net, FAT-Net)
├── make_fig_cmp_all_v2.py                       # qualitative comparison figure (Fig. 4)
├── baseline_wbdm_ecrf/                          # our retraining of WBDM-ECRF (see note below)
├── requirements.txt
└── LICENSE
```

## Installation

```bash
git clone https://github.com/Nishita-Thakur/EBR-Net.git
cd EBR-Net
pip install -r requirements.txt
```

Tested with Python 3 and `torch 2.4.0+cu124`, `torchvision 0.19`, `timm 1.0.30`. The `torch` and `torchvision` versions must match each other.

## Data

The ISIC images are **not included** in this repository. Download them from the ISIC archive (<https://www.isic-archive.com>) and see the paper for the split we used for ISIC 2018.

Expected layout for each dataset:

```
<data_root>/ISIC2017/
    train/{images,masks}
    val/{images,masks}
    test/{images,masks}
```

## Note on the WBDM-ECRF baseline

`baseline_wbdm_ecrf/` contains **our own retraining of WBDM-ECRF** (Ji et al., 2026), written by us because no official code was available. It is not the authors' code, and its numbers may differ from those reported in the original paper.

