#!/usr/bin/env bash
# download_ph2.sh — fetches the PH2 dataset, used only for zero-shot
# cross-dataset evaluation (paper Table 3: train on ISIC2018, eval on
# PH2 with no fine-tuning). Lays out images/masks to match
# configs/base.yaml -> data.datasets.ph2.
#
# Usage:
#   ./scripts/download_ph2.sh <data_root>
set -euo pipefail

DATA_ROOT="${1:-data_root}"
OUT="${DATA_ROOT}/PH2"
mkdir -p "${OUT}/images" "${OUT}/masks"

# PH2 is distributed by ADDI (Universidade do Porto); the archive ships
# per-lesion folders each containing the dermoscopic image and its
# manual segmentation. Set PH2_URL if the ADDI download link changes.
PH2_URL="${PH2_URL:-https://www.fc.up.pt/addi/ph2%20database.html}"

echo "PH2 is distributed via a request form at:"
echo "  ${PH2_URL}"
echo "This script cannot auto-download it without credentials/agreement."
echo "After manually downloading 'PH2Dataset.zip', run:"
echo "  unzip PH2Dataset.zip -d ${DATA_ROOT}/PH2_raw"
echo "then re-run this script with PH2_RAW=${DATA_ROOT}/PH2_raw to reorganize it."

PH2_RAW="${PH2_RAW:-${DATA_ROOT}/PH2_raw}"
if [[ -d "${PH2_RAW}" ]]; then
  echo "Reorganizing ${PH2_RAW} -> ${OUT}"
  find "${PH2_RAW}" -iname "*_Dermoscopic_Image" -type d | while read -r img_dir; do
    case_id="$(basename "${img_dir}" | sed 's/_Dermoscopic_Image//')"
    cp "${img_dir}"/*.bmp "${OUT}/images/${case_id}.bmp" 2>/dev/null || true
  done
  find "${PH2_RAW}" -iname "*_lesion" -type d | while read -r msk_dir; do
    case_id="$(basename "${msk_dir}" | sed 's/_lesion//')"
    cp "${msk_dir}"/*.bmp "${OUT}/masks/${case_id}.bmp" 2>/dev/null || true
  done
  echo "Done: $(ls "${OUT}/images" | wc -l) images, $(ls "${OUT}/masks" | wc -l) masks."
else
  echo "PH2_RAW directory not found (${PH2_RAW}); nothing to reorganize yet."
fi
