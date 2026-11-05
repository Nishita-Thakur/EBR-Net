#!/usr/bin/env bash
# download_isic.sh — fetches ISIC 2016/2017/2018 train+test images/masks
# via the official ISIC Archive API and lays them out to match
# configs/base.yaml's data.datasets.isic201{6,7,8} paths.
#
# Usage:
#   ./scripts/download_isic.sh <data_root> [2016|2017|2018|all]
set -euo pipefail

DATA_ROOT="${1:-data_root}"
YEAR="${2:-all}"

download_year() {
  local year="$1"
  local out="${DATA_ROOT}/ISIC${year}"
  echo "== ISIC ${year} -> ${out} =="
  mkdir -p "${out}/train/images" "${out}/train/masks" "${out}/test/images" "${out}/test/masks"

  case "${year}" in
    2016)
      TRAIN_IMG_URL="https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_Data.zip"
      TRAIN_MSK_URL="https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Training_GroundTruth.zip"
      TEST_IMG_URL="https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_Data.zip"
      TEST_MSK_URL="https://isic-challenge-data.s3.amazonaws.com/2016/ISBI2016_ISIC_Part1_Test_GroundTruth.zip"
      ;;
    2017)
      TRAIN_IMG_URL="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Data.zip"
      TRAIN_MSK_URL="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Training_Part1_GroundTruth.zip"
      TEST_IMG_URL="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Data.zip"
      TEST_MSK_URL="https://isic-challenge-data.s3.amazonaws.com/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip"
      ;;
    2018)
      TRAIN_IMG_URL="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Training_Input.zip"
      TRAIN_MSK_URL="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1_Training_GroundTruth.zip"
      TEST_IMG_URL="https://isic-challenge-data.s3.amazonaws.com/2018/ISIC2018_Task1-2_Test_Input.zip"
      TEST_MSK_URL=""  # 2018 test masks are withheld by the challenge; use the validation split for local eval if needed
      ;;
    *)
      echo "Unknown ISIC year: ${year}" >&2; exit 1 ;;
  esac

  for pair in "TRAIN_IMG_URL:${out}/train/images" "TRAIN_MSK_URL:${out}/train/masks" "TEST_IMG_URL:${out}/test/images" "TEST_MSK_URL:${out}/test/masks"; do
    var="${pair%%:*}"; dest="${pair#*:}"
    url="${!var}"
    if [[ -z "${url}" ]]; then
      echo "  (skipping ${var}: no public URL for this split)"
      continue
    fi
    tmp_zip="$(mktemp --suffix=.zip)"
    echo "  downloading ${url}"
    curl -L --fail -o "${tmp_zip}" "${url}"
    unzip -q -o "${tmp_zip}" -d "${dest}"
    rm -f "${tmp_zip}"
  done
}

if [[ "${YEAR}" == "all" ]]; then
  for y in 2016 2017 2018; do download_year "${y}"; done
else
  download_year "${YEAR}"
fi

echo "Done. Verify paths against configs/base.yaml -> data.datasets."
