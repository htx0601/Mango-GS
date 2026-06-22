#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/render_one_frame.sh <dataset> <scene> <source_path> <model_path> <iteration> <gpu_id> <out_png>

Example:
  scripts/render_one_frame.sh n3v cook_spinach /data/datasets/n3v/cook_spinach weights/n3v/cook_spinach 24000 0 previews/cook_spinach.png
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 7 ]]; then
  usage
  exit 0
fi

DATASET="$1"
SCENE="$2"
SOURCE_PATH="$3"
MODEL_PATH="$4"
ITERATION="$5"
GPU_ID="$6"
OUT_PNG="$7"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_CONFIG="${PROFILE_CONFIG:-${REPO_ROOT}/configs/profiles}"
PYTHON_BIN="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" scripts/tools/render_one_frame.py \
  --source_path "${SOURCE_PATH}" \
  --model_path "${MODEL_PATH}" \
  --iteration "${ITERATION}" \
  --out_path "${OUT_PNG}" \
  --profile_config "${PROFILE_CONFIG}"
