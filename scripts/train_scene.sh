#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/train_scene.sh <dataset> <scene> <source_path> <model_root> <gpu_id> [extra train args...]

Examples:
  scripts/train_scene.sh n3v cook_spinach /data/datasets/n3v/cook_spinach outputs 0
  scripts/train_scene.sh hypernerf vrig-peel-banana /data/datasets/hypernerf/vrig/vrig-peel-banana outputs 1 --iterations 12000

The release profile is selected from <source_path> when possible. Supported datasets:
  n3v, hypernerf
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 5 ]]; then
  usage
  exit 0
fi

DATASET="$1"
SCENE="$2"
SOURCE_PATH="$3"
MODEL_ROOT="$4"
GPU_ID="$5"
shift 5

case "${DATASET}" in
  n3v|hypernerf) ;;
  *)
    echo "Unsupported dataset '${DATASET}'. Use n3v or hypernerf." >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_CONFIG="${PROFILE_CONFIG:-${REPO_ROOT}/configs/profiles}"
PYTHON_BIN="${PYTHON:-python}"
MODEL_PATH="${MODEL_ROOT%/}/${DATASET}_${SCENE}"
mkdir -p "${MODEL_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" train.py \
  --source_path "${SOURCE_PATH}" \
  --model_path "${MODEL_PATH}" \
  --profile_config "${PROFILE_CONFIG}" \
  "$@"
