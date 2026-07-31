#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/eval_metrics.sh <model_path> <gpu_id>

The model path should already contain rendered test images under `test/`.
The release checkpoint or latest rendered training checkpoint is selected
automatically.

Run scripts/render_scene.sh first if those folders do not exist.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  usage
  exit 0
fi

MODEL_PATH="$1"
GPU_ID="$2"

if [[ ! -d "${MODEL_PATH}/test" && -d "${MODEL_PATH}_mango_node/test" ]]; then
  MODEL_PATH="${MODEL_PATH}_mango_node"
fi

if [[ -d "${MODEL_PATH}/test/video_release" ]]; then
  CHECKPOINT_TAG=release
else
  LATEST_DIR="$(find "${MODEL_PATH}/test" -maxdepth 1 -type d -name 'video_*' | sort -V | tail -n 1)"
  if [[ -z "${LATEST_DIR}" ]]; then
    echo "No rendered video checkpoint found under ${MODEL_PATH}/test" >&2
    exit 1
  fi
  CHECKPOINT_TAG="${LATEST_DIR##*/video_}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" metrics.py --model_paths "${MODEL_PATH}"
TPIPS="$("${PYTHON_BIN}" scripts/tools/tpips_tlp.py "${MODEL_PATH}" "${CHECKPOINT_TAG}" --device cuda)"
echo "TPIPS/TLP: ${TPIPS}"
