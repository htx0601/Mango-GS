#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/eval_metrics.sh <model_path> <iteration> <gpu_id>

The model path should already contain rendered test images under:
  <model_path>/test/video_<iteration>/renders
  <model_path>/test/video_<iteration>/gt

Run scripts/render_scene.sh first if those folders do not exist.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 3 ]]; then
  usage
  exit 0
fi

MODEL_PATH="$1"
ITERATION="$2"
GPU_ID="$3"

if [[ ! -d "${MODEL_PATH}/test" && -d "${MODEL_PATH}_mango_node/test" ]]; then
  MODEL_PATH="${MODEL_PATH}_mango_node"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" metrics.py --model_paths "${MODEL_PATH}"
TPIPS="$("${PYTHON_BIN}" scripts/tools/tpips_tlp.py "${MODEL_PATH}" "${ITERATION}" --device cuda)"
echo "TPIPS/TLP: ${TPIPS}"
