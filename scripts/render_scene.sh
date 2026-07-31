#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/render_scene.sh <dataset> <scene> <source_path> <model_path> <gpu_id> [video_fps] [video_window_mode] [extra render args...]

Examples:
  scripts/render_scene.sh n3v cook_spinach /data/datasets/n3v/cook_spinach weights/n3v/cook_spinach 0 20
  scripts/render_scene.sh hypernerf vrig-peel-banana /data/datasets/hypernerf/vrig/vrig-peel-banana weights/hypernerf/vrig-peel-banana 1 10 block
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 5 ]]; then
  usage
  exit 0
fi

DATASET="$1"
SCENE="$2"
SOURCE_PATH="$3"
MODEL_PATH="$4"
GPU_ID="$5"
VIDEO_FPS="${6:-10}"
VIDEO_WINDOW_MODE="${7:-block}"
if [[ "$#" -gt 7 ]]; then
  shift 7
else
  shift "$#"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_CONFIG="${PROFILE_CONFIG:-${REPO_ROOT}/configs/profiles}"
PYTHON_BIN="${PYTHON:-python}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" render.py \
  --source_path "${SOURCE_PATH}" \
  --model_path "${MODEL_PATH}" \
  --iteration -1 \
  --deform_type mango_node \
  --profile_config "${PROFILE_CONFIG}" \
  --video_fps "${VIDEO_FPS}" \
  --video_window_mode "${VIDEO_WINDOW_MODE}" \
  --quiet \
  "$@"
