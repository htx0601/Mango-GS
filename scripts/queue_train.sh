#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/queue_train.sh <jobs.tsv> <gpu_ids_csv> [max_parallel]

Job TSV columns:
  dataset<TAB>scene<TAB>source_path<TAB>model_root<TAB>gpu_or_auto<TAB>extra_args

Rules:
  - Lines beginning with # and empty lines are ignored.
  - gpu_or_auto can be a concrete GPU id, or "auto".
  - extra_args is optional and is split by the shell, so quote paths with spaces.

Example jobs.tsv:
  n3v	cook_spinach	/data/datasets/n3v/cook_spinach	outputs	auto	
  hypernerf	vrig-peel-banana	/data/datasets/hypernerf/vrig/vrig-peel-banana	outputs	auto	--iterations 12000

Example:
  scripts/queue_train.sh jobs.tsv 0,1,2,3 4
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "$#" -lt 2 ]]; then
  usage
  exit 0
fi

JOBS_TSV="$1"
GPU_IDS_CSV="$2"
MAX_PARALLEL="${3:-}"

if [[ ! -f "${JOBS_TSV}" ]]; then
  echo "Job file not found: ${JOBS_TSV}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_POOL <<< "${GPU_IDS_CSV}"
if [[ "${#GPU_POOL[@]}" -eq 0 ]]; then
  echo "No GPU ids provided." >&2
  exit 2
fi
if [[ -z "${MAX_PARALLEL}" ]]; then
  MAX_PARALLEL="${#GPU_POOL[@]}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/queue_logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_ROOT}"

gpu_cursor=0
active_pids=()
active_gpus=()
queue_status=0

array_len() {
  local array_name="$1"
  eval "echo \${#${array_name}[@]}"
}

is_running_pid() {
  local needle="$1"
  local pid
  while read -r pid; do
    [[ "${pid}" == "${needle}" ]] && return 0
  done < <(jobs -r -p)
  return 1
}

reap_finished() {
  local still_running=()
  local still_gpus=()
  local pid
  local gpu
  local status=0
  local idx
  for idx in "${!active_pids[@]}"; do
    pid="${active_pids[$idx]}"
    gpu="${active_gpus[$idx]}"
    if is_running_pid "${pid}"; then
      still_running+=("${pid}")
      still_gpus+=("${gpu}")
    else
      wait "${pid}" || {
        status=$?
        queue_status="${status}"
      }
    fi
  done
  if [[ "${#still_running[@]}" -gt 0 ]]; then
    active_pids=("${still_running[@]}")
  else
    active_pids=()
  fi
  if [[ "${#still_gpus[@]}" -gt 0 ]]; then
    active_gpus=("${still_gpus[@]}")
  else
    active_gpus=()
  fi
  return "${status}"
}

gpu_is_busy() {
  local candidate="$1"
  local gpu
  for gpu in ${active_gpus+"${active_gpus[@]}"}; do
    [[ "${gpu}" == "${candidate}" ]] && return 0
  done
  return 1
}

next_free_auto_gpu() {
  local tries=0
  local gpu
  while [[ "${tries}" -lt "${#GPU_POOL[@]}" ]]; do
    gpu="${GPU_POOL[${gpu_cursor}]}"
    gpu_cursor=$(((gpu_cursor + 1) % ${#GPU_POOL[@]}))
    if ! gpu_is_busy "${gpu}"; then
      echo "${gpu}"
      return 0
    fi
    tries=$((tries + 1))
  done
  return 1
}

wait_for_launch_slot() {
  local requested_gpu="$1"
  while true; do
    reap_finished || true
    if [[ "$(array_len active_pids)" -lt "${MAX_PARALLEL}" ]]; then
      if [[ "${requested_gpu}" != "auto" && -n "${requested_gpu}" ]]; then
        return 0
      fi
      if [[ "$(array_len active_gpus)" -lt "${#GPU_POOL[@]}" ]]; then
        return 0
      fi
    fi
    sleep 5
  done
}

assign_gpu() {
  local requested_gpu="$1"
  if [[ "${requested_gpu}" != "auto" && -n "${requested_gpu}" ]]; then
    echo "${requested_gpu}"
    return 0
  fi
  local gpu
  while true; do
    gpu="$(next_free_auto_gpu || true)"
    if [[ -n "${gpu}" ]]; then
      echo "${gpu}"
      return 0
    fi
    sleep 5
  done
}

launch_job() {
  local dataset="$1"
  local scene="$2"
  local source_path="$3"
  local model_root="$4"
  local requested_gpu="$5"
  local extra_args="$6"
  local gpu_id
  wait_for_launch_slot "${requested_gpu}"
  gpu_id="$(assign_gpu "${requested_gpu}")"
  local tag="${dataset}_${scene}_gpu${gpu_id}_$(date +%H%M%S)"
  local log_path="${LOG_ROOT}/${tag}.log"
  echo "[$(date '+%F %T')] launch ${dataset}/${scene} on GPU ${gpu_id}; log=${log_path}"
  (
    cd "${REPO_ROOT}"
    # shellcheck disable=SC2086
    scripts/train_scene.sh "${dataset}" "${scene}" "${source_path}" "${model_root}" "${gpu_id}" ${extra_args}
  ) >"${log_path}" 2>&1 &
  active_pids+=("$!")
  active_gpus+=("${gpu_id}")
}

while IFS=$'\t' read -r dataset scene source_path model_root gpu_or_auto extra_args rest; do
  [[ -z "${dataset// }" || "${dataset}" =~ ^# ]] && continue
  [[ "${dataset}" == "dataset" && "${scene}" == "scene" ]] && continue
  if [[ -n "${rest:-}" ]]; then
    echo "Ignoring extra TSV columns for ${dataset}/${scene}: ${rest}" >&2
  fi
  launch_job "${dataset}" "${scene}" "${source_path}" "${model_root}" "${gpu_or_auto:-auto}" "${extra_args:-}"
done < "${JOBS_TSV}"

while [[ "$(array_len active_pids)" -gt 0 ]]; do
  reap_finished || true
  sleep 5
done

echo "[$(date '+%F %T')] queue complete; logs=${LOG_ROOT}"
exit "${queue_status}"
