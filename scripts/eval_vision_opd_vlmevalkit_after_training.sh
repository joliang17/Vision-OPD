#!/usr/bin/env bash

set -euo pipefail

# Wait for the current Vision-OPD training process, merge the final FSDP
# checkpoint, serve it with vLLM, and evaluate with VLMEvalKit.
#
# Common overrides:
#   TRAIN_PID=857384 bash scripts/eval_vision_opd_vlmevalkit_after_training.sh
#   VLMEVALKIT_DIR=/path/to/VLMEvalKit VLM_DATASETS="MMStar VStarBench" bash ...
#   CHECKPOINT_ROOT=/path/to/checkpoints/Vision-OPD-Qwen3-VL-4B-Instruct bash ...

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ "${SETUP_PROXY:-0}" == "1" ]]; then
  if [[ -n "${PROXY_SETUP_SCRIPT:-}" ]]; then
    bash "${PROXY_SETUP_SCRIPT}"
  fi
  export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:7890}"
  export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:7890}"
  export http_proxy="${http_proxy:-${HTTP_PROXY}}"
  export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
  export ALL_PROXY="${ALL_PROXY:-socks5h://127.0.0.1:1080}"
  export all_proxy="${all_proxy:-${ALL_PROXY}}"
  export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
  export no_proxy="${no_proxy:-${NO_PROXY}}"
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-Vision-OPD-Qwen3-VL-4B-Instruct}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
MERGED_CKPT="${MERGED_CKPT:-}"
TRAIN_PID="${TRAIN_PID:-857384}"

VLM_DATASETS="${VLM_DATASETS:-MMStar VStarBench}"
VLM_WORK_DIR="${VLM_WORK_DIR:-${PROJECT_ROOT}/vlmevalkit_outputs/${EXPERIMENT_NAME}}"
VLMEVALKIT_DIR="${VLMEVALKIT_DIR:-${PROJECT_ROOT}/external/VLMEvalKit}"
INSTALL_VLMEVALKIT="${INSTALL_VLMEVALKIT:-0}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${EXPERIMENT_NAME}}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_TP_SIZE="${VLLM_TP_SIZE:-$(python3 - <<'PY'
import torch
print(torch.cuda.device_count() or 1)
PY
)}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-9216}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-${PROJECT_ROOT}/logs/vlmevalkit}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-${VLLM_LOG_DIR}/vllm_${EXPERIMENT_NAME}.log}"
RUN_LOG_FILE="${RUN_LOG_FILE:-${VLLM_LOG_DIR}/run_${EXPERIMENT_NAME}.log}"

JUDGE_MODEL="${JUDGE_MODEL:-exact_matching}"
API_NPROC="${API_NPROC:-32}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
TIMEOUT="${TIMEOUT:-1800}"
EXTRA_RUN_ARGS="${EXTRA_RUN_ARGS:-}"

mkdir -p "${VLLM_LOG_DIR}" "${VLM_WORK_DIR}" "${PROJECT_ROOT}/external"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

wait_for_training() {
  if [[ -z "${TRAIN_PID}" ]]; then
    log "TRAIN_PID is empty; skipping process wait."
    return
  fi

  if ! kill -0 "${TRAIN_PID}" 2>/dev/null; then
    log "Training PID ${TRAIN_PID} is not running; continuing."
    return
  fi

  log "Waiting for training PID ${TRAIN_PID} to finish."
  while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 60
  done
  log "Training PID ${TRAIN_PID} finished."
}

find_latest_checkpoint() {
  if [[ -n "${MERGED_CKPT}" ]]; then
    echo "${MERGED_CKPT}"
    return
  fi

  if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    log "Checkpoint root does not exist yet: ${CHECKPOINT_ROOT}" >&2
    exit 1
  fi

  local latest
  latest="$(
    find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' \
      | sort -V \
      | tail -n 1
  )"
  if [[ -z "${latest}" ]]; then
    log "No global_step_* checkpoint found under ${CHECKPOINT_ROOT}" >&2
    exit 1
  fi
  echo "${latest%/}"
}

merge_checkpoint_if_needed() {
  local ckpt_dir="$1"
  if [[ -f "${ckpt_dir}/config.json" ]] && \
     [[ -f "${ckpt_dir}/model.safetensors" || -f "${ckpt_dir}/model.safetensors.index.json" || -f "${ckpt_dir}/pytorch_model.bin.index.json" ]]; then
    log "Checkpoint already looks merged: ${ckpt_dir}"
    return
  fi

  log "Merging checkpoint: ${ckpt_dir}"
  bash "${PROJECT_ROOT}/scripts/merge_checkpoint.sh" "${ckpt_dir}"
}

ensure_vlmevalkit() {
  if python3 -c 'import vlmeval' >/dev/null 2>&1; then
    log "Using installed VLMEvalKit Python package."
    return
  fi

  if [[ ! -d "${VLMEVALKIT_DIR}/.git" ]]; then
    log "Cloning VLMEvalKit into ${VLMEVALKIT_DIR}"
    git clone https://github.com/open-compass/VLMEvalKit.git "${VLMEVALKIT_DIR}"
  fi

  if [[ "${INSTALL_VLMEVALKIT}" == "1" ]]; then
    log "Installing VLMEvalKit in editable mode without dependency upgrades."
    python3 -m pip install -e "${VLMEVALKIT_DIR}" --no-deps
  fi
}

wait_for_vllm() {
  local base_url="http://${VLLM_HOST}:${VLLM_PORT}/v1"
  log "Waiting for vLLM API at ${base_url}"
  for _ in $(seq 1 120); do
    if python3 - "${base_url}" <<'PY' >/dev/null 2>&1
import sys
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url=sys.argv[1])
client.models.list()
PY
    then
      log "vLLM API is ready."
      return
    fi
    sleep 10
  done
  log "Timed out waiting for vLLM. See ${VLLM_LOG_FILE}" >&2
  exit 1
}

start_vllm() {
  local ckpt_dir="$1"
  if python3 - "http://${VLLM_HOST}:${VLLM_PORT}/v1" <<'PY' >/dev/null 2>&1
import sys
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url=sys.argv[1])
client.models.list()
PY
  then
    log "Reusing existing vLLM API on ${VLLM_HOST}:${VLLM_PORT}."
    return
  fi

  log "Starting vLLM for ${ckpt_dir}"
  nohup vllm serve "${ckpt_dir}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${VLLM_TP_SIZE}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --trust-remote-code \
    >"${VLLM_LOG_FILE}" 2>&1 &
  echo $! > "${VLLM_LOG_DIR}/vllm_${EXPERIMENT_NAME}.pid"
  wait_for_vllm
}

run_vlmevalkit() {
  local base_url="http://${VLLM_HOST}:${VLLM_PORT}/v1"
  local run_py="${VLMEVALKIT_DIR}/run.py"
  if [[ ! -f "${run_py}" ]]; then
    run_py="$(python3 - <<'PY'
import pathlib
import vlmeval
print(pathlib.Path(vlmeval.__file__).resolve().parents[1] / 'run.py')
PY
)"
  fi

  log "Running VLMEvalKit datasets: ${VLM_DATASETS}"
  # shellcheck disable=SC2086
  python3 "${run_py}" \
    --data ${VLM_DATASETS} \
    --model "${SERVED_MODEL_NAME}" \
    --base-url "${base_url}" \
    --key EMPTY \
    --work-dir "${VLM_WORK_DIR}" \
    --api-nproc "${API_NPROC}" \
    --judge "${JUDGE_MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --timeout "${TIMEOUT}" \
    --verbose \
    ${EXTRA_RUN_ARGS} \
    2>&1 | tee -a "${RUN_LOG_FILE}"
}

main() {
  log "Vision-OPD VLMEvalKit evaluation wrapper started."
  wait_for_training
  local ckpt_dir
  ckpt_dir="$(find_latest_checkpoint)"
  merge_checkpoint_if_needed "${ckpt_dir}"
  ensure_vlmevalkit
  start_vllm "${ckpt_dir}"
  run_vlmevalkit
  log "VLMEvalKit evaluation finished."
}

main "$@"
