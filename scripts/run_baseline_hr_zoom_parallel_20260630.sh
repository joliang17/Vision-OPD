#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd"
VLMEVAL="${ROOT}/VLMEvalKit"
PYTHON="${PYTHON:-/usr/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
RUN_ID="${RUN_ID:-remaining_vlmeval_20260630_2038}"
OUT="${VLMEVAL}/outputs_vllm_curated/${RUN_ID}"
MODEL="qwen3vl2b_base_reference_temp0_4096_hr_zoom"
WORK="${OUT}/qwen3vl2b_base_hr_zoom_eval"
CONFIG="${OUT}/configs/${MODEL}.json"
JUDGE="${JUDGE:-gpt-5.4-mini-2026-03-17}"
PROVIDER="${PROVIDER:-tiktok_azure}"
KEY_CONF="${KEY_CONF:-${ROOT}/config/key.conf}"
API_NPROC="${API_NPROC:-8}"
RETRY="${RETRY:-6}"
TIMEOUT="${TIMEOUT:-600}"
DATASETS=(HRBench4K HRBench8K tdbench_cs_zoom)
LOG="${OUT}/baseline_hr_zoom_parallel.log"

mkdir -p "${WORK}" "${WORK}/normal_scoring"
exec > >(tee -a "${LOG}") 2>&1

export PYTHONPATH="${VLMEVAL}:${PYTHONPATH:-}"
export TRANSFORMERS_CACHE="${ROOT}/../cache/transformers"
export HF_HOME="${ROOT}/../cache/huggingface"
export VLLM_CACHE_ROOT="${ROOT}/../cache/vllm"

echo "[START baseline parallel] $(date -Is)"
echo "GPU_IDS=${GPU_IDS}"
echo "CONFIG=${CONFIG}"
echo "WORK=${WORK}"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" "${VLMEVAL}/run.py" \
  --config "${CONFIG}" \
  --data "${DATASETS[@]}" \
  --work-dir "${WORK}" \
  --mode infer \
  --api-nproc "${API_NPROC}" \
  --retry "${RETRY}" \
  --reuse \
  --reuse-aux all

score_pred() {
  local dataset="$1"
  local src
  src="$(find "${WORK}/${MODEL}" -type f -name "${MODEL}_${dataset}.xlsx" | sort | tail -1 || true)"
  if [[ ! -f "${src}" ]]; then
    echo "[MISS scoring] ${dataset}"
    return 1
  fi
  local pred="${WORK}/normal_scoring/${MODEL}_${dataset}_normal.xlsx"
  local acc="${WORK}/normal_scoring/${MODEL}_${dataset}_normal_acc.csv"
  if [[ -f "${acc}" ]]; then
    echo "[SKIP scoring] ${dataset} ${acc}"
    return 0
  fi
  cp -f "${src}" "${pred}"
  echo "[SCORING] ${dataset}: ${pred}"
  "${PYTHON}" "${VLMEVAL}/tools/run_normal_eval.py" \
    --dataset "${dataset}" \
    --prediction-file "${pred}" \
    --judge "${JUDGE}" \
    --provider "${PROVIDER}" \
    --nproc "${API_NPROC}" \
    --retry "${RETRY}" \
    --timeout "${TIMEOUT}" \
    --temperature 0.0 \
    --key-conf "${KEY_CONF}" \
    > "${WORK}/normal_scoring/${MODEL}_${dataset}_${JUDGE}.log" 2>&1
}

for dataset in "${DATASETS[@]}"; do
  score_pred "${dataset}" || true
done

echo "[DONE baseline parallel] $(date -Is)"
