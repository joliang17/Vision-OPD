#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd"
VLMEVAL="${ROOT}/VLMEvalKit"
PYTHON="${PYTHON:-/usr/bin/python}"
RUN_ROOT="${VLMEVAL}/outputs_vllm_curated/remaining_vlmeval_20260630_2038"
CONFIG="${RUN_ROOT}/configs/visionopd_qwen3vl2b_step65_temp0_4096_hr_zoom.json"
WORK="${RUN_ROOT}/visionopd_hr4k_rerun_gpu2"
GPU_IDS="${GPU_IDS:-2}"
API_NPROC="${API_NPROC:-8}"
RETRY="${RETRY:-6}"

mkdir -p "${WORK}"
export PYTHONPATH="${VLMEVAL}:${PYTHONPATH:-}"
export TRANSFORMERS_CACHE="${ROOT}/../cache/transformers"
export HF_HOME="${ROOT}/../cache/huggingface"
export VLLM_CACHE_ROOT="${ROOT}/../cache/vllm"
export LMUData="${LMUData:-/home/tiger/LMUData}"

echo "[START] VisionOPD HRBench4K inference on GPU ${GPU_IDS} $(date -Is)"
echo "LMUData=${LMUData}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" "${VLMEVAL}/run.py" \
  --config "${CONFIG}" \
  --data HRBench4K \
  --work-dir "${WORK}" \
  --mode infer \
  --api-nproc "${API_NPROC}" \
  --retry "${RETRY}" \
  --reuse \
  --reuse-aux all
echo "[DONE] VisionOPD HRBench4K inference $(date -Is)"
