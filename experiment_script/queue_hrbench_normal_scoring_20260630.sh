#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd"
VLMEVAL="${ROOT}/VLMEvalKit"
PYTHON="${PYTHON:-/usr/bin/python}"
RUN_ROOT="${VLMEVAL}/outputs_vllm_curated/remaining_vlmeval_20260630_2038"
OUT="${RUN_ROOT}/hrbench_rejudge_gpt54_queued_0630"
JUDGE="${JUDGE:-gpt-5.4-mini-2026-03-17}"
PROVIDER="${PROVIDER:-tiktok_azure}"
KEY_CONF="${KEY_CONF:-${ROOT}/config/key.conf}"
NPROC="${NPROC:-8}"
RETRY="${RETRY:-6}"
TIMEOUT="${TIMEOUT:-600}"

mkdir -p "${OUT}/preds" "${OUT}/logs"
exec > >(tee -a "${OUT}/runner.log") 2>&1

export PYTHONPATH="${VLMEVAL}:${PYTHONPATH:-}"
export KEY_CONF="${KEY_CONF}"
export VLMEVAL_REQUIRE_WORKING_JUDGE=1

echo "[START] queued HRBench normal scoring $(date -Is)"
echo "judge=${JUDGE} provider=${PROVIDER} key_conf=${KEY_CONF}"
echo "output=${OUT}"

while pgrep -f 'run_normal_eval.py --dataset MathVista_MINI' >/dev/null; do
  echo "[WAIT] MathVista scoring still running: $(date -Is)"
  sleep 60
done

run_one() {
  local label="$1"
  local dataset="$2"
  local src="$3"
  local dst="${OUT}/preds/${label}_${dataset}_normal.xlsx"
  local acc="${OUT}/preds/${label}_${dataset}_normal_acc.csv"

  if [[ ! -f "${src}" ]]; then
    echo "[MISS] ${label} ${dataset}: ${src}"
    return 0
  fi
  if [[ -f "${acc}" ]]; then
    echo "[SKIP] ${label} ${dataset}: ${acc}"
    return 0
  fi

  cp -f "${src}" "${dst}"
  echo "[SCORING] ${label} ${dataset}: ${dst}"
  "${PYTHON}" "${VLMEVAL}/tools/run_normal_eval.py" \
    --dataset "${dataset}" \
    --prediction-file "${dst}" \
    --judge "${JUDGE}" \
    --provider "${PROVIDER}" \
    --nproc "${NPROC}" \
    --retry "${RETRY}" \
    --timeout "${TIMEOUT}" \
    --temperature 0.0 \
    --key-conf "${KEY_CONF}" \
    > "${OUT}/logs/${label}_${dataset}_${JUDGE}.log" 2>&1
}

BASE_DIR="${RUN_ROOT}/qwen3vl2b_base_hr_zoom_eval/qwen3vl2b_base_reference_temp0_4096_hr_zoom/T20260630-204443"
VISION_DIR="${RUN_ROOT}/visionopd_hr_zoom_eval/visionopd_qwen3vl2b_step65_temp0_4096_hr_zoom/T20260630-203855"

run_one "qwen3vl2b_base_reference_temp0_4096_hr_rejudge" "HRBench4K" \
  "${BASE_DIR}/qwen3vl2b_base_reference_temp0_4096_hr_zoom_HRBench4K.xlsx"
run_one "qwen3vl2b_base_reference_temp0_4096_hr_rejudge" "HRBench8K" \
  "${BASE_DIR}/qwen3vl2b_base_reference_temp0_4096_hr_zoom_HRBench8K.xlsx"
run_one "visionopd_qwen3vl2b_step65_temp0_4096_hr_rejudge" "HRBench8K" \
  "${VISION_DIR}/visionopd_qwen3vl2b_step65_temp0_4096_hr_zoom_HRBench8K.xlsx"

echo "[DONE] queued HRBench normal scoring $(date -Is)"
