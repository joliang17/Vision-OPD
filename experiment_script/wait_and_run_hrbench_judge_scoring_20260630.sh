#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd"
VLMEVAL="${ROOT}/VLMEvalKit"
PYTHON="${PYTHON:-/usr/bin/python}"
RUN_ROOT="${VLMEVAL}/outputs_vllm_curated/remaining_vlmeval_20260630_2038"
OUT="${RUN_ROOT}/hrbench_rejudge_gpt54_wait_0630"
JUDGE="${JUDGE:-gpt-5.4-mini-2026-03-17}"
PROVIDER="${PROVIDER:-tiktok_azure}"
KEY_CONF="${KEY_CONF:-${ROOT}/config/key.conf}"
NPROC="${NPROC:-8}"
RETRY="${RETRY:-6}"
TIMEOUT="${TIMEOUT:-600}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"

mkdir -p "${OUT}/preds" "${OUT}/logs"
exec > >(tee -a "${OUT}/runner.log") 2>&1

export PYTHONPATH="${VLMEVAL}:${PYTHONPATH:-}"
export KEY_CONF="${KEY_CONF}"
export VLMEVAL_REQUIRE_WORKING_JUDGE=1

echo "[START] wait-and-run HRBench judge scoring $(date -Is)"
echo "judge=${JUDGE} provider=${PROVIDER} key_conf=${KEY_CONF}"
echo "out=${OUT}"

judge_ready() {
  cd "${VLMEVAL}"
  "${PYTHON}" - <<PY
from vlmeval.dataset.utils.judge_util import build_judge
m = build_judge(
    model="${JUDGE}",
    provider="${PROVIDER}",
    key_conf="${KEY_CONF}",
    retry=1,
    timeout=60,
    temperature=0.0,
)
raise SystemExit(0 if m.working() else 1)
PY
}

until judge_ready; do
  echo "[WAIT] judge not working; retry after ${SLEEP_SECONDS}s: $(date -Is)"
  sleep "${SLEEP_SECONDS}"
done

echo "[READY] judge working: $(date -Is)"

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

run_one "qwen3vl2b_base_reference_temp0_4096_hr_judge" "HRBench4K" \
  "${BASE_DIR}/qwen3vl2b_base_reference_temp0_4096_hr_zoom_HRBench4K.xlsx"
run_one "qwen3vl2b_base_reference_temp0_4096_hr_judge" "HRBench8K" \
  "${BASE_DIR}/qwen3vl2b_base_reference_temp0_4096_hr_zoom_HRBench8K.xlsx"
run_one "visionopd_qwen3vl2b_step65_temp0_4096_hr_judge" "HRBench8K" \
  "${VISION_DIR}/visionopd_qwen3vl2b_step65_temp0_4096_hr_zoom_HRBench8K.xlsx"

echo "[DONE] wait-and-run HRBench judge scoring $(date -Is)"
