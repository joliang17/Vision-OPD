#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd"
VLMEVAL="${ROOT}/VLMEvalKit"
PYTHON="${PYTHON:-/usr/bin/python}"
RUN_ROOT="${VLMEVAL}/outputs_vllm_curated/remaining_vlmeval_20260630_2038"
SRC="${RUN_ROOT}/visionopd_hr4k_rerun_gpu2/visionopd_qwen3vl2b_step65_temp0_4096_hr_zoom/T20260630-235534/visionopd_qwen3vl2b_step65_temp0_4096_hr_zoom_HRBench4K.xlsx"
OUT="${RUN_ROOT}/visionopd_hr4k_strict_rejudge_0701"
PRED="${OUT}/preds/visionopd_qwen3vl2b_step65_temp0_4096_hr_judge_HRBench4K_normal.xlsx"
JUDGE="${JUDGE:-gpt-5.4-mini-2026-03-17}"
PROVIDER="${PROVIDER:-tiktok_azure}"
KEY_CONF="${KEY_CONF:-${ROOT}/config/key.conf}"
NPROC="${NPROC:-1}"
RETRY="${RETRY:-10}"
TIMEOUT="${TIMEOUT:-600}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"

mkdir -p "${OUT}/preds" "${OUT}/logs"
export PYTHONPATH="${VLMEVAL}:${PYTHONPATH:-}"
export TRANSFORMERS_CACHE="${ROOT}/../cache/transformers"
export HF_HOME="${ROOT}/../cache/huggingface"
export VLLM_CACHE_ROOT="${ROOT}/../cache/vllm"
export VLMEVAL_REQUIRE_WORKING_JUDGE=1

probe_judge() {
  "${PYTHON}" - <<PY
from vlmeval.dataset.utils import build_judge
model = build_judge(
    model="${JUDGE}",
    provider="${PROVIDER}",
    key_conf="${KEY_CONF}",
    temperature=0.0,
    timeout=60,
    retry=1,
)
raise SystemExit(0 if model.working() else 1)
PY
}

echo "[START] VisionOPD HRBench4K strict scoring $(date -Is)"
cp -f "${SRC}" "${PRED}"
while ! probe_judge; do
  echo "[WAIT] judge not working; retry after ${SLEEP_SECONDS}s: $(date -Is)"
  sleep "${SLEEP_SECONDS}"
done
echo "[READY] judge working: $(date -Is)"

"${PYTHON}" "${VLMEVAL}/tools/run_normal_eval.py" \
  --dataset HRBench4K \
  --prediction-file "${PRED}" \
  --judge "${JUDGE}" \
  --provider "${PROVIDER}" \
  --nproc "${NPROC}" \
  --retry "${RETRY}" \
  --timeout "${TIMEOUT}" \
  --temperature 0.0 \
  --key-conf "${KEY_CONF}" \
  > "${OUT}/logs/visionopd_qwen3vl2b_step65_temp0_4096_hr_judge_HRBench4K_${JUDGE}.log" 2>&1

echo "[DONE] VisionOPD HRBench4K strict scoring $(date -Is)"
