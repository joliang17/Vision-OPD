#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd"
VLMEVAL="${ROOT}/VLMEvalKit"
PYTHON="${PYTHON:-/usr/bin/python}"
RUN_ROOT="${VLMEVAL}/outputs_vllm_curated/remaining_vlmeval_20260630_2038"
SRC_DIR="${RUN_ROOT}/scoring_existing/preds"
OUT="${RUN_ROOT}/visionopd_core_strict_rejudge_0630"
JUDGE="${JUDGE:-gpt-5.4-mini-2026-03-17}"
PROVIDER="${PROVIDER:-tiktok_azure}"
KEY_CONF="${KEY_CONF:-${ROOT}/config/key.conf}"
NPROC="${NPROC:-1}"
RETRY="${RETRY:-10}"
TIMEOUT="${TIMEOUT:-600}"
SLEEP_SECONDS="${SLEEP_SECONDS:-120}"
MODEL="visionopd_qwen3vl2b_step65_temp0_4096_correct"

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

echo "[START] VisionOPD core strict scoring $(date -Is)"
while ! probe_judge; do
  echo "[WAIT] judge not working; retry after ${SLEEP_SECONDS}s: $(date -Is)"
  sleep "${SLEEP_SECONDS}"
done
echo "[READY] judge working: $(date -Is)"

for dataset in BLINK MMStar MMBench_DEV_EN VStarBench MathVista_MINI; do
  src="${SRC_DIR}/${MODEL}_${dataset}_normal.xlsx"
  pred="${OUT}/preds/${MODEL}_${dataset}_strict.xlsx"
  log="${OUT}/logs/${MODEL}_${dataset}_${JUDGE}.log"
  acc="${OUT}/preds/${MODEL}_${dataset}_strict_acc.csv"
  if [[ -f "${acc}" ]]; then
    echo "[SKIP] ${dataset}: ${acc}"
    continue
  fi
  if [[ ! -f "${src}" ]]; then
    echo "[MISS] ${dataset}: ${src}"
    continue
  fi
  cp -f "${src}" "${pred}"
  echo "[SCORING] ${dataset}: ${pred}"
  "${PYTHON}" "${VLMEVAL}/tools/run_normal_eval.py" \
    --dataset "${dataset}" \
    --prediction-file "${pred}" \
    --judge "${JUDGE}" \
    --provider "${PROVIDER}" \
    --nproc "${NPROC}" \
    --retry "${RETRY}" \
    --timeout "${TIMEOUT}" \
    --temperature 0.0 \
    --key-conf "${KEY_CONF}" \
    > "${log}" 2>&1
done

echo "[DONE] VisionOPD core strict scoring $(date -Is)"
