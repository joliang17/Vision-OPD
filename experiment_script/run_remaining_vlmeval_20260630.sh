#!/usr/bin/env bash
set -euo pipefail

ROOT="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd"
VLMEVAL="${ROOT}/VLMEvalKit"
PYTHON="${PYTHON:-/usr/bin/python}"
GPU_IDS="${GPU_IDS:-2}"
JUDGE="${JUDGE:-gpt-5.4-mini-2026-03-17}"
PROVIDER="${PROVIDER:-tiktok_azure}"
KEY_CONF="${KEY_CONF:-${ROOT}/config/key.conf}"
API_NPROC="${API_NPROC:-8}"
RETRY="${RETRY:-6}"
TIMEOUT="${TIMEOUT:-600}"

RUN_ID="${RUN_ID:-remaining_vlmeval_20260630_$(date +%H%M%S)}"
OUT="${VLMEVAL}/outputs_vllm_curated/${RUN_ID}"
LOG="${OUT}/driver.log"

BASE_MODEL_PATH="${ROOT}/../cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203"
VISIONOPD_MODEL_PATH="${ROOT}/Vision-OPD/checkpoints/Vision-OPD-visionopd-Qwen3-VL-2B-Instruct/global_step_65"
VISION_EXISTING_WORK="${VLMEVAL}/outputs_vllm_curated/visionopd_correct_eval_and_base_hrbench_20260630_023111/visionopd_qwen3vl2b_step65_temp0_4096_eval"
VISION_EXISTING_MODEL="visionopd_qwen3vl2b_step65_temp0_4096_correct"
VISION_EXISTING_PRED_DIR="${VISION_EXISTING_WORK}/${VISION_EXISTING_MODEL}/T20260630-023118"

VISION_MODEL="visionopd_qwen3vl2b_step65_temp0_4096_hr_zoom"
BASE_MODEL="qwen3vl2b_base_reference_temp0_4096_hr_zoom"

CORE_DATASETS=(BLINK MMStar MMBench_DEV_EN VStarBench MathVista_MINI)
EXTRA_DATASETS=(HRBench4K HRBench8K tdbench_cs_zoom)

mkdir -p "${OUT}/configs" "${OUT}/logs" "${OUT}/scoring_existing" "${OUT}/summary"
exec > >(tee -a "${LOG}") 2>&1

export PYTHONPATH="${VLMEVAL}:${PYTHONPATH:-}"
export TRANSFORMERS_CACHE="${ROOT}/../cache/transformers"
export HF_HOME="${ROOT}/../cache/huggingface"
export VLLM_CACHE_ROOT="${ROOT}/../cache/vllm"

echo "[START] ${RUN_ID} $(date -Is)"
echo "VLMEvalKit: ${VLMEVAL}"
echo "GPU_IDS: ${GPU_IDS}"
echo "judge: ${JUDGE}, provider: ${PROVIDER}, key_conf: ${KEY_CONF}"
echo "ZoomBench alias: tdbench_cs_zoom"

cat > "${OUT}/configs/${VISION_MODEL}.json" <<JSON
{
  "model": {
    "${VISION_MODEL}": {
      "class": "Qwen3VLChat",
      "model_path": "${VISIONOPD_MODEL_PATH}",
      "use_custom_prompt": false,
      "use_vllm": true,
      "temperature": 0.0,
      "max_new_tokens": 4096,
      "repetition_penalty": 1.0,
      "presence_penalty": 1.5,
      "top_p": 0.8,
      "top_k": 20
    }
  }
}
JSON

cat > "${OUT}/configs/${BASE_MODEL}.json" <<JSON
{
  "model": {
    "${BASE_MODEL}": {
      "class": "Qwen3VLChat",
      "model_path": "${BASE_MODEL_PATH}",
      "use_custom_prompt": false,
      "use_vllm": true,
      "temperature": 0.0,
      "max_new_tokens": 4096,
      "repetition_penalty": 1.0,
      "presence_penalty": 1.5,
      "top_p": 0.8,
      "top_k": 20
    }
  }
}
JSON

score_pred() {
  local dataset="$1"
  local src="$2"
  local out_dir="$3"
  local stem="$4"

  mkdir -p "${out_dir}/preds"
  local pred="${out_dir}/preds/${stem}_${dataset}_normal.xlsx"
  local acc="${out_dir}/preds/${stem}_${dataset}_normal_acc.csv"
  if [[ -f "${acc}" ]]; then
    echo "[SKIP scoring] ${dataset} exists: ${acc}"
    return 0
  fi
  if [[ ! -f "${src}" ]]; then
    echo "[MISS scoring] ${dataset}: ${src}"
    return 1
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
    > "${out_dir}/${stem}_${dataset}_${JUDGE}.log" 2>&1
}

score_existing_visionopd_core() {
  echo "[PHASE] score existing VisionOPD core predictions"
  for ds in "${CORE_DATASETS[@]}"; do
    score_pred \
      "${ds}" \
      "${VISION_EXISTING_PRED_DIR}/${VISION_EXISTING_MODEL}_${ds}.xlsx" \
      "${OUT}/scoring_existing" \
      "${VISION_EXISTING_MODEL}" || true
  done
}

run_infer() {
  local model="$1"
  local config="$2"
  local work="$3"
  shift 3
  local datasets=("$@")
  mkdir -p "${work}"
  echo "[INFER] ${model}: ${datasets[*]}"
  CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" "${VLMEVAL}/run.py" \
    --config "${config}" \
    --data "${datasets[@]}" \
    --work-dir "${work}" \
    --mode infer \
    --api-nproc "${API_NPROC}" \
    --retry "${RETRY}" \
    --reuse \
    --reuse-aux all
}

score_infer_outputs() {
  local model="$1"
  local work="$2"
  shift 2
  local datasets=("$@")
  local score_dir="${work}/normal_scoring"
  mkdir -p "${score_dir}"
  for ds in "${datasets[@]}"; do
    local pred
    pred="$(find "${work}/${model}" -type f -name "${model}_${ds}.xlsx" | sort | tail -1 || true)"
    score_pred "${ds}" "${pred}" "${score_dir}" "${model}" || true
  done
}

summarize() {
  echo "[PHASE] summarize"
  "${PYTHON}" - <<PY
from pathlib import Path
import csv

out = Path("${OUT}")
rows = []
for acc in sorted(out.rglob("*_acc.csv")):
    try:
        with acc.open() as f:
            data = list(csv.reader(f))
    except Exception as exc:
        rows.append([str(acc), "READ_ERROR", str(exc)])
        continue
    if len(data) >= 2 and len(data[0]) >= 2:
        rows.append([str(acc), data[0][-1], data[1][-1]])
    elif data:
        rows.append([str(acc), "raw", " | ".join(",".join(r) for r in data[:3])])
    else:
        rows.append([str(acc), "empty", ""])

summary = out / "summary" / "acc_files.csv"
summary.parent.mkdir(parents=True, exist_ok=True)
with summary.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["acc_file", "metric", "value"])
    w.writerows(rows)
print(f"summary={summary}")
for row in rows:
    print(",".join(row))
PY
}

score_existing_visionopd_core &
SCORING_PID=$!

VISION_WORK="${OUT}/visionopd_hr_zoom_eval"
BASE_WORK="${OUT}/qwen3vl2b_base_hr_zoom_eval"

run_infer "${VISION_MODEL}" "${OUT}/configs/${VISION_MODEL}.json" "${VISION_WORK}" "${EXTRA_DATASETS[@]}"
score_infer_outputs "${VISION_MODEL}" "${VISION_WORK}" "${EXTRA_DATASETS[@]}"

run_infer "${BASE_MODEL}" "${OUT}/configs/${BASE_MODEL}.json" "${BASE_WORK}" "${EXTRA_DATASETS[@]}"
score_infer_outputs "${BASE_MODEL}" "${BASE_WORK}" "${EXTRA_DATASETS[@]}"

wait "${SCORING_PID}" || true
summarize

echo "[DONE] ${RUN_ID} $(date -Is)"
