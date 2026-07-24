#!/usr/bin/env bash
set -euo pipefail

VLMEVAL_ROOT="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/VLMEvalKit"
RUN_ROOT="${VLMEVAL_ROOT}/outputs_vllm_curated/visionopd_correct_eval_and_base_hrbench_20260630_023111"
VISION_WORK_DIR="${RUN_ROOT}/visionopd_qwen3vl2b_step65_temp0_4096_eval"
VISION_MODEL="visionopd_qwen3vl2b_step65_temp0_4096_correct"
VISION_T_DIR="${VISION_WORK_DIR}/${VISION_MODEL}/T20260630-023118"
NORMAL_DIR="${VISION_WORK_DIR}/normal_scoring"
HR_WORK_DIR="${RUN_ROOT}/qwen3vl2b_base_reference_temp0_4096_hrbench_eval"
LOG="${RUN_ROOT}/resume_scoring_hrbench.log"

JUDGE="gpt-5.4-mini-2026-03-17"
JUDGE_PROVIDER="tiktok_azure"
KEY_CONF="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/config/key.conf"

cd "${VLMEVAL_ROOT}"
mkdir -p "${NORMAL_DIR}" "${HR_WORK_DIR}/configs"

{
  echo "[START] resume VisionOPD normal scoring $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  for ds in BLINK MMStar MMBench_DEV_EN VStarBench MathVista_MINI; do
    src="${VISION_T_DIR}/${VISION_MODEL}_${ds}.xlsx"
    pred="${NORMAL_DIR}/${VISION_MODEL}_${ds}_normal.xlsx"
    score_log="${NORMAL_DIR}/${VISION_MODEL}_${ds}_normal_${JUDGE}.log"
    if [[ ! -f "${src}" ]]; then
      echo "Missing prediction file: ${src}" >&2
      exit 1
    fi
    cp -f "${src}" "${pred}"
    echo "[SCORING] ${ds}: ${pred}"
    /usr/bin/python -u tools/run_normal_eval.py \
      --dataset "${ds}" \
      --prediction-file "${pred}" \
      --judge "${JUDGE}" \
      --provider "${JUDGE_PROVIDER}" \
      --nproc 8 \
      --retry 6 \
      --timeout 600 \
      --temperature 0.0 \
      --max-tokens 2048 \
      --key-conf "${KEY_CONF}" > "${score_log}" 2>&1
    perl -0pi -e 's/API Key: [A-Za-z0-9_\-]+/API Key: [REDACTED]/g; s/sk-[A-Za-z0-9_\-]+/[REDACTED]/g' "${score_log}" || true
    grep 'RESULT_JSON=' "${score_log}" || true
  done

  echo "[START] base Qwen3-VL-2B-Instruct HRBench $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  cat > "${HR_WORK_DIR}/configs/qwen3vl2b_base_reference_temp0_4096_hrbench.json" <<'JSON'
{
  "model": {
    "qwen3vl2b_base_reference_temp0_4096_hrbench": {
      "class": "Qwen3VLChat",
      "model_path": "/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd/../cache/transformers/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203",
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

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
  export PYTHONPATH="${VLMEVAL_ROOT}:${PYTHONPATH:-}"
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  export LMUData="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/simple-mmeval/datasets"
  export HF_HOME="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache"
  export HF_HUB_CACHE="/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/cache/transformers"
  export HUGGINGFACE_HUB_CACHE="${HF_HUB_CACHE}"
  export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"

  /usr/bin/python -u run.py \
    --config "${HR_WORK_DIR}/configs/qwen3vl2b_base_reference_temp0_4096_hrbench.json" \
    --data HRBench4K HRBench8K \
    --work-dir "${HR_WORK_DIR}" \
    --mode all \
    --api-nproc 8 \
    --retry 6 \
    --judge "${JUDGE}" \
    --judge-args '{"provider":"tiktok_azure"}' \
    --judge-api-nproc 8 \
    --judge-retry 6 \
    --judge-timeout 600 \
    --reuse \
    --reuse-aux all

  echo "[DONE] resume scoring and HRBench complete $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} 2>&1 | tee -a "${LOG}"
