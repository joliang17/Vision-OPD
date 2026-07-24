set -euo pipefail
# greedy 无 presence penalty 版 eval：temperature=0, presence=0, rep=1.0, max_new_tokens=32768
# 与 sample_t07_16k / greedy_16k 两个脚本成组，用于对比采样参数的影响。

OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CKPT_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30

cd "${OPSD_ROOT}/Vision-OPD"
# 多个任务共用同一 ckpt，避免并发重复 merge 互相踩：已 merge（有 config.json）则跳过
if [ ! -f "${CKPT_DIR}/config.json" ]; then
  bash scripts/merge_checkpoint.sh "${CKPT_DIR}"
fi

cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/${CKPT_DIR}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30_greedy_p0_32k \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
GPU_IDS=0 \
TEMPERATURE=0.0 \
PRESENCE_PENALTY=0.0 \
REPETITION_PENALTY=1.0 \
MAX_NEW_TOKENS=32768 \
bash shell_scripts/eval_via_vllm_server.sh
