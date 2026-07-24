set -euo pipefail
# 注意：不要在这里 unset 代理变量 —— eval_via_vllm_server.sh 需要保存原始代理值、
# 在最后 judge 阶段恢复；两个下游脚本都会自己按阶段管理代理。

OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CKPT_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30

cd "${OPSD_ROOT}/Vision-OPD"
bash scripts/merge_checkpoint.sh "${CKPT_DIR}"

cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/${CKPT_DIR}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30 \
DATASETS=BLINK,MMStar,MMBench_DEV_EN,VStarBench,MathVista_MINI,HRBench4K,HRBench8K,POPE,HallusionBench \
GPU_IDS=0 \
bash shell_scripts/eval_via_vllm_server.sh

cd "${OPSD_ROOT}/Vision-OPD"
bash scripts/run_zoombench_canonical.sh \
  "${CKPT_DIR}" \
  contrast_standard_virl39k_90step_step30 0 8010
