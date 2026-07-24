set -euo pipefail
# 补跑剩余部分：默认参数组的 HRBench8K 已在上一个任务(6c8f6d7ef5b1d225)成功产出
# (75.125%)，本脚本只跑剩下 3 组采样设置的 HRBench8K + ZoomBench canonical 重跑。

OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CKPT_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30

cd "${OPSD_ROOT}/Vision-OPD"
if [ ! -f "${CKPT_DIR}/config.json" ]; then
  bash scripts/merge_checkpoint.sh "${CKPT_DIR}"
fi

cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
MODEL_PATH_ABS="${OPSD_ROOT}/Vision-OPD/${CKPT_DIR}"

MODEL_PATH="${MODEL_PATH_ABS}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30_t07_16k \
DATASETS=HRBench8K GPU_IDS=0 \
TEMPERATURE=0.7 TOP_P=0.8 TOP_K=20 PRESENCE_PENALTY=1.5 REPETITION_PENALTY=1.0 MAX_NEW_TOKENS=16384 \
bash shell_scripts/eval_via_vllm_server.sh

MODEL_PATH="${MODEL_PATH_ABS}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30_greedy_16k \
DATASETS=HRBench8K GPU_IDS=0 \
TEMPERATURE=0.0 PRESENCE_PENALTY=1.5 REPETITION_PENALTY=1.0 MAX_NEW_TOKENS=16384 \
bash shell_scripts/eval_via_vllm_server.sh

MODEL_PATH="${MODEL_PATH_ABS}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30_greedy_p0_32k \
DATASETS=HRBench8K GPU_IDS=0 \
TEMPERATURE=0.0 PRESENCE_PENALTY=0.0 REPETITION_PENALTY=1.0 MAX_NEW_TOKENS=32768 \
bash shell_scripts/eval_via_vllm_server.sh

cd "${OPSD_ROOT}/Vision-OPD"
bash scripts/run_zoombench_canonical.sh \
  "${CKPT_DIR}" \
  contrast_standard_virl39k_90step_step30 0 8010
