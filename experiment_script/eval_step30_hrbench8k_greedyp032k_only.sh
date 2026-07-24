set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
CKPT_DIR=checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-virl39k-filtered-90step/global_step_30
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/${CKPT_DIR}" \
MODEL_NAME=contrast_standard_virl39k_90step_step30_greedy_p0_32k \
DATASETS=HRBench8K GPU_IDS=0 \
TEMPERATURE=0.0 PRESENCE_PENALTY=0.0 REPETITION_PENALTY=1.0 MAX_NEW_TOKENS=32768 \
bash shell_scripts/eval_via_vllm_server.sh
