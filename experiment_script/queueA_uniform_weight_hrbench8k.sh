set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-uniform-weight-Qwen3-VL-2B-Instruct/global_step_62" \
MODEL_NAME=uniform_weight_hrbench8k_rerun \
DATASETS=HRBench8K \
GPU_IDS=0 \
bash shell_scripts/eval_via_vllm_server.sh
