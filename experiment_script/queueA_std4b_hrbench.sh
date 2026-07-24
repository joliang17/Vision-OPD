set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-4B-Instruct/global_step_62" \
MODEL_NAME=std4b_hrbench_rerun \
DATASETS=HRBench4K,HRBench8K \
GPU_IDS=0 \
bash shell_scripts/eval_via_vllm_server.sh
