set -euo pipefail
OPSD_ROOT=/mnt/bn/tns-algo-video-vlm-ruby/yijunliang/project/opsd
cd "${OPSD_ROOT}/VLMEvalKit"
pip3 install -r requirements_arnold.txt
MODEL_PATH="${OPSD_ROOT}/Vision-OPD/checkpoints/Vision-OPD-contrast-standard-Qwen3-VL-2B-Instruct-sr1-filtered/global_step_140" \
MODEL_NAME=std_sr1_step140_hrbench8k_rerun \
DATASETS=HRBench8K \
GPU_IDS=0 \
bash shell_scripts/eval_via_vllm_server.sh
